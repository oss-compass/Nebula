import asyncio
import time
import aiohttp
from typing import Any, Dict, List
from .config import API_URL, MODEL, API_KEY, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, DEFAULT_MAX_TIMEOUT
from .complexity import calculate_complexity_score
from .context import generate_context_summary, generate_context_summary_fast


def extract_comments_summary(comments: List[Dict]) -> str:
    """Extract minimal comment information to reduce token usage"""
    if not comments:
        return "no_comments"
    
    # Only take the first comment and limit its length
    for comment in comments[:1]:  # Only first comment
        text = comment.get("text", "").strip()
        if text:
            # Clean comment symbols and limit length
            text = text.lstrip("#").lstrip("//").lstrip("/*").rstrip("*/").strip()
            if text:
                return text[:50]  # Very short limit
    
    return "no_comments"


def detect_language_from_file(file_path: str) -> str:
    """Detect language based on file path"""
    from pathlib import Path
    from .config import LANGUAGE_MAP
    
    ext = Path(file_path).suffix.lower()
    return LANGUAGE_MAP.get(ext, "python")


def _parse_docstring(docstring: str) -> Dict[str, str]:
    """解析docstring，拆分成描述、参数和返回值三个部�?
    
    Args:
        docstring: 完整的docstring字符�?
        
    Returns:
        包含三个部分的字�?
    """
    try:
        # 初始化结�?
        result = {
            "description": "",
            "args": "",
            "returns": ""
        }
        
        if not docstring:
            return result
        
        # 按行分割
        lines = docstring.strip().split('\n')
        
        current_section = "description"
        section_content = []
        
        for line in lines:
            line = line.strip()
            
            # 跳过空行
            if not line:
                continue
            
            # 检测新的部�?
            if line.lower().startswith('args:'):
                # 保存之前的描述部�?
                if section_content:
                    result["description"] = '\n'.join(section_content).strip()
                current_section = "args"
                section_content = []
                continue
            elif line.lower().startswith('returns:'):
                # 保存之前的参数部�?
                if section_content:
                    result["args"] = '\n'.join(section_content).strip()
                current_section = "returns"
                section_content = []
                continue
            elif line.lower().startswith('raises:') or line.lower().startswith('note:') or line.lower().startswith('example:'):
                # 其他部分，归入描�?
                if current_section == "description":
                    section_content.append(line)
                continue
            
            # 添加到当前部�?
            section_content.append(line)
        
        # 保存最后一个部�?
        if section_content:
            if current_section == "description":
                result["description"] = '\n'.join(section_content).strip()
            elif current_section == "args":
                result["args"] = '\n'.join(section_content).strip()
            elif current_section == "returns":
                result["returns"] = '\n'.join(section_content).strip()
        
        # 如果没有明确的分段，尝试智能分割
        if not result["args"] and not result["returns"]:
            # 尝试查找参数和返回值的模式
            content = result["description"]
            
            # 查找参数部分（通常在描述后的第一段）
            if "(" in content and ")" in content:
                # 简单的启发式分�?
                parts = content.split('\n\n', 1)
                if len(parts) > 1:
                    result["description"] = parts[0].strip()
                    remaining = parts[1].strip()
                    
                    # 尝试进一步分割参数和返回�?
                    if "Args:" in remaining or "Parameters:" in remaining:
                        args_start = remaining.find("Args:") if "Args:" in remaining else remaining.find("Parameters:")
                        if args_start != -1:
                            args_end = remaining.find("Returns:", args_start) if "Returns:" in remaining else len(remaining)
                            result["args"] = remaining[args_start:args_end].strip()
                            
                            if "Returns:" in remaining:
                                returns_start = remaining.find("Returns:")
                                result["returns"] = remaining[returns_start:].strip()
        
        return result
        
    except Exception as e:
        # 如果解析失败，返回原始内容作为描�?
        return {
            "description": docstring,
            "args": "",
            "returns": ""
        }


async def call_model_batch(session: aiohttp.ClientSession, functions_batch: List[Dict], all_functions_data: List[Dict] = None, complexity_cache: Dict = None) -> List[Dict[str, Any]]:
    """批量调用AI模型，一次处理多个函�?""
    if not API_KEY:
        raise EnvironmentError("GITEE_API_KEY is not defined")
    
    from .config import SYSTEM_PROMPT, BATCH_PROMPT_TEMPLATE
    
    # 构建批量提示
    batch_prompt = BATCH_PROMPT_TEMPLATE
    
    batch_data = []
    for func in functions_batch:
        # 使用预计算的复杂度信息或重新计算
        if complexity_cache and id(func) in complexity_cache:
            complexity_info = complexity_cache[id(func)]
        else:
            complexity_info = calculate_complexity_score(func, all_functions_data)
        context_summary = generate_context_summary_fast(func)
        
        function_name = func.get('basic_info', {}).get('function_name', 'unknown')
        source_code = func.get('basic_info', {}).get('source_code', '')[:800]  # 限制长度
        comments = extract_comments_summary(func.get('basic_info', {}).get('comments', []))
        
        batch_prompt += f"函数: {function_name}\n复杂�? {complexity_info['complexity_level']}\n上下�? {context_summary}\n代码: {source_code}\n注释: {comments}\n---\n"
        
        batch_data.append({
            'function_info': func,
            'complexity_info': complexity_info,
            'context_summary': context_summary
        })
    
    # 调用模型
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n请按顺序为每个函数生成Docstring，用---分隔�?},
            {"role": "user", "content": batch_prompt}
        ],
        "temperature": 0.2
    }
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    try:
        async with session.post(API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            resp.raise_for_status()
            result = await resp.json()
            
            # 解析批量响应
            content = result["choices"][0]["message"]["content"]
            docstrings = content.split('---')
            
            # 处理结果
            results = []
            for i, (docstring, batch_item) in enumerate(zip(docstrings, batch_data)):
                if i < len(batch_data):
                    parsed_docstring = _parse_docstring(docstring.strip())
                    results.append({
                        "docstring": parsed_docstring,
                        "duration(s)": 0,  # 批量处理无法精确计时
                        "complexity_info": batch_item['complexity_info'],
                        "context_summary": batch_item['context_summary'],
                        "function_name": batch_item['function_info']['basic_info']['function_name']
                    })
            
            return results
            
    except Exception as e:
        # 如果批量处理失败，返回错误信�?
        return [{"error": str(e)} for _ in batch_data]


async def call_model(session: aiohttp.ClientSession, lang: str, function_info: Dict, all_functions_data: List[Dict] = None, complexity_cache: Dict = None, max_retries: int = None) -> Dict[str, Any]:
    """Call AI model to generate docstring with retry mechanism"""
    if not API_KEY:
        raise EnvironmentError("GITEE_API_KEY is not defined")
    
    if max_retries is None:
        max_retries = DEFAULT_MAX_RETRIES
    
    from .config import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    
    # 使用预计算的复杂度信息或重新计算
    if complexity_cache and id(function_info) in complexity_cache:
        complexity_info = complexity_cache[id(function_info)]
    else:
        complexity_info = calculate_complexity_score(function_info, all_functions_data)
    
    # 生成上下文摘要（使用快速版本）
    context_summary = generate_context_summary_fast(function_info)
    
    # 提取函数信息
    function_name = function_info.get('basic_info', {}).get('function_name', 'unknown')
    source_code = function_info.get('basic_info', {}).get('source_code', '')
    comments = function_info.get('basic_info', {}).get('comments', [])
    
    # 使用源代码和注释摘要
    source_code_summary = source_code[:1000] if source_code else "no_source_code"
    comments_summary = extract_comments_summary(comments)
    
    # 构建用户提示
    user_prompt = USER_PROMPT_TEMPLATE.format(
        complexity_level=complexity_info["complexity_level"],
        complexity_score=complexity_info["complexity_score"],
        context_summary=context_summary,
        function_name=function_name,
        source_code=source_code_summary,
        comments=comments_summary
    )
    
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            # 增加超时时间，使用指数退避策�?
            base_timeout = DEFAULT_TIMEOUT
            timeout_multiplier = 1.5 ** attempt
            current_timeout = min(base_timeout * timeout_multiplier, DEFAULT_MAX_TIMEOUT)
            
            timeout = aiohttp.ClientTimeout(
                total=current_timeout,
                connect=30,
                sock_read=current_timeout
            )
            
            print(f"Attempt {attempt + 1}/{max_retries}: Using timeout {current_timeout}s")
            
            async with session.post(API_URL, headers=headers, json=data, timeout=timeout) as resp:
                if resp.status == 443:
                    wait_time = min(2 ** (attempt + 1), 60)
                    print(f"Encountered 443 error, waiting {wait_time} seconds before retry... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                
                resp.raise_for_status()
                result = await resp.json()
                end_time = time.time()
                duration = round(end_time - start_time, 3)
                
                print(f"Success on attempt {attempt + 1}, duration: {duration}s")
                # 解析生成的docstring，拆分成三个部分
                docstring_content = result["choices"][0]["message"]["content"].strip()
                parsed_docstring = _parse_docstring(docstring_content)
                
                return {
                    "docstring": parsed_docstring,
                    "duration": duration,
                    "complexity_info": complexity_info,
                    "context_summary": context_summary
                }
                
        except aiohttp.ClientError as e:
            if attempt < max_retries - 1:
                wait_time = min(2 ** (attempt + 1), 60)
                print(f"Network error: {e}, waiting {wait_time} seconds before retry... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            else:
                raise Exception(f"Network request failed after {max_retries} retries: {e}")
                
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                wait_time = min(2 ** (attempt + 1), 60)
                print(f"Request timeout, waiting {wait_time} seconds before retry... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            else:
                raise Exception(f"Request timeout after {max_retries} retries")
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = min(2 ** (attempt + 1), 60)
                print(f"Unknown error: {e}, waiting {wait_time} seconds before retry... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            else:
                raise e
    
    raise Exception(f"All retries failed, attempted {max_retries} times")
