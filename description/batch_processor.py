import asyncio
import aiohttp
from typing import Any, Dict, List
from .cache import description_cache
from .complexity import calculate_complexity_score
from .ai_client import call_model, call_model_batch


def _generate_unique_key(function_info: Dict[str, Any]) -> str:
    """生成函数的唯一标识符，避免同名函数覆盖
    
    Args:
        function_info: 函数信息字典
        
    Returns:
        唯一标识符字符串
    """
    try:
        # 获取基本信息
        function_name = function_info.get('basic_info', {}).get('function_name', 'unknown')
        file_path = function_info.get('context', {}).get('file_path', '')
        parent_class = function_info.get('context', {}).get('parent_class', {})
        
        # 构建唯一标识�?
        key_parts = []
        
        # 1. 函数�?
        key_parts.append(function_name)
        
        # 2. 文件路径（去除扩展名和路径前缀�?
        if file_path:
            # 提取文件名（不含扩展名）
            from pathlib import Path
            file_name = Path(file_path).stem
            if file_name:
                key_parts.append(file_name)
        
        # 3. 所属类名（如果有）
        if parent_class and parent_class.get('name'):
            class_name = parent_class['name']
            if class_name:
                key_parts.append(class_name)
        
        # 4. 如果还是不够唯一，添加文件路径的哈希�?
        if len(key_parts) < 3:
            path_hash = str(hash(file_path))[-4:]  # 取最�?�?
            key_parts.append(f"h{path_hash}")
        
        # 组合成唯一标识�?
        unique_key = "::".join(key_parts)
        
        # 如果标识符太长，进行截断
        if len(unique_key) > 100:
            # 保留函数名和文件名，截断其他部分
            if len(key_parts) >= 2:
                unique_key = f"{key_parts[0]}::{key_parts[1]}"
            else:
                unique_key = key_parts[0]
        
        return unique_key
        
    except Exception as e:
        # 如果生成失败，返回带哈希值的函数�?
        function_name = function_info.get('basic_info', {}).get('function_name', 'unknown')
        fallback_hash = str(hash(str(function_info)))[-6:]
        return f"{function_name}_fallback_{fallback_hash}"


def _create_function_index(results: Dict[str, Any]) -> Dict[str, Any]:
    """创建函数索引，便于查找同名函�?
    
    Args:
        results: 处理结果字典
        
    Returns:
        索引字典
    """
    try:
        # 按函数名分组
        name_groups = {}
        
        for unique_key, result in results.items():
            if 'function_name' in result:
                function_name = result['function_name']
                if function_name not in name_groups:
                    name_groups[function_name] = []
                
                # 收集函数信息
                func_info = {
                    "unique_key": unique_key,
                    "file_path": result.get('file_path', ''),
                    "has_error": 'error' in result,
                    "complexity_level": result.get('complexity_info', {}).get('complexity_level', 'unknown'),
                    "complexity_score": result.get('complexity_info', {}).get('complexity_score', 0)
                }
                
                # 如果有错误，添加错误信息
                if 'error' in result:
                    func_info["error"] = result['error']
                
                name_groups[function_name].append(func_info)
        
        # 创建索引结构
        index = {
            "total_functions": len(results),
            "unique_function_names": len(name_groups),
            "functions_with_duplicates": 0,
            "function_groups": {},
            "duplicate_analysis": {}
        }
        
        # 分析每个函数名组
        for function_name, instances in name_groups.items():
            count = len(instances)
            
            if count > 1:
                index["functions_with_duplicates"] += 1
                index["duplicate_analysis"][function_name] = {
                    "count": count,
                    "instances": instances
                }
            
            index["function_groups"][function_name] = {
                "count": count,
                "instances": instances
            }
        
        return index
        
    except Exception as e:
        return {
            "error": f"Failed to create index: {str(e)}",
            "total_functions": len(results)
        }


async def generate(functions_list: List[Dict], concurrent: int = 5, batch_size: int = 10, skip_complexity: bool = False) -> Dict[str, Any]:
    """使用批量处理生成docstrings
    
    Args:
        functions_list: 函数列表
        concurrent: 并发�?
        batch_size: 批次大小
        skip_complexity: 是否跳过复杂度计算，直接使用moderate
    """
    results: Dict[str, Any] = {}
    sem = asyncio.Semaphore(concurrent)
    processed_count = 0
    total_count = len(functions_list)
    
    # 性能优化：预计算所有函数的复杂度，避免重复计算
    if skip_complexity:
        print("跳过复杂度计算，所有函数使用moderate复杂�?..")
        complexity_cache = {}
        simple_functions = []
        complex_functions = []
        
        # 所有函数都使用moderate复杂�?
        for func in functions_list:
            complexity_info = calculate_complexity_score(func, None, skip_calculation=True)
            complexity_cache[id(func)] = complexity_info
            # 所有函数都作为简单函数处理（批量处理�?
            simple_functions.append(func)
        
        print(f"所�?{len(simple_functions)} 个函数将使用批量处理（moderate复杂度）")
    else:
        print("正在预计算函数复杂度...")
        complexity_cache = {}
        simple_functions = []
        complex_functions = []
        
        # 使用快速复杂度计算（不传入all_functions_data避免重复统计�?
        for func in functions_list:
            complexity_info = calculate_complexity_score(func, None)  # 传入None避免重复计算
            complexity_cache[id(func)] = complexity_info
            
            if complexity_info['complexity_level'] == 'simple':
                simple_functions.append(func)
            else:
                complex_functions.append(func)
        
        print(f"Simple functions: {len(simple_functions)}, Complex functions: {len(complex_functions)}")
    
    async with aiohttp.ClientSession() as session:
        # 处理简单函数（批量�?
        async def process_simple_batch(batch: List[Dict]):
            nonlocal processed_count
            async with sem:
                try:
                    # 检查缓�?
                    uncached_batch = []
                    for func in batch:
                        cached_result = description_cache.get(func)
                        if cached_result:
                            unique_key = _generate_unique_key(func)
                            results[unique_key] = cached_result
                            processed_count += 1
                        else:
                            uncached_batch.append(func)
                    
                    if not uncached_batch:
                        return
                    
                    # 批量处理未缓存的函数
                    batch_results = await call_model_batch(session, uncached_batch, functions_list, complexity_cache)
                    
                    for func, result in zip(uncached_batch, batch_results):
                        unique_key = _generate_unique_key(func)
                        if 'error' not in result:
                            # 缓存结果
                            description_cache.set(func, result)
                        
                        results[unique_key] = result
                        processed_count += 1
                        # 减少进度输出频率，每100个函数显示一�?
                        if processed_count % 100 == 0 or processed_count == total_count:
                            print(f"Progress: {processed_count}/{total_count} - {unique_key}: Batch processed")
                        
                except Exception as e:
                    print(f"Batch processing error: {e}")
                    # 如果批量处理失败，回退到单独处�?
                    for func in batch:
                        unique_key = _generate_unique_key(func)
                        results[unique_key] = {"error": f"Batch processing failed: {str(e)}"}
                        processed_count += 1
        
        # 处理复杂函数（单独处理）
        async def process_complex_function(func: Dict):
            nonlocal processed_count
            async with sem:
                try:
                    # 检查缓�?
                    cached_result = description_cache.get(func)
                    if cached_result:
                        unique_key = _generate_unique_key(func)
                        results[unique_key] = cached_result
                        processed_count += 1
                        return
                    
                    # 单独处理复杂函数
                    model_response = await call_model(session, "python", func, functions_list, complexity_cache)
                    unique_key = _generate_unique_key(func)
                    
                    # 缓存结果
                    description_cache.set(func, model_response)
                    
                    results[unique_key] = model_response
                    processed_count += 1
                    # 减少进度输出频率，每100个函数显示一�?
                    if processed_count % 100 == 0 or processed_count == total_count:
                        print(f"Progress: {processed_count}/{total_count} - {unique_key}: Complex function processed")
                    
                except Exception as e:
                    unique_key = _generate_unique_key(func)
                    results[unique_key] = {"error": str(e)}
                    processed_count += 1
                    print(f"Progress: {processed_count}/{total_count} - {unique_key}: Error - {str(e)}")
        
        # 并行处理
        tasks = []
        
        # 批量处理简单函�?
        for i in range(0, len(simple_functions), batch_size):
            batch = simple_functions[i:i + batch_size]
            tasks.append(process_simple_batch(batch))
        
        # 单独处理复杂函数
        for func in complex_functions:
            tasks.append(process_complex_function(func))
        
        await asyncio.gather(*tasks)
    
    return results


def create_complete_data(api_json: Dict[str, Any], description_results: Dict[str, Any], skip_complexity: bool = False) -> Dict[str, Any]:
    """创建包含extract5.py输出和docstring的完整数�?
    
    Args:
        api_json: 原始的extract5.py输出数据
        description_results: 生成的docstring结果
        skip_complexity: 是否跳过了复杂度计算
        
    Returns:
        合并后的完整数据
    """
    import time
    
    try:
        # 创建函数键映射，用于匹配extract5.py输出中的函数和description结果
        def create_function_key(func: Dict[str, Any]) -> str:
            """创建函数键，用于匹配两个JSON文件中的函数"""
            func_name = func.get("basic_info", {}).get("function_name", "")
            
            # 从context中获取file_path
            context = func.get("context", {})
            file_path = context.get("file_path", "")
            
            # 从文件路径中提取文件名（不含扩展名）
            if file_path:
                from pathlib import Path
                file_name = Path(file_path).stem
            else:
                file_name = ""
            
            # 获取父类信息
            parent_class = context.get("parent_class", {})
            class_name = parent_class.get("name", "") if parent_class else ""
            
            # 如果函数在类中，使用 function_name::file_name::class_name 格式
            if class_name:
                return f"{func_name}::{file_name}::{class_name}"
            else:
                return f"{func_name}::{file_name}"
        
        # 创建完整的函数数�?
        complete_functions = []
        functions = api_json.get("functions", [])
        
        for func in functions:
            # 创建函数�?
            func_key = create_function_key(func)
            
            # 查找对应的description结果
            description_info = description_results.get(func_key, {})
            
            # 合并数据，保留extract5.py的所有原始信�?
            complete_func = {
                # 保留原始extract5.py的所有信�?
                "basic_info": func.get("basic_info", {}),
                "complexity": func.get("complexity", {}),
                "context": func.get("context", {}),
                "importance": func.get("importance", {}),
                
                # 添加description信息
                "description_info": {
                    "docstring": description_info.get("docstring", {}),
                    "complexity_level": description_info.get("complexity_info", {}).get("complexity_level", "unknown"),
                    "context_summary": description_info.get("context_summary", ""),
                    "duration": description_info.get("duration(s)", 0),
                    "has_error": "error" in description_info,
                    "error": description_info.get("error", None)
                }
            }
            
            # 如果跳过了复杂度计算，确保复杂度信息正确设置
            if skip_complexity:
                from .config import DEFAULT_IMPORTANCE
                # 设置复杂度为moderate
                complete_func["description_info"]["complexity_level"] = "moderate"
                # 设置默认重要度为moderate
                if not complete_func.get("importance"):
                    complete_func["importance"] = {
                        "importance_level": DEFAULT_IMPORTANCE,
                        "importance_score": 15.0,  # moderate的典型分�?
                        "calculation_skipped": True
                    }
            
            complete_functions.append(complete_func)
        
        # 创建完整的输出结构，保持与extract5.py一致的格式
        complete_data = {
            # 保留extract5.py的原始结�?
            "repository": api_json.get("repository", ""),
            "analysis_timestamp": api_json.get("analysis_timestamp", ""),
            "total_functions": len(complete_functions),
            "function_names": api_json.get("function_names", []),
            "functions": complete_functions,
            
            # 保留extract5.py的类和函数关系分�?
            "class_function_relationship": api_json.get("class_function_relationship", {}),
            
            # 保留extract5.py的API调用关系分析
            "api_call_relationships": api_json.get("api_call_relationships", {}),
            
            # 保留extract5.py的传递调用分�?
            "transitive_calls": api_json.get("transitive_calls", {}),
            
            # 保留extract5.py的过滤信息（如果存在�?
            "filter_info": api_json.get("filter_info", {}),
            
            # 添加description生成信息
            "description_generation": {
                "source_file": "extract5.py output + description results",
                "total_functions": len(complete_functions),
                "functions_with_docstring": len([f for f in complete_functions if not f["description_info"]["has_error"]]),
                "functions_with_errors": len([f for f in complete_functions if f["description_info"]["has_error"]]),
                "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "complexity_distribution": {}
            }
        }
        
        # 添加复杂度统�?
        complexity_stats = {"simple": 0, "moderate": 0, "complex": 0, "unknown": 0}
        for func in complete_functions:
            level = func["description_info"]["complexity_level"]
            complexity_stats[level] += 1
        
        complete_data["description_generation"]["complexity_distribution"] = complexity_stats
        
        return complete_data
        
    except Exception as e:
        print(f"Error creating complete data: {e}")
        return {
            "error": f"Failed to create complete data: {str(e)}",
            "metadata": {
                "source_file": "extract5.py output + description results",
                "total_functions": len(api_json.get("functions", [])),
                "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "functions": []
        }
