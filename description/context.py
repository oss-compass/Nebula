<<<<<<< HEAD
=======
"""
上下文摘要模块
负责生成函数的上下文摘要和智能分析
"""

>>>>>>> d73e9c4add0f6ab55a65312431901567e37244ec
from pathlib import Path
from typing import Any, Dict, List
from .config import (
    DOMAIN_MAPPINGS, FUNCTION_PATTERNS, API_SPECIFIC_PATTERNS, 
    API_PATTERNS, HF_INDICATORS
)


def generate_context_summary(function_info: Dict) -> str:
    """使用context_capture生成智能上下文摘要"""
    try:
        # 导入新的上下文捕捉器
        from context_capture import APIContextCapture
        
        # 获取函数信息
        function_name = function_info.get('basic_info', {}).get('function_name', 'unknown')
        file_path = function_info.get('context', {}).get('file_path', '')
        
        if not file_path:
            return "文件路径未知"
        
        # 获取项目路径（从文件路径推断）
        project_path = Path(file_path).parent
        while project_path.name not in ['src', 'lib', 'app', 'main', 'core'] and len(project_path.parts) > 1:
            project_path = project_path.parent
        
        # 如果找不到合适的项目根目录，使用文件所在目录的父目录
        if len(project_path.parts) <= 1:
            project_path = Path(file_path).parent.parent
        
        # 创建全局上下文捕捉器
        try:
            capture = APIContextCapture(str(project_path), branch="main")
            
            # 捕捉全局上下文
            context_data = capture.capture_global_context(
                function_name, 
                file_path, 
                {"functions": [function_info]}  # 传入当前函数信息作为extract_data
            )
            
            # 生成智能上下文摘要
            summary = _generate_intelligent_summary(function_info, context_data)
            return summary
                
        except Exception as e:
            # 如果上下文捕捉失败，回退到原始方法
            return _fallback_context_summary(function_info)
            
    except ImportError:
        # 如果无法导入context_capture，回退到原始方法
        return _fallback_context_summary(function_info)
    except Exception as e:
        return f"上下文信息提取失败: {str(e)}"


def generate_context_summary_fast(function_info: Dict) -> str:
    """快速生成上下文摘要，减少计算开销"""
    try:
        context = function_info.get('context', {})
        parent_class = context.get('parent_class', {})
        file_path = context.get('file_path', '')
        imports = context.get('imports', [])
        
        summary_parts = []
        
        # 文件位置
        if file_path:
            summary_parts.append(f"文件: {Path(file_path).name}")
        
        # 所属类
        if parent_class and parent_class.get('name'):
            summary_parts.append(f"类: {parent_class['name']}")
        
        # 主要导入（只取前2个）
        if imports:
            main_imports = imports[:2]
            summary_parts.append(f"导入: {', '.join(main_imports)}")
        
        if summary_parts:
            return " | ".join(summary_parts)
        else:
            return "独立函数"
            
    except Exception:
        return "上下文未知"


def _generate_intelligent_summary(function_info: Dict, context_data: Dict) -> str:
    """生成智能上下文摘要，避免过于技术化的描述"""
    try:
        function_name = function_info.get('basic_info', {}).get('function_name', 'unknown')
        file_path = function_info.get('context', {}).get('file_path', '')
        context = function_info.get('context', {})
        
        # 提取关键信息
        imports = context.get('imports', [])
        function_calls = context.get('function_calls', [])
        parent_class = context.get('parent_class', {})
        
        # 分析函数用途和上下文
        summary_parts = []
        
        # 1. 文件位置
        if file_path:
            summary_parts.append(f"文件: {Path(file_path).name}")
        
        # 2. 所属类
        if parent_class and parent_class.get('name'):
            summary_parts.append(f"类: {parent_class['name']}")
        
        # 3. 智能分析主要导入和用途
        # 优先检查 Hugging Face 相关
        hf_context = _analyze_huggingface_context(function_info)
        if hf_context:
            summary_parts.append(hf_context)
        else:
            domain_context = _analyze_domain_context(imports, function_calls, function_name)
            if domain_context:
                summary_parts.append(domain_context)
        
        # 4. 分析调用模式
        call_pattern = _analyze_call_pattern(function_calls, function_name)
        if call_pattern:
            summary_parts.append(call_pattern)
        
        # 5. 分析全局上下文中的调用处
        call_sites = context_data.get('call_sites', [])
        if call_sites:
            usage_context = _analyze_usage_context(call_sites)
            if usage_context:
                summary_parts.append(usage_context)
        
        # 6. 分析常量和配置
        constants = context_data.get('constants_and_configs', [])
        if constants:
            config_context = _analyze_config_context(constants)
            if config_context:
                summary_parts.append(config_context)
        
        if summary_parts:
            return " | ".join(summary_parts)
        else:
            return "独立函数，无特殊上下文"
            
    except Exception as e:
        return _fallback_context_summary(function_info)


def _analyze_huggingface_context(function_info: Dict) -> str:
    """专门分析 Hugging Face 相关的上下文"""
    try:
        function_name = function_info.get('basic_info', {}).get('function_name', '').lower()
        source_code = function_info.get('basic_info', {}).get('source_code', '').lower()
        context = function_info.get('context', {})
        imports = context.get('imports', [])
        
        # 检查是否是 Hugging Face 相关
        is_hf_related = any(any(hf_ind in imp.lower() for hf_ind in HF_INDICATORS) for imp in imports)
        
        if not is_hf_related:
            return ""
        
        # 分析源代码中的关键模式
        if 'client' in function_name or 'init' in function_name:
            if any(keyword in source_code for keyword in ['api_key', 'token', 'auth']):
                return "用途: 配置AI模型服务连接"
            elif any(keyword in source_code for keyword in ['base_url', 'endpoint', 'url']):
                return "用途: 建立模型仓库连接"
            else:
                return "用途: 初始化AI模型客户端"
        
        elif 'download' in function_name or 'fetch' in function_name:
            return "用途: 下载AI模型文件"
        
        elif 'load' in function_name or 'from_pretrained' in function_name:
            return "用途: 加载预训练模型"
        
        elif 'cache' in function_name:
            return "用途: 管理模型缓存"
        
        elif 'tokenizer' in function_name:
            return "用途: 文本预处理"
        
        elif 'pipeline' in function_name:
            return "用途: 创建AI任务流水线"
        
        # 基于源代码内容分析
        if any(keyword in source_code for keyword in ['from_pretrained', 'pipeline']):
            return "用途: AI模型操作"
        elif any(keyword in source_code for keyword in ['download', 'snapshot_download']):
            return "用途: 模型文件下载"
        elif any(keyword in source_code for keyword in ['tokenizer', 'tokenize']):
            return "用途: 文本预处理"
        elif any(keyword in source_code for keyword in ['cache', 'cache_dir']):
            return "用途: 模型缓存管理"
        elif any(keyword in source_code for keyword in ['client', 'hub', 'api_key']):
            return "用途: 模型服务连接"
        
        return "用途: AI模型操作"
        
    except Exception as e:
        return ""


def _analyze_domain_context(imports: List[str], function_calls: List[str], function_name: str) -> str:
    """分析函数的领域上下文，提供更准确的业务描述"""
    try:
        # 分析导入
        detected_domains = []
        for imp in imports:
            imp_lower = imp.lower()
            for lib, domain in DOMAIN_MAPPINGS.items():
                if lib in imp_lower:
                    detected_domains.append(domain)
                    break
        
        # 分析函数调用
        for call in function_calls:
            if isinstance(call, dict):
                call_name = call.get('function_name', '').lower()
            else:
                call_name = str(call).lower()
            
            for lib, domain in DOMAIN_MAPPINGS.items():
                if lib in call_name:
                    detected_domains.append(domain)
                    break
        
        # 特殊函数名模式 - 更准确的业务描述
        for pattern, domain in FUNCTION_PATTERNS.items():
            if pattern in function_name.lower():
                detected_domains.append(domain)
                break
        
        # 优先检查 API 特定模式
        for pattern, domain in API_SPECIFIC_PATTERNS.items():
            if pattern in function_name.lower():
                detected_domains.append(domain)
                break
        
        # 智能组合领域描述
        if detected_domains:
            unique_domains = list(set(detected_domains))
            
            # 特殊处理：如果是 Hugging Face 相关，提供更准确的描述
            if any(any(hf_imp in imp.lower() for hf_imp in HF_INDICATORS) for imp in imports):
                if any(pattern in function_name.lower() for pattern in ['from_pretrained', 'pipeline']):
                    return "用途: AI模型加载"
                elif any(pattern in function_name.lower() for pattern in ['download', 'snapshot_download']):
                    return "用途: 模型文件下载"
                elif any(pattern in function_name.lower() for pattern in ['tokenizer', 'tokenize']):
                    return "用途: 文本预处理"
                elif any(pattern in function_name.lower() for pattern in ['cache', 'cache_dir']):
                    return "用途: 模型缓存管理"
                elif any(pattern in function_name.lower() for pattern in ['client', 'hub']):
                    return "用途: 模型仓库连接"
                else:
                    return "用途: AI模型操作"
            
            # 特殊处理：网络请求相关
            if any('requests' in imp.lower() or 'aiohttp' in imp.lower() for imp in imports):
                if any(pattern in function_name.lower() for pattern in ['get', 'post', 'put', 'delete']):
                    return "用途: API数据交互"
                elif any(pattern in function_name.lower() for pattern in ['client', 'session']):
                    return "用途: 网络连接管理"
                elif any(pattern in function_name.lower() for pattern in ['auth', 'token', 'api_key']):
                    return "用途: 身份认证"
                else:
                    return "用途: 网络通信"
            
            # 特殊处理：结合函数调用分析
            if function_calls:
                # 检查函数调用中是否包含 Hugging Face 相关调用
                hf_calls = []
                for call in function_calls:
                    if isinstance(call, dict):
                        call_name = call.get('function_name', '').lower()
                    else:
                        call_name = str(call).lower()
                    
                    if any(hf_pattern in call_name for hf_pattern in ['from_pretrained', 'pipeline', 'tokenizer', 'download']):
                        hf_calls.append(call_name)
                
                if hf_calls:
                    if any('from_pretrained' in call for call in hf_calls):
                        return "用途: 预训练模型加载"
                    elif any('pipeline' in call for call in hf_calls):
                        return "用途: AI任务流水线"
                    elif any('download' in call for call in hf_calls):
                        return "用途: 模型文件下载"
                    else:
                        return "用途: AI模型操作"
            
            # 默认处理
            if len(unique_domains) == 1:
                return f"用途: {unique_domains[0]}"
            else:
                return f"用途: {', '.join(unique_domains[:2])}"
        
        return ""
        
    except Exception as e:
        return ""


def _analyze_call_pattern(function_calls: List[str], function_name: str) -> str:
    """分析函数调用模式"""
    try:
        if not function_calls:
            return ""
        
        # 分析调用模式
        call_count = len(function_calls)
        
        # 检查是否有特定的调用模式
        for call in function_calls:
            if isinstance(call, dict):
                call_name = call.get('function_name', '')
            else:
                call_name = str(call)
            
            for pattern, description in API_PATTERNS.items():
                if pattern in call_name.lower():
                    return f"调用: {description}"
        
        # 默认描述
        if call_count <= 3:
            return f"调用: {call_count}个函数"
        else:
            return f"调用: {call_count}个函数（复杂调用）"
            
    except Exception as e:
        return ""


def _analyze_usage_context(call_sites: List[Dict]) -> str:
    """分析使用上下文"""
    try:
        if not call_sites:
            return ""
        
        # 分析调用处的分布
        file_count = len(set(site.get('file_path', '') for site in call_sites))
        
        if file_count == 1:
            return "使用: 单文件内调用"
        elif file_count <= 3:
            return f"使用: {file_count}个文件调用"
        else:
            return f"使用: {file_count}个文件调用（广泛使用）"
            
    except Exception as e:
        return ""


def _analyze_config_context(constants: List[Dict]) -> str:
    """分析配置上下文"""
    try:
        if not constants:
            return ""
        
        # 分析配置类型
        config_types = set()
        for const in constants:
            const_type = const.get('type', '')
            if const_type == 'config_item':
                config_types.add('配置项')
            elif const_type == 'constant_definition':
                config_types.add('常量')
        
        if config_types:
            return f"配置: {', '.join(config_types)}"
        
        return ""
        
    except Exception as e:
        return ""


def _fallback_context_summary(function_info: Dict) -> str:
    """回退的上下文摘要生成方法（原始实现）"""
    try:
        context = function_info.get('context', {})
        
        # 提取关键上下文信息
        parent_class = context.get('parent_class', {})
        file_path = context.get('file_path', '')
        imports = context.get('imports', [])
        function_calls = context.get('function_calls', [])
        
        # 构建上下文摘要
        summary_parts = []
        
        # 文件位置
        if file_path:
            summary_parts.append(f"文件: {Path(file_path).name}")
        
        # 所属类
        if parent_class and parent_class.get('name'):
            summary_parts.append(f"类: {parent_class['name']}")
        
        # 主要导入
        if imports:
            main_imports = imports[:3]  # 只取前3个主要导入
            summary_parts.append(f"导入: {', '.join(main_imports)}")
        
        # 函数调用
        if function_calls:
            call_count = len(function_calls)
            summary_parts.append(f"调用: {call_count}个函数")
        
        if summary_parts:
            return " | ".join(summary_parts)
        else:
            return "独立函数，无特殊上下文"
            
    except Exception as e:
        return f"上下文信息提取失败: {str(e)}"
