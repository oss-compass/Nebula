__version__ = "1.0.0"
__author__ = "Description Generation Team"

# 导入主要功能（延迟导入以避免依赖问题�?
try:
    from .main import main
    from .batch_processor import generate, create_complete_data
    from .complexity import calculate_complexity_score
    from .context import generate_context_summary, generate_context_summary_fast
    from .ai_client import call_model, call_model_batch
    from .cache import description_cache, DescriptionCache
    from .config import get_config, API_URL, MODEL, API_KEY
except ImportError as e:
    # 如果导入失败，提供基本的配置信息
    from .config import get_config, API_URL, MODEL, API_KEY
    print(f"警告: 部分功能导入失败，可能缺少依�? {e}")
    print("请安装必要的依赖: pip install aiohttp")

__all__ = [
    # 主要功能
    "main",
    "generate",
    "create_complete_data",
    
    # 复杂度分�?
    "calculate_complexity_score",
    
    # 上下文分�?
    "generate_context_summary",
    "generate_context_summary_fast",
    
    # AI模型调用
    "call_model",
    "call_model_batch",
    
    # 缓存管理
    "description_cache",
    "DescriptionCache",
    
    # 配置
    "get_config",
    "API_URL",
    "MODEL",
    "API_KEY",
]

# 模块信息
MODULE_INFO = {
    "name": "description",
    "version": __version__,
    "description": "AI-powered function description generation toolkit",
    "author": __author__,
    "features": [
        "智能复杂度分�?,
        "上下文感知的docstring生成", 
        "批量处理和缓存优�?,
        "多种输出格式",
        "自适应评分算法",
        "Hugging Face专用优化"
    ],
    "dependencies": [
        "aiohttp",
        "asyncio",
        "pathlib",
        "typing"
    ],
    "optional_dependencies": [
        "context_capture"  # 用于高级上下文分�?
    ]
}

def get_module_info():
    """获取模块信息"""
    return MODULE_INFO.copy()

def check_dependencies():
    """检查依赖是否安�?""
    missing_deps = []
    
    try:
        import aiohttp
    except ImportError:
        missing_deps.append("aiohttp")
    
    try:
        import asyncio
    except ImportError:
        missing_deps.append("asyncio")
    
    try:
        from pathlib import Path
    except ImportError:
        missing_deps.append("pathlib")
    
    # 检查可选依�?
    optional_missing = []
    try:
        import context_capture
    except ImportError:
        optional_missing.append("context_capture")
    
    return {
        "all_dependencies_met": len(missing_deps) == 0,
        "missing_required": missing_deps,
        "missing_optional": optional_missing,
        "api_key_configured": API_KEY is not None
    }

def print_status():
    """打印模块状态信�?""
    deps_status = check_dependencies()
    
    print("Description模块状�?")
    print(f"  版本: {__version__}")
    print(f"  API密钥配置: {'�? if deps_status['api_key_configured'] else '�?}")
    print(f"  必需依赖: {'�? if deps_status['all_dependencies_met'] else '�?}")
    
    if deps_status['missing_required']:
        print(f"  缺少必需依赖: {', '.join(deps_status['missing_required'])}")
    
    if deps_status['missing_optional']:
        print(f"  缺少可选依�? {', '.join(deps_status['missing_optional'])}")
    
    if not deps_status['api_key_configured']:
        print("  请设置环境变�?GITEE_API_KEY")
    
    if not deps_status['all_dependencies_met']:
        print("  请安装缺少的依赖: pip install " + " ".join(deps_status['missing_required']))
