__version__ = "1.0.0"
__author__ = "Code Analysis Team"

# ????????????????????????
try:
    from .main import main, process_repo
    from .utils import clone_repo, detect_language, get_parser
    from .config import EXT_TO_LANG, NODE_TYPES, SUPPORTED_LANGUAGES
    from .complexity_analyzer import analyze_complexity_metrics
    from .info_extractor import extract_function_info, extract_class_info, process_file
    from .call_graph_analyzer import build_api_call_graph, analyze_api_dependencies
    from .importance_calculator import calculate_function_importance
except ImportError as e:
    # ????????????????
    from .config import EXT_TO_LANG, NODE_TYPES, SUPPORTED_LANGUAGES
    print(f"??: ???????????????: {e}")
    print("????????: pip install tree-sitter tree-sitter-languages lizard")

__all__ = [
    # ????
    "main",
    "process_repo",
    
    # ????
    "clone_repo",
    "detect_language", 
    "get_parser",
    
    # ??
    "EXT_TO_LANG",
    "NODE_TYPES", 
    "SUPPORTED_LANGUAGES",
    
    # ????
    "analyze_complexity_metrics",
    "extract_function_info",
    "extract_class_info", 
    "process_file",
    "build_api_call_graph",
    "analyze_api_dependencies",
    "calculate_function_importance",
]