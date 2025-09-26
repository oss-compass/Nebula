"""
extract - 代码分析工具包

这是一个功能强大的代码分析工具，主要用于从GitHub仓库中提取和分析API信息。

主要功能：
1. 代码复杂度分析（语义、语法、结构复杂度）
2. API信息提取（函数、类、导入、调用关系）
3. 调用关系图构建和分析
4. 传递调用关系分析
5. 函数重要度计算
6. 依赖关系分析和循环依赖检测

模块结构：
- config.py: 配置和常量定义
- utils.py: 基础工具函数
- complexity_analyzer.py: 复杂度分析模块
- info_extractor.py: 信息提取模块
- call_graph_analyzer.py: 调用图分析模块
- importance_calculator.py: 重要度计算模块
- main.py: 主入口文件

使用方法：
    python -m extract.main <GitHub仓库链接> [选项]

示例：
    # 完整克隆 + 过滤大函数
    python -m extract.main https://github.com/user/repo --filter-large-functions
    
    # 稀疏检出src目录
    python -m extract.main https://github.com/user/repo src
"""

__version__ = "1.0.0"
__author__ = "Code Analysis Team"

# 导入主要功能（延迟导入以避免依赖问题）
try:
    from .main import main, process_repo
    from .utils import clone_repo, detect_language, get_parser
    from .config import EXT_TO_LANG, NODE_TYPES, SUPPORTED_LANGUAGES
    from .complexity_analyzer import analyze_complexity_metrics
    from .info_extractor import extract_function_info, extract_class_info, process_file
    from .call_graph_analyzer import build_api_call_graph, analyze_api_dependencies
    from .importance_calculator import calculate_function_importance
except ImportError as e:
    # 如果导入失败，提供基本的配置信息
    from .config import EXT_TO_LANG, NODE_TYPES, SUPPORTED_LANGUAGES
    print(f"警告: 部分功能导入失败，可能缺少依赖: {e}")
    print("请安装必要的依赖: pip install tree-sitter tree-sitter-languages lizard")

__all__ = [
    # 主要功能
    "main",
    "process_repo",
    
    # 工具函数
    "clone_repo",
    "detect_language", 
    "get_parser",
    
    # 配置
    "EXT_TO_LANG",
    "NODE_TYPES", 
    "SUPPORTED_LANGUAGES",
    
    # 分析功能
    "analyze_complexity_metrics",
    "extract_function_info",
    "extract_class_info", 
    "process_file",
    "build_api_call_graph",
    "analyze_api_dependencies",
    "calculate_function_importance",
]
