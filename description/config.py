<<<<<<< HEAD
=======
"""
Description模块配置和常量定义
包含所有系统提示、API配置、默认参数等
"""

>>>>>>> d73e9c4add0f6ab55a65312431901567e37244ec
import os
from typing import Dict, Any

# API配置
API_URL = "https://ai.gitee.com/v1/chat/completions"
MODEL = "Qwen3-4B"
API_KEY = os.getenv("GITEE_API_KEY")

# 默认参数
DEFAULT_CONCURRENT = 10
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_TIMEOUT = 180

# 复杂度阈值
COMPLEXITY_THRESHOLDS = {
    "simple": 10,
    "moderate": 25,
    "complex": 50
}

# 默认重要度配置
DEFAULT_IMPORTANCE = "moderate"

# 默认权重配置
DEFAULT_WEIGHTS = {
    'cyclomatic': 0.4,
    'lines': 0.01,
    'branches': 0.3,
    'params': 0.1
}

# 语言映射
LANGUAGE_MAP = {
    ".py": "python",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".js": "javascript",
    ".ts": "typescript"
}

# 领域映射 - 更注重业务用途而非技术实现
DOMAIN_MAPPINGS = {
    # 机器学习/AI - 更准确的业务描述
    'transformers': 'AI模型处理',
    'torch': '深度学习框架',
    'tensorflow': '机器学习框架',
    'sklearn': '机器学习工具',
    'huggingface': 'AI模型库',
    'datasets': '数据管理',
    'tokenizers': '文本预处理',
    
    # Web开发
    'flask': 'Web服务',
    'django': 'Web应用框架',
    'fastapi': 'API服务',
    'requests': '网络通信',
    'aiohttp': '异步网络',
    
    # 数据处理
    'pandas': '数据分析',
    'numpy': '数值计算',
    'matplotlib': '图表绘制',
    'seaborn': '统计图表',
    
    # 数据库
    'sqlalchemy': '数据库管理',
    'pymongo': '文档数据库',
    'redis': '缓存服务',
    
    # 工具库
    'logging': '日志管理',
    'json': '数据格式',
    'os': '系统接口',
    'pathlib': '文件路径',
}

# 函数模式映射
FUNCTION_PATTERNS = {
    'download': '资源下载',
    'load': '数据加载',
    'save': '数据保存',
    'train': '模型训练',
    'predict': '模型预测',
    'transform': '数据转换',
    'process': '数据处理',
    'validate': '数据验证',
    'parse': '数据解析',
    'encode': '数据编码',
    'decode': '数据解码',
    'generate': '内容生成',
    'create': '对象创建',
    'build': '构建操作',
    'setup': '环境配置',
    'init': '初始化',
    'configure': '参数配置',
}

# API特定模式识别
API_SPECIFIC_PATTERNS = {
    # Hugging Face 相关
    'from_pretrained': '预训练模型加载',
    'pipeline': 'AI任务流水线',
    'model': 'AI模型操作',
    'tokenizer': '文本分词',
    'huggingface': 'AI模型库',
    'transformers': 'AI模型处理',
    'hf_hub': '模型仓库',
    'snapshot_download': '模型下载',
    'download': '模型下载',
    'cache_dir': '模型缓存',
    'local_files_only': '本地模型',
    
    # 网络请求相关
    'get': '数据获取',
    'post': '数据提交',
    'put': '数据更新',
    'delete': '数据删除',
    'request': '网络请求',
    'fetch': '数据获取',
    'client': 'API客户端',
    'session': '会话管理',
    'auth': '身份验证',
    'token': '访问令牌',
    'api_key': 'API密钥',
    'endpoint': '服务端点',
    'base_url': '基础URL',
    
    # 文件操作相关
    'read': '文件读取',
    'write': '文件写入',
    'open': '文件打开',
    'close': '文件关闭',
}

# API调用模式
API_PATTERNS = {
    'get': '数据获取',
    'post': '数据提交',
    'put': '数据更新',
    'delete': '数据删除',
    'create': '创建操作',
    'update': '更新操作',
    'find': '查找操作',
    'search': '搜索操作',
}

# Hugging Face 相关指示器
HF_INDICATORS = ['huggingface', 'transformers', 'hf_hub', 'datasets']

# 系统提示
SYSTEM_PROMPT = """
你是一名资深 Python 代码文档编写专家，需要为给定函数生成符合 PEP 257 规范的 Docstring。

## 编写原则
1. **根据函数复杂度选择文档详略**
   - 如果函数包含多分支、复杂参数类型、多个可能的返回情况 → 生成**详细版文档**（列出核心逻辑步骤/分类分支）。
   - 如果函数简单（例如一两行代码、单一分支、简单返回）→ 生成**简洁版文档**（一句话功能说明 + 简单参数与返回值）。
2. **描述风格**
   - 直接用动词短语开头描述功能，避免使用"This function"、"该方法"等套话。
   - 重点描述函数的**业务用途**和**实际功能**，而非技术实现细节。
   - 避免过于技术化的描述，如"发送HTTP请求"应描述为"获取数据"或"提交信息"。
   - 对于AI/ML相关函数，描述其业务价值而非技术实现：
     * "初始化客户端并配置与服务器的连接" → "建立AI模型服务连接"
     * "设置基础URL和API密钥" → "配置模型访问权限"
     * "建立用于后续API交互的HTTP客户端" → "准备模型下载环境"
   - 对复杂函数，按逻辑结构或分类清单列出关键处理分支，便于后续进行语义检索。
   - 对简单函数，功能描述一句话即可，避免冗余。
3. **Args 部分**
   - 按照函数签名顺序列出每个参数。
   - 标注参数类型，并说明作用、取值范围或特殊行为。
   - 重点说明参数的**业务含义**，而非技术细节。
4. **Returns 部分**
   - 指出返回值的类型。
   - 对于复杂函数，列出不同返回分支的含义；简单函数只需一句说明。
   - 描述返回值的**业务含义**，如"返回模型对象"而非"返回HTTP响应"。
5. **其他规范**
   - 输出必须是中文 Docstring 格式。
   - 不要在 Docstring 外输出任何额外解释。
   - 遵循 PEP 257 格式：
       \"\"\"  
       Summary.

       Args:
           param (type): description

       Returns:
           type: description
       \"\"\"
"""

# 上下文摘要提示
CONTEXT_SUMMARY_PROMPT = """
基于以下代码和上下文信息，生成一段简短的上下文摘要（不超过100字），说明函数的作用域、依赖关系和主要用途。
不要输出自然语言描述，只输出结构化的上下文摘要。
"""

# 用户提示模板
USER_PROMPT_TEMPLATE = """
请阅读以下代码信息，并根据复杂度选择生成详细版或简洁版 Docstring。

## 复杂度信息
**复杂度等级:** {complexity_level}
**复杂度评分:** {complexity_score}

## 上下文摘要
{context_summary}

## 输入信息
**函数名称:** {function_name}  
**函数原代码:**  
{source_code}  
**函数注释:**  
{comments}  

## 要求
- 复杂度等级为 "simple"：生成简洁功能描述，用一句话说明功能即可。
- 复杂度等级为 "moderate"：生成中等详细功能描述，用 2–3 句话说明功能、主要流程和典型输入输出。
- 复杂度等级为 "complex"：生成详细功能描述，包含核心逻辑步骤、关键分支条件、可能的异常处理。
- 复杂度等级为 "unknown"：生成占位描述，提示功能尚不明确，并建议补充更多上下文或文档。
- 根据复杂度调整 Args 和 Returns 部分的详细程度
- **重要**：重点描述函数的实际功能，避免过于技术化的描述
- 对于 AI/ML 相关函数，描述其业务价值：
  * "初始化客户端并配置与服务器的连接" → "建立AI模型服务连接"
  * "设置基础URL和API密钥" → "配置模型访问权限"  
  * "建立用于后续API交互的HTTP客户端" → "准备模型下载环境"
- 对于网络请求函数，描述其业务目的（如"获取用户数据"而非"发送GET请求"）
- 对于模型相关函数，描述其AI用途（如"加载预训练模型"而非"发送HTTP请求"）
"""

# 批量处理提示
BATCH_PROMPT_TEMPLATE = "请为以下函数生成Docstring，每个函数用---分隔：\n\n"

# 性能优化说明
PERFORMANCE_NOTES = """
性能优化说明：
1. 批量处理：简单函数按批次处理，减少API调用次数
2. 智能缓存：避免重复处理相同函数，大幅提升速度
3. 并发优化：复杂函数单独处理，简单函数批量处理
4. 快速上下文：简化上下文分析，减少计算开销

使用方法：
- 启用缓存：python description4.py input.json --use-cache
- 清除缓存：python description4.py input.json --clear-cache
- 调整批次大小：python description4.py input.json --batch-size 15
- 调整并发数：python description4.py input.json --concurrent 8

预期性能提升：5-20倍（取决于函数数量和复杂度分布）
"""

def get_config() -> Dict[str, Any]:
    """获取完整配置信息"""
    return {
        "api": {
            "url": API_URL,
            "model": MODEL,
            "key": API_KEY
        },
        "defaults": {
            "concurrent": DEFAULT_CONCURRENT,
            "batch_size": DEFAULT_BATCH_SIZE,
            "max_retries": DEFAULT_MAX_RETRIES,
            "timeout": DEFAULT_TIMEOUT,
            "max_timeout": DEFAULT_MAX_TIMEOUT
        },
        "complexity": {
            "thresholds": COMPLEXITY_THRESHOLDS,
            "weights": DEFAULT_WEIGHTS
        },
        "language_map": LANGUAGE_MAP,
        "domain_mappings": DOMAIN_MAPPINGS,
        "function_patterns": FUNCTION_PATTERNS,
        "api_patterns": API_PATTERNS,
        "prompts": {
            "system": SYSTEM_PROMPT,
            "context_summary": CONTEXT_SUMMARY_PROMPT,
            "user_template": USER_PROMPT_TEMPLATE,
            "batch_template": BATCH_PROMPT_TEMPLATE
        }
    }
