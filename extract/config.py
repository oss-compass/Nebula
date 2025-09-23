# 文件扩展名到语言的映�?
EXT_TO_LANG = {
    # Python
    ".py": "python",
    ".pyi": "python",
    
    # Java
    ".java": "java",
    
    # C/C++
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".h++": "cpp",
    
    # JavaScript/TypeScript
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "typescript",
    
    # C#
    ".cs": "c_sharp",
    
    # Go
    ".go": "go",
    
    # Rust
    ".rs": "rust",
    
    # Ruby
    ".rb": "ruby",
    
    # PHP
    ".php": "php",
    ".php3": "php",
    ".php4": "php",
    ".php5": "php",
    ".phtml": "php",
    
    # Swift
    ".swift": "swift",
    
    # Scala
    ".scala": "scala",
    ".sc": "scala",
    
    # Lua
    ".lua": "lua",
    
    # R
    ".r": "r",
    ".R": "r",
    
    # Elixir
    ".ex": "elixir",
    ".exs": "elixir",
    
    # Bash/Shell
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "bash",
    
    # Protocol Buffers
    ".proto": "proto",
    
    # CodeQL
    ".ql": "codeql",
    ".qll": "codeql",
    
    # Starlark
    ".star": "starlark",
    ".bzl": "starlark"
}

# Tree-sitter节点类型配置
NODE_TYPES = {
    "function_definitions": {
        "python": "function_definition",
        "java": "method_declaration",
        "c": "function_definition",
        "cpp": "function_definition",
        "javascript": ["function", "function_declaration", "function_expression", "method_definition", "arrow_function", "generator_function", "generator_function_declaration"],
        "typescript": ["function_declaration", "function_expression", "method_definition", "arrow_function", "generator_function", "generator_function_declaration"],
        "jsx": ["function_declaration", "function_expression", "method_definition", "arrow_function", "generator_function", "generator_function_declaration"],
        "c_sharp": "method_declaration",
        "go": "function_declaration",
        "rust": "function_item",
        "ruby": "method",
        "php": "method_declaration",
        "swift": "function_declaration",
        "scala": "function_declaration",
        "lua": "function_declaration",
        "r": "function_definition",
        "elixir": "function_declaration",
        "bash": "function_definition",
        "proto": "rpc",
        "codeql": "predicate",
        "starlark": "function_definition"
    },
    "class_definitions": {
        "python": "class_definition",
        "java": "class_declaration",
        "c": "struct_declaration",
        "cpp": "class_specifier",
        "javascript": "class_declaration", 
        "typescript": "class_declaration",
        "jsx": "class_declaration",
        "c_sharp": "class_declaration",
        "go": "type_declaration",
        "rust": "struct_item",
        "ruby": "class",
        "php": "class_declaration",
        "swift": "class_declaration",
        "scala": "class_declaration",
        "lua": "table_constructor",
        "r": "class_definition",
        "elixir": "defmodule",
        "bash": None,  # Bash没有类概�?
        "proto": "message",
        "codeql": "class",
        "starlark": None  # Starlark没有类概�?
    },
    "comments": {
        "python": ["comment", "block_comment"],
        "java": ["line_comment", "block_comment"],
        "c": ["comment", "block_comment"],
        "cpp": ["comment", "block_comment"],
        "javascript": ["comment", "block_comment"],
        "typescript": ["comment", "block_comment"],
        "jsx": ["comment", "block_comment"],
        "c_sharp": ["comment", "block_comment"],
        "go": ["comment", "block_comment"],
        "rust": ["comment", "block_comment"],
        "ruby": ["comment", "block_comment"],
        "php": ["comment", "block_comment"],
        "swift": ["comment", "block_comment"],
        "scala": ["comment", "block_comment"],
        "lua": ["comment", "block_comment"],
        "r": ["comment", "block_comment"],
        "elixir": ["comment", "block_comment"],
        "bash": ["comment"],
        "proto": ["comment", "block_comment"],
        "codeql": ["comment", "block_comment"],
        "starlark": ["comment", "block_comment"]
    },
    "function_calls": {
        "python": "call",
        "java": "method_invocation",
        "c": "call_expression",
        "cpp": "call_expression",
        "javascript": "call_expression",
        "typescript": "call_expression",
        "jsx": "call_expression",
        "c_sharp": "invocation_expression",
        "go": "call_expression",
        "rust": "call_expression",
        "ruby": "call",
        "php": "function_call_expression",
        "swift": "call_expression",
        "scala": "call_expression",
        "lua": "function_call",
        "r": "call",
        "elixir": "call",
        "bash": "command",
        "proto": None,  # Protocol Buffers没有函数调用
        "codeql": "call",
        "starlark": "call"
    }
}

# 支持的语言列表
SUPPORTED_LANGUAGES = [
    "python", "java", "c", "cpp", "javascript", "typescript", "jsx",
    "c_sharp", "go", "rust", "ruby", "php", "swift", "scala", "lua",
    "r", "elixir", "bash", "proto", "codeql", "starlark"
]

# 默认目标项目目录
DEFAULT_TARGET_DIR = "target_project"

# 大函数过滤阈�?
LARGE_FUNCTION_THRESHOLD = 200

# 重要度等级阈�?
IMPORTANCE_THRESHOLDS = {
    "Critical": 20,
    "High": 15,
    "Medium": 10,
    "Low": 5,
    "Minimal": 0
}

# 传递调用分析的最大深度限�?
MAX_TRANSITIVE_DEPTH = 20

