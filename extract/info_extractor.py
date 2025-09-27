from pathlib import Path
from tree_sitter import Node
from typing import Dict, List, Optional, Set

from .config import NODE_TYPES
from .utils import (
    get_node_text, get_node_location, find_parent_class, 
    extract_comments, get_parser
)
from .complexity_analyzer import analyze_complexity_metrics


def extract_imports(file_path: str, lang: str, repo_path: Path) -> List[str]:
    try:
        full_path = repo_path / file_path
        if not full_path.exists():
            print(f"文件不存在: {full_path}")
            return []
        
        parser = get_parser(lang)
        if not parser:
            return []
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        tree = parser.parse(content.encode('utf-8'))
        if not tree.root_node:
            return []
        
        imports = []
        
        def collect_imports(node: Node, depth: int = 0):
            if depth > 1000:  
                return
                
            if lang == "python":
                if node.type == "import_statement":
                    for child in node.children:
                        if child.type == "dotted_name":
                            imports.append(get_node_text(child, content.encode('utf-8')))
                        elif child.type == "aliased_import":
                            name_node = child.child_by_field_name("name")
                            if name_node:
                                imports.append(get_node_text(name_node, content.encode('utf-8')))
                
                elif node.type == "import_from_statement":
                    module_node = node.child_by_field_name("module")
                    if module_node:
                        module_name = get_node_text(module_node, content.encode('utf-8'))
                        for child in node.children:
                            if child.type == "import_list":
                                for import_item in child.children:
                                    if import_item.type == "aliased_import":
                                        name_node = import_item.child_by_field_name("name")
                                        if name_node:
                                            imports.append(f"{module_name}.{get_node_text(name_node, content.encode('utf-8'))}")
                                    elif import_item.type == "dotted_name":
                                        imports.append(f"{module_name}.{get_node_text(import_item, content.encode('utf-8'))}")
            
            elif lang == "java":
                if node.type == "import_declaration":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        imports.append(get_node_text(name_node, content.encode('utf-8')))
            elif lang in ["c", "cpp"]:
                if node.type == "preproc_include":
                    name_node = node.child_by_field_name("path")
                    if name_node:
                        imports.append(get_node_text(name_node, content.encode('utf-8')))
            elif lang in ["javascript", "typescript", "jsx"]:
                if node.type == "import_statement":
                    for child in node.children:
                        if child.type == "import_clause":
                            for import_item in child.children:
                                if import_item.type == "identifier":
                                    imports.append(get_node_text(import_item, content.encode('utf-8')))
                        elif child.type == "string":
                            imports.append(get_node_text(child, content.encode('utf-8')))
            elif lang == "c_sharp":
                if node.type == "using_directive":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        imports.append(get_node_text(name_node, content.encode('utf-8')))
            elif lang == "go":
                if node.type == "import_declaration":
                    for child in node.children:
                        if child.type == "import_spec_list":
                            for spec in child.children:
                                if spec.type == "import_spec":
                                    name_node = spec.child_by_field_name("name")
                                    if name_node:
                                        imports.append(get_node_text(name_node, content.encode('utf-8')))
            elif lang == "rust":
                if node.type == "use_declaration":
                    for child in node.children:
                        if child.type == "scoped_use_list":
                            for item in child.children:
                                if item.type == "scoped_identifier":
                                    imports.append(get_node_text(item, content.encode('utf-8')))
            elif lang == "ruby":
                if node.type == "call":
                    fn_node = node.child_by_field_name("method")
                    if fn_node and get_node_text(fn_node, content.encode('utf-8')) in ["require", "require_relative"]:
                        for child in node.children:
                            if child.type == "string":
                                imports.append(get_node_text(child, content.encode('utf-8')))
            elif lang == "php":
                if node.type == "use_declaration":
                    for child in node.children:
                        if child.type == "name":
                            imports.append(get_node_text(child, content.encode('utf-8')))
            elif lang == "swift":
                if node.type == "import_declaration":
                    for child in node.children:
                        if child.type == "import_path":
                            imports.append(get_node_text(child, content.encode('utf-8')))
            elif lang == "scala":
                if node.type == "import_declaration":
                    for child in node.children:
                        if child.type == "import_expr":
                            imports.append(get_node_text(child, content.encode('utf-8')))
            elif lang == "lua":
                if node.type == "call":
                    fn_node = node.child_by_field_name("function")
                    if fn_node and get_node_text(fn_node, content.encode('utf-8')) == "require":
                        for child in node.children:
                            if child.type == "string":
                                imports.append(get_node_text(child, content.encode('utf-8')))
            
            elif lang == "r":
                if node.type == "call":
                    fn_node = node.child_by_field_name("function")
                    if fn_node and get_node_text(fn_node, content.encode('utf-8')) in ["library", "require"]:
                        for child in node.children:
                            if child.type == "string":
                                imports.append(get_node_text(child, content.encode('utf-8')))
            elif lang == "elixir":
                if node.type == "alias":
                    for child in node.children:
                        if child.type == "alias":
                            imports.append(get_node_text(child, content.encode('utf-8')))
            elif lang == "bash":
                if node.type == "command":
                    for child in node.children:
                        if child.type == "command_name":
                            cmd_name = get_node_text(child, content.encode('utf-8'))
                            if cmd_name in ["source", "."]:
                                for sibling in node.children:
                                    if sibling.type == "argument":
                                        imports.append(get_node_text(sibling, content.encode('utf-8')))
            elif lang == "proto":
                if node.type == "import_statement":
                    for child in node.children:
                        if child.type == "string":
                            imports.append(get_node_text(child, content.encode('utf-8')))
            elif lang == "codeql":
                if node.type == "import":
                    for child in node.children:
                        if child.type == "module_name":
                            imports.append(get_node_text(child, content.encode('utf-8')))
            elif lang == "starlark":
                if node.type == "load_statement":
                    for child in node.children:
                        if child.type == "string":
                            imports.append(get_node_text(child, content.encode('utf-8')))
        
            for child in node.children:
                collect_imports(child, depth + 1)
        
        collect_imports(tree.root_node, 0)
        return imports
        
    except Exception as e:
        print(f"提取imports失败: {e}")
        return []


def find_function_calls(node: Node, code_bytes: bytes, lang: str, all_function_names: Optional[Set[str]] = None) -> Dict:
    internal_calls = []
    external_calls = []
    call_type = NODE_TYPES["function_calls"].get(lang)
    
    if not call_type:
        return {"internal_calls": internal_calls, "external_calls": external_calls}
    
    def collect_calls(n: Node, depth: int = 0):
        if depth > 1000:
            return
            
        if n.type == call_type:
            function_name = ""
            if lang == "python":
                fn_node = n.child_by_field_name("function")
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang in ["javascript", "typescript", "jsx"]:
                fn_node = n.child(0)
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "java":
                fn_node = n.child_by_field_name("name")
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang in ["c", "cpp"]:
                fn_node = n.child(0)
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "c_sharp":
                fn_node = n.child_by_field_name("expression")
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "go":
                fn_node = n.child(0)
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "rust":
                fn_node = n.child(0)
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "ruby":
                fn_node = n.child_by_field_name("method")
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "php":
                fn_node = n.child_by_field_name("name")
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "swift":
                fn_node = n.child(0)
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "scala":
                fn_node = n.child(0)
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "lua":
                fn_node = n.child_by_field_name("function")
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "r":
                fn_node = n.child_by_field_name("function")
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "elixir":
                fn_node = n.child(0)
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "bash":
                fn_node = n.child_by_field_name("command_name")
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "codeql":
                fn_node = n.child(0)
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            elif lang == "starlark":
                fn_node = n.child(0)
                if fn_node:
                    function_name = get_node_text(fn_node, code_bytes)
            
            if function_name:
                call_info = {
                    "function_name": function_name,
                    "full_call": get_node_text(n, code_bytes),
                    "comments": extract_comments(n, code_bytes, lang)
                }
                
                if all_function_names and function_name in all_function_names:
                    internal_calls.append(call_info)
                else:
                    external_calls.append(call_info)
        
        for child in n.children:
            collect_calls(child, depth + 1)
    
    collect_calls(node, 0)
    return {"internal_calls": internal_calls, "external_calls": external_calls}


def _should_filter_js_function(function_name: str) -> bool:
    if not function_name:
        return True
    if function_name.startswith("window[") or function_name.startswith("window."):
        return True
    global_objects = ["document", "console", "navigator", "location", "history", "screen", "global", "self"]
    for obj in global_objects:
        if function_name.startswith(f"{obj}[") or function_name.startswith(f"{obj}."):
            return True
    if "[" in function_name and "]" in function_name:
        if not (function_name.count("[") == 1 and function_name.count("]") == 1):
            return True
        bracket_start = function_name.find("[")
        bracket_end = function_name.find("]")
        if bracket_start != -1 and bracket_end != -1:
            inside_brackets = function_name[bracket_start+1:bracket_end].strip()
            if '"' in inside_brackets or "'" in inside_brackets:
                return True

    if function_name.count(".") > 1:
        return True
    if function_name.startswith("$") and len(function_name) > 1:
        return True
    if any(char in function_name for char in ["'", '"', "`"]):
        return True
    
    return False


def extract_js_function_name(node: Node, code_bytes: bytes) -> str:
    function_name = ""
    
    if node.type == "function_declaration":
        name_node = node.child_by_field_name("name")
        if name_node:
            function_name = get_node_text(name_node, code_bytes)
    
    elif node.type == "function":
        if node.parent:
            if node.parent.type == "variable_declarator":
                name_node = node.parent.child_by_field_name("name")
                if name_node:
                    function_name = get_node_text(name_node, code_bytes)
            elif node.parent.type == "assignment_expression":
                name_node = node.parent.child_by_field_name("left")
                if name_node:
                    left_text = get_node_text(name_node, code_bytes)
                    if not _should_filter_js_function(left_text):
                        function_name = left_text
            elif node.parent.type == "property_assignment":
                name_node = node.parent.child_by_field_name("name")
                if name_node:
                    function_name = get_node_text(name_node, code_bytes)
    
    elif node.type == "function_expression":
        if node.parent:
            if node.parent.type == "variable_declarator":
                name_node = node.parent.child_by_field_name("name")
                if name_node:
                    function_name = get_node_text(name_node, code_bytes)
            elif node.parent.type == "assignment_expression":
                name_node = node.parent.child_by_field_name("left")
                if name_node:
                    left_text = get_node_text(name_node, code_bytes)
                    if not _should_filter_js_function(left_text):
                        function_name = left_text
            elif node.parent.type == "property_assignment":
                name_node = node.parent.child_by_field_name("name")
                if name_node:
                    function_name = get_node_text(name_node, code_bytes)
    
    elif node.type == "arrow_function":
        if node.parent:
            if node.parent.type == "variable_declarator":
                name_node = node.parent.child_by_field_name("name")
                if name_node:
                    function_name = get_node_text(name_node, code_bytes)
            elif node.parent.type == "assignment_expression":
                name_node = node.parent.child_by_field_name("left")
                if name_node:
                    left_text = get_node_text(name_node, code_bytes)
                    if not _should_filter_js_function(left_text):
                        function_name = left_text
            elif node.parent.type == "property_assignment":
                name_node = node.parent.child_by_field_name("name")
                if name_node:
                    function_name = get_node_text(name_node, code_bytes)
            elif node.parent.type == "object":
                if node.parent.parent and node.parent.parent.type == "pair":
                    name_node = node.parent.parent.child_by_field_name("key")
                    if name_node:
                        function_name = get_node_text(name_node, code_bytes)
    
    elif node.type == "method_definition":
        name_node = node.child_by_field_name("name")
        if name_node:
            function_name = get_node_text(name_node, code_bytes)
    
    elif node.type in ["generator_function", "generator_function_declaration"]:
        # 生成器函数: function* genFunc() {}
        name_node = node.child_by_field_name("name")
        if name_node:
            function_name = get_node_text(name_node, code_bytes)
    
    # 处理特殊情况：构造函数和原型方法
    if not function_name and node.parent:
        # 检查是否是构造函数: function MyClass() {}
        if node.type == "function_declaration" and node.parent.type == "program":
            # 这可能是全局构造函数
            name_node = node.child_by_field_name("name")
            if name_node:
                function_name = get_node_text(name_node, code_bytes)
        
        # 检查是否是原型方法: MyClass.prototype.method = function() {}
        elif node.parent.type == "assignment_expression":
            left_node = node.parent.child_by_field_name("left")
            if left_node and left_node.type == "member_expression":
                # 提取原型方法名
                property_node = left_node.child_by_field_name("property")
                if property_node:
                    function_name = get_node_text(property_node, code_bytes)
    
    # 最后检查是否需要过滤这个函数名
    if _should_filter_js_function(function_name):
        return ""
    
    return function_name


def extract_function_info(node: Node, code_bytes: bytes, lang: str, file_path: str, repo_path: Path, all_function_names: Optional[Set[str]] = None) -> Dict:
    function_name = ""
    name_node = None
    
    if lang == "python":
        name_node = node.child_by_field_name("name")
    elif lang in ["javascript", "typescript", "jsx"]:
        # 使用专门的JavaScript函数名提取函数
        function_name = extract_js_function_name(node, code_bytes)
        if function_name:
            # 创建一个虚拟的name_node来保持兼容性
            class DummyNode:
                def __init__(self, text):
                    self.text = text
            name_node = DummyNode(function_name)
    elif lang == "java":
        if node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
        elif node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
        elif node.type == "function_expression":
            # 对于函数表达式，需要查找父节点来获取函数名
            # 例如: var funcName = function() {}
            if node.parent:
                if node.parent.type == "variable_declarator":
                    name_node = node.parent.child_by_field_name("name")
                elif node.parent.type == "assignment_expression":
                    name_node = node.parent.child_by_field_name("left")
                elif node.parent.type == "property_assignment":
                    name_node = node.parent.child_by_field_name("name")
        elif node.type == "method_definition":
            name_node = node.child_by_field_name("name")
    elif lang in ["c", "cpp"]:
        # C/C++语言函数名通常在declarator中
        declarator = node.child_by_field_name("declarator")
        if declarator:
            name_node = declarator.child_by_field_name("declarator")
    elif lang == "c_sharp":
        name_node = node.child_by_field_name("name")
    elif lang == "go":
        name_node = node.child_by_field_name("name")
    elif lang == "rust":
        name_node = node.child_by_field_name("name")
    elif lang == "ruby":
        name_node = node.child_by_field_name("name")
    elif lang == "php":
        name_node = node.child_by_field_name("name")
    elif lang == "swift":
        name_node = node.child_by_field_name("name")
    elif lang == "scala":
        name_node = node.child_by_field_name("name")
    elif lang == "lua":
        name_node = node.child_by_field_name("name")
    elif lang == "r":
        name_node = node.child_by_field_name("name")
    elif lang == "elixir":
        name_node = node.child_by_field_name("name")
    elif lang == "bash":
        name_node = node.child_by_field_name("name")
    elif lang == "proto":
        name_node = node.child_by_field_name("name")
    elif lang == "codeql":
        name_node = node.child_by_field_name("name")
    elif lang == "starlark":
        name_node = node.child_by_field_name("name")
    
    if name_node:
        if hasattr(name_node, 'text') and not hasattr(name_node, 'type'):
            # 这是我们的DummyNode
            function_name = name_node.text
        else:
            # 这是Tree-sitter节点
            function_name = get_node_text(name_node, code_bytes)
    
    # 获取参数信息
    parameters = []
    params_node = None
    
    if lang == "python":
        params_node = node.child_by_field_name("parameters")
    elif lang in ["java", "javascript", "typescript", "jsx"]:
        params_node = node.child_by_field_name("parameters")
    elif lang in ["c", "cpp"]:
        params_node = node.child_by_field_name("parameters")
    elif lang == "c_sharp":
        params_node = node.child_by_field_name("parameters")
    elif lang == "go":
        params_node = node.child_by_field_name("parameters")
    elif lang == "rust":
        params_node = node.child_by_field_name("parameters")
    elif lang == "ruby":
        params_node = node.child_by_field_name("parameters")
    elif lang == "php":
        params_node = node.child_by_field_name("parameters")
    elif lang == "swift":
        params_node = node.child_by_field_name("parameters")
    elif lang == "scala":
        params_node = node.child_by_field_name("parameters")
    elif lang == "lua":
        params_node = node.child_by_field_name("parameters")
    elif lang == "r":
        params_node = node.child_by_field_name("parameters")
    elif lang == "elixir":
        params_node = node.child_by_field_name("parameters")
    elif lang == "bash":
        params_node = node.child_by_field_name("parameters")
    elif lang == "proto":
        params_node = node.child_by_field_name("parameters")
    elif lang == "codeql":
        params_node = node.child_by_field_name("parameters")
    elif lang == "starlark":
        params_node = node.child_by_field_name("parameters")
    
    if params_node:
        for param in params_node.children:
            if param.is_named:
                param_info = {
                    "name": "",
                    "type": "",
                    "default": ""
                }
                
                if lang == "python":
                    name_field = param.child_by_field_name("name")
                    type_field = param.child_by_field_name("type")
                    default_field = param.child_by_field_name("default")
                    
                    if name_field:
                        param_info["name"] = get_node_text(name_field, code_bytes)
                    if type_field:
                        param_info["type"] = get_node_text(type_field, code_bytes)
                    if default_field:
                        param_info["default"] = get_node_text(default_field, code_bytes)
                
                elif lang in ["java", "c", "cpp", "c_sharp", "go", "rust", "swift", "scala"]:
                    type_field = param.child_by_field_name("type")
                    name_field = param.child_by_field_name("name")
                    
                    if type_field:
                        param_info["type"] = get_node_text(type_field, code_bytes)
                    if name_field:
                        param_info["name"] = get_node_text(name_field, code_bytes)
                
                elif lang in ["javascript", "typescript", "jsx"]:
                    name_field = param.child_by_field_name("name")
                    type_field = param.child_by_field_name("type")
                    default_field = param.child_by_field_name("default")
                    
                    if name_field:
                        param_info["name"] = get_node_text(name_field, code_bytes)
                    if type_field:
                        param_info["type"] = get_node_text(type_field, code_bytes)
                    if default_field:
                        param_info["default"] = get_node_text(default_field, code_bytes)
                
                elif lang in ["ruby", "php", "lua", "r", "elixir", "bash", "starlark"]:
                    name_field = param.child_by_field_name("name")
                    if name_field:
                        param_info["name"] = get_node_text(name_field, code_bytes)
                
                elif lang in ["proto", "codeql"]:
                    type_field = param.child_by_field_name("type")
                    name_field = param.child_by_field_name("name")
                    
                    if type_field:
                        param_info["type"] = get_node_text(type_field, code_bytes)
                    if name_field:
                        param_info["name"] = get_node_text(name_field, code_bytes)
                
                if param_info["name"]:
                    parameters.append(param_info)
    
    # 获取返回类型
    return_type = ""
    if lang in ["java", "c", "cpp", "c_sharp", "go", "rust", "swift", "scala", "proto", "codeql"]:
        return_type_node = node.child_by_field_name("type")
        if return_type_node:
            return_type = get_node_text(return_type_node, code_bytes)
    
    # 获取源代码
    source_code = get_node_text(node, code_bytes)
    
    # 分析复杂度
    complexity = analyze_complexity_metrics(node, code_bytes, source_code, lang)
    
    # 构建API信息
    api_info = {
        # (1) 基础信息
        "basic_info": {
            "function_name": function_name,
            "source_code": source_code,
            "code_location": get_node_location(node),
            "comments": extract_comments(node, code_bytes, lang),
            "parameters": parameters,
            "return_type": return_type
        },
        
        # (2) 复杂度信息
        "complexity": complexity,
        
        # (3) 上下文信息
        "context": {
            "parent_class": find_parent_class(node, code_bytes),
            "file_path": file_path,
            "imports": extract_imports(file_path, lang, repo_path),
            "function_calls": find_function_calls(node, code_bytes, lang, all_function_names),
            "return_value_usage": []  # 暂时为空，需要跨文件分析
        }
    }
    
    # 注意：重要度分析需要等所有函数都提取完成后才能计算
    # 这里先添加一个占位符，后续会更新
    api_info["importance"] = {
        "status": "pending",
        "message": "重要度分析将在所有函数提取完成后计算"
    }
    
    return api_info


def extract_class_info(node: Node, code_bytes: bytes, lang: str, file_path: str, all_function_names: Optional[Set[str]] = None) -> Dict:
    """
    提取类的完整信息
    
    Args:
        node: Tree-sitter节点
        code_bytes: 源代码字节
        lang: 语言名称
        file_path: 文件路径
        all_function_names: 所有函数名集合
        
    Returns:
        类信息字典
    """
    # 获取类名
    class_name = ""
    name_node = node.child_by_field_name("name")
    if name_node:
        class_name = get_node_text(name_node, code_bytes)
    
    # 获取基类信息
    bases = []
    if lang == "python":
        base_node = node.child_by_field_name("base_classes")
        if base_node:
            for base in base_node.children:
                if base.is_named:
                    bases.append(get_node_text(base, code_bytes))
    elif lang in ["java", "javascript", "typescript", "jsx", "c_sharp", "swift", "scala"]:
        base_node = node.child_by_field_name("superclass")
        if base_node:
            bases.append(get_node_text(base_node, code_bytes))
    elif lang == "cpp":
        # C++基类在base_clause中
        base_clause = node.child_by_field_name("base_clause")
        if base_clause:
            for base in base_clause.children:
                if base.is_named and base.type == "base_class_specifier":
                    name_node = base.child_by_field_name("name")
                    if name_node:
                        bases.append(get_node_text(name_node, code_bytes))
    elif lang == "go":
        # Go没有继承概念，但有接口实现
        pass
    elif lang == "rust":
        # Rust trait实现
        pass
    elif lang == "ruby":
        # Ruby继承
        base_node = node.child_by_field_name("superclass")
        if base_node:
            bases.append(get_node_text(base_node, code_bytes))
    elif lang == "php":
        # PHP继承
        base_node = node.child_by_field_name("superclass")
        if base_node:
            bases.append(get_node_text(base_node, code_bytes))
    elif lang == "lua":
        # Lua没有类继承概念
        pass
    elif lang == "r":
        # R S3/S4类系统
        pass
    elif lang == "elixir":
        # Elixir没有类继承概念
        pass
    elif lang == "bash":
        # Bash没有类概念
        pass
    elif lang == "proto":
        # Protocol Buffers继承
        base_node = node.child_by_field_name("superclass")
        if base_node:
            bases.append(get_node_text(base_node, code_bytes))
    elif lang == "codeql":
        # CodeQL类继承
        base_node = node.child_by_field_name("superclass")
        if base_node:
            bases.append(get_node_text(base_node, code_bytes))
    elif lang == "starlark":
        # Starlark没有类概念
        pass
    
    # 获取类的方法
    methods = []
    function_types = NODE_TYPES["function_definitions"].get(lang)
    if function_types:
        # 确保function_types是列表
        if isinstance(function_types, str):
            function_types = [function_types]
        
        for child in node.children:
            if child.type in function_types:
                # 需要从file_path构建完整的repo_path
                # 这里假设file_path是相对于repo_path的路径
                repo_path = Path("target_project")
                method_info = extract_function_info(child, code_bytes, lang, file_path, repo_path, all_function_names)
                if method_info["basic_info"]["function_name"]:
                    methods.append(method_info)
    
    # 构建类信息
    class_info = {
        "class_name": class_name,
        "file_path": file_path,
        "location": get_node_location(node),
        "bases": bases,
        "methods": methods,
    }
    
    return class_info


def process_file(file_path: Path, repo_path: Path, all_function_names: Optional[Set[str]] = None) -> Dict:
    """
    处理单个文件，提取API信息
    
    Args:
        file_path: 文件路径
        repo_path: 仓库路径
        all_function_names: 所有函数名集合
        
    Returns:
        文件分析结果
    """
    from .utils import detect_language, get_parser
    
    lang = detect_language(file_path)
    if not lang:
        return {"classes": [], "functions": []}
    
    parser = get_parser(lang)
    if not parser:
        return {"classes": [], "functions": []}
    
    try:
        code_bytes = file_path.read_bytes()
        tree = parser.parse(code_bytes)
        
        if not tree.root_node:
            return {"classes": [], "functions": []}
        
        classes = []
        functions = []
        
        function_types = NODE_TYPES["function_definitions"].get(lang)
        class_type = NODE_TYPES["class_definitions"].get(lang)
        
        # 确保function_types是列表
        if isinstance(function_types, str):
            function_types = [function_types]
        
        def find_nodes(node: Node, depth: int = 0):
            # 防止递归深度过深
            if depth > 1000:  # 设置最大递归深度
                return
                
            if node.type == class_type:
                rel_path = file_path.relative_to(repo_path).as_posix()
                class_info = extract_class_info(node, code_bytes, lang, rel_path, all_function_names)
                if class_info["class_name"]:
                    classes.append(class_info)
            elif node.type in function_types:
                rel_path = file_path.relative_to(repo_path).as_posix()
                function_info = extract_function_info(node, code_bytes, lang, rel_path, repo_path, all_function_names)
                if function_info["basic_info"]["function_name"]:
                    functions.append(function_info)
            
            for child in node.children:
                find_nodes(child, depth + 1)
        
        find_nodes(tree.root_node, 0)
        return {"classes": classes, "functions": functions}
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return {"classes": [], "functions": []}

