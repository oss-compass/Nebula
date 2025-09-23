import math
import lizard
from tree_sitter import Node
from typing import Dict

from .config import NODE_TYPES


def analyze_semantic_complexity(source_code: str, lang: str) -> Dict:
    """
    使用lizard工具分析语义复杂?
    
    Args:
        source_code: 源代?
        lang: 语言名称
        
    Returns:
        语义复杂度信?
    """
    try:
        if lang == "python":
            # 使用lizard分析Python代码
            analysis = lizard.analyze_file.analyze_source_code("temp.py", source_code)
            if analysis.function_list:
                func = analysis.function_list[0]  # 取第一个函?
                return {
                    "cyclomatic_complexity": func.cyclomatic_complexity,
                    "lines_of_code": func.nloc,
                    "parameters": func.parameter_count,
                    "token_count": func.token_count,
                    "complexity_score": func.cyclomatic_complexity * func.nloc / 10.0
                }
        elif lang in ["c", "cpp"]:
            # 使用lizard分析C/C++代码
            file_ext = "temp.cpp" if lang == "cpp" else "temp.c"
            analysis = lizard.analyze_file.analyze_source_code(file_ext, source_code)
            if analysis.function_list:
                func = analysis.function_list[0]  # 取第一个函?
                return {
                    "cyclomatic_complexity": func.cyclomatic_complexity,
                    "lines_of_code": func.nloc,
                    "parameters": func.parameter_count,
                    "token_count": func.token_count,
                    "complexity_score": func.cyclomatic_complexity * func.nloc / 10.0
                }
        elif lang == "java":
            # 使用lizard分析Java代码
            analysis = lizard.analyze_file.analyze_source_code("temp.java", source_code)
            if analysis.function_list:
                func = analysis.function_list[0]  # 取第一个函?
                return {
                    "cyclomatic_complexity": func.cyclomatic_complexity,
                    "lines_of_code": func.nloc,
                    "parameters": func.parameter_count,
                    "token_count": func.token_count,
                    "complexity_score": func.cyclomatic_complexity * func.nloc / 10.0
                }
        elif lang in ["javascript", "typescript", "jsx"]:
            # 使用lizard分析JavaScript/TypeScript代码
            file_ext = "temp.js" if lang == "javascript" else ("temp.jsx" if lang == "jsx" else "temp.ts")
            analysis = lizard.analyze_file.analyze_source_code(file_ext, source_code)
            if analysis.function_list:
                func = analysis.function_list[0]  # 取第一个函?
                return {
                    "cyclomatic_complexity": func.cyclomatic_complexity,
                    "lines_of_code": func.nloc,
                    "parameters": func.parameter_count,
                    "token_count": func.token_count,
                    "complexity_score": func.cyclomatic_complexity * func.nloc / 10.0
                }
        elif lang in ["c_sharp", "go", "rust", "ruby", "php", "swift", "scala", "lua", "r", "elixir", "bash", "proto", "codeql", "starlark"]:
            # 对于其他语言，尝试使用通用分析
            try:
                # 根据语言选择合适的文件扩展?
                file_ext_map = {
                    "c_sharp": "temp.cs",
                    "go": "temp.go", 
                    "rust": "temp.rs",
                    "ruby": "temp.rb",
                    "php": "temp.php",
                    "swift": "temp.swift",
                    "scala": "temp.scala",
                    "lua": "temp.lua",
                    "r": "temp.r",
                    "elixir": "temp.ex",
                    "bash": "temp.sh",
                    "proto": "temp.proto",
                    "codeql": "temp.ql",
                    "starlark": "temp.star"
                }
                file_ext = file_ext_map.get(lang, "temp.txt")
                analysis = lizard.analyze_file.analyze_source_code(file_ext, source_code)
                if analysis.function_list:
                    func = analysis.function_list[0]  # 取第一个函?
                    return {
                        "cyclomatic_complexity": func.cyclomatic_complexity,
                        "lines_of_code": func.nloc,
                        "parameters": func.parameter_count,
                        "token_count": func.token_count,
                        "complexity_score": func.cyclomatic_complexity * func.nloc / 10.0
                    }
            except Exception as e:
                print(f"lizard分析{lang}语言失败: {e}")
        # 其他语言暂时返回基础信息
        return {
            "cyclomatic_complexity": 0,
            "lines_of_code": len(source_code.split('\n')),
            "parameters": 0,
            "token_count": 0,
            "complexity_score": 0
        }
    except Exception as e:
        print(f"语义复杂度分析失? {e}")
        return {
            "cyclomatic_complexity": 0,
            "lines_of_code": len(source_code.split('\n')),
            "parameters": 0,
            "token_count": 0,
            "complexity_score": 0
        }


def analyze_tree_sitter_complexity(node: Node, code_bytes: bytes, lang: str) -> Dict:
    """
    使用tree-sitter分析语法复杂?
    
    Args:
        node: Tree-sitter节点
        code_bytes: 源代码字?
        lang: 语言名称
        
    Returns:
        语法复杂度信?
    """
    try:
        return_types_count = 0
        external_dependencies_count = 0
        parameters_count = 0
        max_depth = 0
        
        def traverse_complexity(n: Node, depth: int):
            nonlocal max_depth, return_types_count, external_dependencies_count, parameters_count
            max_depth = max(max_depth, depth)
            
            # 统计参数数量
            if n.type in ["parameters", "formal_parameters", "parameter_list"]:
                parameters_count += 1
            
            # 统计返回类型注解
            if lang == "python" and n.type == "type":
                return_types_count += 1
            elif lang in ["java", "c", "cpp", "c_sharp", "go", "rust", "swift", "scala", "proto", "codeql"] and n.type == "type_identifier":
                return_types_count += 1
            elif lang in ["typescript", "jsx"] and n.type in ["type_annotation", "type_identifier"]:
                return_types_count += 1
            
            # 统计外部依赖（函数调用）
            if n.type in ["call", "method_invocation", "call_expression", "invocation_expression", 
                          "function_call", "command", "rpc"]:
                external_dependencies_count += 1
            
            # 递归遍历子节?
            for child in n.children:
                traverse_complexity(child, depth + 1)
        
        traverse_complexity(node, 0)
        
        return {
            "return_types_count": return_types_count,
            "external_dependencies_count": external_dependencies_count,
            "parameters_count": parameters_count,
            "syntax_depth": max_depth
        }
    except Exception as e:
        print(f"Tree-sitter复杂度分析失? {e}")
        return {
            "return_types_count": 0,
            "external_dependencies_count": 0,
            "parameters_count": 0,
            "syntax_depth": 0
        }


def count_branches_and_depth(node: Node, code_bytes: bytes) -> Dict:
    """
    统计分支数和语法层数
    
    Args:
        node: Tree-sitter节点
        code_bytes: 源代码字?
        
    Returns:
        分支和深度信?
    """
    branch_count = 0
    max_depth = 0
    
    def traverse_branches(n: Node, depth: int):
        nonlocal max_depth, branch_count
        max_depth = max(max_depth, depth)
        
        # 统计分支结构
        if n.type in ["if_statement", "else_clause", "for_statement", "while_statement", 
                      "switch_statement", "case_clause", "try_statement", "catch_clause"]:
            branch_count += 1
        
        for child in n.children:
            traverse_branches(child, depth + 1)
    
    traverse_branches(node, 0)
    
    return {
        "branch_count": branch_count,
        "syntax_depth": max_depth
    }


def analyze_complexity_metrics(node: Node, code_bytes: bytes, source_code: str, lang: str) -> Dict:
    """
    综合分析复杂度指?
    
    Args:
        node: Tree-sitter节点
        code_bytes: 源代码字?
        source_code: 源代码字符串
        lang: 语言名称
        
    Returns:
        完整的复杂度分析结果
    """
    # 语义复杂度分?
    semantic_complexity = analyze_semantic_complexity(source_code, lang)
    
    # 语法复杂度分?
    tree_sitter_complexity = analyze_tree_sitter_complexity(node, code_bytes, lang)
    
    # 结构复杂度分?
    branch_info = count_branches_and_depth(node, code_bytes)
    
    return {
        "semantic_complexity": {
            "cyclomatic_complexity": semantic_complexity["cyclomatic_complexity"],
            "lines_of_code": semantic_complexity["lines_of_code"],
            "parameters": semantic_complexity["parameters"],
            "token_count": semantic_complexity["token_count"],
            "complexity_score": semantic_complexity["complexity_score"]
        },
        "syntax_complexity": {
            "return_types_count": tree_sitter_complexity["return_types_count"],
            "external_dependencies_count": tree_sitter_complexity["external_dependencies_count"],
            "parameters_count": tree_sitter_complexity["parameters_count"],
            "syntax_depth": tree_sitter_complexity["syntax_depth"]
        },
        "structural_complexity": {
            "branch_count": branch_info["branch_count"],
            "syntax_depth": branch_info["syntax_depth"]
        }
    }

