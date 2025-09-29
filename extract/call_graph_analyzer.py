
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, DefaultDict

from .config import MAX_TRANSITIVE_DEPTH


def build_api_call_graph(all_functions: List[Dict], repo_path: Path) -> Dict:

    function_map = {}
    for func in all_functions:
        func_name = func["basic_info"]["function_name"]
        if func_name:
            # 确保func_name是字符串类型
            if isinstance(func_name, bytes):
                func_name = func_name.decode('utf-8')
            function_map[func_name] = func

    call_graph = {
        "nodes": [],
        "edges": [],
        "statistics": {
            "total_nodes": 0,
            "total_edges": 0,
            "isolated_functions": 0,
            "most_called_functions": [],
            "functions_with_most_calls": []
        }
    }
    
    # 添加所有函数作为节点
    for func in all_functions:
        func_name = func["basic_info"]["function_name"]
        if func_name:
            node = {
                "id": func_name,
                "name": func_name,
                "file_path": func["context"]["file_path"],
                "parent_class": func["context"]["parent_class"]["name"] if func["context"]["parent_class"] else None,
                "complexity_score": func["complexity"]["semantic_complexity"]["complexity_score"],
                "lines_of_code": func["complexity"]["semantic_complexity"]["lines_of_code"],
                "cyclomatic_complexity": func["complexity"]["semantic_complexity"]["cyclomatic_complexity"],
                "parameters_count": func["complexity"]["semantic_complexity"]["parameters"]
            }
            call_graph["nodes"].append(node)
    
    # 分析调用关系
    call_count = defaultdict(int)  # 记录每个函数被调用的次数
    outgoing_calls = defaultdict(int)  # 记录每个函数调用其他函数的次数
    
    for func in all_functions:
        caller_name = func["basic_info"]["function_name"]
        if not caller_name:
            continue
            
        # 获取函数内部的调用（只处理内部API调用）
        function_calls = func["context"]["function_calls"]["internal_calls"]
        
        for call in function_calls:
            callee_name = call["function_name"]
            
            # 检查被调用的函数是否在当前分析的函数集合中
            if callee_name in function_map:
                # 创建调用边
                edge = {
                    "source": caller_name,
                    "target": callee_name,
                    "source_file": func["context"]["file_path"],
                    "target_file": function_map[callee_name]["context"]["file_path"],
                    "call_location": call.get("location", {}),
                    "call_comments": call.get("comments", [])
                }
                call_graph["edges"].append(edge)
                
                # 统计调用次数
                call_count[callee_name] += 1
                outgoing_calls[caller_name] += 1
    
    # 更新统计信息
    call_graph["statistics"]["total_nodes"] = len(call_graph["nodes"])
    call_graph["statistics"]["total_edges"] = len(call_graph["edges"])
    
    # 计算孤立函数（没有被其他函数调用的函数）
    called_functions = set(edge["target"] for edge in call_graph["edges"])
    isolated_functions = [node["id"] for node in call_graph["nodes"] if node["id"] not in called_functions]
    call_graph["statistics"]["isolated_functions"] = len(isolated_functions)
    
    # 找出被调用最多的函数
    most_called = sorted(call_count.items(), key=lambda x: x[1], reverse=True)[:10]
    call_graph["statistics"]["most_called_functions"] = [
        {"function_name": name, "call_count": count} for name, count in most_called
    ]
    
    # 找出调用其他函数最多的函数
    most_outgoing = sorted(outgoing_calls.items(), key=lambda x: x[1], reverse=True)[:10]
    call_graph["statistics"]["functions_with_most_calls"] = [
        {"function_name": name, "call_count": count} for name, count in most_outgoing
    ]
    
    return call_graph


def analyze_api_dependencies(all_functions: List[Dict], repo_path: Path) -> Dict:
    dependencies = {
        "file_dependencies": defaultdict(set),
        "class_dependencies": defaultdict(set),
        "module_dependencies": defaultdict(set),
        "external_dependencies": defaultdict(set),
        "dependency_chains": [],
        "circular_dependencies": []
    }
    
    # 创建函数映射
    function_map = {}
    for func in all_functions:
        func_name = func["basic_info"]["function_name"]
        if func_name:
            # 确保func_name是字符串类型
            if isinstance(func_name, bytes):
                func_name = func_name.decode('utf-8')
            function_map[func_name] = func
    
    # 分析依赖关系
    for func in all_functions:
        caller_name = func["basic_info"]["function_name"]
        caller_file = func["context"]["file_path"]
        caller_class = func["context"]["parent_class"]["name"] if func["context"]["parent_class"] else None
        
        # 分析内部API函数调用
        for call in func["context"]["function_calls"]["internal_calls"]:
            callee_name = call["function_name"]
            
            if callee_name in function_map:
                callee_func = function_map[callee_name]
                callee_file = callee_func["context"]["file_path"]
                callee_class = callee_func["context"]["parent_class"]["name"] if callee_func["context"]["parent_class"] else None
                
                # 文件级依赖
                if caller_file != callee_file:
                    dependencies["file_dependencies"][caller_file].add(callee_file)
                
                # 类级依赖
                if caller_class and callee_class and caller_class != callee_class:
                    dependencies["class_dependencies"][caller_class].add(callee_class)
                
                # 模块级依赖（基于文件路径）
                caller_module = Path(caller_file).parent.as_posix()
                callee_module = Path(callee_file).parent.as_posix()
                if caller_module != callee_module:
                    dependencies["module_dependencies"][caller_module].add(callee_module)
        
        # 分析外部函数调用
        for call in func["context"]["function_calls"]["external_calls"]:
            callee_name = call["function_name"]
            dependencies["external_dependencies"][caller_file].add(callee_name)
    
    # 检测循环依赖
    def detect_circular_dependencies(dep_graph):
        """检测循环依赖"""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in dep_graph.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # 找到循环
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        for node in dep_graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    # 检测各种级别的循环依赖
    dependencies["circular_dependencies"] = {
        "file_level": detect_circular_dependencies(dict(dependencies["file_dependencies"])),
        "class_level": detect_circular_dependencies(dict(dependencies["class_dependencies"])),
        "module_level": detect_circular_dependencies(dict(dependencies["module_dependencies"]))
    }
    
    # 将set转换为list以便JSON序列化
    dependencies["file_dependencies"] = {
        k: list(v) for k, v in dependencies["file_dependencies"].items()
    }
    dependencies["class_dependencies"] = {
        k: list(v) for k, v in dependencies["class_dependencies"].items()
    }
    dependencies["module_dependencies"] = {
        k: list(v) for k, v in dependencies["module_dependencies"].items()
    }
    dependencies["external_dependencies"] = {
        k: list(v) for k, v in dependencies["external_dependencies"].items()
    }
    
    return dependencies


def analyze_transitive_calls(all_functions: List[Dict], call_graph: Dict) -> Dict:
    """
    分析传递调用逻辑 - 找出间接调用关系
    
    Args:
        all_functions: 所有函数列表
        call_graph: 调用关系图
        
    Returns:
        传递调用分析结果
    """
    print("正在分析传递调用逻辑...")
    
    # 创建函数名到函数信息的映射
    function_map = {}
    for func in all_functions:
        func_name = func["basic_info"]["function_name"]
        if func_name:
            # 确保func_name是字符串类型
            if isinstance(func_name, bytes):
                func_name = func_name.decode('utf-8')
            function_map[func_name] = func
    
    # 构建直接调用图（邻接表）
    direct_calls = defaultdict(set)
    for edge in call_graph["edges"]:
        source = edge["source"]
        target = edge["target"]
        direct_calls[source].add(target)
    
    # 分析传递调用关系
    transitive_calls = defaultdict(set)
    transitive_paths = defaultdict(list)
    max_depth = 0
    
    def find_transitive_calls(start_func: str, current_path: List[str], depth: int):
        """递归查找传递调用关系"""
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        
        if depth > MAX_TRANSITIVE_DEPTH:  # 防止无限递归
            return
        
        current_path.append(start_func)
        
        # 查找直接调用的函数
        for callee in direct_calls.get(start_func, set()):
            if callee not in current_path:  # 避免循环调用
                # 记录传递调用关系
                transitive_calls[start_func].add(callee)
                
                # 记录调用路径
                path_key = f"{start_func} -> {callee}"
                if path_key not in transitive_paths:
                    transitive_paths[path_key] = current_path + [callee]
                
                # 递归查找更深层的传递调用
                find_transitive_calls(callee, current_path.copy(), depth + 1)
        
        current_path.pop()
    
    # 为每个函数查找传递调用关系
    for func in all_functions:
        func_name = func["basic_info"]["function_name"]
        if func_name:
            find_transitive_calls(func_name, [], 0)
    
    # 分析传递调用的统计信息
    transitive_stats = {
        "total_transitive_calls": 0,
        "max_transitive_depth": max_depth,
        "functions_with_transitive_calls": 0,
        "most_transitive_functions": [],
        "transitive_call_chains": []
    }
    
    # 统计传递调用数量
    for func_name, transitive_set in transitive_calls.items():
        transitive_stats["total_transitive_calls"] += len(transitive_set)
        if transitive_set:
            transitive_stats["functions_with_transitive_calls"] += 1
    
    # 找出传递调用最多的函数
    transitive_counts = [(name, len(transitive_set)) for name, transitive_set in transitive_calls.items()]
    transitive_counts.sort(key=lambda x: x[1], reverse=True)
    transitive_stats["most_transitive_functions"] = [
        {"function_name": name, "transitive_call_count": count} 
        for name, count in transitive_counts[:10]
    ]
    
    # 找出最长的传递调用链
    longest_chains = []
    for path_key, path in transitive_paths.items():
        longest_chains.append((path_key, len(path), path))
    
    longest_chains.sort(key=lambda x: x[1], reverse=True)
    transitive_stats["transitive_call_chains"] = [
        {
            "chain": path,
            "length": length,
            "description": f"{path[0]} -> ... -> {path[-1]}"
        }
        for _, length, path in longest_chains[:10]
    ]
    
    # 构建传递调用图
    transitive_graph = {
        "nodes": [],
        "edges": [],
        "statistics": transitive_stats
    }
    
    # 添加所有函数作为节点
    for func in all_functions:
        func_name = func["basic_info"]["function_name"]
        if func_name:
            node = {
                "id": func_name,
                "name": func_name,
                "file_path": func["context"]["file_path"],
                "parent_class": func["context"]["parent_class"]["name"] if func["context"]["parent_class"] else None,
                "direct_calls": len(direct_calls.get(func_name, set())),
                "transitive_calls": len(transitive_calls.get(func_name, set())),
                "total_reachable": len(direct_calls.get(func_name, set())) + len(transitive_calls.get(func_name, set()))
            }
            transitive_graph["nodes"].append(node)
    
    # 添加传递调用边
    for source, targets in transitive_calls.items():
        for target in targets:
            edge = {
                "source": source,
                "target": target,
                "source_file": function_map[source]["context"]["file_path"] if source in function_map else "",
                "target_file": function_map[target]["context"]["file_path"] if target in function_map else "",
                "call_type": "transitive",
                "call_chain": transitive_paths.get(f"{source} -> {target}", [])
            }
            transitive_graph["edges"].append(edge)
    
    # 分析传递调用的影响范围
    impact_analysis = {
        "high_impact_functions": [],
        "isolated_functions": [],
        "hub_functions": [],
        "leaf_functions": []
    }
    
    # 找出高影响函数（被很多函数间接调用的函数）
    impact_counts = defaultdict(int)
    for targets in transitive_calls.values():
        for target in targets:
            impact_counts[target] += 1
    
    high_impact = sorted(impact_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    impact_analysis["high_impact_functions"] = [
        {"function_name": name, "impact_count": count} for name, count in high_impact
    ]
    
    # 找出孤立函数（没有传递调用的函数）
    isolated = [name for name in function_map.keys() if name not in transitive_calls and name not in direct_calls]
    impact_analysis["isolated_functions"] = isolated
    
    # 找出枢纽函数（既有直接调用又有传递调用的函数）
    hub_functions = []
    for func_name in function_map.keys():
        direct_count = len(direct_calls.get(func_name, set()))
        transitive_count = len(transitive_calls.get(func_name, set()))
        if direct_count > 0 and transitive_count > 0:
            hub_functions.append({
                "function_name": func_name,
                "direct_calls": direct_count,
                "transitive_calls": transitive_count,
                "total_influence": direct_count + transitive_count
            })
    
    hub_functions.sort(key=lambda x: x["total_influence"], reverse=True)
    impact_analysis["hub_functions"] = hub_functions[:10]
    
    # 找出叶子函数（只被调用，不调用其他函数的函数）
    leaf_functions = []
    for func_name in function_map.keys():
        if func_name not in direct_calls and func_name not in transitive_calls:
            # 检查是否被其他函数调用
            is_called = any(func_name in targets for targets in direct_calls.values()) or \
                       any(func_name in targets for targets in transitive_calls.values())
            if is_called:
                leaf_functions.append(func_name)
    
    impact_analysis["leaf_functions"] = leaf_functions[:20]  # 限制数量
    
    return {
        "transitive_graph": transitive_graph,
        "impact_analysis": impact_analysis,
        "direct_calls": dict(direct_calls),
        "transitive_calls": {k: list(v) for k, v in transitive_calls.items()},
        "transitive_paths": transitive_paths
    }


def analyze_api_metrics(all_functions: List[Dict], call_graph: Dict) -> Dict:
    """
    分析API指标
    
    Args:
        all_functions: 所有函数列表
        call_graph: 调用关系图
        
    Returns:
        API指标分析结果
    """
    print("正在分析API指标...")
    
    metrics = {
        "function_metrics": {},
        "file_metrics": defaultdict(lambda: {
            "function_count": 0,
            "total_lines": 0,
            "avg_complexity": 0,
            "max_complexity": 0,
            "total_calls": 0,
            "internal_calls": 0,
            "external_calls": 0
        }),
        "class_metrics": defaultdict(lambda: {
            "method_count": 0,
            "total_lines": 0,
            "avg_complexity": 0,
            "max_complexity": 0,
            "public_methods": 0,
            "private_methods": 0
        }),
        "overall_metrics": {
            "total_functions": len(all_functions),
            "avg_function_size": 0,
            "avg_complexity": 0,
            "most_complex_functions": [],
            "largest_functions": [],
            "functions_with_most_parameters": []
        }
    }
    
    # 计算函数级指标
    total_lines = 0
    total_complexity = 0
    function_sizes = []
    function_complexities = []
    function_parameters = []
    
    for func in all_functions:
        func_name = func["basic_info"]["function_name"]
        if not func_name:
            continue
            
        lines = func["complexity"]["semantic_complexity"]["lines_of_code"]
        complexity = func["complexity"]["semantic_complexity"]["cyclomatic_complexity"]
        params = func["complexity"]["semantic_complexity"]["parameters"]
        
        # 函数级指标
        internal_calls_count = len(func["context"]["function_calls"]["internal_calls"])
        external_calls_count = len(func["context"]["function_calls"]["external_calls"])
        
        metrics["function_metrics"][func_name] = {
            "lines_of_code": lines,
            "cyclomatic_complexity": complexity,
            "parameters_count": params,
            "internal_calls_count": internal_calls_count,
            "external_calls_count": external_calls_count,
            "total_calls_count": internal_calls_count + external_calls_count,
            "comments_count": len(func["basic_info"]["comments"]),
            "imports_count": len(func["context"]["imports"])
        }
        
        # 累计统计
        total_lines += lines
        total_complexity += complexity
        function_sizes.append((func_name, lines))
        function_complexities.append((func_name, complexity))
        function_parameters.append((func_name, params))
        
        # 文件级指标
        file_path = func["context"]["file_path"]
        file_metrics = metrics["file_metrics"][file_path]
        file_metrics["function_count"] += 1
        file_metrics["total_lines"] += lines
        file_metrics["max_complexity"] = max(file_metrics["max_complexity"], complexity)
        file_metrics["total_calls"] += internal_calls_count + external_calls_count
        file_metrics["internal_calls"] += internal_calls_count
        file_metrics["external_calls"] += external_calls_count
        
        # 类级指标
        if func["context"]["parent_class"]:
            class_name = func["context"]["parent_class"]["name"]
            class_metrics = metrics["class_metrics"][class_name]
            class_metrics["method_count"] += 1
            class_metrics["total_lines"] += lines
            class_metrics["max_complexity"] = max(class_metrics["max_complexity"], complexity)
            
            # 简单判断公共/私有方法（基于命名约定）
            # 确保func_name是字符串类型
            if isinstance(func_name, bytes):
                func_name_str = func_name.decode('utf-8', errors='ignore')
            else:
                func_name_str = str(func_name)
            
            if func_name_str.startswith('_'):
                class_metrics["private_methods"] += 1
            else:
                class_metrics["public_methods"] += 1
    
    # 计算平均值
    if all_functions:
        metrics["overall_metrics"]["avg_function_size"] = total_lines / len(all_functions)
        metrics["overall_metrics"]["avg_complexity"] = total_complexity / len(all_functions)
    
    # 计算文件级平均值
    for file_metrics in metrics["file_metrics"].values():
        if file_metrics["function_count"] > 0:
            file_metrics["avg_complexity"] = file_metrics["total_lines"] / file_metrics["function_count"]
    
    # 计算类级平均值
    for class_metrics in metrics["class_metrics"].values():
        if class_metrics["method_count"] > 0:
            class_metrics["avg_complexity"] = class_metrics["total_lines"] / class_metrics["method_count"]
    
    # 排序并获取最复杂的函数
    function_complexities.sort(key=lambda x: x[1], reverse=True)
    metrics["overall_metrics"]["most_complex_functions"] = [
        {"function_name": name, "complexity": complexity} 
        for name, complexity in function_complexities[:10]
    ]
    
    # 排序并获取最大的函数
    function_sizes.sort(key=lambda x: x[1], reverse=True)
    metrics["overall_metrics"]["largest_functions"] = [
        {"function_name": name, "lines": lines} 
        for name, lines in function_sizes[:10]
    ]
    
    # 排序并获取参数最多的函数
    function_parameters.sort(key=lambda x: x[1], reverse=True)
    metrics["overall_metrics"]["functions_with_most_parameters"] = [
        {"function_name": name, "parameters": params} 
        for name, params in function_parameters[:10]
    ]
    
    return metrics

