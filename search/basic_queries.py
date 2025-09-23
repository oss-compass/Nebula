import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from .base import BaseSearch

logger = logging.getLogger(__name__)

class BasicQueries(BaseSearch):
    """基础查询功能�?""
    
    def find_api_callers(self, 
                        api_name: str, 
                        max_depth: int = 3,
                        include_external: bool = True) -> Dict[str, Any]:
        """
        查找API的调用�?
        
        Args:
            api_name: API名称
            max_depth: 最大搜索深�?
            include_external: 是否包含外部调用
            
        Returns:
            调用者信息字�?
        """
        logger.info(f"查找API调用�? {api_name}, 深度: {max_depth}")
        
        # 查找直接调用�?
        direct_callers_query = """
        MATCH (caller:Function)-[r:INTERNAL_CALLS|CALLS]->(target:Function {name: $api_name})
        RETURN caller.id as caller_id, 
               caller.name as caller_name,
               caller.filepath as caller_file,
               caller.start_line as caller_start_line,
               caller.end_line as caller_end_line,
               r.full_call as call_expression,
               type(r) as relationship_type
        ORDER BY caller.name
        """
        
        direct_callers = self._run_query(direct_callers_query, {"api_name": api_name})
        
        # 查找间接调用者（通过调用链）
        indirect_callers = []
        if max_depth > 1:
            indirect_callers_query = f"""
            MATCH path = (caller:Function)-[:INTERNAL_CALLS|CALLS*2..{max_depth}]->(target:Function {{name: $api_name}})
            RETURN caller.id as caller_id,
                   caller.name as caller_name,
                   caller.filepath as caller_file,
                   caller.start_line as caller_start_line,
                   caller.end_line as caller_end_line,
                   length(path) as call_depth,
                   [node in nodes(path)[1..-1] | node.name] as call_chain
            ORDER BY call_depth, caller.name
            """
            
            indirect_callers = self._run_query(indirect_callers_query, {"api_name": api_name})
        
        # 查找外部调用（如果启用）
        external_callers = []
        if include_external:
            external_callers_query = """
            MATCH (caller:Function)
            WHERE caller.calls CONTAINS $api_name
            RETURN caller.id as caller_id,
                   caller.name as caller_name,
                   caller.filepath as caller_file,
                   caller.start_line as caller_start_line,
                   caller.end_line as caller_end_line,
                   'external' as relationship_type
            ORDER BY caller.name
            """
            
            external_callers = self._run_query(external_callers_query, {"api_name": api_name})
        
        # 统计信息
        total_callers = len(direct_callers) + len(indirect_callers) + len(external_callers)
        
        # 按文件分�?
        callers_by_file = defaultdict(list)
        for caller in direct_callers + indirect_callers + external_callers:
            file_path = caller.get("caller_file", "unknown")
            callers_by_file[file_path].append(caller)
        
        return {
            "api_name": api_name,
            "total_callers": total_callers,
            "direct_callers": direct_callers,
            "indirect_callers": indirect_callers,
            "external_callers": external_callers,
            "callers_by_file": dict(callers_by_file),
            "max_depth": max_depth,
            "include_external": include_external
        }
    
    def find_api_callees(self, 
                        api_name: str, 
                        max_depth: int = 3,
                        include_external: bool = True) -> Dict[str, Any]:
        """
        查找API的被调用者（API调用的其他函数）
        
        Args:
            api_name: API名称
            max_depth: 最大搜索深�?
            include_external: 是否包含外部调用
            
        Returns:
            被调用者信息字�?
        """
        logger.info(f"查找API被调用�? {api_name}, 深度: {max_depth}")
        
        # 查找直接被调用�?
        direct_callees_query = """
        MATCH (source:Function {name: $api_name})-[r:INTERNAL_CALLS|CALLS]->(callee:Function)
        RETURN callee.id as callee_id,
               callee.name as callee_name,
               callee.filepath as callee_file,
               callee.start_line as callee_start_line,
               callee.end_line as callee_end_line,
               r.full_call as call_expression,
               type(r) as relationship_type
        ORDER BY callee.name
        """
        
        direct_callees = self._run_query(direct_callees_query, {"api_name": api_name})
        
        # 查找间接被调用�?
        indirect_callees = []
        if max_depth > 1:
            indirect_callees_query = f"""
            MATCH path = (source:Function {{name: $api_name}})-[:INTERNAL_CALLS|CALLS*2..{max_depth}]->(callee:Function)
            RETURN callee.id as callee_id,
                   callee.name as callee_name,
                   callee.filepath as callee_file,
                   callee.start_line as callee_start_line,
                   callee.end_line as callee_end_line,
                   length(path) as call_depth,
                   [node in nodes(path)[1..-1] | node.name] as call_chain
            ORDER BY call_depth, callee.name
            """
            
            indirect_callees = self._run_query(indirect_callees_query, {"api_name": api_name})
        
        # 统计信息
        total_callees = len(direct_callees) + len(indirect_callees)
        
        # 按文件分�?
        callees_by_file = defaultdict(list)
        for callee in direct_callees + indirect_callees:
            file_path = callee.get("callee_file", "unknown")
            callees_by_file[file_path].append(callee)
        
        return {
            "api_name": api_name,
            "total_callees": total_callees,
            "direct_callees": direct_callees,
            "indirect_callees": indirect_callees,
            "callees_by_file": dict(callees_by_file),
            "max_depth": max_depth,
            "include_external": include_external
        }
    
    def get_dependency_list(self, 
                           function_name: str,
                           include_transitive: bool = True,
                           max_depth: int = 5) -> Dict[str, Any]:
        """
        获取函数的依赖清�?
        
        Args:
            function_name: 函数�?
            include_transitive: 是否包含传递依�?
            max_depth: 最大依赖深�?
            
        Returns:
            依赖清单字典
        """
        logger.info(f"获取依赖清单: {function_name}, 传递依�? {include_transitive}")
        
        # 获取直接依赖
        direct_deps_query = """
        MATCH (source:Function {name: $function_name})-[r:INTERNAL_CALLS|CALLS]->(dep:Function)
        RETURN dep.id as dep_id,
               dep.name as dep_name,
               dep.filepath as dep_file,
               dep.start_line as dep_start_line,
               dep.end_line as dep_end_line,
               dep.complexity_level as dep_complexity,
               dep.cyclomatic_complexity as dep_cyclomatic_complexity,
               type(r) as relationship_type,
               r.full_call as call_expression
        ORDER BY dep.name
        """
        
        direct_deps = self._run_query(direct_deps_query, {"function_name": function_name})
        
        # 获取传递依�?
        transitive_deps = []
        if include_transitive and max_depth > 1:
            transitive_deps_query = f"""
            MATCH path = (source:Function {{name: $function_name}})-[:INTERNAL_CALLS|CALLS*2..{max_depth}]->(dep:Function)
            RETURN dep.id as dep_id,
                   dep.name as dep_name,
                   dep.filepath as dep_file,
                   dep.start_line as dep_start_line,
                   dep.end_line as dep_end_line,
                   dep.complexity_level as dep_complexity,
                   dep.cyclomatic_complexity as dep_cyclomatic_complexity,
                   length(path) as dependency_depth,
                   [node in nodes(path)[1..-1] | node.name] as dependency_chain
            ORDER BY dependency_depth, dep.name
            """
            
            transitive_deps = self._run_query(transitive_deps_query, {"function_name": function_name})
        
        # 获取外部依赖（从calls属性中�?
        external_deps_query = """
        MATCH (source:Function {name: $function_name})
        WHERE source.calls IS NOT NULL
        RETURN source.calls as external_calls
        """
        
        external_result = self._run_query_single(external_deps_query, {"function_name": function_name})
        external_deps = external_result.get("external_calls", []) if external_result else []
        
        # 统计信息
        total_deps = len(direct_deps) + len(transitive_deps) + len(external_deps)
        
        # 按文件分�?
        deps_by_file = defaultdict(list)
        for dep in direct_deps + transitive_deps:
            file_path = dep.get("dep_file", "unknown")
            deps_by_file[file_path].append(dep)
        
        # 复杂度统�?
        complexity_stats = self._calculate_dependency_complexity(direct_deps + transitive_deps)
        
        return {
            "function_name": function_name,
            "total_dependencies": total_deps,
            "direct_dependencies": direct_deps,
            "transitive_dependencies": transitive_deps,
            "external_dependencies": external_deps,
            "dependencies_by_file": dict(deps_by_file),
            "complexity_statistics": complexity_stats,
            "include_transitive": include_transitive,
            "max_depth": max_depth
        }
    
    def _calculate_dependency_complexity(self, dependencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算依赖的复杂度统计"""
        if not dependencies:
            return {}
        
        complexities = [dep.get("dep_cyclomatic_complexity") for dep in dependencies 
                       if dep.get("dep_cyclomatic_complexity") is not None]
        
        if not complexities:
            return {}
        
        return {
            "total_dependencies": len(dependencies),
            "dependencies_with_complexity": len(complexities),
            "average_complexity": sum(complexities) / len(complexities),
            "max_complexity": max(complexities),
            "min_complexity": min(complexities),
            "high_complexity_deps": [dep for dep in dependencies 
                                   if dep.get("dep_cyclomatic_complexity", 0) > 10]
        }
    
    def find_function_by_name(self, function_name: str, exact_match: bool = True) -> List[Dict[str, Any]]:
        """
        根据函数名查找函�?
        
        Args:
            function_name: 函数�?
            exact_match: 是否精确匹配
            
        Returns:
            函数信息列表
        """
        if exact_match:
            query = """
            MATCH (fn:Function {name: $function_name})
            RETURN fn
            ORDER BY fn.filepath, fn.start_line
            """
        else:
            query = """
            MATCH (fn:Function)
            WHERE fn.name CONTAINS $function_name
            RETURN fn
            ORDER BY fn.name, fn.filepath, fn.start_line
            """
        
        result = self._run_query(query, {"function_name": function_name})
        return [dict(record["fn"]) for record in result]
    
    def get_function_call_graph(self, 
                               function_name: str, 
                               max_depth: int = 3,
                               direction: str = "both") -> Dict[str, Any]:
        """
        获取函数的调用图
        
        Args:
            function_name: 函数�?
            max_depth: 最大深�?
            direction: 方向 ("incoming", "outgoing", "both")
            
        Returns:
            调用图信�?
        """
        logger.info(f"获取调用�? {function_name}, 深度: {max_depth}, 方向: {direction}")
        
        nodes = set()
        edges = []
        
        if direction in ["incoming", "both"]:
            # 查找调用�?
            incoming_query = f"""
            MATCH path = (caller:Function)-[:INTERNAL_CALLS|CALLS*1..{max_depth}]->(target:Function {{name: $function_name}})
            RETURN path
            """
            
            incoming_results = self._run_query(incoming_query, {"function_name": function_name})
            
            for record in incoming_results:
                path = record["path"]
                for rel in path.relationships:
                    source = rel.start_node
                    target = rel.end_node
                    
                    nodes.add((source["id"], source["name"], source.get("filepath", "")))
                    nodes.add((target["id"], target["name"], target.get("filepath", "")))
                    
                    edges.append({
                        "source": source["name"],
                        "target": target["name"],
                        "type": rel.type,
                        "properties": dict(rel)
                    })
        
        if direction in ["outgoing", "both"]:
            # 查找被调用�?
            outgoing_query = f"""
            MATCH path = (source:Function {{name: $function_name}})-[:INTERNAL_CALLS|CALLS*1..{max_depth}]->(callee:Function)
            RETURN path
            """
            
            outgoing_results = self._run_query(outgoing_query, {"function_name": function_name})
            
            for record in outgoing_results:
                path = record["path"]
                for rel in path.relationships:
                    source = rel.start_node
                    target = rel.end_node
                    
                    nodes.add((source["id"], source["name"], source.get("filepath", "")))
                    nodes.add((target["id"], target["name"], target.get("filepath", "")))
                    
                    edges.append({
                        "source": source["name"],
                        "target": target["name"],
                        "type": rel.type,
                        "properties": dict(rel)
                    })
        
        return {
            "function_name": function_name,
            "nodes": [{"id": node_id, "name": name, "filepath": filepath} 
                     for node_id, name, filepath in nodes],
            "edges": edges,
            "max_depth": max_depth,
            "direction": direction,
            "total_nodes": len(nodes),
            "total_edges": len(edges)
        }
    
    def get_file_dependencies(self, file_path: str) -> Dict[str, Any]:
        """
        获取文件的依赖关�?
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件依赖信息
        """
        logger.info(f"获取文件依赖: {file_path}")
        
        # 获取文件中的函数
        functions_query = """
        MATCH (f:File {path: $file_path})-[:DECLARES]->(fn:Function)
        RETURN fn.id as function_id,
               fn.name as function_name,
               fn.start_line as start_line,
               fn.end_line as end_line,
               fn.complexity_level as complexity_level
        ORDER BY fn.start_line
        """
        
        functions = self._run_query(functions_query, {"file_path": file_path})
        
        # 获取文件间的依赖关系
        file_deps_query = """
        MATCH (f1:File {path: $file_path})-[:DECLARES]->(fn1:Function)-[:INTERNAL_CALLS|CALLS]->(fn2:Function)<-[:DECLARES]-(f2:File)
        WHERE f1 <> f2
        RETURN DISTINCT f2.path as dependent_file,
               count(fn1) as dependency_count
        ORDER BY dependency_count DESC
        """
        
        file_deps = self._run_query(file_deps_query, {"file_path": file_path})
        
        return {
            "file_path": file_path,
            "functions": functions,
            "file_dependencies": file_deps,
            "total_functions": len(functions),
            "total_dependent_files": len(file_deps)
        }
