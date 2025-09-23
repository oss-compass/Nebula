import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("networkx not available, graph analysis will be limited")

from .base import BaseSearch
from .config import config

logger = logging.getLogger(__name__)

class GraphAnalysis(BaseSearch):
    """图分析功能类"""
    
    def __init__(self, 
                 neo4j_uri: Optional[str] = None,
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 neo4j_database: Optional[str] = None):
        """
        初始化图分析�?
        
        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_user: Neo4j用户�?
            neo4j_password: Neo4j密码
            neo4j_database: 数据库名�?
        """
        super().__init__(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
        
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX不可用，部分图分析功能将被禁�?)
        
        logger.info("图分析模块初始化完成")
    
    def calculate_centrality(self, 
                           algorithm: str = "pagerank",
                           top_k: int = 20,
                           include_weights: bool = True) -> Dict[str, Any]:
        """
        计算中心性指�?
        
        Args:
            algorithm: 中心性算�?("pagerank", "betweenness", "closeness", "eigenvector")
            top_k: 返回前k个结�?
            include_weights: 是否考虑边的权重
            
        Returns:
            中心性分析结�?
        """
        logger.info(f"计算中心�? {algorithm}, top_k: {top_k}")
        
        if not NETWORKX_AVAILABLE:
            return self._calculate_centrality_neo4j(algorithm, top_k)
        
        # 构建NetworkX�?
        G = self._build_networkx_graph(include_weights)
        
        if G.number_of_nodes() == 0:
            logger.warning("图中没有节点")
            return {"algorithm": algorithm, "results": [], "total_nodes": 0}
        
        # 计算中心�?
        if algorithm == "pagerank":
            centrality_scores = nx.pagerank(G, weight='weight' if include_weights else None)
        elif algorithm == "betweenness":
            centrality_scores = nx.betweenness_centrality(G, weight='weight' if include_weights else None)
        elif algorithm == "closeness":
            centrality_scores = nx.closeness_centrality(G, distance='weight' if include_weights else None)
        elif algorithm == "eigenvector":
            try:
                centrality_scores = nx.eigenvector_centrality(G, weight='weight' if include_weights else None)
            except nx.PowerIterationFailedConvergence:
                logger.warning("特征向量中心性计算未收敛，使用度中心性替�?)
                centrality_scores = nx.degree_centrality(G)
        else:
            raise ValueError(f"不支持的中心性算�? {algorithm}")
        
        # 排序并返回前k个结�?
        sorted_scores = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (node_id, score) in enumerate(sorted_scores[:top_k]):
            # 获取节点详细信息
            node_info = self._get_node_info(node_id)
            results.append({
                "rank": i + 1,
                "node_id": node_id,
                "centrality_score": float(score),
                **node_info
            })
        
        return {
            "algorithm": algorithm,
            "results": results,
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "include_weights": include_weights
        }
    
    def _calculate_centrality_neo4j(self, algorithm: str, top_k: int) -> Dict[str, Any]:
        """使用Neo4j内置算法计算中心�?""
        logger.info(f"使用Neo4j计算中心�? {algorithm}")
        
        if algorithm == "pagerank":
            query = f"""
            CALL gds.pageRank.stream('function-graph')
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id as node_id,
                   gds.util.asNode(nodeId).name as node_name,
                   gds.util.asNode(nodeId).filepath as filepath,
                   score
            ORDER BY score DESC
            LIMIT {top_k}
            """
        elif algorithm == "betweenness":
            query = f"""
            CALL gds.betweenness.stream('function-graph')
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id as node_id,
                   gds.util.asNode(nodeId).name as node_name,
                   gds.util.asNode(nodeId).filepath as filepath,
                   score
            ORDER BY score DESC
            LIMIT {top_k}
            """
        else:
            # 回退到简单的度中心�?
            query = f"""
            MATCH (fn:Function)
            OPTIONAL MATCH (fn)-[r:INTERNAL_CALLS|CALLS]->()
            WITH fn, count(r) as out_degree
            OPTIONAL MATCH ()-[r:INTERNAL_CALLS|CALLS]->(fn)
            WITH fn, out_degree, count(r) as in_degree
            WITH fn, out_degree + in_degree as total_degree
            RETURN fn.id as node_id,
                   fn.name as node_name,
                   fn.filepath as filepath,
                   total_degree as score
            ORDER BY total_degree DESC
            LIMIT {top_k}
            """
        
        try:
            results = self._run_query(query)
            formatted_results = []
            for i, record in enumerate(results):
                formatted_results.append({
                    "rank": i + 1,
                    "node_id": record.get("node_id"),
                    "centrality_score": float(record.get("score", 0)),
                    "node_name": record.get("node_name"),
                    "filepath": record.get("filepath")
                })
            
            return {
                "algorithm": algorithm,
                "results": formatted_results,
                "total_nodes": len(results),
                "include_weights": False
            }
        except Exception as e:
            logger.error(f"Neo4j中心性计算失�? {e}")
            return {"algorithm": algorithm, "results": [], "error": str(e)}
    
    def _build_networkx_graph(self, include_weights: bool = True) -> 'nx.DiGraph':
        """构建NetworkX�?""
        G = nx.DiGraph()
        
        # 获取所有节�?
        nodes_query = """
        MATCH (fn:Function)
        RETURN fn.id as id, fn.name as name, fn.filepath as filepath,
               fn.complexity_level as complexity_level,
               fn.cyclomatic_complexity as cyclomatic_complexity
        """
        
        nodes = self._run_query(nodes_query)
        for node in nodes:
            G.add_node(
                node["id"],
                name=node["name"],
                filepath=node["filepath"],
                complexity_level=node["complexity_level"],
                cyclomatic_complexity=node["cyclomatic_complexity"]
            )
        
        # 获取所有边
        edges_query = """
        MATCH (source:Function)-[r:INTERNAL_CALLS|CALLS]->(target:Function)
        RETURN source.id as source, target.id as target, type(r) as relationship_type
        """
        
        edges = self._run_query(edges_query)
        for edge in edges:
            weight = 1.0
            if include_weights:
                # 可以根据关系类型设置不同权重
                if edge["relationship_type"] == "INTERNAL_CALLS":
                    weight = 1.0
                elif edge["relationship_type"] == "CALLS":
                    weight = 0.5
            
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=weight,
                relationship_type=edge["relationship_type"]
            )
        
        logger.info(f"构建NetworkX图完�? {G.number_of_nodes()}个节�? {G.number_of_edges()}条边")
        return G
    
    def _get_node_info(self, node_id: str) -> Dict[str, Any]:
        """获取节点详细信息"""
        query = """
        MATCH (fn:Function {id: $node_id})
        RETURN fn.name as node_name,
               fn.filepath as filepath,
               fn.complexity_level as complexity_level,
               fn.cyclomatic_complexity as cyclomatic_complexity,
               fn.lines_of_code as lines_of_code
        """
        
        result = self._run_query_single(query, {"node_id": node_id})
        return dict(result) if result else {}
    
    def find_communities(self, 
                        algorithm: str = "louvain",
                        min_community_size: int = 2,
                        resolution: float = 1.0) -> Dict[str, Any]:
        """
        社区发现
        
        Args:
            algorithm: 社区发现算法 ("louvain", "leiden", "label_propagation")
            min_community_size: 最小社区大�?
            resolution: 分辨率参数（仅对louvain和leiden有效�?
            
        Returns:
            社区发现结果
        """
        logger.info(f"社区发现: {algorithm}, 最小社区大�? {min_community_size}")
        
        if not NETWORKX_AVAILABLE:
            return self._find_communities_neo4j(algorithm, min_community_size)
        
        # 构建无向图（社区发现通常在无向图上进行）
        G = self._build_networkx_graph(include_weights=True)
        G_undirected = G.to_undirected()
        
        if G_undirected.number_of_nodes() == 0:
            logger.warning("图中没有节点")
            return {"algorithm": algorithm, "communities": [], "total_nodes": 0}
        
        # 执行社区发现
        if algorithm == "louvain":
            try:
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.louvain_communities(G_undirected, weight='weight', resolution=resolution)
            except ImportError:
                logger.warning("louvain算法不可用，使用标签传播")
                communities = list(nx_comm.label_propagation_communities(G_undirected))
        elif algorithm == "leiden":
            try:
                import leidenalg
                import igraph as ig
                # 转换为igraph格式
                ig_graph = ig.Graph.from_networkx(G_undirected)
                partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition, resolution_parameter=resolution)
                communities = [list(partition[i]) for i in range(len(partition))]
            except ImportError:
                logger.warning("leiden算法不可用，使用louvain")
                import networkx.algorithms.community as nx_comm
                communities = nx_comm.louvain_communities(G_undirected, weight='weight')
        elif algorithm == "label_propagation":
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.label_propagation_communities(G_undirected))
        else:
            raise ValueError(f"不支持的社区发现算法: {algorithm}")
        
        # 过滤小社�?
        filtered_communities = [comm for comm in communities if len(comm) >= min_community_size]
        
        # 格式化结�?
        community_results = []
        for i, community in enumerate(filtered_communities):
            community_info = {
                "community_id": i + 1,
                "size": len(community),
                "nodes": []
            }
            
            for node_id in community:
                node_info = self._get_node_info(node_id)
                community_info["nodes"].append({
                    "node_id": node_id,
                    **node_info
                })
            
            community_results.append(community_info)
        
        # 按社区大小排�?
        community_results.sort(key=lambda x: x["size"], reverse=True)
        
        return {
            "algorithm": algorithm,
            "communities": community_results,
            "total_communities": len(community_results),
            "total_nodes": G_undirected.number_of_nodes(),
            "min_community_size": min_community_size,
            "resolution": resolution
        }
    
    def _find_communities_neo4j(self, algorithm: str, min_community_size: int) -> Dict[str, Any]:
        """使用Neo4j内置算法进行社区发现"""
        logger.info(f"使用Neo4j进行社区发现: {algorithm}")
        
        if algorithm == "louvain":
            query = """
            CALL gds.louvain.stream('function-graph')
            YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId).id as node_id,
                   gds.util.asNode(nodeId).name as node_name,
                   gds.util.asNode(nodeId).filepath as filepath,
                   communityId
            ORDER BY communityId, node_name
            """
        else:
            # 回退到弱连通分�?
            query = """
            CALL gds.wcc.stream('function-graph')
            YIELD nodeId, componentId
            RETURN gds.util.asNode(nodeId).id as node_id,
                   gds.util.asNode(nodeId).name as node_name,
                   gds.util.asNode(nodeId).filepath as filepath,
                   componentId
            ORDER BY componentId, node_name
            """
        
        try:
            results = self._run_query(query)
            
            # 按社区分�?
            communities_dict = defaultdict(list)
            for record in results:
                community_id = record["componentId"] if algorithm != "louvain" else record["communityId"]
                communities_dict[community_id].append({
                    "node_id": record["node_id"],
                    "node_name": record["node_name"],
                    "filepath": record["filepath"]
                })
            
            # 过滤小社区并格式�?
            community_results = []
            for i, (community_id, nodes) in enumerate(communities_dict.items()):
                if len(nodes) >= min_community_size:
                    community_results.append({
                        "community_id": i + 1,
                        "size": len(nodes),
                        "nodes": nodes
                    })
            
            # 按社区大小排�?
            community_results.sort(key=lambda x: x["size"], reverse=True)
            
            return {
                "algorithm": algorithm,
                "communities": community_results,
                "total_communities": len(community_results),
                "total_nodes": len(results),
                "min_community_size": min_community_size
            }
        except Exception as e:
            logger.error(f"Neo4j社区发现失败: {e}")
            return {"algorithm": algorithm, "communities": [], "error": str(e)}
    
    def calculate_similarity_matrix(self, 
                                  function_names: List[str],
                                  similarity_type: str = "structural") -> Dict[str, Any]:
        """
        计算函数间的相似度矩�?
        
        Args:
            function_names: 函数名列�?
            similarity_type: 相似度类�?("structural", "semantic", "jaccard")
            
        Returns:
            相似度矩�?
        """
        logger.info(f"计算相似度矩�? {len(function_names)}个函�? 类型: {similarity_type}")
        
        if similarity_type == "structural":
            return self._calculate_structural_similarity(function_names)
        elif similarity_type == "semantic":
            return self._calculate_semantic_similarity(function_names)
        elif similarity_type == "jaccard":
            return self._calculate_jaccard_similarity(function_names)
        else:
            raise ValueError(f"不支持的相似度类�? {similarity_type}")
    
    def _calculate_structural_similarity(self, function_names: List[str]) -> Dict[str, Any]:
        """计算结构相似�?""
        # 获取函数的调用关�?
        functions_data = {}
        for func_name in function_names:
            # 获取函数的调用和被调用关�?
            query = """
            MATCH (fn:Function {name: $func_name})
            OPTIONAL MATCH (fn)-[:INTERNAL_CALLS|CALLS]->(callee:Function)
            OPTIONAL MATCH (caller:Function)-[:INTERNAL_CALLS|CALLS]->(fn)
            RETURN fn.name as name,
                   collect(DISTINCT callee.name) as callees,
                   collect(DISTINCT caller.name) as callers,
                   fn.cyclomatic_complexity as complexity,
                   fn.lines_of_code as lines
            """
            
            result = self._run_query_single(query, {"func_name": func_name})
            if result:
                functions_data[func_name] = {
                    "callees": set(result["callees"]) if result["callees"] else set(),
                    "callers": set(result["callers"]) if result["callers"] else set(),
                    "complexity": result["complexity"] or 0,
                    "lines": result["lines"] or 0
                }
        
        # 计算相似度矩�?
        n = len(function_names)
        similarity_matrix = np.zeros((n, n))
        
        for i, func1 in enumerate(function_names):
            for j, func2 in enumerate(function_names):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                elif func1 in functions_data and func2 in functions_data:
                    data1 = functions_data[func1]
                    data2 = functions_data[func2]
                    
                    # 计算Jaccard相似度（基于调用关系�?
                    callees_intersection = len(data1["callees"] & data2["callees"])
                    callees_union = len(data1["callees"] | data2["callees"])
                    callers_intersection = len(data1["callers"] & data2["callers"])
                    callers_union = len(data1["callers"] | data2["callers"])
                    
                    callees_sim = callees_intersection / callees_union if callees_union > 0 else 0
                    callers_sim = callers_intersection / callers_union if callers_union > 0 else 0
                    
                    # 复杂度相似度
                    complexity_sim = 1 - abs(data1["complexity"] - data2["complexity"]) / max(data1["complexity"], data2["complexity"], 1)
                    
                    # 综合相似�?
                    similarity_matrix[i][j] = (callees_sim + callers_sim + complexity_sim) / 3
        
        return {
            "similarity_type": "structural",
            "function_names": function_names,
            "similarity_matrix": similarity_matrix.tolist(),
            "functions_data": {name: {
                "callees": list(data["callees"]),
                "callers": list(data["callers"]),
                "complexity": data["complexity"],
                "lines": data["lines"]
            } for name, data in functions_data.items()}
        }
    
    def _calculate_semantic_similarity(self, function_names: List[str]) -> Dict[str, Any]:
        """计算语义相似度（需要嵌入模型）"""
        # 这里需要结合SemanticSearch类的功能
        # 暂时返回结构相似�?
        logger.warning("语义相似度计算需要嵌入模型，回退到结构相似度")
        return self._calculate_structural_similarity(function_names)
    
    def _calculate_jaccard_similarity(self, function_names: List[str]) -> Dict[str, Any]:
        """计算Jaccard相似�?""
        # 获取每个函数的特征集�?
        functions_features = {}
        for func_name in function_names:
            query = """
            MATCH (fn:Function {name: $func_name})
            OPTIONAL MATCH (fn)-[:INTERNAL_CALLS|CALLS]->(callee:Function)
            RETURN fn.name as name,
                   collect(DISTINCT callee.name) as callees
            """
            
            result = self._run_query_single(query, {"func_name": func_name})
            if result:
                functions_features[func_name] = set(result["callees"]) if result["callees"] else set()
        
        # 计算Jaccard相似度矩�?
        n = len(function_names)
        similarity_matrix = np.zeros((n, n))
        
        for i, func1 in enumerate(function_names):
            for j, func2 in enumerate(function_names):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                elif func1 in functions_features and func2 in functions_features:
                    set1 = functions_features[func1]
                    set2 = functions_features[func2]
                    
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    
                    similarity_matrix[i][j] = intersection / union if union > 0 else 0
        
        return {
            "similarity_type": "jaccard",
            "function_names": function_names,
            "similarity_matrix": similarity_matrix.tolist(),
            "functions_features": {name: list(features) for name, features in functions_features.items()}
        }
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图统计信�?""
        logger.info("获取图统计信�?)
        
        # 基本统计
        basic_stats_query = """
        MATCH (fn:Function)
        OPTIONAL MATCH (fn)-[r:INTERNAL_CALLS|CALLS]->()
        WITH fn, count(r) as out_degree
        OPTIONAL MATCH ()-[r:INTERNAL_CALLS|CALLS]->(fn)
        WITH fn, out_degree, count(r) as in_degree
        RETURN count(fn) as total_nodes,
               sum(out_degree) as total_edges,
               avg(out_degree + in_degree) as avg_degree,
               max(out_degree + in_degree) as max_degree,
               min(out_degree + in_degree) as min_degree
        """
        
        basic_stats = self._run_query_single(basic_stats_query)
        
        # 连通性统�?
        connectivity_query = """
        CALL gds.wcc.stream('function-graph')
        YIELD componentId
        WITH componentId, count(*) as component_size
        RETURN count(componentId) as total_components,
               max(component_size) as largest_component_size,
               avg(component_size) as avg_component_size
        """
        
        try:
            connectivity_stats = self._run_query_single(connectivity_query)
        except Exception as e:
            logger.warning(f"连通性统计失�? {e}")
            connectivity_stats = {}
        
        # 复杂度分�?
        complexity_query = """
        MATCH (fn:Function)
        WHERE fn.complexity_level IS NOT NULL
        RETURN fn.complexity_level as level, count(fn) as count
        ORDER BY count DESC
        """
        
        complexity_dist = self._run_query(complexity_query)
        
        return {
            "basic_statistics": dict(basic_stats) if basic_stats else {},
            "connectivity_statistics": dict(connectivity_stats) if connectivity_stats else {},
            "complexity_distribution": [dict(record) for record in complexity_dist],
            "analysis_timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else None
        }
