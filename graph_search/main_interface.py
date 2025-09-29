
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

from .basic_queries import BasicQueries
from .semantic_search import SemanticSearch
from .graph_analysis import GraphAnalysis
from .config import config

logger = logging.getLogger(__name__)

class Neo4jSearchEngine:
    
    def __init__(self, 
                 neo4j_uri: Optional[str] = None,
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 neo4j_database: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 openai_api_key: Optional[str] = None):
        """
        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            neo4j_database: 数据库名称
            embedding_model: 嵌入模型名称
            openai_api_key: OpenAI API密钥
        """
        # 初始化各个功能模块
        self.basic_queries = BasicQueries(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
        self.semantic_search = SemanticSearch(neo4j_uri, neo4j_user, neo4j_password, neo4j_database, embedding_model, openai_api_key)
        self.graph_analysis = GraphAnalysis(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
        
        logger.info("Neo4j搜索引擎初始化完成")
    
    def close(self):
        """关闭所有连接"""
        self.basic_queries.close()
        self.semantic_search.close()
        self.graph_analysis.close()
        logger.info("所有连接已关闭")
    
    # ==================== 基础查询功能 ====================
    
    def find_api_callers(self, api_name: str, max_depth: int = 3, include_external: bool = True) -> Dict[str, Any]:
        """查找API的调用者"""
        return self.basic_queries.find_api_callers(api_name, max_depth, include_external)
    
    def find_api_callees(self, api_name: str, max_depth: int = 3, include_external: bool = True) -> Dict[str, Any]:
        """查找API的被调用者"""
        return self.basic_queries.find_api_callees(api_name, max_depth, include_external)
    
    def get_dependency_list(self, function_name: str, include_transitive: bool = True, max_depth: int = 5) -> Dict[str, Any]:
        """获取函数的依赖清单"""
        return self.basic_queries.get_dependency_list(function_name, include_transitive, max_depth)
    
    def find_function_by_name(self, function_name: str, exact_match: bool = True) -> List[Dict[str, Any]]:
        """根据函数名查找函数"""
        return self.basic_queries.find_function_by_name(function_name, exact_match)
    
    def get_function_call_graph(self, function_name: str, max_depth: int = 3, direction: str = "both") -> Dict[str, Any]:
        """获取函数的调用图"""
        return self.basic_queries.get_function_call_graph(function_name, max_depth, direction)
    
    def get_file_dependencies(self, file_path: str) -> Dict[str, Any]:
        """获取文件的依赖关系"""
        return self.basic_queries.get_file_dependencies(file_path)
    
    # ==================== 语义搜索功能 ====================
    
    def search_by_natural_language(self, 
                                  query: str, 
                                  limit: int = 10,
                                  search_type: str = "hybrid",
                                  similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """使用自然语言搜索代码"""
        return self.semantic_search.search_by_natural_language(query, limit, search_type, similarity_threshold)
    
    def search_by_complexity(self, 
                           complexity_level: Optional[str] = None,
                           min_lines: Optional[int] = None,
                           max_lines: Optional[int] = None,
                           min_complexity: Optional[float] = None,
                           max_complexity: Optional[float] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """按复杂度搜索函数"""
        return self.semantic_search.search_by_complexity(complexity_level, min_lines, max_lines, min_complexity, max_complexity, limit)
    
    def search_similar_functions(self, 
                               function_name: str,
                               limit: int = 10,
                               similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """查找相似函数"""
        return self.semantic_search.search_similar_functions(function_name, limit, similarity_threshold)
    
    # ==================== 图分析功能 ====================
    
    def calculate_centrality(self, 
                           algorithm: str = "pagerank",
                           top_k: int = 20,
                           include_weights: bool = True) -> Dict[str, Any]:
        """计算中心性指标"""
        return self.graph_analysis.calculate_centrality(algorithm, top_k, include_weights)
    
    def find_communities(self, 
                        algorithm: str = "louvain",
                        min_community_size: int = 2,
                        resolution: float = 1.0) -> Dict[str, Any]:
        """社区发现"""
        return self.graph_analysis.find_communities(algorithm, min_community_size, resolution)
    
    def calculate_similarity_matrix(self, 
                                  function_names: List[str],
                                  similarity_type: str = "structural") -> Dict[str, Any]:
        """计算函数间的相似度矩阵"""
        return self.graph_analysis.calculate_similarity_matrix(function_names, similarity_type)
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        return self.graph_analysis.get_graph_statistics()
    
    # ==================== 综合分析功能 ====================
    
    def analyze_function_importance(self, 
                                  function_name: str,
                                  include_centrality: bool = True,
                                  include_community: bool = True,
                                  include_dependencies: bool = True) -> Dict[str, Any]:
        """
        综合分析函数重要性
        
        Args:
            function_name: 函数名
            include_centrality: 是否包含中心性分析
            include_community: 是否包含社区分析
            include_dependencies: 是否包含依赖分析
            
        Returns:
            综合分析结果
        """
        logger.info(f"分析函数重要性: {function_name}")
        
        analysis_result = {
            "function_name": function_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_components": []
        }
        
        # 基础信息
        functions = self.find_function_by_name(function_name, exact_match=True)
        if not functions:
            analysis_result["error"] = f"未找到函数: {function_name}"
            return analysis_result
        
        function_info = functions[0]
        analysis_result["function_info"] = function_info
        
        # 中心性分析
        if include_centrality:
            try:
                centrality_result = self.calculate_centrality("pagerank", top_k=100)
                # 查找目标函数在中心性排名中的位置
                function_rank = None
                for i, result in enumerate(centrality_result["results"]):
                    if result["node_name"] == function_name:
                        function_rank = i + 1
                        break
                
                analysis_result["centrality_analysis"] = {
                    "function_rank": function_rank,
                    "total_functions": centrality_result["total_nodes"],
                    "top_centrality_functions": centrality_result["results"][:10]
                }
                analysis_result["analysis_components"].append("centrality")
            except Exception as e:
                logger.error(f"中心性分析失败: {e}")
                analysis_result["centrality_analysis"] = {"error": str(e)}
        
        # 社区分析
        if include_community:
            try:
                community_result = self.find_communities("louvain", min_community_size=2)
                # 查找目标函数所属的社区
                function_community = None
                for community in community_result["communities"]:
                    for node in community["nodes"]:
                        if node["node_name"] == function_name:
                            function_community = community
                            break
                    if function_community:
                        break
                
                analysis_result["community_analysis"] = {
                    "function_community": function_community,
                    "total_communities": community_result["total_communities"],
                    "largest_communities": community_result["communities"][:5]
                }
                analysis_result["analysis_components"].append("community")
            except Exception as e:
                logger.error(f"社区分析失败: {e}")
                analysis_result["community_analysis"] = {"error": str(e)}
        
        # 依赖分析
        if include_dependencies:
            try:
                dependency_result = self.get_dependency_list(function_name, include_transitive=True, max_depth=3)
                analysis_result["dependency_analysis"] = dependency_result
                analysis_result["analysis_components"].append("dependencies")
            except Exception as e:
                logger.error(f"依赖分析失败: {e}")
                analysis_result["dependency_analysis"] = {"error": str(e)}
        
        return analysis_result
    
    def comprehensive_search(self, 
                           query: str,
                           search_type: str = "hybrid",
                           include_analysis: bool = False,
                           limit: int = 10) -> Dict[str, Any]:
        """
        综合搜索功能
        
        Args:
            query: 搜索查询
            search_type: 搜索类型
            include_analysis: 是否包含图分析
            limit: 结果数量限制
            
        Returns:
            综合搜索结果
        """
        logger.info(f"执行综合搜索: {query}")
        
        search_result = {
            "query": query,
            "search_type": search_type,
            "search_timestamp": datetime.now().isoformat(),
            "results": []
        }
        
        # 执行语义搜索
        try:
            semantic_results = self.search_by_natural_language(query, limit, search_type)
            search_result["semantic_results"] = semantic_results
            search_result["total_semantic_results"] = len(semantic_results)
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            search_result["semantic_results"] = []
            search_result["semantic_search_error"] = str(e)
        
        # 如果启用分析，对前几个结果进行重要性分析
        if include_analysis and semantic_results:
            analysis_results = []
            for i, result in enumerate(semantic_results[:3]):  # 只分析前3个结果
                try:
                    function_name = result.get("name")
                    if function_name:
                        importance_analysis = self.analyze_function_importance(
                            function_name, 
                            include_centrality=True, 
                            include_community=True, 
                            include_dependencies=True
                        )
                        analysis_results.append(importance_analysis)
                except Exception as e:
                    logger.error(f"重要性分析失败: {e}")
            
            search_result["importance_analysis"] = analysis_results
        
        return search_result
    
    def export_results(self, 
                      results: Union[Dict[str, Any], List[Dict[str, Any]]], 
                      output_file: str, 
                      format: str = "json") -> None:
        """
        导出搜索结果
        
        Args:
            results: 搜索结果
            output_file: 输出文件路径
            format: 输出格式 ("json", "markdown")
        """
        logger.info(f"导出结果到: {output_file}, 格式: {format}")
        
        if format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        elif format == "markdown":
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# 搜索结果\n\n")
                f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if isinstance(results, dict):
                    self._export_dict_to_markdown(f, results, level=1)
                elif isinstance(results, list):
                    for i, result in enumerate(results, 1):
                        f.write(f"## 结果 {i}\n\n")
                        self._export_dict_to_markdown(f, result, level=2)
        
        logger.info(f"结果已导出到: {output_file}")
    
    def _export_dict_to_markdown(self, f, data: Dict[str, Any], level: int = 1) -> None:
        """将字典数据导出为Markdown格式"""
        for key, value in data.items():
            if isinstance(value, dict):
                f.write(f"{'#' * level} {key}\n\n")
                self._export_dict_to_markdown(f, value, level + 1)
            elif isinstance(value, list):
                f.write(f"**{key}**:\n\n")
                for i, item in enumerate(value, 1):
                    if isinstance(item, dict):
                        f.write(f"{i}. ")
                        self._export_dict_to_markdown(f, item, level + 1)
                    else:
                        f.write(f"{i}. {item}\n")
                f.write("\n")
            else:
                f.write(f"**{key}**: {value}\n\n")
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库基本信息"""
        return self.basic_queries.get_database_info()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
