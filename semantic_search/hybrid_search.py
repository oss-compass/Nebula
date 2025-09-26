#!/usr/bin/env python3
"""
混合搜索模块
结合向量搜索和图数据库搜索，提供更强大的代码搜索功能
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# 尝试导入Neo4j
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from .semantic_search import SearchQuery, SearchResult, SemanticSearcher
from .sync_indexer import SyncSemanticIndexer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """混合搜索配置"""
    # 权重配置
    vector_weight: float = 0.7
    graph_weight: float = 0.3
    
    # 图数据库配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "90879449Drq"
    neo4j_database: Optional[str] = None
    
    # 搜索配置
    max_results: int = 50
    enable_relationship_boost: bool = True
    relationship_boost_factor: float = 1.2
    
    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1小时


@dataclass
class HybridSearchResult:
    """混合搜索结果"""
    embedding: Any  # CodeEmbedding
    vector_similarity: float
    graph_score: float
    combined_score: float
    rank: int
    highlights: List[str] = None
    context: Dict[str, Any] = None
    graph_context: Dict[str, Any] = None


class GraphSearcher:
    """图数据库搜索器"""
    
    def __init__(self, config: HybridSearchConfig):
        self.config = config
        self.driver = None
        
        if NEO4J_AVAILABLE:
            self._init_neo4j()
        else:
            logger.warning("Neo4j not available, graph search will be disabled")
    
    def _init_neo4j(self):
        """初始化Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            # 测试连接
            with self.driver.session(database=self.config.neo4j_database) as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def search_by_function_name(self, function_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """按函数名搜索"""
        if not self.driver:
            return []
        
        query = """
        MATCH (fn:Function)
        WHERE fn.name CONTAINS $function_name
        RETURN fn.name, fn.filepath, fn.complexity_score, fn.importance_score,
               fn.parent_class_name, fn.function_type, fn.is_async, fn.is_test
        ORDER BY fn.importance_score DESC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, {"function_name": function_name, "limit": limit})
            return [dict(record) for record in result]
    
    def search_by_file_path(self, file_pattern: str, limit: int = 20) -> List[Dict[str, Any]]:
        """按文件路径搜索"""
        if not self.driver:
            return []
        
        query = """
        MATCH (fn:Function)
        WHERE fn.filepath CONTAINS $file_pattern
        RETURN fn.name, fn.filepath, fn.complexity_score, fn.importance_score,
               fn.parent_class_name, fn.function_type, fn.is_async, fn.is_test
        ORDER BY fn.importance_score DESC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, {"file_pattern": file_pattern, "limit": limit})
            return [dict(record) for record in result]
    
    def search_by_complexity(self, min_complexity: float, max_complexity: float = None, 
                           limit: int = 20) -> List[Dict[str, Any]]:
        """按复杂度搜索"""
        if not self.driver:
            return []
        
        if max_complexity is None:
            query = """
            MATCH (fn:Function)
            WHERE fn.complexity_score >= $min_complexity
            RETURN fn.name, fn.filepath, fn.complexity_score, fn.importance_score,
                   fn.parent_class_name, fn.function_type, fn.is_async, fn.is_test
            ORDER BY fn.complexity_score DESC
            LIMIT $limit
            """
            params = {"min_complexity": min_complexity, "limit": limit}
        else:
            query = """
            MATCH (fn:Function)
            WHERE fn.complexity_score >= $min_complexity AND fn.complexity_score <= $max_complexity
            RETURN fn.name, fn.filepath, fn.complexity_score, fn.importance_score,
                   fn.parent_class_name, fn.function_type, fn.is_async, fn.is_test
            ORDER BY fn.complexity_score DESC
            LIMIT $limit
            """
            params = {"min_complexity": min_complexity, "max_complexity": max_complexity, "limit": limit}
        
        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, params)
            return [dict(record) for record in result]
    
    def search_by_function_type(self, function_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        """按函数类型搜索"""
        if not self.driver:
            return []
        
        query = """
        MATCH (fn:Function)
        WHERE fn.function_type = $function_type
        RETURN fn.name, fn.filepath, fn.complexity_score, fn.importance_score,
               fn.parent_class_name, fn.function_type, fn.is_async, fn.is_test
        ORDER BY fn.importance_score DESC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, {"function_type": function_type, "limit": limit})
            return [dict(record) for record in result]
    
    def search_by_class(self, class_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """按类搜索"""
        if not self.driver:
            return []
        
        query = """
        MATCH (c:Class {name: $class_name})-[:HAS_METHOD]->(fn:Function)
        RETURN fn.name, fn.filepath, fn.complexity_score, fn.importance_score,
               fn.parent_class_name, fn.function_type, fn.is_async, fn.is_test
        ORDER BY fn.importance_score DESC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, {"class_name": class_name, "limit": limit})
            return [dict(record) for record in result]
    
    def search_by_calls(self, function_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """搜索调用指定函数的函数"""
        if not self.driver:
            return []
        
        query = """
        MATCH (caller:Function)-[:CALLS]->(callee:Function {name: $function_name})
        RETURN caller.name, caller.filepath, caller.complexity_score, caller.importance_score,
               caller.parent_class_name, caller.function_type, caller.is_async, caller.is_test
        ORDER BY caller.importance_score DESC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, {"function_name": function_name, "limit": limit})
            return [dict(record) for record in result]
    
    def search_by_relationships(self, function_name: str, relationship_types: List[str] = None,
                              limit: int = 20) -> List[Dict[str, Any]]:
        """基于关系搜索"""
        if not self.driver:
            return []
        
        if relationship_types is None:
            relationship_types = ["CALLS", "HAS_METHOD"]
        
        # 构建关系查询
        rel_conditions = []
        for rel_type in relationship_types:
            rel_conditions.append(f"(fn)-[:{rel_type}]->(related)")
            rel_conditions.append(f"(related)-[:{rel_type}]->(fn)")
        
        rel_query = " OR ".join(rel_conditions)
        
        query = f"""
        MATCH (fn:Function {{name: $function_name}})
        MATCH {rel_query}
        RETURN DISTINCT related.name as name, related.filepath as filepath, 
               related.complexity_score as complexity_score, related.importance_score as importance_score,
               related.parent_class_name as parent_class_name, related.function_type as function_type,
               related.is_async as is_async, related.is_test as is_test
        ORDER BY related.importance_score DESC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, {"function_name": function_name, "limit": limit})
            return [dict(record) for record in result]
    
    def get_function_context(self, function_name: str, file_path: str = None) -> Dict[str, Any]:
        """获取函数的图数据库上下文"""
        if not self.driver:
            return {}
        
        # 构建查询条件
        where_conditions = ["fn.name = $function_name"]
        params = {"function_name": function_name}
        
        if file_path:
            where_conditions.append("fn.filepath = $file_path")
            params["file_path"] = file_path
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
        MATCH (fn:Function)
        WHERE {where_clause}
        OPTIONAL MATCH (fn)<-[:DECLARES]-(f:File)
        OPTIONAL MATCH (fn)<-[:HAS_METHOD]-(c:Class)
        OPTIONAL MATCH (fn)-[:CALLS]->(callee:Function)
        OPTIONAL MATCH (caller:Function)-[:CALLS]->(fn)
        RETURN fn, f, c, 
               collect(DISTINCT callee.name) as called_functions,
               collect(DISTINCT caller.name) as calling_functions
        """
        
        with self.driver.session(database=self.config.neo4j_database) as session:
            result = session.run(query, params)
            record = result.single()
            
            if not record:
                return {}
            
            fn = record["fn"]
            f = record["f"]
            c = record["c"]
            
            context = {
                "function_name": fn.get("name"),
                "file_path": fn.get("filepath"),
                "complexity_score": fn.get("complexity_score", 0),
                "importance_score": fn.get("importance_score", 0),
                "parent_class": c.get("name") if c else None,
                "file_info": {
                    "path": f.get("path") if f else None,
                    "file_type": f.get("file_type") if f else None
                } if f else None,
                "called_functions": record["called_functions"],
                "calling_functions": record["calling_functions"],
                "call_count": len(record["called_functions"]) + len(record["calling_functions"])
            }
            
            return context
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()


class HybridSearcher:
    """混合搜索器"""
    
    def __init__(self, semantic_searcher: SemanticSearcher, 
                 graph_searcher: GraphSearcher, config: HybridSearchConfig):
        self.semantic_searcher = semantic_searcher
        self.graph_searcher = graph_searcher
        self.config = config
        self.search_cache = {}  # 简单的内存缓存
    
    def search(self, query: SearchQuery, graph_filters: Dict[str, Any] = None) -> List[HybridSearchResult]:
        """执行混合搜索"""
        logger.info(f"Hybrid search: {query.query}")
        
        # 检查缓存
        cache_key = self._get_cache_key(query, graph_filters)
        if self.config.enable_cache and cache_key in self.search_cache:
            logger.info("Returning cached hybrid search results")
            return self.search_cache[cache_key]
        
        # 执行向量搜索
        vector_results = self.semantic_searcher.search(query)
        
        # 执行图数据库搜索
        graph_results = self._execute_graph_search(query, graph_filters)
        
        # 合并结果
        hybrid_results = self._merge_results(vector_results, graph_results)
        
        # 缓存结果
        if self.config.enable_cache:
            self._cache_results(cache_key, hybrid_results)
        
        return hybrid_results
    
    def _get_cache_key(self, query: SearchQuery, graph_filters: Dict[str, Any] = None) -> str:
        """生成缓存键"""
        import hashlib
        key_data = f"{query.query}_{query.query_type}_{query.top_k}_{query.threshold}_{json.dumps(graph_filters or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _execute_graph_search(self, query: SearchQuery, graph_filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行图数据库搜索"""
        if not graph_filters:
            # 基于查询文本推断图搜索策略
            graph_filters = self._infer_graph_filters(query.query)
        
        graph_results = []
        
        # 按函数名搜索
        if "function_name" in graph_filters:
            results = self.graph_searcher.search_by_function_name(
                graph_filters["function_name"], 
                self.config.max_results
            )
            graph_results.extend(results)
        
        # 按文件路径搜索
        if "file_pattern" in graph_filters:
            results = self.graph_searcher.search_by_file_path(
                graph_filters["file_pattern"], 
                self.config.max_results
            )
            graph_results.extend(results)
        
        # 按复杂度搜索
        if "min_complexity" in graph_filters:
            results = self.graph_searcher.search_by_complexity(
                graph_filters["min_complexity"],
                graph_filters.get("max_complexity"),
                self.config.max_results
            )
            graph_results.extend(results)
        
        # 按函数类型搜索
        if "function_type" in graph_filters:
            results = self.graph_searcher.search_by_function_type(
                graph_filters["function_type"], 
                self.config.max_results
            )
            graph_results.extend(results)
        
        # 按类搜索
        if "class_name" in graph_filters:
            results = self.graph_searcher.search_by_class(
                graph_filters["class_name"], 
                self.config.max_results
            )
            graph_results.extend(results)
        
        # 去重
        seen = set()
        unique_results = []
        for result in graph_results:
            key = (result.get("name"), result.get("filepath"))
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results
    
    def _infer_graph_filters(self, query_text: str) -> Dict[str, Any]:
        """从查询文本推断图搜索过滤器"""
        query_lower = query_text.lower()
        filters = {}
        
        # 检测函数名模式
        if any(word in query_lower for word in ["function", "method", "def"]):
            # 尝试提取函数名
            words = query_text.split()
            for word in words:
                if word.isidentifier() and not word.lower() in ["function", "method", "def", "class"]:
                    filters["function_name"] = word
                    break
        
        # 检测文件模式
        if any(word in query_lower for word in ["file", "path", "module"]):
            # 尝试提取文件路径
            words = query_text.split()
            for word in words:
                if "/" in word or "." in word:
                    filters["file_pattern"] = word
                    break
        
        # 检测复杂度模式
        if any(word in query_lower for word in ["complex", "difficult", "hard"]):
            filters["min_complexity"] = 5.0
        elif any(word in query_lower for word in ["simple", "easy", "basic"]):
            filters["max_complexity"] = 3.0
        
        # 检测函数类型模式
        if any(word in query_lower for word in ["test", "testing"]):
            filters["function_type"] = "test"
        elif any(word in query_lower for word in ["async", "await"]):
            filters["function_type"] = "async"
        elif any(word in query_lower for word in ["decorator", "decorate"]):
            filters["function_type"] = "decorator"
        
        # 检测类模式
        if any(word in query_lower for word in ["class", "object", "instance"]):
            words = query_text.split()
            for word in words:
                if word[0].isupper() and word.isidentifier():
                    filters["class_name"] = word
                    break
        
        return filters
    
    def _merge_results(self, vector_results: List[SearchResult], 
                      graph_results: List[Dict[str, Any]]) -> List[HybridSearchResult]:
        """合并向量和图数据库搜索结果"""
        # 创建函数名到向量的映射
        vector_map = {}
        for result in vector_results:
            func_name = result.embedding.metadata.get("name", "")
            file_path = result.embedding.metadata.get("filepath", "")
            key = (func_name, file_path)
            vector_map[key] = result
        
        # 创建函数名到图结果的映射
        graph_map = {}
        for result in graph_results:
            func_name = result.get("name", "")
            file_path = result.get("filepath", "")
            key = (func_name, file_path)
            graph_map[key] = result
        
        # 合并结果
        hybrid_results = []
        all_keys = set(vector_map.keys()) | set(graph_map.keys())
        
        for key in all_keys:
            vector_result = vector_map.get(key)
            graph_result = graph_map.get(key)
            
            # 计算分数
            vector_similarity = vector_result.similarity if vector_result else 0.0
            graph_score = self._calculate_graph_score(graph_result)
            
            # 组合分数
            combined_score = (
                vector_similarity * self.config.vector_weight + 
                graph_score * self.config.graph_weight
            )
            
            # 关系增强
            if self.config.enable_relationship_boost and graph_result:
                relationship_boost = self._calculate_relationship_boost(graph_result)
                combined_score *= relationship_boost
            
            # 创建混合结果
            if vector_result:
                hybrid_result = HybridSearchResult(
                    embedding=vector_result.embedding,
                    vector_similarity=vector_similarity,
                    graph_score=graph_score,
                    combined_score=combined_score,
                    rank=0,  # 稍后设置
                    highlights=vector_result.highlights,
                    context=vector_result.context,
                    graph_context=graph_result
                )
            else:
                # 只有图结果，创建虚拟嵌入
                virtual_embedding = self._create_virtual_embedding(graph_result)
                hybrid_result = HybridSearchResult(
                    embedding=virtual_embedding,
                    vector_similarity=0.0,
                    graph_score=graph_score,
                    combined_score=combined_score,
                    rank=0,  # 稍后设置
                    highlights=[],
                    context=self._create_context_from_graph(graph_result),
                    graph_context=graph_result
                )
            
            hybrid_results.append(hybrid_result)
        
        # 按组合分数排序
        hybrid_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # 设置排名
        for i, result in enumerate(hybrid_results, 1):
            result.rank = i
        
        return hybrid_results
    
    def _calculate_graph_score(self, graph_result: Dict[str, Any]) -> float:
        """计算图数据库分数"""
        if not graph_result:
            return 0.0
        
        score = 0.0
        
        # 重要性分数 - 即使为0也给予基础分数
        importance_score = graph_result.get("importance_score", 0)
        if importance_score > 0:
            score += min(importance_score / 10.0, 1.0) * 0.4
        else:
            # 即使importance_score为0，也给予基础分数
            score += 0.1
        
        # 复杂度分数 - 即使为0也给予基础分数
        complexity_score = graph_result.get("complexity_score", 0)
        if complexity_score > 0:
            score += min(complexity_score / 20.0, 1.0) * 0.3
        else:
            # 即使complexity_score为0，也给予基础分数
            score += 0.05
        
        # 函数类型奖励
        function_type = graph_result.get("function_type", "")
        if function_type in ["test", "async", "decorator"]:
            score += 0.1
        
        # 类方法奖励
        if graph_result.get("parent_class_name"):
            score += 0.1
        
        # 测试函数奖励
        if graph_result.get("is_test"):
            score += 0.1
        
        # 基础存在奖励 - 确保每个函数都有最小分数
        score += 0.05
        
        return min(score, 1.0)
    
    def _calculate_relationship_boost(self, graph_result: Dict[str, Any]) -> float:
        """计算关系增强因子"""
        boost = 1.0
        
        # 基于调用关系增强
        # 这里可以添加更复杂的逻辑
        
        return boost
    
    def _create_virtual_embedding(self, graph_result: Dict[str, Any]):
        """为图结果创建虚拟嵌入"""
        from .vector_embedding import CodeEmbedding
        
        # 创建虚拟内容
        content = f"def {graph_result.get('name', 'unknown')}():"
        if graph_result.get("parent_class_name"):
            content = f"class {graph_result['parent_class_name']}:\n    {content}"
        
        # 创建元数据
        metadata = {
            "name": graph_result.get("name", "unknown"),
            "filepath": graph_result.get("filepath", "unknown"),
            "function_type": graph_result.get("function_type", "regular"),
            "complexity_score": graph_result.get("complexity_score", 0),
            "importance_score": graph_result.get("importance_score", 0),
            "parent_class_name": graph_result.get("parent_class_name"),
            "is_async": graph_result.get("is_async", False),
            "is_test": graph_result.get("is_test", False)
        }
        
        # 创建虚拟嵌入向量（全零向量）
        embedding_vector = np.zeros(384)  # 默认维度
        
        return CodeEmbedding(
            id=f"graph_{graph_result.get('name', 'unknown')}_{graph_result.get('filepath', 'unknown')}",
            content=content,
            embedding=embedding_vector,
            metadata=metadata,
            model_name="virtual",
            timestamp=datetime.now().isoformat()
        )
    
    def _create_context_from_graph(self, graph_result: Dict[str, Any]) -> Dict[str, Any]:
        """从图结果创建上下文"""
        return {
            "function_name": graph_result.get("name", "Unknown"),
            "file_path": graph_result.get("filepath", "Unknown"),
            "language": "unknown",
            "function_type": graph_result.get("function_type", "regular"),
            "complexity_score": graph_result.get("complexity_score", 0),
            "importance_score": graph_result.get("importance_score", 0),
            "parent_class": graph_result.get("parent_class_name"),
            "is_async": graph_result.get("is_async", False),
            "is_test": graph_result.get("is_test", False)
        }
    
    def _cache_results(self, cache_key: str, results: List[HybridSearchResult]):
        """缓存搜索结果"""
        if len(self.search_cache) >= 100:  # 限制缓存大小
            # 删除最旧的缓存项
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]
        
        self.search_cache[cache_key] = results
    
    def search_by_example_with_graph(self, example_code: str, example_metadata: Dict[str, Any] = None,
                                   graph_filters: Dict[str, Any] = None) -> List[HybridSearchResult]:
        """基于示例代码的混合搜索"""
        # 创建搜索查询
        query = SearchQuery(
            query=example_code,
            query_type="semantic",
            top_k=20,
            threshold=0.0,
            include_context=True,
            include_highlights=True
        )
        
        return self.search(query, graph_filters)
    
    def get_function_ecosystem(self, function_name: str, file_path: str = None) -> Dict[str, Any]:
        """获取函数的生态系统（调用关系、类关系等）"""
        # 获取图数据库上下文
        graph_context = self.graph_searcher.get_function_context(function_name, file_path)
        
        # 获取向量相似函数
        vector_query = SearchQuery(
            query=function_name,
            query_type="semantic",
            top_k=10,
            threshold=0.3
        )
        vector_results = self.semantic_searcher.search(vector_query)
        
        # 组织生态系统信息
        ecosystem = {
            "function_name": function_name,
            "file_path": file_path,
            "graph_context": graph_context,
            "similar_functions": [
                {
                    "name": result.embedding.metadata.get("name"),
                    "file_path": result.embedding.metadata.get("filepath"),
                    "similarity": result.similarity,
                    "context": result.context
                }
                for result in vector_results
            ],
            "ecosystem_stats": {
                "called_functions_count": len(graph_context.get("called_functions", [])),
                "calling_functions_count": len(graph_context.get("calling_functions", [])),
                "similar_functions_count": len(vector_results),
                "complexity_score": graph_context.get("complexity_score", 0),
                "importance_score": graph_context.get("importance_score", 0)
            }
        }
        
        return ecosystem
    
    def get_code_insights(self) -> Dict[str, Any]:
        """获取代码洞察"""
        # 获取向量搜索分析
        vector_analytics = self.semantic_searcher.get_search_analytics()
        
        # 获取图数据库统计
        graph_stats = self._get_graph_stats()
        
        # 获取复杂度分析
        complexity_analysis = self._get_complexity_analysis()
        
        insights = {
            "vector_analytics": vector_analytics,
            "graph_stats": graph_stats,
            "complexity_analysis": complexity_analysis,
            "hybrid_stats": {
                "total_searches": len(self.search_cache),
                "cache_hit_rate": len(self.search_cache) / max(1, vector_analytics.get("total_searches", 1)),
                "vector_weight": self.config.vector_weight,
                "graph_weight": self.config.graph_weight
            }
        }
        
        return insights
    
    def _get_graph_stats(self) -> Dict[str, Any]:
        """获取图数据库统计"""
        if not self.graph_searcher.driver:
            return {"error": "Graph database not available"}
        
        try:
            with self.graph_searcher.driver.session(database=self.config.neo4j_database) as session:
                # 获取基本统计
                result = session.run("""
                    MATCH (fn:Function)
                    RETURN count(fn) as total_functions,
                           avg(fn.complexity_score) as avg_complexity,
                           max(fn.complexity_score) as max_complexity,
                           min(fn.complexity_score) as min_complexity
                """)
                
                stats = dict(result.single())
                
                # 获取函数类型分布
                result = session.run("""
                    MATCH (fn:Function)
                    RETURN fn.function_type, count(fn) as count
                    ORDER BY count DESC
                """)
                
                function_types = {record["fn.function_type"]: record["count"] for record in result}
                stats["function_types"] = function_types
                
                # 获取关系统计
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as rel_type, count(r) as count
                    ORDER BY count DESC
                """)
                
                relationships = {record["rel_type"]: record["count"] for record in result}
                stats["relationships"] = relationships
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {"error": str(e)}
    
    def _get_complexity_analysis(self) -> Dict[str, Any]:
        """获取复杂度分析"""
        if not self.graph_searcher.driver:
            return {"error": "Graph database not available"}
        
        try:
            with self.graph_searcher.driver.session(database=self.config.neo4j_database) as session:
                # 复杂度分布
                result = session.run("""
                    MATCH (fn:Function)
                    RETURN 
                      CASE 
                        WHEN fn.complexity_score < 2 THEN 'low'
                        WHEN fn.complexity_score < 5 THEN 'medium'
                        WHEN fn.complexity_score < 10 THEN 'high'
                        ELSE 'very_high'
                      END as complexity_level,
                      count(fn) as count
                    ORDER BY count DESC
                """)
                
                complexity_distribution = {record["complexity_level"]: record["count"] for record in result}
                
                # 最复杂的函数
                result = session.run("""
                    MATCH (fn:Function)
                    WHERE fn.complexity_score > 5
                    RETURN fn.name, fn.filepath, fn.complexity_score
                    ORDER BY fn.complexity_score DESC
                    LIMIT 10
                """)
                
                most_complex = [dict(record) for record in result]
                
                return {
                    "complexity_distribution": complexity_distribution,
                    "most_complex_functions": most_complex
                }
                
        except Exception as e:
            logger.error(f"Error getting complexity analysis: {e}")
            return {"error": str(e)}
    
    def close(self):
        """关闭资源"""
        if self.graph_searcher:
            self.graph_searcher.close()


def create_hybrid_searcher(semantic_searcher: SemanticSearcher, 
                          config: HybridSearchConfig) -> HybridSearcher:
    """创建混合搜索器"""
    graph_searcher = GraphSearcher(config)
    return HybridSearcher(semantic_searcher, graph_searcher, config)


def create_default_hybrid_config() -> HybridSearchConfig:
    """创建默认混合搜索配置"""
    return HybridSearchConfig(
        vector_weight=0.7,
        graph_weight=0.3,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="90879449Drq",
        neo4j_database=None,
        max_results=50,
        enable_relationship_boost=True,
        relationship_boost_factor=1.2,
        enable_cache=True,
        cache_ttl=3600
    )


if __name__ == "__main__":
    # 测试代码
    from .sync_indexer import create_default_sync_config
    from .semantic_search import create_advanced_searcher
    import tempfile
    import shutil
    
    # 创建测试环境
    test_dir = tempfile.mkdtemp()
    test_file = os.path.join(test_dir, "test.py")
    
    with open(test_file, 'w') as f:
        f.write("""
def calculate_fibonacci(n):
    '''Calculate the nth Fibonacci number'''
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def binary_search(arr, target):
    '''Binary search in sorted array'''
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""")
    
    try:
        # 创建索引器和搜索器
        config = create_default_sync_config(test_dir)
        indexer = SyncSemanticIndexer(config)
        indexer.index_repository()
        
        semantic_searcher = create_advanced_searcher(indexer)
        
        # 创建混合搜索器
        hybrid_config = create_default_hybrid_config()
        hybrid_searcher = create_hybrid_searcher(semantic_searcher, hybrid_config)
        
        # 测试混合搜索
        query = SearchQuery(
            query="find element in array",
            query_type="semantic",
            top_k=5
        )
        
        results = hybrid_searcher.search(query)
        print("Hybrid search results:")
        for result in results:
            print(f"  Rank {result.rank}: {result.embedding.metadata.get('name')} "
                  f"(vector: {result.vector_similarity:.4f}, "
                  f"graph: {result.graph_score:.4f}, "
                  f"combined: {result.combined_score:.4f})")
        
        # 测试函数生态系统
        ecosystem = hybrid_searcher.get_function_ecosystem("binary_search")
        print(f"\nFunction ecosystem for 'binary_search':")
        print(f"  Graph context: {ecosystem['graph_context']}")
        print(f"  Similar functions: {len(ecosystem['similar_functions'])}")
        
        # 测试代码洞察
        insights = hybrid_searcher.get_code_insights()
        print(f"\nCode insights:")
        print(f"  Hybrid stats: {insights['hybrid_stats']}")
        
        hybrid_searcher.close()
        indexer.close()
        
    finally:
        # 清理测试目录
        shutil.rmtree(test_dir)
