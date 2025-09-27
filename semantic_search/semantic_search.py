#!/usr/bin/env python3

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path

from .vector_embedding import CodeEmbedding, CodeEmbeddingManager, EmbeddingConfig
from .sync_indexer import SyncSemanticIndexer, SyncIndexerConfig

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    embedding: CodeEmbedding
    similarity: float
    rank: int
    highlights: List[str] = None
    context: Dict[str, Any] = None


@dataclass
class SearchQuery:
    """搜索查询"""
    query: str
    query_type: str = "semantic"  # semantic, keyword, hybrid
    filters: Dict[str, Any] = None
    top_k: int = 10
    threshold: float = 0.0
    include_context: bool = True
    include_highlights: bool = True


@dataclass
class SearchConfig:
    """搜索配置"""
    # 基础配置
    max_results: int = 100
    default_threshold: float = 0.0
    enable_hybrid_search: bool = True
    
    # 高亮配置
    highlight_max_length: int = 200
    highlight_context_lines: int = 3
    
    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1小时
    
    # 推荐配置
    enable_recommendations: bool = True
    recommendation_threshold: float = 0.7
    max_recommendations: int = 5


class SemanticSearcher:
    """语义搜索器"""
    
    def __init__(self, indexer: SyncSemanticIndexer, config: SearchConfig = None):
        self.indexer = indexer
        self.config = config or SearchConfig()
        self.search_cache = {}  # 简单的内存缓存
        self.search_history = []  # 搜索历史
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """执行搜索"""
        logger.info(f"Searching: {query.query} (type: {query.query_type})")
        
        # 检查缓存
        cache_key = self._get_cache_key(query)
        if self.config.enable_cache and cache_key in self.search_cache:
            logger.info("Returning cached results")
            return self.search_cache[cache_key]
        
        # 执行搜索
        if query.query_type == "semantic":
            results = self._semantic_search(query)
        elif query.query_type == "keyword":
            results = self._keyword_search(query)
        elif query.query_type == "hybrid":
            results = self._hybrid_search(query)
        else:
            raise ValueError(f"Unsupported query type: {query.query_type}")
        
        # 后处理结果
        processed_results = self._post_process_results(results, query)
        
        # 缓存结果
        if self.config.enable_cache:
            self._cache_results(cache_key, processed_results)
        
        # 记录搜索历史
        self._record_search(query, processed_results)
        
        return processed_results
    
    def _get_cache_key(self, query: SearchQuery) -> str:
        """生成缓存键"""
        import hashlib
        key_data = f"{query.query}_{query.query_type}_{query.top_k}_{query.threshold}_{json.dumps(query.filters or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _semantic_search(self, query: SearchQuery) -> List[Tuple[CodeEmbedding, float]]:
        """语义搜索"""
        return self.indexer.search_semantic(
            query.query,
            top_k=query.top_k,
            threshold=query.threshold,
            filters=query.filters
        )
    
    def _keyword_search(self, query: SearchQuery) -> List[Tuple[CodeEmbedding, float]]:
        """关键词搜索"""
        # 简单的关键词匹配
        results = []
        query_lower = query.query.lower()
        
        for embedding in self.indexer.embedding_manager.embeddings_index.values():
            # 检查函数名、文件路径、内容中是否包含关键词
            score = 0.0
            
            # 函数名匹配
            func_name = embedding.metadata.get('name', '').lower()
            if query_lower in func_name:
                score += 0.8
            
            # 文件路径匹配
            filepath = embedding.metadata.get('filepath', '').lower()
            if query_lower in filepath:
                score += 0.6
            
            # 内容匹配
            content = embedding.content.lower()
            if query_lower in content:
                score += 0.4
            
            # 文档字符串匹配
            docstring = embedding.metadata.get('docstring_description', '').lower()
            if query_lower in docstring:
                score += 0.7
            
            if score > 0:
                results.append((embedding, score))
        
        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:query.top_k]
    
    def _hybrid_search(self, query: SearchQuery) -> List[Tuple[CodeEmbedding, float]]:
        """混合搜索（语义 + 关键词）"""
        # 执行语义搜索
        semantic_results = self._semantic_search(query)
        
        # 执行关键词搜索
        keyword_results = self._keyword_search(query)
        
        # 合并结果
        combined_results = {}
        
        # 添加语义搜索结果（权重 0.7）
        for embedding, similarity in semantic_results:
            combined_results[embedding.id] = (embedding, similarity * 0.7)
        
        # 添加关键词搜索结果（权重 0.3）
        for embedding, score in keyword_results:
            if embedding.id in combined_results:
                # 如果已存在，增加分数
                existing_embedding, existing_score = combined_results[embedding.id]
                combined_results[embedding.id] = (embedding, existing_score + score * 0.3)
            else:
                combined_results[embedding.id] = (embedding, score * 0.3)
        
        # 转换为列表并排序
        results = list(combined_results.values())
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:query.top_k]
    
    def _post_process_results(self, results: List[Tuple[CodeEmbedding, float]], 
                            query: SearchQuery) -> List[SearchResult]:
        """后处理搜索结果"""
        processed_results = []
        
        for rank, (embedding, similarity) in enumerate(results, 1):
            # 生成高亮
            highlights = []
            if query.include_highlights:
                highlights = self._generate_highlights(embedding, query.query)
            
            # 生成上下文
            context = {}
            if query.include_context:
                context = self._generate_context(embedding)
            
            result = SearchResult(
                embedding=embedding,
                similarity=similarity,
                rank=rank,
                highlights=highlights,
                context=context
            )
            processed_results.append(result)
        
        return processed_results
    
    def _generate_highlights(self, embedding: CodeEmbedding, query: str) -> List[str]:
        """生成高亮文本"""
        highlights = []
        query_lower = query.lower()
        content = embedding.content
        
        # 按行分割内容
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if query_lower in line.lower():
                # 找到包含查询的行
                start_line = max(0, i - self.config.highlight_context_lines)
                end_line = min(len(lines), i + self.config.highlight_context_lines + 1)
                
                context_lines = lines[start_line:end_line]
                highlight_text = '\n'.join(context_lines)
                
                # 截断过长的文本
                if len(highlight_text) > self.config.highlight_max_length:
                    highlight_text = highlight_text[:self.config.highlight_max_length] + "..."
                
                highlights.append(highlight_text)
        
        return highlights[:3]  # 最多返回3个高亮
    
    def _generate_context(self, embedding: CodeEmbedding) -> Dict[str, Any]:
        """生成上下文信息"""
        metadata = embedding.metadata
        
        context = {
            'function_name': metadata.get('name', 'Unknown'),
            'file_path': metadata.get('filepath', 'Unknown'),
            'language': metadata.get('language', 'unknown'),
            'function_type': metadata.get('function_type', 'regular'),
            'complexity_score': metadata.get('complexity_score', 0),
            'start_line': metadata.get('start_line', 0),
            'end_line': metadata.get('end_line', 0),
            'docstring': metadata.get('docstring_description', ''),
            'parameters_count': metadata.get('parameters_count', 0),
            'return_type': metadata.get('return_type', ''),
            'parent_class': metadata.get('parent_class_name', ''),
            'is_async': metadata.get('is_async', False),
            'is_test': metadata.get('is_test', False),
            'is_decorator': metadata.get('is_decorator', False),
            'repo_name': metadata.get('repo_name', '')
        }
        
        return context
    
    def _cache_results(self, cache_key: str, results: List[SearchResult]):
        """缓存搜索结果"""
        if len(self.search_cache) >= self.config.cache_size:
            # 删除最旧的缓存项
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]
        
        self.search_cache[cache_key] = results
    
    def _record_search(self, query: SearchQuery, results: List[SearchResult]):
        """记录搜索历史"""
        search_record = {
            'query': query.query,
            'query_type': query.query_type,
            'timestamp': datetime.now().isoformat(),
            'result_count': len(results),
            'top_similarity': results[0].similarity if results else 0.0
        }
        
        self.search_history.append(search_record)
        
        # 限制历史记录数量
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]
    
    def get_recommendations(self, embedding: CodeEmbedding, 
                          top_k: int = None) -> List[SearchResult]:
        """获取推荐结果"""
        if not self.config.enable_recommendations:
            return []
        
        top_k = top_k or self.config.max_recommendations
        
        # 查找相似的嵌入
        similar_embeddings = self.indexer.embedding_manager.vectorizer.find_similar(
            embedding.embedding,
            list(self.indexer.embedding_manager.embeddings_index.values()),
            top_k=top_k + 1,  # +1 因为会包含自己
            threshold=self.config.recommendation_threshold
        )
        
        # 过滤掉自己
        recommendations = []
        for similar_emb, similarity in similar_embeddings:
            if similar_emb.id != embedding.id:
                result = SearchResult(
                    embedding=similar_emb,
                    similarity=similarity,
                    rank=len(recommendations) + 1
                )
                recommendations.append(result)
        
        return recommendations[:top_k]
    
    def get_search_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """获取搜索建议"""
        suggestions = set()
        partial_lower = partial_query.lower()
        
        # 从函数名中提取建议
        for embedding in self.indexer.embedding_manager.embeddings_index.values():
            func_name = embedding.metadata.get('name', '')
            if partial_lower in func_name.lower():
                suggestions.add(func_name)
            
            # 从文档字符串中提取关键词
            docstring = embedding.metadata.get('docstring_description', '')
            if docstring and partial_lower in docstring.lower():
                # 提取包含查询的单词
                words = docstring.split()
                for word in words:
                    if partial_lower in word.lower() and len(word) > len(partial_query):
                        suggestions.add(word.strip('.,!?;:'))
        
        # 从搜索历史中提取建议
        for record in self.search_history:
            query = record['query']
            if partial_lower in query.lower() and query not in suggestions:
                suggestions.add(query)
        
        # 排序并返回
        suggestions_list = list(suggestions)
        suggestions_list.sort(key=lambda x: len(x))  # 按长度排序
        return suggestions_list[:limit]
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """获取搜索分析"""
        if not self.search_history:
            return {'message': 'No search history available'}
        
        # 统计信息
        total_searches = len(self.search_history)
        query_types = {}
        avg_result_count = 0
        avg_similarity = 0
        
        for record in self.search_history:
            query_type = record['query_type']
            query_types[query_type] = query_types.get(query_type, 0) + 1
            avg_result_count += record['result_count']
            avg_similarity += record['top_similarity']
        
        # 热门查询
        query_counts = {}
        for record in self.search_history:
            query = record['query']
            query_counts[query] = query_counts.get(query, 0) + 1
        
        popular_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        analytics = {
            'total_searches': total_searches,
            'query_type_distribution': query_types,
            'average_result_count': avg_result_count / total_searches if total_searches > 0 else 0,
            'average_top_similarity': avg_similarity / total_searches if total_searches > 0 else 0,
            'popular_queries': popular_queries,
            'cache_hit_rate': len(self.search_cache) / max(1, total_searches),
            'index_stats': self.indexer.get_stats()
        }
        
        return analytics
    
    def clear_cache(self):
        """清除缓存"""
        self.search_cache.clear()
        logger.info("Search cache cleared")
    
    def clear_history(self):
        """清除搜索历史"""
        self.search_history.clear()
        logger.info("Search history cleared")


class AdvancedSemanticSearcher(SemanticSearcher):
    """高级语义搜索器，提供更多功能"""
    
    def __init__(self, indexer: SyncSemanticIndexer, config: SearchConfig = None):
        super().__init__(indexer, config)
        self.cluster_cache = {}  # 聚类缓存
    
    def search_by_example(self, example_code: str, example_metadata: Dict[str, Any] = None,
                         top_k: int = 10, threshold: float = 0.0) -> List[SearchResult]:
        """基于示例代码搜索"""
        # 为示例代码生成嵌入
        example_embedding = self.indexer.embedding_manager.vectorizer.embed_single(
            example_code, example_metadata or {}
        )
        
        # 查找相似的代码
        similar_embeddings = self.indexer.embedding_manager.vectorizer.find_similar(
            example_embedding.embedding,
            list(self.indexer.embedding_manager.embeddings_index.values()),
            top_k=top_k,
            threshold=threshold
        )
        
        # 转换为搜索结果
        results = []
        for rank, (embedding, similarity) in enumerate(similar_embeddings, 1):
            result = SearchResult(
                embedding=embedding,
                similarity=similarity,
                rank=rank,
                highlights=self._generate_highlights(embedding, example_code),
                context=self._generate_context(embedding)
            )
            results.append(result)
        
        return results
    
    def search_by_function_signature(self, function_name: str, parameters: List[str] = None,
                                   return_type: str = None, top_k: int = 10) -> List[SearchResult]:
        """基于函数签名搜索"""
        # 构建查询
        query_parts = [f"function {function_name}"]
        
        if parameters:
            query_parts.append(f"parameters: {', '.join(parameters)}")
        
        if return_type:
            query_parts.append(f"returns: {return_type}")
        
        query_text = " ".join(query_parts)
        
        # 执行搜索
        search_query = SearchQuery(
            query=query_text,
            query_type="semantic",
            top_k=top_k,
            threshold=0.3
        )
        
        return self.search(search_query)
    
    def find_duplicate_code(self, threshold: float = 0.95) -> List[List[SearchResult]]:
        """查找重复代码"""
        duplicates = []
        processed_ids = set()
        
        embeddings_list = list(self.indexer.embedding_manager.embeddings_index.values())
        
        for i, embedding1 in enumerate(embeddings_list):
            if embedding1.id in processed_ids:
                continue
            
            similar_embeddings = self.indexer.embedding_manager.vectorizer.find_similar(
                embedding1.embedding,
                embeddings_list[i+1:],  # 只检查后面的嵌入
                top_k=100,
                threshold=threshold
            )
            
            if similar_embeddings:
                duplicate_group = [SearchResult(
                    embedding=embedding1,
                    similarity=1.0,
                    rank=1,
                    context=self._generate_context(embedding1)
                )]
                
                for similar_emb, similarity in similar_embeddings:
                    duplicate_group.append(SearchResult(
                        embedding=similar_emb,
                        similarity=similarity,
                        rank=len(duplicate_group) + 1,
                        context=self._generate_context(similar_emb)
                    ))
                    processed_ids.add(similar_emb.id)
                
                duplicates.append(duplicate_group)
                processed_ids.add(embedding1.id)
        
        return duplicates
    
    def cluster_similar_functions(self, n_clusters: int = 10) -> Dict[int, List[SearchResult]]:
        """聚类相似函数"""
        from sklearn.cluster import KMeans
        
        embeddings_list = list(self.indexer.embedding_manager.embeddings_index.values())
        
        if len(embeddings_list) < n_clusters:
            n_clusters = len(embeddings_list)
        
        # 提取嵌入向量
        vectors = np.array([emb.embedding for emb in embeddings_list])
        
        # 执行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(vectors)
        
        # 组织结果
        clusters = {}
        for i, (embedding, label) in enumerate(zip(embeddings_list, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            
            result = SearchResult(
                embedding=embedding,
                similarity=0.0,  # 聚类没有相似度概念
                rank=len(clusters[label]) + 1,
                context=self._generate_context(embedding)
            )
            clusters[label].append(result)
        
        return clusters
    
    def get_code_complexity_analysis(self) -> Dict[str, Any]:
        """获取代码复杂度分析"""
        embeddings_list = list(self.indexer.embedding_manager.embeddings_index.values())
        
        complexity_scores = []
        function_types = {}
        languages = {}
        
        for embedding in embeddings_list:
            metadata = embedding.metadata
            complexity_score = metadata.get('complexity_score', 0)
            complexity_scores.append(complexity_score)
            
            function_type = metadata.get('function_type', 'unknown')
            function_types[function_type] = function_types.get(function_type, 0) + 1
            
            language = metadata.get('language', 'unknown')
            languages[language] = languages.get(language, 0) + 1
        
        # 计算统计信息
        if complexity_scores:
            avg_complexity = np.mean(complexity_scores)
            max_complexity = np.max(complexity_scores)
            min_complexity = np.min(complexity_scores)
            std_complexity = np.std(complexity_scores)
        else:
            avg_complexity = max_complexity = min_complexity = std_complexity = 0
        
        # 复杂度分布
        complexity_distribution = {
            'low': len([s for s in complexity_scores if s < 2]),
            'medium': len([s for s in complexity_scores if 2 <= s < 5]),
            'high': len([s for s in complexity_scores if 5 <= s < 10]),
            'very_high': len([s for s in complexity_scores if s >= 10])
        }
        
        analysis = {
            'total_functions': len(embeddings_list),
            'average_complexity': avg_complexity,
            'max_complexity': max_complexity,
            'min_complexity': min_complexity,
            'std_complexity': std_complexity,
            'complexity_distribution': complexity_distribution,
            'function_type_distribution': function_types,
            'language_distribution': languages
        }
        
        return analysis


def create_semantic_searcher(indexer: SyncSemanticIndexer, 
                           config: SearchConfig = None) -> SemanticSearcher:
    """创建语义搜索器"""
    return SemanticSearcher(indexer, config)


def create_advanced_searcher(indexer: SyncSemanticIndexer, 
                           config: SearchConfig = None) -> AdvancedSemanticSearcher:
    """创建高级语义搜索器"""
    return AdvancedSemanticSearcher(indexer, config)


if __name__ == "__main__":
    # 测试代码
    from .sync_indexer import create_default_sync_config
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

def linear_search(arr, target):
    '''Linear search in array'''
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1
""")
    
    try:
        # 创建索引器和搜索器
        config = create_default_sync_config(test_dir)
        indexer = SyncSemanticIndexer(config)
        indexer.index_repository()
        
        searcher = create_advanced_searcher(indexer)
        
        # 测试搜索
        query = SearchQuery(
            query="find element in array",
            query_type="semantic",
            top_k=3
        )
        
        results = searcher.search(query)
        print("Search results:")
        for result in results:
            print(f"  Rank {result.rank}: {result.embedding.metadata.get('name')} "
                  f"(similarity: {result.similarity:.4f})")
        
        # 测试基于示例的搜索
        example_code = "def search(arr, target):\n    for i, item in enumerate(arr):\n        if item == target:\n            return i\n    return -1"
        example_results = searcher.search_by_example(example_code, top_k=2)
        print("\nExample-based search results:")
        for result in example_results:
            print(f"  {result.embedding.metadata.get('name')} "
                  f"(similarity: {result.similarity:.4f})")
        
        # 测试复杂度分析
        complexity_analysis = searcher.get_code_complexity_analysis()
        print("\nComplexity analysis:")
        print(json.dumps(complexity_analysis, indent=2))
        
        # 测试搜索分析
        analytics = searcher.get_search_analytics()
        print("\nSearch analytics:")
        print(json.dumps(analytics, indent=2))
        
        indexer.close()
        
    finally:
        # 清理测试目录
        shutil.rmtree(test_dir)
