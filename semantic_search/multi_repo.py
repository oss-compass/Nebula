import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
import concurrent.futures
from threading import Lock

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, semantic search will be disabled")

from .vector_embedding import CodeVectorizer, EmbeddingConfig, CodeEmbedding
from .sync_indexer import SyncSemanticIndexer, SyncIndexerConfig
from .single_repo import SingleRepoSearch, SearchQuery, SearchResult, SearchConfig

logger = logging.getLogger(__name__)

@dataclass
class MultiRepoSearchConfig:
    """多库搜索配置"""
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    max_results_per_repo: int = 20
    max_total_results: int = 100
    enable_parallel_search: bool = True
    max_workers: int = 4
    enable_cache: bool = True
    cross_repo_similarity_threshold: float = 0.8

@dataclass
class RepoConfig:
    """仓库配置"""
    repo_name: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None

@dataclass
class CrossRepoResult:
    """跨库搜索结果"""
    repo_name: str
    results: List[SearchResult]
    repo_similarity_score: float
    total_matches: int

@dataclass
class MultiRepoSearchResult:
    """多库搜索结果"""
    query: str
    total_repos_searched: int
    total_results: int
    cross_repo_results: List[CrossRepoResult]
    global_similarity_score: float
    search_timestamp: str

class MultiRepoSearch:
    """多库语义搜索�?""
    
    def __init__(self, 
                 repo_configs: List[RepoConfig],
                 config: Optional[MultiRepoSearchConfig] = None):
        """
        初始化多库搜索器
        
        Args:
            repo_configs: 仓库配置列表
            config: 多库搜索配置
        """
        self.config = config or MultiRepoSearchConfig()
        self.repo_configs = repo_configs
        self.repo_searchers = {}
        self.search_cache = {}
        self.search_history = []
        self.cache_lock = Lock()
        
        # 初始化各个仓库的搜索�?
        self._init_repo_searchers()
        
        # 初始化全局向量化器
        self.global_vectorizer = None
        self._init_global_vectorizer()
        
        logger.info(f"多库搜索器初始化完成，支�?{len(self.repo_configs)} 个仓�?)
    
    def _init_repo_searchers(self):
        """初始化各个仓库的搜索�?""
        for repo_config in self.repo_configs:
            try:
                search_config = SearchConfig(
                    embedding_model=self.config.embedding_model,
                    similarity_threshold=self.config.similarity_threshold,
                    max_results=self.config.max_results_per_repo
                )
                
                searcher = SingleRepoSearch(
                    neo4j_uri=repo_config.neo4j_uri,
                    neo4j_user=repo_config.neo4j_user,
                    neo4j_password=repo_config.neo4j_password,
                    neo4j_database=repo_config.neo4j_database,
                    config=search_config
                )
                
                self.repo_searchers[repo_config.repo_name] = searcher
                logger.info(f"仓库搜索器初始化成功: {repo_config.repo_name}")
                
            except Exception as e:
                logger.error(f"仓库搜索器初始化失败 {repo_config.repo_name}: {e}")
    
    def _init_global_vectorizer(self):
        """初始化全局向量化器"""
        try:
            embedding_config = EmbeddingConfig(
                model_name=self.config.embedding_model,
                device="cpu"
            )
            self.global_vectorizer = CodeVectorizer(embedding_config)
            logger.info(f"全局向量化器初始化成功，模型: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"全局向量化器初始化失�? {e}")
            self.global_vectorizer = None
    
    def search_across_repos(self, 
                           query: str,
                           search_type: str = "hybrid",
                           similarity_threshold: float = None,
                           target_repos: Optional[List[str]] = None,
                           enable_cross_repo_analysis: bool = True) -> MultiRepoSearchResult:
        """
        跨库搜索
        
        Args:
            query: 搜索查询
            search_type: 搜索类型
            similarity_threshold: 相似度阈�?
            target_repos: 目标仓库列表，None表示搜索所有仓�?
            enable_cross_repo_analysis: 是否启用跨库分析
            
        Returns:
            多库搜索结果
        """
        logger.info(f"执行跨库搜索: {query}")
        
        # 检查缓�?
        cache_key = self._get_cache_key(query, search_type, target_repos)
        if self.config.enable_cache:
            with self.cache_lock:
                if cache_key in self.search_cache:
                    logger.info("返回缓存结果")
                    return self.search_cache[cache_key]
        
        # 确定搜索的仓�?
        repos_to_search = target_repos or list(self.repo_searchers.keys())
        
        # 执行并行搜索
        cross_repo_results = []
        if self.config.enable_parallel_search:
            cross_repo_results = self._parallel_search(query, search_type, similarity_threshold, repos_to_search)
        else:
            cross_repo_results = self._sequential_search(query, search_type, similarity_threshold, repos_to_search)
        
        # 跨库分析
        if enable_cross_repo_analysis:
            cross_repo_results = self._analyze_cross_repo_similarity(cross_repo_results, query)
        
        # 计算全局相似度分�?
        global_similarity_score = self._calculate_global_similarity(cross_repo_results)
        
        # 构建结果
        result = MultiRepoSearchResult(
            query=query,
            total_repos_searched=len(repos_to_search),
            total_results=sum(len(crr.results) for crr in cross_repo_results),
            cross_repo_results=cross_repo_results,
            global_similarity_score=global_similarity_score,
            search_timestamp=json.dumps({"timestamp": "now"})
        )
        
        # 缓存结果
        if self.config.enable_cache:
            with self.cache_lock:
                self.search_cache[cache_key] = result
        
        # 记录搜索历史
        self._record_search(query, result)
        
        return result
    
    def _parallel_search(self, 
                        query: str, 
                        search_type: str, 
                        similarity_threshold: float,
                        repos_to_search: List[str]) -> List[CrossRepoResult]:
        """并行搜索多个仓库"""
        cross_repo_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交搜索任务
            future_to_repo = {}
            for repo_name in repos_to_search:
                if repo_name in self.repo_searchers:
                    future = executor.submit(
                        self._search_single_repo,
                        repo_name,
                        query,
                        search_type,
                        similarity_threshold
                    )
                    future_to_repo[future] = repo_name
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                try:
                    results = future.result()
                    if results:
                        cross_repo_result = CrossRepoResult(
                            repo_name=repo_name,
                            results=results,
                            repo_similarity_score=0.0,  # 将在后续分析中计�?
                            total_matches=len(results)
                        )
                        cross_repo_results.append(cross_repo_result)
                except Exception as e:
                    logger.error(f"仓库 {repo_name} 搜索失败: {e}")
        
        return cross_repo_results
    
    def _sequential_search(self, 
                          query: str, 
                          search_type: str, 
                          similarity_threshold: float,
                          repos_to_search: List[str]) -> List[CrossRepoResult]:
        """顺序搜索多个仓库"""
        cross_repo_results = []
        
        for repo_name in repos_to_search:
            if repo_name in self.repo_searchers:
                try:
                    results = self._search_single_repo(
                        repo_name, query, search_type, similarity_threshold
                    )
                    if results:
                        cross_repo_result = CrossRepoResult(
                            repo_name=repo_name,
                            results=results,
                            repo_similarity_score=0.0,  # 将在后续分析中计�?
                            total_matches=len(results)
                        )
                        cross_repo_results.append(cross_repo_result)
                except Exception as e:
                    logger.error(f"仓库 {repo_name} 搜索失败: {e}")
        
        return cross_repo_results
    
    def _search_single_repo(self, 
                           repo_name: str, 
                           query: str, 
                           search_type: str,
                           similarity_threshold: float) -> List[SearchResult]:
        """搜索单个仓库"""
        searcher = self.repo_searchers[repo_name]
        
        search_query = SearchQuery(
            query=query,
            query_type=search_type,
            top_k=self.config.max_results_per_repo,
            threshold=similarity_threshold or self.config.similarity_threshold
        )
        
        return searcher.search(search_query)
    
    def _analyze_cross_repo_similarity(self, 
                                     cross_repo_results: List[CrossRepoResult],
                                     query: str) -> List[CrossRepoResult]:
        """分析跨库相似�?""
        if not self.global_vectorizer:
            logger.warning("全局向量化器未初始化，跳过跨库分�?)
            return cross_repo_results
        
        try:
            # 生成查询向量
            query_embedding = self.global_vectorizer.embed_text(query)
            
            # 为每个仓库计算相似度分数
            for cross_repo_result in cross_repo_results:
                if cross_repo_result.results:
                    # 计算仓库级别的相似度分数
                    repo_similarity_scores = []
                    for result in cross_repo_result.results:
                        if hasattr(result.code_embedding, 'embedding') and result.code_embedding.embedding:
                            similarity = self._calculate_similarity(
                                query_embedding, result.code_embedding.embedding
                            )
                            repo_similarity_scores.append(similarity)
                    
                    if repo_similarity_scores:
                        cross_repo_result.repo_similarity_score = np.mean(repo_similarity_scores)
                    else:
                        cross_repo_result.repo_similarity_score = 0.0
            
            # 按仓库相似度分数排序
            cross_repo_results.sort(key=lambda x: x.repo_similarity_score, reverse=True)
            
            return cross_repo_results
            
        except Exception as e:
            logger.error(f"跨库相似性分析失�? {e}")
            return cross_repo_results
    
    def _calculate_global_similarity(self, cross_repo_results: List[CrossRepoResult]) -> float:
        """计算全局相似度分�?""
        if not cross_repo_results:
            return 0.0
        
        # 计算所有仓库相似度分数的加权平�?
        total_weight = 0
        weighted_sum = 0
        
        for cross_repo_result in cross_repo_results:
            weight = cross_repo_result.total_matches
            weighted_sum += cross_repo_result.repo_similarity_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def find_similar_apis_across_repos(self, 
                                     api_name: str,
                                     similarity_threshold: float = None,
                                     target_repos: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        跨库查找相似API
        
        Args:
            api_name: API名称
            similarity_threshold: 相似度阈�?
            target_repos: 目标仓库列表
            
        Returns:
            相似API分析结果
        """
        logger.info(f"跨库查找相似API: {api_name}")
        
        # 构建搜索查询
        query = f"API similar to {api_name}"
        
        # 执行跨库搜索
        search_result = self.search_across_repos(
            query=query,
            search_type="semantic",
            similarity_threshold=similarity_threshold or self.config.cross_repo_similarity_threshold,
            target_repos=target_repos,
            enable_cross_repo_analysis=True
        )
        
        # 分析结果
        api_analysis = {
            "target_api": api_name,
            "total_repos_analyzed": search_result.total_repos_searched,
            "total_similar_apis": search_result.total_results,
            "global_similarity_score": search_result.global_similarity_score,
            "repo_analysis": []
        }
        
        for cross_repo_result in search_result.cross_repo_results:
            repo_analysis = {
                "repo_name": cross_repo_result.repo_name,
                "similarity_score": cross_repo_result.repo_similarity_score,
                "api_count": cross_repo_result.total_matches,
                "similar_apis": []
            }
            
            for result in cross_repo_result.results:
                api_info = {
                    "name": getattr(result.code_embedding, 'name', 'Unknown'),
                    "similarity_score": result.similarity_score,
                    "content": result.code_embedding.content[:200] + "...",
                    "file_path": getattr(result.code_embedding, 'file_path', 'Unknown'),
                    "explanation": result.explanation
                }
                repo_analysis["similar_apis"].append(api_info)
            
            api_analysis["repo_analysis"].append(repo_analysis)
        
        return api_analysis
    
    def analyze_api_usage_patterns(self, 
                                 api_name: str,
                                 target_repos: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        分析API使用模式
        
        Args:
            api_name: API名称
            target_repos: 目标仓库列表
            
        Returns:
            API使用模式分析结果
        """
        logger.info(f"分析API使用模式: {api_name}")
        
        usage_patterns = {
            "api_name": api_name,
            "total_repos_analyzed": 0,
            "usage_statistics": {},
            "repo_usage_patterns": []
        }
        
        repos_to_analyze = target_repos or list(self.repo_searchers.keys())
        usage_patterns["total_repos_analyzed"] = len(repos_to_analyze)
        
        for repo_name in repos_to_analyze:
            if repo_name in self.repo_searchers:
                try:
                    searcher = self.repo_searchers[repo_name]
                    
                    # 搜索API使用
                    usage_results = searcher.search_by_natural_language(
                        query=f"usage of {api_name}",
                        limit=50,
                        search_type="keyword"
                    )
                    
                    if usage_results:
                        repo_pattern = {
                            "repo_name": repo_name,
                            "usage_count": len(usage_results),
                            "usage_examples": []
                        }
                        
                        for result in usage_results[:10]:  # 只取�?0个例�?
                            usage_example = {
                                "function_name": result.get("name", "Unknown"),
                                "file_path": result.get("file_path", "Unknown"),
                                "content_preview": result.get("content", "")[:150] + "...",
                                "similarity_score": result.get("similarity_score", 0.0)
                            }
                            repo_pattern["usage_examples"].append(usage_example)
                        
                        usage_patterns["repo_usage_patterns"].append(repo_pattern)
                        
                        # 更新统计信息
                        if repo_name not in usage_patterns["usage_statistics"]:
                            usage_patterns["usage_statistics"][repo_name] = 0
                        usage_patterns["usage_statistics"][repo_name] += len(usage_results)
                
                except Exception as e:
                    logger.error(f"分析仓库 {repo_name} 的API使用模式失败: {e}")
        
        return usage_patterns
    
    def recommend_similar_repos(self, 
                              query: str,
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """
        推荐相似仓库
        
        Args:
            query: 查询描述
            top_k: 返回的仓库数�?
            
        Returns:
            相似仓库推荐列表
        """
        logger.info(f"推荐相似仓库: {query}")
        
        # 执行跨库搜索
        search_result = self.search_across_repos(
            query=query,
            search_type="hybrid",
            enable_cross_repo_analysis=True
        )
        
        # 构建推荐结果
        recommendations = []
        for cross_repo_result in search_result.cross_repo_results:
            if cross_repo_result.repo_similarity_score > 0.5:  # 只推荐相似度较高的仓�?
                recommendation = {
                    "repo_name": cross_repo_result.repo_name,
                    "similarity_score": cross_repo_result.repo_similarity_score,
                    "match_count": cross_repo_result.total_matches,
                    "description": self._get_repo_description(cross_repo_result.repo_name),
                    "tags": self._get_repo_tags(cross_repo_result.repo_name)
                }
                recommendations.append(recommendation)
        
        # 按相似度排序并限制数�?
        recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
        return recommendations[:top_k]
    
    def _get_repo_description(self, repo_name: str) -> str:
        """获取仓库描述"""
        for repo_config in self.repo_configs:
            if repo_config.repo_name == repo_name:
                return repo_config.description or f"Repository: {repo_name}"
        return f"Repository: {repo_name}"
    
    def _get_repo_tags(self, repo_name: str) -> List[str]:
        """获取仓库标签"""
        for repo_config in self.repo_configs:
            if repo_config.repo_name == repo_name:
                return repo_config.tags or []
        return []
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个向量的相似度"""
        try:
            # 使用余弦相似�?
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"计算相似度失�? {e}")
            return 0.0
    
    def _get_cache_key(self, query: str, search_type: str, target_repos: Optional[List[str]]) -> str:
        """生成缓存�?""
        import hashlib
        repos_str = json.dumps(sorted(target_repos or []), sort_keys=True)
        key_data = f"{query}_{search_type}_{repos_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _record_search(self, query: str, result: MultiRepoSearchResult):
        """记录搜索历史"""
        search_record = {
            "timestamp": json.dumps({"timestamp": "now"}),
            "query": query,
            "total_repos": result.total_repos_searched,
            "total_results": result.total_results,
            "global_similarity": result.global_similarity_score
        }
        self.search_history.append(search_record)
        
        # 限制历史记录数量
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """获取搜索历史"""
        return self.search_history.copy()
    
    def clear_cache(self):
        """清空缓存"""
        with self.cache_lock:
            self.search_cache.clear()
        logger.info("多库搜索缓存已清�?)
    
    def get_repo_statistics(self) -> Dict[str, Any]:
        """获取仓库统计信息"""
        stats = {
            "total_repos": len(self.repo_configs),
            "active_searchers": len(self.repo_searchers),
            "repo_details": []
        }
        
        for repo_config in self.repo_configs:
            repo_detail = {
                "repo_name": repo_config.repo_name,
                "description": repo_config.description,
                "tags": repo_config.tags,
                "searcher_available": repo_config.repo_name in self.repo_searchers
            }
            stats["repo_details"].append(repo_detail)
        
        return stats
    
    def close(self):
        """关闭多库搜索�?""
        for searcher in self.repo_searchers.values():
            searcher.close()
        logger.info("多库搜索器已关闭")

def create_multi_repo_search(repo_configs: List[RepoConfig],
                           config: MultiRepoSearchConfig = None) -> MultiRepoSearch:
    """创建多库搜索器实�?""
    return MultiRepoSearch(repo_configs, config)
