import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, semantic search will be disabled")

try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logging.warning("jieba not available, keyword extraction will be limited")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("openai not available, AI-enhanced search will be disabled")

from .vector_embedding import CodeVectorizer, EmbeddingConfig, CodeEmbedding
from .sync_indexer import SyncSemanticIndexer, SyncIndexerConfig

logger = logging.getLogger(__name__)

@dataclass
class SearchQuery:
    """搜索查询"""
    query: str
    query_type: str = "semantic"  # semantic, keyword, hybrid, ai
    top_k: int = 10
    threshold: float = 0.7
    filters: Optional[Dict[str, Any]] = None

@dataclass
class SearchResult:
    """搜索结果"""
    code_embedding: CodeEmbedding
    similarity_score: float
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SearchConfig:
    """搜索配置"""
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    max_results: int = 50
    enable_cache: bool = True
    enable_ai_enhancement: bool = False
    openai_api_key: Optional[str] = None
    recommendation_threshold: float = 0.7
    max_recommendations: int = 5

class SingleRepoSearch:
    """单库语义搜索�?""
    
    def __init__(self, 
                 neo4j_uri: Optional[str] = None,
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 neo4j_database: Optional[str] = None,
                 config: Optional[SearchConfig] = None):
        """
        初始化单库搜索器
        
        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_user: Neo4j用户�?            neo4j_password: Neo4j密码
            neo4j_database: 数据库名�?            config: 搜索配置
        """
        self.config = config or SearchConfig()
        self.search_cache = {}
        self.search_history = []
        
        # 初始化Neo4j连接
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        
        # 初始化向量化�?        self.vectorizer = None
        self._init_vectorizer()
        
        # 初始化索引器
        self.indexer = None
        self._init_indexer()
        
        logger.info("单库语义搜索器初始化完成")
    
    def _init_vectorizer(self):
        """初始化向量化�?""
        try:
            embedding_config = EmbeddingConfig(
                model_name=self.config.embedding_model,
                device="cpu"
            )
            self.vectorizer = CodeVectorizer(embedding_config)
            logger.info(f"向量化器初始化成功，模型: {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"向量化器初始化失�? {e}")
            self.vectorizer = None
    
    def _init_indexer(self):
        """初始化索引器"""
        try:
            if self.neo4j_uri:
                # 使用默认配置创建索引�?                from .sync_indexer import create_default_sync_config
                indexer_config = create_default_sync_config(self.neo4j_uri)
                self.indexer = SyncSemanticIndexer(indexer_config)
                logger.info("索引器初始化成功")
            else:
                logger.warning("未提供Neo4j配置，索引器未初始化")
        except Exception as e:
            logger.error(f"索引器初始化失败: {e}")
            self.indexer = None
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            
        Returns:
            搜索结果列表
        """
        logger.info(f"执行搜索: {query.query} (类型: {query.query_type})")
        
        # 检查缓�?        cache_key = self._get_cache_key(query)
        if self.config.enable_cache and cache_key in self.search_cache:
            logger.info("返回缓存结果")
            return self.search_cache[cache_key]
        
        # 执行搜索
        if query.query_type == "semantic":
            results = self._semantic_search(query)
        elif query.query_type == "keyword":
            results = self._keyword_search(query)
        elif query.query_type == "hybrid":
            results = self._hybrid_search(query)
        elif query.query_type == "ai":
            results = self._ai_enhanced_search(query)
        else:
            raise ValueError(f"不支持的查询类型: {query.query_type}")
        
        # 后处理结�?        processed_results = self._post_process_results(results, query)
        
        # 缓存结果
        if self.config.enable_cache:
            self._cache_results(cache_key, processed_results)
        
        # 记录搜索历史
        self._record_search(query, processed_results)
        
        return processed_results
    
    def search_by_natural_language(self, 
                                  query: str, 
                                  limit: int = 10,
                                  search_type: str = "hybrid",
                                  similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """
        使用自然语言搜索代码
        
        Args:
            query: 自然语言查询
            limit: 返回结果数量限制
            search_type: 搜索类型 ("semantic", "keyword", "hybrid", "ai")
            similarity_threshold: 相似度阈�?            
        Returns:
            搜索结果列表
        """
        search_query = SearchQuery(
            query=query,
            query_type=search_type,
            top_k=limit,
            threshold=similarity_threshold or self.config.similarity_threshold
        )
        
        results = self.search(search_query)
        
        # 转换为字典格�?        return [self._result_to_dict(result) for result in results]
    
    def search_similar_functions(self, 
                               function_name: str,
                               limit: int = 10,
                               similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """
        查找相似函数
        
        Args:
            function_name: 函数�?            limit: 返回结果数量限制
            similarity_threshold: 相似度阈�?            
        Returns:
            相似函数列表
        """
        logger.info(f"查找相似函数: {function_name}")
        
        # 首先获取目标函数的信�?        target_function = self._get_function_info(function_name)
        if not target_function:
            logger.warning(f"未找到函�? {function_name}")
            return []
        
        # 构建搜索查询
        search_query = SearchQuery(
            query=f"function similar to {function_name}",
            query_type="semantic",
            top_k=limit,
            threshold=similarity_threshold or self.config.similarity_threshold,
            filters={"exclude_function": function_name}
        )
        
        results = self.search(search_query)
        
        # 转换为字典格�?        return [self._result_to_dict(result) for result in results]
    
    def search_by_complexity(self, 
                           complexity_level: Optional[str] = None,
                           min_lines: Optional[int] = None,
                           max_lines: Optional[int] = None,
                           min_complexity: Optional[float] = None,
                           max_complexity: Optional[float] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        按复杂度搜索函数
        
        Args:
            complexity_level: 复杂度级�?("low", "medium", "high")
            min_lines: 最小行�?            max_lines: 最大行�?            min_complexity: 最小复杂度
            max_complexity: 最大复杂度
            limit: 返回结果数量限制
            
        Returns:
            搜索结果列表
        """
        logger.info(f"按复杂度搜索: {complexity_level}")
        
        # 构建复杂度过滤器
        filters = {}
        if complexity_level:
            filters["complexity_level"] = complexity_level
        if min_lines is not None:
            filters["min_lines"] = min_lines
        if max_lines is not None:
            filters["max_lines"] = max_lines
        if min_complexity is not None:
            filters["min_complexity"] = min_complexity
        if max_complexity is not None:
            filters["max_complexity"] = max_complexity
        
        search_query = SearchQuery(
            query="complex functions",
            query_type="keyword",
            top_k=limit,
            filters=filters
        )
        
        results = self.search(search_query)
        
        # 转换为字典格�?        return [self._result_to_dict(result) for result in results]
    
    def _semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """执行语义搜索"""
        if not self.vectorizer:
            logger.error("向量化器未初始化")
            return []
        
        try:
            # 生成查询向量
            query_embedding = self.vectorizer.embed_text(query.query)
            
            # 从索引器获取候选结�?            candidates = self._get_candidates_from_indexer(query)
            
            # 如果没有索引器或没有候选结果，使用默认的代码示�?            if not candidates:
                logger.warning("索引器未初始化或无候选结果，使用默认代码示例")
                candidates = self._get_default_code_samples()
            
            # 计算相似�?            results = []
            for candidate in candidates:
                if hasattr(candidate, 'embedding') and candidate.embedding is not None:
                    similarity = self._calculate_similarity(query_embedding, candidate.embedding)
                    if similarity >= query.threshold:
                        result = SearchResult(
                            code_embedding=candidate,
                            similarity_score=similarity,
                            explanation=f"语义相似�? {similarity:.3f}"
                        )
                        results.append(result)
            
            # 按相似度排序
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return results[:query.top_k]
            
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return []
    
    def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """执行关键词搜�?""
        try:
            # 提取关键�?            keywords = self._extract_keywords(query.query)
            
            # 从索引器搜索
            candidates = self._search_by_keywords(keywords, query)
            
            # 转换为搜索结�?            results = []
            for candidate in candidates:
                result = SearchResult(
                    code_embedding=candidate,
                    similarity_score=1.0,  # 关键词匹配给满分
                    explanation="关键词匹�?
                )
                results.append(result)
            
            return results[:query.top_k]
            
        except Exception as e:
            logger.error(f"关键词搜索失�? {e}")
            return []
    
    def _hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """执行混合搜索"""
        try:
            # 执行语义搜索
            semantic_results = self._semantic_search(query)
            
            # 执行关键词搜�?            keyword_results = self._keyword_search(query)
            
            # 合并结果
            all_results = semantic_results + keyword_results
            
            # 去重并重新排�?            unique_results = self._deduplicate_results(all_results)
            unique_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return unique_results[:query.top_k]
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return []
    
    def _ai_enhanced_search(self, query: SearchQuery) -> List[SearchResult]:
        """执行AI增强搜索"""
        if not OPENAI_AVAILABLE or not self.config.openai_api_key:
            logger.warning("OpenAI不可用，回退到混合搜�?)
            return self._hybrid_search(query)
        
        try:
            # 首先执行混合搜索
            base_results = self._hybrid_search(query)
            
            # 使用AI重新排序和解�?            enhanced_results = self._ai_enhance_results(query, base_results)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"AI增强搜索失败: {e}")
            return self._hybrid_search(query)
    
    def _get_candidates_from_indexer(self, query: SearchQuery) -> List[CodeEmbedding]:
        """从索引器获取候选结�?""
        if not self.indexer:
            logger.warning("索引器未初始�?)
            return []
        
        try:
            # 这里需要根据实际的索引器接口调�?            # 假设索引器有search方法
            if hasattr(self.indexer, 'search'):
                return self.indexer.search(query.query, limit=query.top_k * 2)
            else:
                logger.warning("索引器不支持搜索功能")
                return []
        except Exception as e:
            logger.error(f"从索引器获取候选结果失�? {e}")
            return []
    
    def _get_default_code_samples(self) -> List[CodeEmbedding]:
        """获取默认代码示例（当没有索引器时使用�?""
        default_samples = [
            "def calculate_sum(a, b): return a + b",
            "def multiply_numbers(x, y): return x * y",
            "def find_max_value(numbers): return max(numbers)",
            "def process_user_input(data): return data.strip().lower()",
            "def validate_email(email): return '@' in email and '.' in email",
            "def connect_to_database(host, port): return database.connect(host, port)",
            "def handle_error(error): print(f'Error: {error}')",
            "def save_to_file(filename, content): open(filename, 'w').write(content)",
            "def read_file(filename): return open(filename, 'r').read()",
            "def sort_list(items): return sorted(items)",
            "def filter_data(data, condition): return [item for item in data if condition(item)]",
            "def format_string(template, **kwargs): return template.format(**kwargs)"
        ]
        
        # 为每个示例生成嵌入向�?        embeddings = self.vectorizer.embed_texts(default_samples)
        
        # 转换为CodeEmbedding对象
        code_embeddings = []
        for i, (sample, embedding) in enumerate(zip(default_samples, embeddings)):
            code_embedding = CodeEmbedding(
                id=f"default_sample_{i}",
                content=sample,
                embedding=embedding.embedding,
                metadata={
                    "file_path": "default_samples.py",
                    "line_number": i + 1,
                    "function_name": sample.split('(')[0].replace('def ', '') if 'def ' in sample else f"sample_{i}",
                    "class_name": None,
                    "docstring": None,
                    "complexity_score": 1.0
                },
                model_name=self.vectorizer.config.model_name,
                timestamp="2024-01-01T00:00:00Z"
            )
            code_embeddings.append(code_embedding)
        
        logger.info(f"生成�?{len(code_embeddings)} 个默认代码示�?)
        return code_embeddings
    
    def _search_by_keywords(self, keywords: List[str], query: SearchQuery) -> List[CodeEmbedding]:
        """根据关键词搜�?""
        if not self.indexer:
            return []
        
        try:
            # 这里需要根据实际的索引器接口调�?            # 假设索引器有keyword_search方法
            if hasattr(self.indexer, 'keyword_search'):
                return self.indexer.keyword_search(keywords, limit=query.top_k)
            else:
                logger.warning("索引器不支持关键词搜�?)
                return []
        except Exception as e:
            logger.error(f"关键词搜索失�? {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键�?""
        if JIEBA_AVAILABLE:
            # 使用jieba提取关键�?            keywords = jieba.analyse.extract_tags(text, topK=10)
            return keywords
        else:
            # 简单的关键词提�?            words = re.findall(r'\b\w+\b', text.lower())
            return list(set(words))
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个向量的相似度"""
        try:
            # 使用余弦相似�?            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"计算相似度失�? {e}")
            return 0.0
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """去重搜索结果"""
        seen = set()
        unique_results = []
        
        for result in results:
            # 使用代码内容的哈希作为唯一标识
            content_hash = hash(result.code_embedding.content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _ai_enhance_results(self, query: SearchQuery, results: List[SearchResult]) -> List[SearchResult]:
        """使用AI增强搜索结果"""
        if not OPENAI_AVAILABLE or not self.config.openai_api_key:
            return results
        
        try:
            openai.api_key = self.config.openai_api_key
            
            # 构建提示
            prompt = f"""
            请分析以下代码搜索结果，并重新排序和解释�?            
            查询: {query.query}
            
            搜索结果:
            """
            
            for i, result in enumerate(results[:5]):  # 只处理前5个结�?                prompt += f"\n{i+1}. {result.code_embedding.content[:200]}..."
            
            prompt += "\n\n请提供重新排序的建议和每个结果的解释�?
            
            # 调用OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            # 解析响应并更新结�?            ai_explanation = response.choices[0].message.content
            
            # 为每个结果添加AI解释
            for result in results:
                result.explanation = ai_explanation
            
            return results
            
        except Exception as e:
            logger.error(f"AI增强失败: {e}")
            return results
    
    def _post_process_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """后处理搜索结�?""
        # 应用过滤�?        if query.filters:
            results = self._apply_filters(results, query.filters)
        
        # 限制结果数量
        results = results[:query.top_k]
        
        return results
    
    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """应用过滤�?""
        filtered_results = []
        
        for result in results:
            if self._matches_filters(result, filters):
                filtered_results.append(result)
        
        return filtered_results
    
    def _matches_filters(self, result: SearchResult, filters: Dict[str, Any]) -> bool:
        """检查结果是否匹配过滤器"""
        # 这里需要根据实际的CodeEmbedding结构来实�?        # 假设CodeEmbedding有metadata属�?        if not hasattr(result.code_embedding, 'metadata'):
            return True
        
        metadata = result.code_embedding.metadata
        
        for key, value in filters.items():
            if key == "exclude_function":
                if metadata.get("function_name") == value:
                    return False
            elif key == "complexity_level":
                if metadata.get("complexity_level") != value:
                    return False
            elif key == "min_lines":
                if metadata.get("lines", 0) < value:
                    return False
            elif key == "max_lines":
                if metadata.get("lines", 0) > value:
                    return False
            elif key == "min_complexity":
                if metadata.get("complexity", 0) < value:
                    return False
            elif key == "max_complexity":
                if metadata.get("complexity", 0) > value:
                    return False
        
        return True
    
    def _get_function_info(self, function_name: str) -> Optional[Dict[str, Any]]:
        """获取函数信息"""
        if not self.indexer:
            return None
        
        try:
            # 这里需要根据实际的索引器接口调�?            if hasattr(self.indexer, 'get_function_info'):
                return self.indexer.get_function_info(function_name)
            else:
                logger.warning("索引器不支持获取函数信息")
                return None
        except Exception as e:
            logger.error(f"获取函数信息失败: {e}")
            return None
    
    def _result_to_dict(self, result: SearchResult) -> Dict[str, Any]:
        """将搜索结果转换为字典"""
        # 从metadata中获取函数名，如果没有则从content中提�?        function_name = 'Unknown'
        file_path = 'Unknown'
        line_number = 0
        
        if hasattr(result.code_embedding, 'metadata') and result.code_embedding.metadata:
            function_name = result.code_embedding.metadata.get('function_name', 'Unknown')
            file_path = result.code_embedding.metadata.get('file_path', 'Unknown')
            line_number = result.code_embedding.metadata.get('line_number', 0)
        
        # 如果metadata中没有函数名，尝试从content中提�?        if function_name == 'Unknown' and result.code_embedding.content:
            import re
            match = re.search(r'def\s+(\w+)', result.code_embedding.content)
            if match:
                function_name = match.group(1)
        
        return {
            "name": function_name,
            "content": result.code_embedding.content,
            "similarity_score": result.similarity_score,
            "explanation": result.explanation,
            "metadata": result.metadata or {},
            "file_path": file_path,
            "line_number": line_number
        }
    
    def _get_cache_key(self, query: SearchQuery) -> str:
        """生成缓存�?""
        import hashlib
        key_data = f"{query.query}_{query.query_type}_{query.top_k}_{query.threshold}_{json.dumps(query.filters or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cache_results(self, cache_key: str, results: List[SearchResult]):
        """缓存搜索结果"""
        self.search_cache[cache_key] = results
    
    def _record_search(self, query: SearchQuery, results: List[SearchResult]):
        """记录搜索历史"""
        search_record = {
            "timestamp": json.dumps({"timestamp": "now"}),
            "query": query.query,
            "query_type": query.query_type,
            "result_count": len(results)
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
        self.search_cache.clear()
        logger.info("搜索缓存已清�?)
    
    def close(self):
        """关闭搜索�?""
        if self.indexer:
            self.indexer.close()
        logger.info("单库搜索器已关闭")

def create_single_repo_search(neo4j_uri: str = None,
                             neo4j_user: str = None,
                             neo4j_password: str = None,
                             neo4j_database: str = None,
                             config: SearchConfig = None) -> SingleRepoSearch:
    """创建单库搜索器实�?""
    return SingleRepoSearch(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        config=config
    )
