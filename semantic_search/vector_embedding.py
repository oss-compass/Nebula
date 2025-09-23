#!/usr/bin/env python3

import os
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 尝试导入各种嵌入�?try:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入模型缓存管理�?from .model_cache import get_model_cache_manager, cached_sentence_transformer


@dataclass
class EmbeddingConfig:
    """嵌入配置"""
    model_name: str = "all-MiniLM-L6-v2"
    model_type: str = "sentence_transformers"  # sentence_transformers, openai, transformers, tfidf
    dimension: int = 384
    batch_size: int = 32
    max_length: int = 512
    cache_dir: str = "./embedding_cache"
    use_cache: bool = True
    openai_api_key: Optional[str] = None
    openai_model: str = "text-embedding-ada-002"
    device: str = "cpu"


@dataclass
class CodeEmbedding:
    """代码嵌入结果"""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    model_name: str
    timestamp: str


class CodeVectorizer:
    """代码向量化器"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.cache_db = None
        
        # 创建缓存目录
        os.makedirs(config.cache_dir, exist_ok=True)
        
        # 初始化缓存数据库
        if config.use_cache:
            self._init_cache_db()
        
        # 加载模型
        self._load_model()
    
    def _init_cache_db(self):
        """初始化缓存数据库"""
        cache_db_path = os.path.join(self.config.cache_dir, "embeddings.db")
        self.cache_db = sqlite3.connect(cache_db_path)
        
        # 创建�?        self.cache_db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                content_hash TEXT,
                embedding BLOB,
                metadata TEXT,
                model_name TEXT,
                timestamp TEXT
            )
        """)
        self.cache_db.commit()
    
    def _load_model(self):
        """加载嵌入模型"""
        try:
            if self.config.model_type == "sentence_transformers":
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    raise ImportError("sentence-transformers not available")
                
                # 使用改进的缓存管理器
                logger.info(f"正在加载SentenceTransformer模型: {self.config.model_name}")
                self.model = cached_sentence_transformer(self.config.model_name)
                self.config.dimension = self.model.get_sentence_embedding_dimension()
                
            elif self.config.model_type == "openai":
                if not OPENAI_AVAILABLE:
                    raise ImportError("openai not available")
                if not self.config.openai_api_key:
                    raise ValueError("OpenAI API key required")
                openai.api_key = self.config.openai_api_key
                logger.info("OpenAI API configured")
                
            elif self.config.model_type == "transformers":
                if not TRANSFORMERS_AVAILABLE:
                    raise ImportError("transformers not available")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.model = AutoModel.from_pretrained(self.config.model_name)
                logger.info(f"Loaded Transformers model: {self.config.model_name}")
                
            elif self.config.model_type == "tfidf":
                if not SKLEARN_AVAILABLE:
                    raise ImportError("scikit-learn not available")
                self.vectorizer = TfidfVectorizer(
                    max_features=10000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                logger.info("TF-IDF vectorizer initialized")
                
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_content_hash(self, content: str) -> str:
        """生成内容哈希"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_cache_key(self, content: str) -> str:
        """生成缓存�?""
        content_hash = self._get_content_hash(content)
        return f"{self.config.model_name}_{self.config.model_type}_{content_hash}"
    
    def _get_from_cache(self, content: str) -> Optional[CodeEmbedding]:
        """从缓存获取嵌�?""
        if not self.config.use_cache or not self.cache_db:
            return None
        
        cache_key = self._get_cache_key(content)
        cursor = self.cache_db.execute(
            "SELECT embedding, metadata, timestamp FROM embeddings WHERE id = ?",
            (cache_key,)
        )
        row = cursor.fetchone()
        
        if row:
            embedding_data, metadata_str, timestamp = row
            embedding = pickle.loads(embedding_data)
            metadata = json.loads(metadata_str)
            
            return CodeEmbedding(
                id=cache_key,
                content=content,
                embedding=embedding,
                metadata=metadata,
                model_name=self.config.model_name,
                timestamp=timestamp
            )
        return None
    
    def _save_to_cache(self, embedding: CodeEmbedding):
        """保存嵌入到缓�?""
        if not self.config.use_cache or not self.cache_db:
            return
        
        embedding_data = pickle.dumps(embedding.embedding)
        metadata_str = json.dumps(embedding.metadata)
        
        self.cache_db.execute(
            """INSERT OR REPLACE INTO embeddings 
               (id, content_hash, embedding, metadata, model_name, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                embedding.id,
                self._get_content_hash(embedding.content),
                embedding_data,
                metadata_str,
                embedding.model_name,
                embedding.timestamp
            )
        )
        self.cache_db.commit()
    
    def _preprocess_code(self, code: str, metadata: Dict[str, Any]) -> str:
        """预处理代码文�?""
        # 提取关键信息
        processed_parts = []
        
        # 添加函数�?        if 'name' in metadata:
            processed_parts.append(f"Function: {metadata['name']}")
        
        # 添加文档字符�?        if 'docstring_description' in metadata and metadata['docstring_description']:
            processed_parts.append(f"Documentation: {metadata['docstring_description']}")
        
        # 添加源代�?        processed_parts.append(f"Code: {code}")
        
        # 添加文件路径
        if 'filepath' in metadata:
            processed_parts.append(f"File: {metadata['filepath']}")
        
        # 添加类信�?        if 'parent_class_name' in metadata and metadata['parent_class_name']:
            processed_parts.append(f"Class: {metadata['parent_class_name']}")
        
        # 添加参数信息
        if 'parameters_count' in metadata:
            processed_parts.append(f"Parameters: {metadata['parameters_count']}")
        
        # 添加返回类型
        if 'return_type' in metadata and metadata['return_type']:
            processed_parts.append(f"Returns: {metadata['return_type']}")
        
        return "\n".join(processed_parts)
    
    def _embed_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """使用 SentenceTransformers 生成嵌入"""
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """使用 OpenAI API 生成嵌入"""
        embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            response = openai.Embedding.create(
                input=batch,
                model=self.config.openai_model
            )
            
            batch_embeddings = [data['embedding'] for data in response['data']]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _embed_transformers(self, texts: List[str]) -> np.ndarray:
        """使用 Transformers 生成嵌入"""
        embeddings = []
        
        for text in texts:
            # 截断文本
            if len(text) > self.config.max_length:
                text = text[:self.config.max_length]
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用 [CLS] token 的嵌�?                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def _embed_tfidf(self, texts: List[str]) -> np.ndarray:
        """使用 TF-IDF 生成嵌入"""
        if not hasattr(self, '_fitted_vectorizer') or not self._fitted_vectorizer:
            logger.info(f"Fitting TF-IDF vectorizer with {len(texts)} texts")
            self.vectorizer.fit(texts)
            self._fitted_vectorizer = True
            logger.info("TF-IDF vectorizer fitted successfully")
        
        embeddings = self.vectorizer.transform(texts).toarray()
        return embeddings
    
    def embed_texts(self, texts: List[str], metadata_list: List[Dict[str, Any]] = None) -> List[CodeEmbedding]:
        """批量生成文本嵌入"""
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        logger.info(f"embed_texts called with {len(texts)} texts, model_type={self.config.model_type}")
        
        # 预处理文�?        processed_texts = []
        for i, text in enumerate(texts):
            processed_text = self._preprocess_code(text, metadata_list[i])
            processed_texts.append(processed_text)
        
        # 检查缓�?        cached_embeddings = []
        uncached_indices = []
        uncached_texts = []
        uncached_metadata = []
        
        for i, text in enumerate(processed_texts):
            cached = self._get_from_cache(text)
            if cached:
                cached_embeddings.append(cached)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                uncached_metadata.append(metadata_list[i])
        
        # 生成未缓存的嵌入
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} texts")
            
            if self.config.model_type == "sentence_transformers":
                embeddings_array = self._embed_sentence_transformers(uncached_texts)
            elif self.config.model_type == "openai":
                embeddings_array = self._embed_openai(uncached_texts)
            elif self.config.model_type == "transformers":
                embeddings_array = self._embed_transformers(uncached_texts)
            elif self.config.model_type == "tfidf":
                embeddings_array = self._embed_tfidf(uncached_texts)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # 创建嵌入对象
            new_embeddings = []
            for i, (text, metadata, embedding) in enumerate(zip(uncached_texts, uncached_metadata, embeddings_array)):
                cache_key = self._get_cache_key(text)
                embedding_obj = CodeEmbedding(
                    id=cache_key,
                    content=text,
                    embedding=embedding,
                    metadata=metadata,
                    model_name=self.config.model_name,
                    timestamp=str(np.datetime64('now'))
                )
                new_embeddings.append(embedding_obj)
                self._save_to_cache(embedding_obj)
            
            # 合并结果
            all_embeddings = cached_embeddings + new_embeddings
            
            # 按原始顺序排�?            result = [None] * len(texts)
            cached_idx = 0
            new_idx = 0
            
            for i in range(len(texts)):
                if i in uncached_indices:
                    result[i] = new_embeddings[new_idx]
                    new_idx += 1
                else:
                    result[i] = cached_embeddings[cached_idx]
                    cached_idx += 1
            
            return result
        
        return cached_embeddings
    
    def embed_single(self, text: str, metadata: Dict[str, Any] = None) -> CodeEmbedding:
        """生成单个文本的嵌�?""
        if metadata is None:
            metadata = {}
        
        return self.embed_texts([text], [metadata])[0]
    
    def embed_text(self, text: str, metadata: Dict[str, Any] = None) -> np.ndarray:
        """生成单个文本的嵌入向量（兼容性方法）"""
        embedding = self.embed_single(text, metadata)
        return embedding.embedding
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个嵌入的余弦相似度"""
        # 确保是二维数�?        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        # 计算余弦相似�?        similarity = np.dot(embedding1, embedding2.T) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return float(similarity[0, 0])
    
    def find_similar(self, query_embedding: np.ndarray, candidate_embeddings: List[CodeEmbedding], 
                    top_k: int = 10, threshold: float = 0.0) -> List[Tuple[CodeEmbedding, float]]:
        """查找相似的嵌�?""
        similarities = []
        
        for candidate in candidate_embeddings:
            similarity = self.compute_similarity(query_embedding, candidate.embedding)
            if similarity >= threshold:
                similarities.append((candidate, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save_vectorizer(self, filepath: str):
        """保存向量化器状�?""
        if self.config.model_type == "tfidf" and hasattr(self, '_fitted_vectorizer') and self._fitted_vectorizer:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            logger.info(f"TF-IDF vectorizer saved to {filepath}")
        else:
            logger.warning(f"Cannot save vectorizer: model_type={self.config.model_type}, fitted={getattr(self, '_fitted_vectorizer', False)}")
    
    def load_vectorizer(self, filepath: str):
        """加载向量化器状�?""
        if self.config.model_type == "tfidf":
            import pickle
            try:
                with open(filepath, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self._fitted_vectorizer = True
            except FileNotFoundError:
                logger.warning(f"Vectorizer file not found: {filepath}")
    
    def close(self):
        """关闭资源"""
        if self.cache_db:
            self.cache_db.close()


class CodeEmbeddingManager:
    """代码嵌入管理�?""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.vectorizer = CodeVectorizer(config)
        self.embeddings_index = {}  # id -> CodeEmbedding
        self.embeddings_array = None  # 所有嵌入的数组
        self.embeddings_ids = []  # 对应的ID列表
    
    def add_embeddings(self, embeddings: List[CodeEmbedding]):
        """添加嵌入到索�?""
        for embedding in embeddings:
            self.embeddings_index[embedding.id] = embedding
        
        # 重建数组索引
        self._rebuild_array_index()
    
    def _rebuild_array_index(self):
        """重建数组索引"""
        if not self.embeddings_index:
            self.embeddings_array = None
            self.embeddings_ids = []
            return
        
        embeddings_list = list(self.embeddings_index.values())
        self.embeddings_array = np.array([emb.embedding for emb in embeddings_list])
        self.embeddings_ids = [emb.id for emb in embeddings_list]
    
    def search(self, query: str, query_metadata: Dict[str, Any] = None, 
              top_k: int = 10, threshold: float = 0.0) -> List[Tuple[CodeEmbedding, float]]:
        """搜索相似的代�?""
        if not self.embeddings_index:
            return []
        
        # 生成查询嵌入
        query_embedding = self.vectorizer.embed_single(query, query_metadata or {})
        
        # 计算相似�?        similarities = []
        for i, candidate_id in enumerate(self.embeddings_ids):
            candidate_embedding = self.embeddings_index[candidate_id]
            similarity = self.vectorizer.compute_similarity(
                query_embedding.embedding, 
                candidate_embedding.embedding
            )
            
            if similarity >= threshold:
                similarities.append((candidate_embedding, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save_index(self, filepath: str):
        """保存索引到文�?""
        data = {
            'config': self.config.__dict__,
            'embeddings': {
                emb_id: {
                    'id': emb.id,
                    'content': emb.content,
                    'embedding': emb.embedding.tolist(),
                    'metadata': emb.metadata,
                    'model_name': emb.model_name,
                    'timestamp': emb.timestamp
                }
                for emb_id, emb in self.embeddings_index.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_index(self, filepath: str):
        """从文件加载索�?""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建嵌入对象
        self.embeddings_index = {}
        for emb_id, emb_data in data['embeddings'].items():
            embedding = CodeEmbedding(
                id=emb_data['id'],
                content=emb_data['content'],
                embedding=np.array(emb_data['embedding']),
                metadata=emb_data['metadata'],
                model_name=emb_data['model_name'],
                timestamp=emb_data['timestamp']
            )
            self.embeddings_index[emb_id] = embedding
        
        # 重建数组索引
        self._rebuild_array_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if not self.embeddings_index:
            return {'total_embeddings': 0}
        
        embeddings = list(self.embeddings_index.values())
        
        # 统计信息
        stats = {
            'total_embeddings': len(embeddings),
            'model_name': self.config.model_name,
            'model_type': self.config.model_type,
            'dimension': self.config.dimension,
            'file_types': {},
            'function_types': {},
            'complexity_distribution': {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        }
        
        for emb in embeddings:
            metadata = emb.metadata
            
            # 文件类型统计
            file_type = metadata.get('file_type', 'unknown')
            stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            
            # 函数类型统计
            function_type = metadata.get('function_type', 'unknown')
            stats['function_types'][function_type] = stats['function_types'].get(function_type, 0) + 1
            
            # 复杂度分�?            complexity_score = metadata.get('complexity_score', 0)
            if complexity_score < 2:
                stats['complexity_distribution']['low'] += 1
            elif complexity_score < 5:
                stats['complexity_distribution']['medium'] += 1
            elif complexity_score < 10:
                stats['complexity_distribution']['high'] += 1
            else:
                stats['complexity_distribution']['very_high'] += 1
        
        return stats
    
    def close(self):
        """关闭资源"""
        if self.vectorizer:
            self.vectorizer.close()


def create_default_config() -> EmbeddingConfig:
    """创建默认配置"""
    return EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        model_type="sentence_transformers",
        dimension=384,
        batch_size=32,
        max_length=512,
        cache_dir="./embedding_cache",
        use_cache=True
    )


def create_openai_config(api_key: str) -> EmbeddingConfig:
    """创建 OpenAI 配置"""
    return EmbeddingConfig(
        model_name="text-embedding-ada-002",
        model_type="openai",
        dimension=1536,
        batch_size=100,
        max_length=8191,
        cache_dir="./embedding_cache",
        use_cache=True,
        openai_api_key=api_key,
        openai_model="text-embedding-ada-002"
    )


if __name__ == "__main__":
    # 测试代码
    config = create_default_config()
    manager = CodeEmbeddingManager(config)
    
    # 测试数据
    test_codes = [
        "def calculate_fibonacci(n):\n    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
    ]
    
    test_metadata = [
        {"name": "calculate_fibonacci", "filepath": "math.py", "function_type": "recursive"},
        {"name": "binary_search", "filepath": "search.py", "function_type": "algorithm"}
    ]
    
    # 生成嵌入
    embeddings = manager.vectorizer.embed_texts(test_codes, test_metadata)
    manager.add_embeddings(embeddings)
    
    # 搜索测试
    query = "find element in sorted array"
    results = manager.search(query, top_k=2)
    
    print(f"Query: {query}")
    print("Results:")
    for embedding, similarity in results:
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Function: {embedding.metadata.get('name', 'Unknown')}")
        print(f"  File: {embedding.metadata.get('filepath', 'Unknown')}")
        print()
    
    # 显示统计信息
    stats = manager.get_stats()
    print("Index Statistics:")
    print(json.dumps(stats, indent=2))
    
    manager.close()
