#!/usr/bin/env python3
"""
FAISS向量索引器
使用FAISS进行高效的向量相似度搜索，替代numpy线性搜索
"""

import os
import json
import numpy as np
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging

# 尝试导入FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    # 创建一个模拟的FAISS类用于开发
    class MockFaiss:
        class IndexFlatIP:
            def __init__(self, dimension):
                self.dimension = dimension
                self.vectors = []
                self.ids = []
            
            def add(self, vectors):
                start_id = len(self.ids)
                for i, vector in enumerate(vectors):
                    self.ids.append(start_id + i)
                    self.vectors.append(vector)
                return start_id
            
            def search(self, query_vector, k):
                if not self.vectors:
                    return np.array([]), np.array([])
                
                # 简单的余弦相似度计算
                similarities = []
                for i, vector in enumerate(self.vectors):
                    similarity = np.dot(query_vector, vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(vector)
                    )
                    similarities.append((i, similarity))
                
                # 排序并返回top-k
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_k = min(k, len(similarities))
                
                indices = np.array([sim[0] for sim in similarities[:top_k]])
                scores = np.array([sim[1] for sim in similarities[:top_k]])
                
                return indices, scores
        
        class IndexIVFFlat:
            def __init__(self, quantizer, dimension, nlist):
                self.quantizer = quantizer
                self.dimension = dimension
                self.nlist = nlist
                self.vectors = []
                self.ids = []
                self.is_trained = False
            
            def train(self, vectors):
                self.is_trained = True
            
            def add(self, vectors):
                start_id = len(self.ids)
                for i, vector in enumerate(vectors):
                    self.ids.append(start_id + i)
                    self.vectors.append(vector)
                return start_id
            
            def search(self, query_vector, k):
                if not self.vectors:
                    return np.array([]), np.array([])
                
                # 简单的余弦相似度计算
                similarities = []
                for i, vector in enumerate(self.vectors):
                    similarity = np.dot(query_vector, vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(vector)
                    )
                    similarities.append((i, similarity))
                
                # 排序并返回top-k
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_k = min(k, len(similarities))
                
                indices = np.array([sim[0] for sim in similarities[:top_k]])
                scores = np.array([sim[1] for sim in similarities[:top_k]])
                
                return indices, scores
    
    faiss = MockFaiss()

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FAISSIndexerConfig:
    """FAISS索引器配置"""
    # 索引类型配置
    index_type: str = "flat"  # flat, ivf, hnsw
    dimension: int = 384
    metric: str = "ip"  # ip (inner product), l2 (euclidean)
    
    # IVF配置
    nlist: int = 100  # IVF聚类数量
    nprobe: int = 10  # IVF搜索时检查的聚类数量
    
    # HNSW配置
    m: int = 16  # HNSW连接数
    ef_construction: int = 200  # HNSW构建时的搜索范围
    ef_search: int = 50  # HNSW搜索时的搜索范围
    
    # 性能配置
    use_gpu: bool = False
    gpu_id: int = 0
    batch_size: int = 1000
    
    # 存储配置
    index_path: str = "./faiss_index"
    save_vectors: bool = True  # 是否保存原始向量


class FAISSVectorIndexer:
    """FAISS向量索引器"""
    
    def __init__(self, config: FAISSIndexerConfig):
        self.config = config
        self.index = None
        self.vector_ids = []  # 向量ID列表
        self.vector_metadata = {}  # 向量ID到元数据的映射
        self.vectors_array = None  # 原始向量数组（用于保存/加载）
        
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, using mock implementation")
        
        # 初始化索引
        self._init_index()
    
    def _init_index(self):
        """初始化FAISS索引"""
        try:
            if self.config.index_type == "flat":
                if self.config.metric == "ip":
                    self.index = faiss.IndexFlatIP(self.config.dimension)
                else:  # l2
                    self.index = faiss.IndexFlatL2(self.config.dimension)
                logger.info(f"Created FAISS IndexFlat{self.config.metric.upper()} with dimension {self.config.dimension}")
                
            elif self.config.index_type == "ivf":
                # 创建量化器
                if self.config.metric == "ip":
                    quantizer = faiss.IndexFlatIP(self.config.dimension)
                else:
                    quantizer = faiss.IndexFlatL2(self.config.dimension)
                
                # 创建IVF索引
                self.index = faiss.IndexIVFFlat(quantizer, self.config.dimension, self.config.nlist)
                logger.info(f"Created FAISS IndexIVFFlat with nlist={self.config.nlist}")
                
            elif self.config.index_type == "hnsw":
                # HNSW索引
                self.index = faiss.IndexHNSWFlat(self.config.dimension, self.config.m)
                self.index.hnsw.efConstruction = self.config.ef_construction
                self.index.hnsw.efSearch = self.config.ef_search
                logger.info(f"Created FAISS IndexHNSWFlat with m={self.config.m}")
                
            else:
                raise ValueError(f"Unsupported index type: {self.config.index_type}")
            
            # GPU支持
            if self.config.use_gpu and FAISS_AVAILABLE:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, self.config.gpu_id, self.index)
                    logger.info(f"FAISS index moved to GPU {self.config.gpu_id}")
                except Exception as e:
                    logger.warning(f"Failed to move index to GPU: {e}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]] = None) -> List[int]:
        """添加向量到索引"""
        if vectors is None or len(vectors) == 0:
            return []
        
        # 确保向量是二维数组
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # 检查维度
        if vectors.shape[1] != self.config.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.config.dimension}")
        
        # 对于IVF索引，需要先训练
        if self.config.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            # 使用部分数据训练
            train_vectors = vectors[:min(len(vectors), 10000)]
            self.index.train(train_vectors.astype(np.float32))
            logger.info("IVF index training completed")
        
        # 添加向量到索引
        start_id = len(self.vector_ids)
        vector_ids = list(range(start_id, start_id + len(vectors)))
        
        # 转换数据类型
        vectors_float32 = vectors.astype(np.float32)
        
        # 添加到FAISS索引
        self.index.add(vectors_float32)
        
        # 更新ID列表
        self.vector_ids.extend(vector_ids)
        
        # 保存元数据
        if metadata_list:
            for i, metadata in enumerate(metadata_list):
                vector_id = vector_ids[i]
                self.vector_metadata[vector_id] = metadata
        
        # 保存原始向量（如果需要）
        if self.config.save_vectors:
            if self.vectors_array is None:
                self.vectors_array = vectors_float32
            else:
                self.vectors_array = np.vstack([self.vectors_array, vectors_float32])
        
        logger.info(f"Added {len(vectors)} vectors to FAISS index")
        return vector_ids
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[int, float]]:
        """搜索相似向量"""
        if self.index is None or len(self.vector_ids) == 0:
            return []
        
        # 确保查询向量是二维数组
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # 转换数据类型
        query_vector = query_vector.astype(np.float32)
        
        # 执行搜索
        if self.config.index_type == "ivf":
            # 设置搜索参数
            self.index.nprobe = self.config.nprobe
        
        # 搜索
        scores, indices = self.index.search(query_vector, min(top_k, len(self.vector_ids)))
        
        # 处理结果
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and score >= threshold:  # 有效结果
                vector_id = self.vector_ids[idx]
                results.append((vector_id, float(score)))
        
        return results
    
    def get_vector_metadata(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """获取向量元数据"""
        return self.vector_metadata.get(vector_id)
    
    def get_vector_by_id(self, vector_id: int) -> Optional[np.ndarray]:
        """根据ID获取向量"""
        if not self.config.save_vectors or self.vectors_array is None:
            return None
        
        try:
            idx = self.vector_ids.index(vector_id)
            return self.vectors_array[idx]
        except ValueError:
            return None
    
    def remove_vectors(self, vector_ids: List[int]) -> int:
        """移除向量（注意：FAISS不支持直接删除，这里只是从映射中移除）"""
        removed_count = 0
        for vector_id in vector_ids:
            if vector_id in self.vector_metadata:
                del self.vector_metadata[vector_id]
                removed_count += 1
        
        logger.info(f"Removed {removed_count} vectors from metadata")
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        stats = {
            'total_vectors': len(self.vector_ids),
            'index_type': self.config.index_type,
            'dimension': self.config.dimension,
            'metric': self.config.metric,
            'use_gpu': self.config.use_gpu,
            'is_trained': getattr(self.index, 'is_trained', True) if self.index else False
        }
        
        # 添加索引特定统计
        if self.config.index_type == "ivf" and self.index:
            stats['nlist'] = self.index.nlist
            stats['nprobe'] = self.index.nprobe
        
        return stats
    
    def save_index(self, filepath: str):
        """保存索引到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存FAISS索引
        index_file = filepath + ".faiss"
        faiss.write_index(self.index, index_file)
        
        # 保存元数据
        metadata_file = filepath + ".metadata"
        metadata_data = {
            'config': self.config.__dict__,
            'vector_ids': self.vector_ids,
            'vector_metadata': self.vector_metadata,
            'vectors_array': self.vectors_array.tolist() if self.vectors_array is not None else None
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"FAISS index saved to {index_file}")
        logger.info(f"Metadata saved to {metadata_file}")
    
    def load_index(self, filepath: str):
        """从文件加载索引"""
        # 加载FAISS索引
        index_file = filepath + ".faiss"
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
            logger.info(f"FAISS index loaded from {index_file}")
        
        # 加载元数据
        metadata_file = filepath + ".metadata"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_data = json.load(f)
            
            self.vector_ids = metadata_data.get('vector_ids', [])
            self.vector_metadata = metadata_data.get('vector_metadata', {})
            
            vectors_data = metadata_data.get('vectors_array')
            if vectors_data:
                self.vectors_array = np.array(vectors_data, dtype=np.float32)
            
            logger.info(f"Metadata loaded from {metadata_file}")
    
    def clear(self):
        """清空索引"""
        self.vector_ids = []
        self.vector_metadata = {}
        self.vectors_array = None
        
        # 重新初始化索引
        self._init_index()
        
        logger.info("FAISS index cleared")
    
    def close(self):
        """关闭索引器"""
        # FAISS索引会自动清理
        self.index = None
        self.vector_ids = []
        self.vector_metadata = {}
        self.vectors_array = None
        
        logger.info("FAISS indexer closed")


def create_default_faiss_config(dimension: int = 384) -> FAISSIndexerConfig:
    """创建默认FAISS配置"""
    return FAISSIndexerConfig(
        index_type="flat",
        dimension=dimension,
        metric="ip",
        use_gpu=False,
        save_vectors=True
    )


def create_ivf_config(dimension: int = 384, nlist: int = 100) -> FAISSIndexerConfig:
    """创建IVF配置"""
    return FAISSIndexerConfig(
        index_type="ivf",
        dimension=dimension,
        metric="ip",
        nlist=nlist,
        nprobe=10,
        use_gpu=False,
        save_vectors=True
    )


def create_hnsw_config(dimension: int = 384, m: int = 16) -> FAISSIndexerConfig:
    """创建HNSW配置"""
    return FAISSIndexerConfig(
        index_type="hnsw",
        dimension=dimension,
        metric="ip",
        m=m,
        ef_construction=200,
        ef_search=50,
        use_gpu=False,
        save_vectors=True
    )


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    # 创建测试数据
    dimension = 384
    n_vectors = 1000
    test_vectors = np.random.rand(n_vectors, dimension).astype(np.float32)
    
    # 创建索引器
    config = create_default_faiss_config(dimension)
    indexer = FAISSVectorIndexer(config)
    
    # 添加向量
    vector_ids = indexer.add_vectors(test_vectors)
    print(f"Added {len(vector_ids)} vectors")
    
    # 搜索测试
    query_vector = np.random.rand(1, dimension).astype(np.float32)
    results = indexer.search(query_vector, top_k=5)
    
    print("Search results:")
    for vector_id, score in results:
        print(f"  Vector ID: {vector_id}, Score: {score:.4f}")
    
    # 显示统计信息
    stats = indexer.get_stats()
    print("\nIndex statistics:")
    print(json.dumps(stats, indent=2))
    
    # 保存和加载测试
    indexer.save_index("./test_faiss_index")
    
    # 创建新的索引器并加载
    new_indexer = FAISSVectorIndexer(config)
    new_indexer.load_index("./test_faiss_index")
    
    # 验证加载结果
    new_results = new_indexer.search(query_vector, top_k=5)
    print("\nLoaded index search results:")
    for vector_id, score in new_results:
        print(f"  Vector ID: {vector_id}, Score: {score:.4f}")
    
    indexer.close()
    new_indexer.close()

