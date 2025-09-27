#!/usr/bin/env python3
"""
模型缓存管理器
实现SentenceTransformer模型的全局缓存和预加载，避免重复加载
"""

import os
import logging
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time

# 尝试导入sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """模型信息"""
    model: Optional[SentenceTransformer] = None
    model_name: str = ""
    is_loaded: bool = False
    load_time: float = 0.0
    last_used: float = 0.0
    error: Optional[str] = None


class ModelCache:
    """模型缓存管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._models: Dict[str, ModelInfo] = {}
        self._default_model_name = "all-MiniLM-L6-v2"
        self._loading_lock = threading.Lock()
        
        logger.info("ModelCache initialized")
    
    def get_model(self, model_name: str = None) -> Optional[SentenceTransformer]:
        """获取模型，如果未加载则自动加载"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available")
            return None
        
        model_name = model_name or self._default_model_name
        
        # 检查模型是否已加载
        if model_name in self._models:
            model_info = self._models[model_name]
            if model_info.is_loaded and model_info.model is not None:
                model_info.last_used = time.time()
                logger.debug(f"Using cached model: {model_name}")
                return model_info.model
            elif model_info.error:
                logger.error(f"Model {model_name} has error: {model_info.error}")
                return None
        
        # 模型未加载，开始加载
        return self._load_model(model_name)
    
    def _load_model(self, model_name: str) -> Optional[SentenceTransformer]:
        """加载模型"""
        with self._loading_lock:
            # 双重检查，避免重复加载
            if model_name in self._models and self._models[model_name].is_loaded:
                return self._models[model_name].model
            
            logger.info(f"Loading model: {model_name}")
            start_time = time.time()
            
            try:
                # 创建模型信息
                model_info = ModelInfo(
                    model_name=model_name,
                    is_loaded=False,
                    load_time=0.0,
                    last_used=time.time()
                )
                
                # 加载模型
                model = SentenceTransformer(model_name)
                model_info.model = model
                model_info.is_loaded = True
                model_info.load_time = time.time() - start_time
                model_info.last_used = time.time()
                
                # 保存到缓存
                self._models[model_name] = model_info
                
                logger.info(f"Model {model_name} loaded successfully in {model_info.load_time:.2f}s")
                return model
                
            except Exception as e:
                error_msg = f"Failed to load model {model_name}: {str(e)}"
                logger.error(error_msg)
                
                # 保存错误信息
                if model_name not in self._models:
                    self._models[model_name] = ModelInfo(model_name=model_name)
                self._models[model_name].error = error_msg
                
                return None
    
    def preload_model(self, model_name: str = None, background: bool = True):
        """预加载模型"""
        model_name = model_name or self._default_model_name
        
        if background:
            # 后台线程加载
            def load_in_background():
                self._load_model(model_name)
            
            thread = threading.Thread(target=load_in_background, daemon=True)
            thread.start()
            logger.info(f"Started background loading of model: {model_name}")
        else:
            # 同步加载
            return self._load_model(model_name)
    
    def is_model_loaded(self, model_name: str = None) -> bool:
        """检查模型是否已加载"""
        model_name = model_name or self._default_model_name
        return (model_name in self._models and 
                self._models[model_name].is_loaded and 
                self._models[model_name].model is not None)
    
    def get_model_info(self, model_name: str = None) -> Optional[ModelInfo]:
        """获取模型信息"""
        model_name = model_name or self._default_model_name
        return self._models.get(model_name)
    
    def unload_model(self, model_name: str = None):
        """卸载模型"""
        model_name = model_name or self._default_model_name
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"Model {model_name} unloaded")
    
    def clear_cache(self):
        """清空所有模型缓存"""
        self._models.clear()
        logger.info("Model cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            "total_models": len(self._models),
            "loaded_models": 0,
            "models": {}
        }
        
        for name, info in self._models.items():
            model_stats = {
                "is_loaded": info.is_loaded,
                "load_time": info.load_time,
                "last_used": info.last_used,
                "has_error": info.error is not None,
                "error": info.error
            }
            stats["models"][name] = model_stats
            
            if info.is_loaded:
                stats["loaded_models"] += 1
        
        return stats


# 全局模型缓存实例
model_cache = ModelCache()


def get_cached_model(model_name: str = None) -> Optional[SentenceTransformer]:
    """获取缓存的模型"""
    return model_cache.get_model(model_name)


def preload_model(model_name: str = None, background: bool = True):
    """预加载模型"""
    return model_cache.preload_model(model_name, background)


def is_model_ready(model_name: str = None) -> bool:
    """检查模型是否准备就绪"""
    return model_cache.is_model_loaded(model_name)


def get_model_stats() -> Dict[str, Any]:
    """获取模型统计信息"""
    return model_cache.get_cache_stats()


if __name__ == "__main__":
    # 测试代码
    print("Testing ModelCache...")
    
    # 测试模型加载
    print("Loading model...")
    model = get_cached_model()
    if model:
        print(f"Model loaded successfully: {model}")
        print(f"Model dimension: {model.get_sentence_embedding_dimension()}")
    else:
        print("Failed to load model")
    
    # 测试统计信息
    stats = get_model_stats()
    print(f"Cache stats: {stats}")
    
    # 测试重复获取
    print("Testing cached access...")
    start_time = time.time()
    model2 = get_cached_model()
    end_time = time.time()
    print(f"Cached access took {end_time - start_time:.4f}s")
