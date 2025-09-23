#!/usr/bin/env python3

import os
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

class ModelCacheManager:
    """模型缓存管理�?- 单例模式"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.cache_dir = Path("./model_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # 内存缓存
        self._memory_cache = {}
        
        # 模型文件缓存路径
        self._model_paths = {}
        
        self._initialized = True
        logger.info(f"模型缓存管理器初始化完成，缓存目�? {self.cache_dir}")
    
    def get_model_path(self, model_name: str, model_type: str = "sentence_transformers") -> str:
        """获取模型缓存路径"""
        cache_key = f"{model_type}_{model_name}"
        
        if cache_key in self._model_paths:
            return self._model_paths[cache_key]
        
        # 生成模型缓存路径
        model_hash = hashlib.md5(f"{model_type}_{model_name}".encode()).hexdigest()[:8]
        model_cache_path = self.cache_dir / f"{model_type}_{model_hash}"
        
        self._model_paths[cache_key] = str(model_cache_path)
        return str(model_cache_path)
    
    def is_model_cached(self, model_name: str, model_type: str = "sentence_transformers") -> bool:
        """检查模型是否已缓存"""
        model_path = self.get_model_path(model_name, model_type)
        return os.path.exists(model_path) and os.path.isdir(model_path)
    
    def get_cached_model(self, model_name: str, model_type: str = "sentence_transformers"):
        """从缓存获取模�?""
        cache_key = f"{model_type}_{model_name}"
        
        # 先检查内存缓�?
        if cache_key in self._memory_cache:
            logger.info(f"从内存缓存加载模�? {model_name}")
            return self._memory_cache[cache_key]
        
        # 检查磁盘缓�?
        if self.is_model_cached(model_name, model_type):
            try:
                if model_type == "sentence_transformers":
                    from sentence_transformers import SentenceTransformer
                    model_path = self.get_model_path(model_name, model_type)
                    logger.info(f"从磁盘缓存加载模�? {model_name}")
                    model = SentenceTransformer(model_path)
                    
                    # 缓存到内�?
                    self._memory_cache[cache_key] = model
                    return model
                    
            except Exception as e:
                logger.warning(f"从缓存加载模型失�? {e}")
                return None
        
        return None
    
    def cache_model(self, model, model_name: str, model_type: str = "sentence_transformers"):
        """缓存模型"""
        cache_key = f"{model_type}_{model_name}"
        
        # 缓存到内�?
        self._memory_cache[cache_key] = model
        
        # 缓存到磁盘（如果模型支持保存�?
        try:
            if hasattr(model, 'save') and model_type == "sentence_transformers":
                model_path = self.get_model_path(model_name, model_type)
                logger.info(f"保存模型到磁盘缓�? {model_name}")
                model.save(model_path)
        except Exception as e:
            logger.warning(f"保存模型到磁盘失�? {e}")
    
    def clear_cache(self):
        """清空缓存"""
        self._memory_cache.clear()
        logger.info("内存缓存已清�?)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        info = {
            "memory_cache_size": len(self._memory_cache),
            "memory_cache_models": list(self._memory_cache.keys()),
            "disk_cache_dir": str(self.cache_dir),
            "disk_cache_size": 0
        }
        
        # 计算磁盘缓存大小
        if self.cache_dir.exists():
            total_size = 0
            for file_path in self.cache_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            info["disk_cache_size"] = total_size
        
        return info

# 全局缓存管理器实�?
_cache_manager = ModelCacheManager()

def get_model_cache_manager() -> ModelCacheManager:
    """获取全局模型缓存管理�?""
    return _cache_manager

def cached_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """带缓存的SentenceTransformer加载�?""
    cache_manager = get_model_cache_manager()
    
    # 尝试从缓存获�?
    model = cache_manager.get_cached_model(model_name, "sentence_transformers")
    if model is not None:
        return model
    
    # 如果缓存中没有，则下载并缓存
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"正在下载并缓存SentenceTransformer模型: {model_name}")
        logger.info("首次下载可能需要几分钟，请耐心等待...")
        
        model = SentenceTransformer(model_name)
        cache_manager.cache_model(model, model_name, "sentence_transformers")
        
        logger.info(f"�?模型下载并缓存完�? {model_name}")
        return model
        
    except Exception as e:
        logger.error(f"下载模型失败: {e}")
        raise

@lru_cache(maxsize=1)
def get_default_sentence_transformer():
    """获取默认的SentenceTransformer模型（带LRU缓存�?""
    return cached_sentence_transformer("all-MiniLM-L6-v2")

def clear_model_cache():
    """清空模型缓存"""
    cache_manager = get_model_cache_manager()
    cache_manager.clear_cache()
    get_default_sentence_transformer.cache_clear()

def get_cache_status() -> str:
    """获取缓存状态信�?""
    cache_manager = get_model_cache_manager()
    info = cache_manager.get_cache_info()
    
    status = []
    status.append("📊 模型缓存状�?)
    status.append("=" * 40)
    status.append(f"内存缓存: {info['memory_cache_size']} 个模�?)
    status.append(f"磁盘缓存: {info['disk_cache_size']/1024/1024:.2f} MB")
    status.append(f"缓存目录: {info['disk_cache_dir']}")
    
    if info['memory_cache_models']:
        status.append("\n已缓存的模型:")
        for model in info['memory_cache_models']:
            status.append(f"  - {model}")
    
    return "\n".join(status)

if __name__ == "__main__":
    # 测试缓存功能
    print("🧪 测试模型缓存功能")
    print("=" * 40)
    
    # 获取缓存状�?
    print(get_cache_status())
    
    # 测试加载模型
    print("\n📥 测试加载模型...")
    try:
        model = get_default_sentence_transformer()
        print(f"�?模型加载成功: {type(model).__name__}")
        
        # 再次获取缓存状�?
        print("\n📊 加载后的缓存状�?")
        print(get_cache_status())
        
    except Exception as e:
        print(f"�?模型加载失败: {e}")
