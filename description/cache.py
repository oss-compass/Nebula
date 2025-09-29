import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


class DescriptionCache:
    """函数描述缓存管理器"""
    
    def __init__(self, cache_file: str = "description_cache.pkl"):
        """初始化缓存管理器
        
        Args:
            cache_file: 缓存文件路径
        """
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """加载缓存数据"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_cache(self):
        """保存缓存数据到文件"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def _generate_cache_key(self, function_info: Dict) -> str:
        """生成缓存键
        
        Args:
            function_info: 函数信息字典
            
        Returns:
            缓存键字符串
        """
        func_name = function_info.get('basic_info', {}).get('function_name', '')
        source_code = function_info.get('basic_info', {}).get('source_code', '')
        
        # 使用函数名和源代码的哈希作为缓存键
        content = f"{func_name}:{source_code}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, function_info: Dict) -> Optional[Dict[str, Any]]:
        """获取缓存的结果
        
        Args:
            function_info: 函数信息字典
            
        Returns:
            缓存的结果，如果不存在则返回None
        """
        key = self._generate_cache_key(function_info)
        return self.cache.get(key)
    
    def set(self, function_info: Dict, result: Dict[str, Any]):
        """设置缓存结果
        
        Args:
            function_info: 函数信息字典
            result: 要缓存的结果
        """
        key = self._generate_cache_key(function_info)
        self.cache[key] = result
        self._save_cache()
    
    def clear(self):
        """清除缓存"""
        self.cache.clear()
        if self.cache_file.exists():
            try:
                self.cache_file.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete cache file: {e}")
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "cache_size": len(self.cache),
            "cache_file": str(self.cache_file),
            "cache_file_exists": self.cache_file.exists(),
            "cache_file_size": self.cache_file.stat().st_size if self.cache_file.exists() else 0
        }


# 全局缓存实例
description_cache = DescriptionCache()
