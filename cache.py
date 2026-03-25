#!/usr/bin/env python3
"""
MLX Service - KV Cache Management (预留，目前未启用)
"""
import threading
from typing import Optional
from loguru import logger


class PrefixCacheManager:
    """前缀缓存管理器（预留）"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, enabled: bool = True, **kwargs):
        if self._initialized:
            return
        self._initialized = True
        self.enabled = enabled
        logger.info(f"📦 Cache: {'enabled' if enabled else 'disabled'}")
    
    def get_stats(self) -> dict:
        return {"enabled": self.enabled, "entries": 0, "memory_gb": 0.0}
    
    def clear(self):
        pass


cache_manager: Optional[PrefixCacheManager] = None


def init_cache(config) -> PrefixCacheManager:
    global cache_manager
    cache_manager = PrefixCacheManager(enabled=config.ENABLE_PREFIX_CACHE)
    return cache_manager


def get_cache() -> Optional[PrefixCacheManager]:
    return cache_manager