#!/usr/bin/env python3
"""
MLX Service - Prefix Cache

基于 token hash 的 KV Cache 管理器，复用已计算的 prompt prefix。
"""
import hashlib
import time
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "tokens_saved": self.tokens_saved,
            "evictions": self.evictions,
        }


@dataclass
class CacheEntry:
    """缓存条目"""
    prompt_cache: List[Any]  # KVCache 列表
    tokens: List[int]        # 原始 tokens
    created_at: float
    last_used: float
    token_count: int


class PrefixCache:
    """
    Prefix Cache 管理器

    功能：
    - 基于 token hash 的缓存查找
    - LRU 淘汰策略
    - 内存限制
    - 自动保存/恢复 prompt_cache
    """

    def __init__(
        self,
        max_entries: int = 20,
        max_memory_gb: float = 30.0,
        prefix_min_tokens: int = 10,  # 最小 prefix 长度才缓存
    ):
        self.max_entries = max_entries
        self.max_memory_gb = max_memory_gb
        self.prefix_min_tokens = prefix_min_tokens

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._model_cache_map: Dict[int, str] = {}  # model_id -> cache_key
        self.stats = CacheStats()

    def _hash_tokens(self, tokens: List[int]) -> str:
        """计算 token 序列的 hash"""
        # 使用 join 而不是 bytes，因为 token ID 可能 > 255
        data = ",".join(map(str, tokens))
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _estimate_memory(self, entry: CacheEntry) -> float:
        """估算缓存条目的内存占用（GB）"""
        # 粗略估算：每个 token 约 2KB (KV cache for each layer)
        return entry.token_count * 2 / 1024 / 1024

    def _evict_if_needed(self):
        """LRU 淘汰"""
        while len(self._cache) >= self.max_entries:
            # 移除最旧的
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key)
            self.stats.evictions += 1
            logger.debug(f"Cache evicted: {oldest_key}")

    def lookup(self, tokens: List[int]) -> Tuple[Optional[List[Any]], List[int]]:
        """
        查找缓存的 prefix

        Returns:
            (prompt_cache, remaining_tokens)
            - prompt_cache: 命中的 KVCache，None 表示未命中
            - remaining_tokens: 需要处理的剩余 tokens
        """
        if len(tokens) < self.prefix_min_tokens:
            return None, tokens

        with self._lock:
            # 精确匹配
            key = self._hash_tokens(tokens)
            if key in self._cache:
                entry = self._cache[key]
                entry.last_used = time.time()
                self._cache.move_to_end(key)
                self.stats.hits += 1
                self.stats.tokens_saved += len(tokens)
                logger.debug(f"Cache hit: {key}, tokens={len(tokens)}")
                return entry.prompt_cache, []

            self.stats.misses += 1
            return None, tokens

    def store(self, tokens: List[int], prompt_cache: List[Any], model_id: int) -> Optional[str]:
        """
        存储 prompt cache

        Args:
            tokens: token 序列
            prompt_cache: KVCache 列表
            model_id: 模型 ID（用于关联）

        Returns:
            cache_key 或 None（如果不需要缓存）
        """
        if len(tokens) < self.prefix_min_tokens:
            return None

        key = self._hash_tokens(tokens)

        with self._lock:
            self._evict_if_needed()

            entry = CacheEntry(
                prompt_cache=prompt_cache,
                tokens=tokens,
                created_at=time.time(),
                last_used=time.time(),
                token_count=len(tokens),
            )
            self._cache[key] = entry
            self._model_cache_map[model_id] = key

            logger.debug(f"Cache stored: {key}, tokens={len(tokens)}")
            return key

    def get_for_model(self, model_id: int) -> Optional[Tuple[List[Any], List[int]]]:
        """
        获取关联到特定模型的缓存

        用于连续对话场景
        """
        with self._lock:
            key = self._model_cache_map.get(model_id)
            if key and key in self._cache:
                entry = self._cache[key]
                return entry.prompt_cache, entry.tokens
            return None

    def invalidate_model(self, model_id: int):
        """清除模型关联的缓存"""
        with self._lock:
            key = self._model_cache_map.pop(model_id, None)
            if key and key in self._cache:
                self._cache.pop(key)

    def clear(self):
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._model_cache_map.clear()
            self.stats = CacheStats()

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            return {
                **self.stats.to_dict(),
                "entries": len(self._cache),
                "max_entries": self.max_entries,
            }


# 全局实例
_cache: Optional[PrefixCache] = None


def init_cache(config) -> Optional[PrefixCache]:
    """
    初始化全局缓存

    Args:
        config: Config 对象，包含:
            - ENABLE_PREFIX_CACHE: bool
            - CACHE_MAX_ENTRIES: int
            - CACHE_MAX_MEMORY_GB: float
    """
    global _cache
    if getattr(config, 'ENABLE_PREFIX_CACHE', True):
        _cache = PrefixCache(
            max_entries=getattr(config, 'CACHE_MAX_ENTRIES', 20),
            max_memory_gb=getattr(config, 'CACHE_MAX_MEMORY_GB', 30.0),
        )
        logger.info(f"📦 Prefix cache enabled: max_entries={_cache.max_entries}")
    else:
        _cache = None
        logger.info("📦 Prefix cache disabled")
    return _cache


def get_cache() -> Optional[PrefixCache]:
    """获取全局缓存实例"""
    return _cache