#!/usr/bin/env python3
"""
MLX Service - Prefix Cache with SSD Persistence

功能：
1. 内存 LRU 缓存（热缓存）
2. SSD 持久化（冷缓存）
3. 服务重启恢复
4. 统计信息
"""
import hashlib
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    from mlx_lm.models.cache import save_prompt_cache, load_prompt_cache
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False


@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    evictions: int = 0
    ssd_saves: int = 0
    ssd_loads: int = 0

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
            "ssd_saves": self.ssd_saves,
            "ssd_loads": self.ssd_loads,
        }


@dataclass
class CacheEntry:
    """内存缓存条目"""
    prompt_cache: List[Any]  # KVCache 列表
    tokens: List[int]        # 原始 tokens
    token_count: int
    model_id: int            # 关联的模型 ID
    created_at: float
    last_used: float


class PrefixCache:
    """
    Prefix Cache 管理器（支持 SSD 持久化）
    
    两层架构：
    - Hot tier: 内存 LRU 缓存
    - Cold tier: SSD safetensors 文件
    
    淘汰时写入 SSD，命中时从 SSD 加载
    """

    def __init__(
        self,
        max_entries: int = 20,
        ssd_cache_dir: Optional[Path] = None,
        prefix_min_tokens: int = 10,
    ):
        self.max_entries = max_entries
        self.prefix_min_tokens = prefix_min_tokens
        self.ssd_cache_dir = ssd_cache_dir

        # 内存缓存
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        
        # 模型 ID -> cache key 映射
        self._model_cache_map: Dict[int, str] = {}
        
        # 统计
        self.stats = CacheStats()

        # 初始化 SSD 缓存目录
        if self.ssd_cache_dir:
            self.ssd_cache_dir.mkdir(parents=True, exist_ok=True)
            self._scan_ssd_cache()
            logger.info(f"📦 SSD cache: {self.ssd_cache_dir}")
        else:
            logger.info("📦 SSD cache: disabled")

    def _hash_tokens(self, tokens: List[int]) -> str:
        """计算 token 序列的 hash"""
        data = ",".join(map(str, tokens))
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _ssd_path(self, key: str) -> Path:
        """获取 SSD 缓存文件路径"""
        return self.ssd_cache_dir / f"{key}.safetensors"

    def _scan_ssd_cache(self):
        """扫描 SSD 缓存目录，恢复已有缓存索引"""
        if not self.ssd_cache_dir:
            return
        
        count = 0
        for f in self.ssd_cache_dir.glob("*.safetensors"):
            key = f.stem
            # 读取元数据
            try:
                _, metadata = load_prompt_cache(str(f), return_metadata=True)
                # 记录到索引（不加载实际数据到内存）
                # 后续 lookup 时再加载
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load SSD cache {f}: {e}")
        
        if count > 0:
            logger.info(f"📦 SSD cache: found {count} cached entries")

    def lookup(self, tokens: List[int]) -> Tuple[Optional[List[Any]], List[int]]:
        """
        查找缓存的 prefix
        
        Returns:
            (prompt_cache, remaining_tokens)
        """
        if len(tokens) < self.prefix_min_tokens:
            return None, tokens

        key = self._hash_tokens(tokens)

        with self._lock:
            # 1. 内存命中
            if key in self._cache:
                entry = self._cache[key]
                entry.last_used = time.time()
                self._cache.move_to_end(key)
                self.stats.hits += 1
                self.stats.tokens_saved += len(tokens)
                logger.debug(f"Cache hit (memory): {key}")
                return entry.prompt_cache, []

            # 2. SSD 命中
            if self.ssd_cache_dir:
                ssd_path = self._ssd_path(key)
                if ssd_path.exists():
                    try:
                        prompt_cache, _ = load_prompt_cache(str(ssd_path), return_metadata=True)
                        self.stats.hits += 1
                        self.stats.ssd_loads += 1
                        self.stats.tokens_saved += len(tokens)
                        logger.debug(f"Cache hit (SSD): {key}")
                        
                        # 加载到内存
                        entry = CacheEntry(
                            prompt_cache=prompt_cache,
                            tokens=tokens,
                            token_count=len(tokens),
                            model_id=0,  # 未知模型
                            created_at=time.time(),
                            last_used=time.time(),
                        )
                        self._cache[key] = entry
                        
                        return prompt_cache, []
                    except Exception as e:
                        logger.warning(f"Failed to load from SSD: {e}")

            self.stats.misses += 1
            return None, tokens

    def store(self, tokens: List[int], prompt_cache: List[Any], model_id: int) -> Optional[str]:
        """
        存储 prompt cache
        
        Args:
            tokens: token 序列
            prompt_cache: KVCache 列表
            model_id: 模型 ID
            
        Returns:
            cache_key 或 None
        """
        if len(tokens) < self.prefix_min_tokens:
            return None

        key = self._hash_tokens(tokens)

        with self._lock:
            # 检查是否已存在
            if key in self._cache:
                self._cache[key].last_used = time.time()
                self._cache.move_to_end(key)
                return key

            # LRU 淘汰
            while len(self._cache) >= self.max_entries:
                self._evict_lru()

            # 存入内存
            entry = CacheEntry(
                prompt_cache=prompt_cache,
                tokens=tokens,
                token_count=len(tokens),
                model_id=model_id,
                created_at=time.time(),
                last_used=time.time(),
            )
            self._cache[key] = entry
            self._model_cache_map[model_id] = key

            logger.debug(f"Cache stored: {key}, tokens={len(tokens)}")
            return key

    def _evict_lru(self):
        """LRU 淘汰，写入 SSD"""
        if not self._cache:
            return

        # 获取最旧的条目
        oldest_key = next(iter(self._cache))
        entry = self._cache[oldest_key]

        # 写入 SSD
        if self.ssd_cache_dir and HAS_MLX_LM:
            try:
                ssd_path = self._ssd_path(oldest_key)
                metadata = {
                    "token_count": str(entry.token_count),
                    "model_id": str(entry.model_id),
                    "created_at": str(entry.created_at),
                }
                save_prompt_cache(str(ssd_path), entry.prompt_cache, metadata)
                self.stats.ssd_saves += 1
                logger.debug(f"Cache evicted to SSD: {oldest_key}")
            except Exception as e:
                logger.warning(f"Failed to save to SSD: {e}")

        # 从内存移除
        del self._cache[oldest_key]
        self.stats.evictions += 1

    def invalidate_model(self, model_id: int):
        """清除模型关联的缓存"""
        with self._lock:
            key = self._model_cache_map.pop(model_id, None)
            if key and key in self._cache:
                del self._cache[key]

    def clear(self):
        """清空所有缓存（内存 + SSD）"""
        with self._lock:
            self._cache.clear()
            self._model_cache_map.clear()
            
            # 清空 SSD 缓存
            if self.ssd_cache_dir:
                for f in self.ssd_cache_dir.glob("*.safetensors"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
            
            self.stats = CacheStats()

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            stats = self.stats.to_dict()
            stats["memory_entries"] = len(self._cache)
            stats["max_entries"] = self.max_entries
            stats["ssd_enabled"] = self.ssd_cache_dir is not None
            
            # SSD 缓存大小
            if self.ssd_cache_dir:
                ssd_size = sum(f.stat().st_size for f in self.ssd_cache_dir.glob("*.safetensors"))
                stats["ssd_size_mb"] = round(ssd_size / 1024 / 1024, 1)
            
            return stats


# 全局实例
_cache: Optional[PrefixCache] = None


def init_cache(config) -> Optional[PrefixCache]:
    """
    初始化全局缓存
    
    Args:
        config: Config 对象
    """
    global _cache
    
    if not getattr(config, 'ENABLE_PREFIX_CACHE', True):
        logger.info("📦 Prefix cache: disabled")
        return None
    
    # SSD 缓存目录
    ssd_cache_dir = None
    if getattr(config, 'ENABLE_CACHE_PERSISTENCE', True):
        ssd_cache_dir = getattr(config, 'CACHE_DIR', None)
    
    _cache = PrefixCache(
        max_entries=getattr(config, 'CACHE_MAX_ENTRIES', 20),
        ssd_cache_dir=ssd_cache_dir,
    )
    
    return _cache


def get_cache() -> Optional[PrefixCache]:
    """获取全局缓存实例"""
    return _cache