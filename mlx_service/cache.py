#!/usr/bin/env python3
"""
MLX Service - Tiered Prefix Cache

两层缓存架构：
- Hot tier: 内存 LRU 缓存 + 写入缓冲区
- Cold tier: SSD safetensors 文件

特性：
1. 异步写入 SSD（不阻塞推理）
2. Hot buffer 写入缓冲（写入时立即可读）
3. 服务重启恢复
4. 统计信息
"""
import hashlib
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

try:
    from mlx_lm.models.cache import save_prompt_cache, load_prompt_cache
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False


# ============================================================================
# Statistics
# ============================================================================

@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    evictions: int = 0
    ssd_saves: int = 0
    ssd_loads: int = 0
    async_writes: int = 0
    write_queue_size: int = 0

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
            "async_writes": self.async_writes,
            "write_queue_size": self.write_queue_size,
        }


# ============================================================================
# Cache Entry
# ============================================================================

@dataclass
class CacheEntry:
    """内存缓存条目"""
    prompt_cache: List[Any]  # KVCache 列表
    tokens: List[int]
    token_count: int
    model_id: int
    created_at: float
    last_used: float


@dataclass
class PendingWrite:
    """待写入 SSD 的条目"""
    key: str
    prompt_cache: List[Any]
    tokens: List[int]
    token_count: int
    model_id: int
    created_at: float


# ============================================================================
# SSD Cache Manager
# ============================================================================

class SSDCacheManager:
    """
    SSD 缓存管理器
    
    负责：
    - 异步写入队列
    - SSD 文件管理
    - 索引维护
    """
    
    def __init__(
        self,
        cache_dir: Path,
        max_size_gb: float = 30.0,
        max_queue_size: int = 64,
    ):
        self.cache_dir = cache_dir
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)
        self.max_queue_size = max_queue_size
        
        # 索引：key -> (file_path, token_count, created_at, model_id)
        self._index: Dict[str, Tuple[Path, int, float, int]] = {}
        self._total_size: int = 0
        self._lock = threading.Lock()
        
        # 异步写入队列
        self._write_queue: queue.Queue[Optional[PendingWrite]] = queue.Queue(maxsize=max_queue_size)
        self._running = True
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        
        # Pending writes buffer（写入前暂存，支持立即读取）
        self._pending: Dict[str, PendingWrite] = {}
        self._pending_lock = threading.Lock()
        
        # 初始化
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._scan_existing()
    
    def _scan_existing(self):
        """扫描现有缓存文件"""
        count = 0
        for f in self.cache_dir.glob("*.safetensors"):
            key = f.stem
            try:
                _, metadata = load_prompt_cache(str(f), return_metadata=True)
                token_count = int(metadata.get("token_count", 0))
                created_at = float(metadata.get("created_at", time.time()))
                model_id = int(metadata.get("model_id", 0))
                self._index[key] = (f, token_count, created_at, model_id)
                self._total_size += f.stat().st_size
                count += 1
            except Exception as e:
                logger.warning(f"Failed to scan {f}: {e}")
        
        if count > 0:
            logger.info(f"📦 SSD cache: {count} entries, {self._total_size / 1024 / 1024:.1f} MB")
    
    def _writer_loop(self):
        """后台写入线程"""
        while self._running:
            try:
                item = self._write_queue.get(timeout=1.0)
                if item is None:
                    continue
                
                self._write_to_ssd(item)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Writer thread error: {e}")
    
    def _write_to_ssd(self, item: PendingWrite):
        """实际写入 SSD"""
        if not HAS_MLX_LM:
            return
        
        file_path = self.cache_dir / f"{item.key}.safetensors"
        
        try:
            metadata = {
                "token_count": str(item.token_count),
                "model_id": str(item.model_id),
                "created_at": str(item.created_at),
            }
            save_prompt_cache(str(file_path), item.prompt_cache, metadata)
            
            # 更新索引
            with self._lock:
                self._index[item.key] = (file_path, item.token_count, item.created_at, item.model_id)
                self._total_size += file_path.stat().st_size
            
            # 从 pending buffer 移除
            with self._pending_lock:
                self._pending.pop(item.key, None)
            
            logger.debug(f"SSD write complete: {item.key}")
            
        except Exception as e:
            logger.warning(f"Failed to write to SSD: {e}")
    
    def enqueue_write(self, item: PendingWrite) -> bool:
        """
        将写入任务加入队列
        
        同时存入 pending buffer，支持立即读取
        """
        # 先存入 pending buffer
        with self._pending_lock:
            self._pending[item.key] = item
        
        # 加入写入队列
        try:
            self._write_queue.put_nowait(item)
            return True
        except queue.Full:
            logger.warning("Write queue full, falling back to sync write")
            self._write_to_ssd(item)
            return True
    
    def load(self, key: str) -> Optional[List[Any]]:
        """
        从 SSD 加载缓存
        
        先检查 pending buffer，再检查 SSD
        """
        # 1. 检查 pending buffer
        with self._pending_lock:
            if key in self._pending:
                return self._pending[key].prompt_cache
        
        # 2. 检查 SSD
        with self._lock:
            if key not in self._index:
                return None
            file_path, _, _ = self._index[key]
        
        if not file_path.exists():
            return None
        
        try:
            prompt_cache, _ = load_prompt_cache(str(file_path), return_metadata=True)
            return prompt_cache
        except Exception as e:
            logger.warning(f"Failed to load from SSD: {e}")
            return None
    
    def contains(self, key: str) -> bool:
        """检查缓存是否存在"""
        # 检查 pending buffer
        with self._pending_lock:
            if key in self._pending:
                return True
        
        # 检查 SSD 索引
        with self._lock:
            return key in self._index
    
    def evict_lru(self, target_size: int = 0) -> int:
        """
        LRU 淘汰 SSD 缓存
        
        Args:
            target_size: 目标大小，0 表示淘汰一个
            
        Returns:
            淘汰的字节数
        """
        if target_size == 0:
            target_size = self._total_size - int(self.max_size_bytes * 0.9)
        
        if target_size <= 0:
            return 0
        
        with self._lock:
            # 按 created_at 排序
            sorted_items = sorted(self._index.items(), key=lambda x: x[1][2])
            
            freed = 0
            for key, (file_path, _, _) in sorted_items:
                if freed >= target_size:
                    break
                
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    del self._index[key]
                    self._total_size -= size
                    freed += size
                except Exception:
                    pass
            
            return freed
    
    def clear(self):
        """清空所有缓存"""
        # 停止写入线程
        self._running = False
        self._write_queue.put(None)
        
        # 清空 pending
        with self._pending_lock:
            self._pending.clear()
        
        # 清空 SSD
        with self._lock:
            for file_path, _, _ in self._index.values():
                try:
                    file_path.unlink()
                except Exception:
                    pass
            self._index.clear()
            self._total_size = 0
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            with self._pending_lock:
                return {
                    "entries": len(self._index),
                    "pending_writes": len(self._pending),
                    "total_size_mb": round(self._total_size / 1024 / 1024, 1),
                    "max_size_gb": self.max_size_bytes / 1024 ** 3,
                    "queue_size": self._write_queue.qsize(),
                }


# ============================================================================
# Tiered Cache Manager
# ============================================================================

class TieredCache:
    """
    分层缓存管理器
    
    Hot tier: 内存 LRU 缓存
    Cold tier: SSD 异步持久化
    """
    
    def __init__(
        self,
        hot_max_entries: int = 20,
        ssd_cache_dir: Optional[Path] = None,
        ssd_max_size_gb: float = 30.0,
        prefix_min_tokens: int = 10,
    ):
        self.hot_max_entries = hot_max_entries
        self.prefix_min_tokens = prefix_min_tokens
        
        # Hot cache (内存)
        self._hot_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hot_lock = threading.Lock()
        
        # Cold cache (SSD)
        self._ssd_cache: Optional[SSDCacheManager] = None
        if ssd_cache_dir and HAS_MLX_LM:
            self._ssd_cache = SSDCacheManager(
                cache_dir=ssd_cache_dir,
                max_size_gb=ssd_max_size_gb,
            )
        
        # 统计
        self.stats = CacheStats()
        
        # 模型映射
        self._model_map: Dict[int, str] = {}
    
    def _hash_tokens(self, tokens: List[int], model_id: int = 0) -> str:
        """计算 token hash（包含 model_id 防止跨模型污染）"""
        data = f"{model_id}:" + ",".join(map(str, tokens))
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def lookup(self, tokens: List[int], model_id: int = 0) -> Tuple[Optional[List[Any]], List[int]]:
        """
        查找缓存
        
        查找顺序：Hot → Pending buffer → SSD
        
        Args:
            tokens: token 列表
            model_id: 模型 ID（用于区分不同模型的缓存）
        
        Returns:
            (prompt_cache, remaining_tokens)
        """
        if len(tokens) < self.prefix_min_tokens:
            return None, tokens
        
        key = self._hash_tokens(tokens, model_id)
        
        # 1. Hot cache 命中
        with self._hot_lock:
            if key in self._hot_cache:
                entry = self._hot_cache[key]
                entry.last_used = time.time()
                self._hot_cache.move_to_end(key)
                self.stats.hits += 1
                self.stats.tokens_saved += len(tokens)
                logger.debug(f"Cache hit (hot): {key}")
                return entry.prompt_cache, []
        
        # 2. Cold cache (SSD) 命中
        if self._ssd_cache:
            prompt_cache = self._ssd_cache.load(key)
            if prompt_cache is not None:
                self.stats.hits += 1
                self.stats.ssd_loads += 1
                self.stats.tokens_saved += len(tokens)
                logger.debug(f"Cache hit (cold): {key}")
                
                # 提升到 hot cache
                with self._hot_lock:
                    self._promote_to_hot(key, prompt_cache, tokens, model_id)
                
                return prompt_cache, []
        
        self.stats.misses += 1
        return None, tokens
    
    def store(self, tokens: List[int], prompt_cache: List[Any], model_id: int) -> Optional[str]:
        """
        存储缓存
        
        存入 hot cache，如果触发淘汰则异步写入 SSD
        """
        if len(tokens) < self.prefix_min_tokens:
            return None
        
        key = self._hash_tokens(tokens, model_id)
        
        with self._hot_lock:
            # 已存在
            if key in self._hot_cache:
                self._hot_cache[key].last_used = time.time()
                self._hot_cache.move_to_end(key)
                return key
            
            # 淘汰
            while len(self._hot_cache) >= self.hot_max_entries:
                self._evict_hot()
            
            # 存入 hot
            entry = CacheEntry(
                prompt_cache=prompt_cache,
                tokens=tokens,
                token_count=len(tokens),
                model_id=model_id,
                created_at=time.time(),
                last_used=time.time(),
            )
            self._hot_cache[key] = entry
            self._model_map[model_id] = key
        
        logger.debug(f"Cache stored (hot): {key}, tokens={len(tokens)}")
        return key
    
    def _promote_to_hot(self, key: str, prompt_cache: List[Any], tokens: List[int], model_id: int = 0):
        """从 cold 提升到 hot（需要在 hot_lock 内调用）"""
        # 淘汰
        while len(self._hot_cache) >= self.hot_max_entries:
            self._evict_hot()
        
        entry = CacheEntry(
            prompt_cache=prompt_cache,
            tokens=tokens,
            token_count=len(tokens),
            model_id=model_id,
            created_at=time.time(),
            last_used=time.time(),
        )
        self._hot_cache[key] = entry
    
    def _evict_hot(self):
        """淘汰 hot cache 条目到 cold cache（需要在 hot_lock 内调用）"""
        if not self._hot_cache:
            return
        
        oldest_key = next(iter(self._hot_cache))
        entry = self._hot_cache[oldest_key]
        
        # 异步写入 SSD
        if self._ssd_cache:
            pending = PendingWrite(
                key=oldest_key,
                prompt_cache=entry.prompt_cache,
                tokens=entry.tokens,
                token_count=entry.token_count,
                model_id=entry.model_id,
                created_at=entry.created_at,
            )
            self._ssd_cache.enqueue_write(pending)
            self.stats.async_writes += 1
            self.stats.ssd_saves += 1
            logger.debug(f"Cache evicted (hot→cold): {oldest_key}")
        
        del self._hot_cache[oldest_key]
        self.stats.evictions += 1
    
    def clear(self):
        """清空所有缓存"""
        with self._hot_lock:
            self._hot_cache.clear()
            self._model_map.clear()
        
        if self._ssd_cache:
            self._ssd_cache.clear()
        
        self.stats = CacheStats()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = self.stats.to_dict()
        
        with self._hot_lock:
            stats["hot_entries"] = len(self._hot_cache)
            stats["hot_max_entries"] = self.hot_max_entries
        
        if self._ssd_cache:
            ssd_stats = self._ssd_cache.get_stats()
            stats["ssd_entries"] = ssd_stats["entries"]
            stats["ssd_pending"] = ssd_stats["pending_writes"]
            stats["ssd_size_mb"] = ssd_stats["total_size_mb"]
            stats["write_queue_size"] = ssd_stats["queue_size"]
            stats["ssd_enabled"] = True
        else:
            stats["ssd_enabled"] = False
        
        return stats


# ============================================================================
# Global Instance
# ============================================================================

_cache: Optional[TieredCache] = None


def init_cache(config) -> Optional[TieredCache]:
    """初始化全局缓存"""
    global _cache
    
    if not getattr(config, 'ENABLE_PREFIX_CACHE', True):
        logger.info("📦 Prefix cache: disabled")
        return None
    
    ssd_cache_dir = None
    if getattr(config, 'ENABLE_CACHE_PERSISTENCE', True):
        ssd_cache_dir = getattr(config, 'CACHE_DIR', None)
    
    _cache = TieredCache(
        hot_max_entries=getattr(config, 'CACHE_MAX_ENTRIES', 20),
        ssd_cache_dir=ssd_cache_dir,
        ssd_max_size_gb=getattr(config, 'CACHE_MAX_MEMORY_GB', 30.0),
    )
    
    logger.info(f"📦 Tiered cache initialized: hot={_cache.hot_max_entries}")
    return _cache


def get_cache() -> Optional[TieredCache]:
    """获取全局缓存实例"""
    return _cache