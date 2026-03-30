#!/usr/bin/env python3
"""MLX Service - Model Manager"""
import time
import threading
from collections import OrderedDict
from typing import Tuple
from loguru import logger

import mlx.core as mx
from mlx_lm import load as load_lm
from mlx_vlm import load as load_vlm

from mlx_service.models.registry import ModelRegistry
from mlx_service.models.types import LoadedModel
from mlx_service.capabilities import Capability


class ModelManager:
    """模型管理器 - 按需加载、LRU 淘汰、空闲卸载、内存预算控制"""
    
    def __init__(self, registry: ModelRegistry, max_loaded: int = 2, idle_timeout: int = 1800, max_memory_gb: float = 120.0):
        self.registry = registry
        self.max_loaded = max_loaded
        self.idle_timeout = idle_timeout
        self.max_memory_gb = max_memory_gb
        
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.RLock()
        
        self._running = True
        self._checker = threading.Thread(target=self._idle_checker, daemon=True)
        self._checker.start()
    
    def _current_memory(self) -> float:
        """计算当前已加载模型的内存占用"""
        return sum(loaded.memory_gb for loaded in self._loaded.values())
    
    def _idle_checker(self):
        """定期检查并卸载空闲模型"""
        while self._running:
            time.sleep(60)
            
            with self._lock:
                now = time.time()
                to_unload = []
                
                for name, loaded in self._loaded.items():
                    if now - loaded.last_used > self.idle_timeout:
                        to_unload.append(name)
                
                for name in to_unload:
                    logger.info(f"⏰ 模型 {name} 空闲超时，卸载")
                    del self._loaded[name]
                    mx.clear_cache()
    
    def get(self, name: str) -> Tuple:
        """获取模型（按需加载）"""
        with self._lock:
            if name in self._loaded:
                self._loaded.move_to_end(name)
                self._loaded[name].touch()
                return self._loaded[name].model, self._loaded[name].processor
            
            mx.clear_cache()
            time.sleep(0.5)
            
            new_model_memory = self.registry.get_model_size(name)
            current_memory = self._current_memory()
            
            logger.debug(f"📊 内存预算: 当前 {current_memory:.1f}GB + 新模型 {new_model_memory:.1f}GB / 上限 {self.max_memory_gb}GB")
            
            if new_model_memory > self.max_memory_gb:
                logger.warning(f"⚠️ 模型 {name} 内存占用 {new_model_memory:.1f}GB 超过预算 {self.max_memory_gb}GB")
            
            while current_memory + new_model_memory > self.max_memory_gb and len(self._loaded) > 0:
                oldest_name = next(iter(self._loaded))
                oldest_memory = self._loaded[oldest_name].memory_gb
                logger.info(f"🔄 卸载模型 {oldest_name} ({oldest_memory:.1f}GB) - 内存预算控制")
                del self._loaded[oldest_name]
                mx.clear_cache()
                current_memory = self._current_memory()
            
            if len(self._loaded) >= self.max_loaded:
                oldest_name = next(iter(self._loaded))
                logger.info(f"🔄 卸载模型 {oldest_name}（LRU）")
                del self._loaded[oldest_name]
                mx.clear_cache()
            
            model_path = self.registry.resolve(name)
            if model_path is None:
                raise ValueError(f"模型不存在: {name}")
            
            model_type = self.registry.get_model_type(name)
            is_vl = model_type.get("is_vl", False)
            is_audio = model_type.get("is_audio", False)
            
            model_type_str = "(VL)" if is_vl else "(Audio)" if is_audio else ""
            logger.info(f"🔄 加载模型: {name} <- {model_path} {model_type_str}")
            start = time.time()
            
            if is_audio:
                from mlx_audio.stt.utils import load_model as load_audio
                model, processor = load_audio(str(model_path)), None
            elif is_vl:
                model, processor = load_vlm(str(model_path))
            else:
                model, processor = load_lm(str(model_path))
            
            load_time = time.time() - start
            logger.info(f"✅ 模型 {name} 加载完成 ({load_time:.1f}s)")
            
            loaded = LoadedModel(
                name=name,
                model=model,
                processor=processor,
                config=getattr(model, 'config', None),
                loaded_at=time.time(),
                last_used=time.time(),
                is_vl=is_vl,
                is_audio=is_audio,
                is_moe=model_type.get("is_moe", False),
                memory_gb=new_model_memory,
                capabilities=model_type.get("capabilities", Capability.TEXT),
            )
            self._loaded[name] = loaded
            
            return model, processor
    
    def unload(self, name: str) -> bool:
        """卸载模型"""
        with self._lock:
            if name in self._loaded:
                del self._loaded[name]
                mx.clear_cache()
                logger.info(f"✅ 模型 {name} 已卸载")
                return True
            return False
    
    def list_loaded(self) -> dict:
        """列出已加载的模型"""
        with self._lock:
            result = [
                {
                    "name": name,
                    "is_vl": loaded.is_vl,
                    "is_audio": loaded.is_audio,
                    "is_moe": loaded.is_moe,
                    "memory_gb": round(loaded.memory_gb, 1),
                    "loaded_at": loaded.loaded_at,
                    "last_used": loaded.last_used,
                }
                for name, loaded in self._loaded.items()
            ]
            return {
                "models": result,
                "total_memory_gb": round(self._current_memory(), 1),
                "max_memory_gb": self.max_memory_gb,
            }
    
    def is_loaded(self, name: str) -> bool:
        """检查模型是否已加载"""
        with self._lock:
            return name in self._loaded
    
    def shutdown(self):
        """关闭管理器"""
        self._running = False
        with self._lock:
            self._loaded.clear()
            mx.clear_cache()
    
    def has_capability(self, model_id: str, cap: Capability) -> bool:
        """检查模型是否有指定能力"""
        with self._lock:
            loaded = self._loaded.get(model_id)
            if not loaded:
                return False
            return bool(loaded.capabilities & cap)
    
    def is_vl(self, model_id: str) -> bool:
        """向后兼容：检查是否为 VL 模型"""
        return self.has_capability(model_id, Capability.VISION)
    
    def is_audio(self, model_id: str) -> bool:
        """向后兼容：检查是否为音频模型"""
        return self.has_capability(model_id, Capability.AUDIO)