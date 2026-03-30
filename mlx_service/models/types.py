#!/usr/bin/env python3
"""MLX Service - Model Types"""
import time
from dataclasses import dataclass
from typing import Any

from mlx_service.capabilities import Capability


@dataclass
class LoadedModel:
    """已加载的模型"""
    name: str
    model: Any
    processor: Any
    config: Any
    loaded_at: float
    last_used: float
    is_vl: bool = False
    is_audio: bool = False
    is_moe: bool = False
    memory_gb: float = 0.0
    capabilities: Capability = Capability.TEXT
    
    def touch(self):
        """更新最后使用时间"""
        self.last_used = time.time()