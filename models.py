#!/usr/bin/env python3
"""
MLX Service - Model Management

功能：
1. 自动扫描模型目录
2. 按需加载
3. LRU 淘汰
4. 空闲自动卸载
5. 基于 config.json 识别模型类型
"""
import json
import time
import threading
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import OrderedDict
from loguru import logger

import mlx.core as mx
from mlx_lm import load as load_lm


def detect_model_type(model_path: Path) -> dict:
    """
    从 config.json 检测模型类型
    
    Returns:
        {
            "is_vl": bool,  # 是否是视觉语言模型
            "is_moe": bool, # 是否是 MoE 模型
            "arch": str,    # 架构名称
        }
    """
    config_path = model_path / "config.json"
    
    if not config_path.exists():
        return {"is_vl": False, "is_moe": False, "arch": "unknown"}
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        arch = config.get("architectures", ["unknown"])[0] if config.get("architectures") else "unknown"
        model_type = config.get("model_type", "")
        
        # 判断是否是 VL 模型
        is_vl = bool(config.get("vision_config")) or \
                "vl" in model_type.lower() or \
                "ForConditionalGeneration" in arch
        
        # 判断是否是 MoE 模型
        text_config = config.get("text_config", {})
        num_experts = config.get("num_experts", text_config.get("num_experts", 0))
        is_moe = num_experts > 0 or "moe" in model_type.lower()
        
        return {
            "is_vl": is_vl,
            "is_moe": is_moe,
            "arch": arch,
        }
    except Exception as e:
        logger.warning(f"读取模型配置失败 {model_path}: {e}")
        return {"is_vl": False, "is_moe": False, "arch": "unknown"}


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
    is_moe: bool = False
    
    def touch(self):
        """更新最后使用时间"""
        self.last_used = time.time()


class ModelRegistry:
    """模型注册表 - 扫描和管理可用模型"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._models: Dict[str, Path] = {}
        self._model_types: Dict[str, dict] = {}
        self._aliases: Dict[str, str] = {}
        self._scan()
    
    def _scan(self):
        """扫描模型目录"""
        if not self.models_dir.exists():
            logger.warning(f"模型目录不存在: {self.models_dir}")
            return
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                # 检查是否有模型文件
                has_safetensors = any(model_dir.glob("*.safetensors"))
                has_config = (model_dir / "config.json").exists()
                
                if has_safetensors or has_config:
                    name = model_dir.name.lower()
                    self._models[name] = model_dir
                    
                    # 检测模型类型
                    model_info = detect_model_type(model_dir)
                    self._model_types[name] = model_info
                    
                    # 生成别名
                    parts = name.split("-")
                    if len(parts) >= 2:
                        short_name = parts[0] + "-" + parts[1].split(".")[0]
                        if short_name not in self._models:
                            self._aliases[short_name] = name
        
        logger.info(f"📦 扫描到 {len(self._models)} 个模型")
    
    def resolve(self, name: str) -> Optional[Path]:
        """解析模型名称到路径"""
        name_lower = name.lower()
        
        # 直接匹配
        if name_lower in self._models:
            return self._models[name_lower]
        
        # 别名匹配
        if name_lower in self._aliases:
            return self._models[self._aliases[name_lower]]
        
        # 模糊匹配
        for model_name, path in self._models.items():
            if name_lower in model_name or model_name.startswith(name_lower):
                return path
        
        return None
    
    def get_model_type(self, name: str) -> dict:
        """获取模型类型信息"""
        name_lower = name.lower()
        
        # 直接匹配
        if name_lower in self._model_types:
            return self._model_types[name_lower]
        
        # 别名匹配
        if name_lower in self._aliases:
            actual_name = self._aliases[name_lower]
            return self._model_types.get(actual_name, {"is_vl": False, "is_moe": False, "arch": "unknown"})
        
        # 模糊匹配
        for model_name, info in self._model_types.items():
            if name_lower in model_name or model_name.startswith(name_lower):
                return info
        
        return {"is_vl": False, "is_moe": False, "arch": "unknown"}
    
    def list_models(self) -> list:
        """列出所有可用模型"""
        result = []
        for name, path in self._models.items():
            model_type = self._model_types.get(name, {})
            result.append({
                "name": name,
                "path": str(path),
                "is_vl": model_type.get("is_vl", False),
                "is_moe": model_type.get("is_moe", False),
                "arch": model_type.get("arch", "unknown"),
            })
        return result


class ModelManager:
    """模型管理器 - 按需加载、LRU 淘汰、空闲卸载"""
    
    def __init__(self, registry: ModelRegistry, max_loaded: int = 2, idle_timeout: int = 1800):
        self.registry = registry
        self.max_loaded = max_loaded
        self.idle_timeout = idle_timeout
        
        # 已加载的模型 (name -> LoadedModel)
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.Lock()
        
        # 启动空闲检查线程
        self._running = True
        self._checker = threading.Thread(target=self._idle_checker, daemon=True)
        self._checker.start()
    
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
    
    def get(self, name: str) -> Tuple[Any, Any]:
        """获取模型（按需加载）"""
        with self._lock:
            # 已加载
            if name in self._loaded:
                self._loaded.move_to_end(name)
                self._loaded[name].touch()
                return self._loaded[name].model, self._loaded[name].processor
            
            # 检查是否超过最大数量
            if len(self._loaded) >= self.max_loaded:
                oldest_name = next(iter(self._loaded))
                logger.info(f"🔄 卸载模型 {oldest_name}（LRU）")
                del self._loaded[oldest_name]
                mx.clear_cache()
            
            # 加载模型
            model_path = self.registry.resolve(name)
            if model_path is None:
                raise ValueError(f"模型不存在: {name}")
            
            logger.info(f"🔄 加载模型: {name} <- {model_path}")
            start = time.time()
            
            model, tokenizer = load_lm(str(model_path))
            
            load_time = time.time() - start
            logger.info(f"✅ 模型 {name} 加载完成 ({load_time:.1f}s)")
            
            # 从 config.json 获取模型类型
            model_type = self.registry.get_model_type(name)
            
            loaded = LoadedModel(
                name=name,
                model=model,
                processor=tokenizer,
                config=getattr(model, 'config', None),
                loaded_at=time.time(),
                last_used=time.time(),
                is_vl=model_type.get("is_vl", False),
                is_moe=model_type.get("is_moe", False),
            )
            self._loaded[name] = loaded
            
            return model, tokenizer
    
    def unload(self, name: str) -> bool:
        """卸载模型"""
        with self._lock:
            if name in self._loaded:
                del self._loaded[name]
                mx.clear_cache()
                logger.info(f"✅ 模型 {name} 已卸载")
                return True
            return False
    
    def list_loaded(self) -> list:
        """列出已加载的模型"""
        with self._lock:
            return [
                {
                    "name": name,
                    "is_vl": loaded.is_vl,
                    "is_moe": loaded.is_moe,
                    "loaded_at": loaded.loaded_at,
                    "last_used": loaded.last_used,
                }
                for name, loaded in self._loaded.items()
            ]
    
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