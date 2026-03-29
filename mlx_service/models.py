#!/usr/bin/env python3
"""
MLX Service - Model Management

功能：
1. 自动扫描模型目录
2. 按需加载
3. LRU 淘汰
4. 空闲自动卸载
5. 基于 config.json 识别模型类型
6. 内存预算控制
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
from mlx_vlm import load as load_vlm


@staticmethod
def _check_vision_weights(model_path: Path) -> bool:
    """检查模型权重是否包含 vision tower"""
    try:
        # 检查 model.safetensors.index.json
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            import json
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            return any("vision" in k.lower() for k in weight_map.keys())
        
        # 如果没有 index 文件，检查单个 safetensors 文件
        safetensors = list(model_path.glob("*.safetensors"))
        if safetensors:
            from safetensors import safe_open
            with safe_open(safetensors[0], framework="mlx") as f:
                return any("vision" in k.lower() for k in f.keys())
    except Exception as e:
        logger.warning(f"Failed to check vision weights: {e}")
    
    return False


def detect_model_type(model_path: Path) -> dict:
    """
    从 config.json 检测模型类型
    
    Returns:
        {
            "is_vl": bool,    # 是否是视觉语言模型
            "is_audio": bool, # 是否是音频模型
            "is_moe": bool,   # 是否是 MoE 模型
            "arch": str,      # 架构名称
        }
    """
    config_path = model_path / "config.json"
    
    if not config_path.exists():
        return {"is_vl": False, "is_audio": False, "is_moe": False, "arch": "unknown"}
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        arch = config.get("architectures", ["unknown"])[0] if config.get("architectures") else "unknown"
        model_type = config.get("model_type", "")
        
        # 判断是否是 Audio 模型（Whisper 等）
        is_audio = model_type.lower() == "whisper" or "whisper" in model_type.lower()
        
        # 判断是否是 VL 模型
        # 条件：
        # 1. 顶层有 image_token_id 且权重文件包含 vision weights
        # 2. 或者架构名称包含 ForConditionalGeneration 且有实际的 vision_config
        # 3. 或者模型类型明确包含 vl
        is_vl = False
        
        # 最可靠的判断：顶层有 image_token_id 且有权重
        if config.get("image_token_id") is not None:
            # 还需要检查权重文件是否真的有 vision weights
            # 有些 distill 模型有 image_token_id 但移除了 vision tower
            has_vision_weights = _check_vision_weights(model_path)
            if has_vision_weights:
                is_vl = True
        # 或者有 vision_start_token_id
        elif config.get("vision_start_token_id") is not None:
            has_vision_weights = _check_vision_weights(model_path)
            if has_vision_weights:
                is_vl = True
        # 架构判断
        elif "ForConditionalGeneration" in arch:
            vision_config = config.get("vision_config", {})
            if vision_config and vision_config.get("in_channels", 0) > 0:
                is_vl = True
        # 模型类型包含 vl
        if not is_vl and "vl" in model_type.lower():
            is_vl = True
        
        # 判断是否是 MoE 模型
        text_config = config.get("text_config", {})
        num_experts = config.get("num_experts", text_config.get("num_experts", 0))
        is_moe = num_experts > 0 or "moe" in model_type.lower()
        
        return {
            "is_vl": is_vl,
            "is_audio": is_audio,
            "is_moe": is_moe,
            "arch": arch,
        }
    except Exception as e:
        logger.warning(f"读取模型配置失败 {model_path}: {e}")
        return {"is_vl": False, "is_audio": False, "is_moe": False, "arch": "unknown"}


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
    memory_gb: float = 0.0  # 模型内存占用估算
    
    def touch(self):
        """更新最后使用时间"""
        self.last_used = time.time()


class ModelRegistry:
    """模型注册表 - 扫描和管理可用模型"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._models: Dict[str, Path] = {}
        self._model_types: Dict[str, dict] = {}
        self._model_sizes: Dict[str, float] = {}  # 模型大小 (GB)
        self._aliases: Dict[str, str] = {}  # 短别名 -> 完整名称
        self._short_names: Dict[str, str] = {}  # 完整名称 -> 短别名
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
                    
                    # 估算模型大小（磁盘大小作为内存估算）
                    try:
                        size_kb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1024
                        size_gb = size_kb / (1024 * 1024)
                        # 运行时额外开销约 10%
                        self._model_sizes[name] = size_gb * 1.1
                    except Exception:
                        self._model_sizes[name] = 0.0
                    
                    # 生成别名
                    parts = name.split("-")
                    if len(parts) >= 2:
                        short_name = parts[0] + "-" + parts[1].split(".")[0]
                        if short_name not in self._models:
                            self._aliases[short_name] = name
                            self._short_names[name] = short_name
        
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
            return self._model_types.get(actual_name, {"is_vl": False, "is_audio": False, "is_moe": False, "arch": "unknown"})
        
        # 模糊匹配
        for model_name, info in self._model_types.items():
            if name_lower in model_name or model_name.startswith(name_lower):
                return info
        
        return {"is_vl": False, "is_audio": False, "is_moe": False, "arch": "unknown"}
    
    def get_model_size(self, name: str) -> float:
        """获取模型内存占用估算 (GB)"""
        name_lower = name.lower()
        
        # 直接匹配
        if name_lower in self._model_sizes:
            return self._model_sizes[name_lower]
        
        # 别名匹配
        if name_lower in self._aliases:
            actual_name = self._aliases[name_lower]
            return self._model_sizes.get(actual_name, 0.0)
        
        # 模糊匹配
        for model_name, size in self._model_sizes.items():
            if name_lower in model_name or model_name.startswith(name_lower):
                return size
        
        return 0.0
    
    def list_models(self) -> list:
        """列出所有可用模型"""
        result = []
        for name, path in self._models.items():
            model_type = self._model_types.get(name, {})
            # 优先使用短别名
            short_name = self._short_names.get(name, name)
            result.append({
                "name": short_name,
                "full_name": name,
                "path": str(path),
                "is_vl": model_type.get("is_vl", False),
                "is_audio": model_type.get("is_audio", False),
                "is_moe": model_type.get("is_moe", False),
                "arch": model_type.get("arch", "unknown"),
                "memory_gb": round(self._model_sizes.get(name, 0.0), 1),
            })
        return result


class ModelManager:
    """模型管理器 - 按需加载、LRU 淘汰、空闲卸载、内存预算控制"""
    
    def __init__(self, registry: ModelRegistry, max_loaded: int = 2, idle_timeout: int = 1800, max_memory_gb: float = 120.0):
        self.registry = registry
        self.max_loaded = max_loaded
        self.idle_timeout = idle_timeout
        self.max_memory_gb = max_memory_gb
        
        # 已加载的模型 (name -> LoadedModel)
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.RLock()  # 可递归获取，防止死锁
        
        # 启动空闲检查线程
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
    
    def get(self, name: str) -> Tuple[Any, Any]:
        """获取模型（按需加载）"""
        with self._lock:
            # 已加载
            if name in self._loaded:
                self._loaded.move_to_end(name)
                self._loaded[name].touch()
                return self._loaded[name].model, self._loaded[name].processor
            
            # 清理 GPU 缓存并等待（防止 Metal 并发冲突）
            mx.clear_cache()
            time.sleep(0.5)  # 等待 GPU 操作完成
            
            # 获取新模型的内存估算
            new_model_memory = self.registry.get_model_size(name)
            current_memory = self._current_memory()
            
            logger.debug(f"📊 内存预算: 当前 {current_memory:.1f}GB + 新模型 {new_model_memory:.1f}GB / 上限 {self.max_memory_gb}GB")
            
            # 检查单个模型是否就超过预算
            if new_model_memory > self.max_memory_gb:
                logger.warning(f"⚠️ 模型 {name} 内存占用 {new_model_memory:.1f}GB 超过预算 {self.max_memory_gb}GB")
            
            # 检查是否需要卸载模型（内存预算）
            while current_memory + new_model_memory > self.max_memory_gb and len(self._loaded) > 0:
                oldest_name = next(iter(self._loaded))
                oldest_memory = self._loaded[oldest_name].memory_gb
                logger.info(f"🔄 卸载模型 {oldest_name} ({oldest_memory:.1f}GB) - 内存预算控制")
                del self._loaded[oldest_name]
                mx.clear_cache()
                current_memory = self._current_memory()
            
            # 检查是否超过最大数量（作为备用限制）
            if len(self._loaded) >= self.max_loaded:
                oldest_name = next(iter(self._loaded))
                logger.info(f"🔄 卸载模型 {oldest_name}（LRU）")
                del self._loaded[oldest_name]
                mx.clear_cache()
            
            # 加载模型
            model_path = self.registry.resolve(name)
            if model_path is None:
                raise ValueError(f"模型不存在: {name}")
            
            # 从 config.json 获取模型类型（加载前）
            model_type = self.registry.get_model_type(name)
            is_vl = model_type.get("is_vl", False)
            is_audio = model_type.get("is_audio", False)
            
            model_type_str = "(VL)" if is_vl else "(Audio)" if is_audio else ""
            logger.info(f"🔄 加载模型: {name} <- {model_path} {model_type_str}")
            start = time.time()
            
            if is_audio:
                # 使用 mlx-audio 加载音频模型
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
    
    def list_loaded(self) -> list:
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