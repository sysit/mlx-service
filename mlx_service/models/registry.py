#!/usr/bin/env python3
"""MLX Service - Model Registry"""
from pathlib import Path
from typing import Dict, Optional
from loguru import logger

from mlx_service.models.detector import detect_model_type
from mlx_service.capabilities import Capability


class ModelRegistry:
    """模型注册表 - 扫描和管理可用模型"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._models: Dict[str, Path] = {}
        self._model_types: Dict[str, dict] = {}
        self._model_sizes: Dict[str, float] = {}
        self._aliases: Dict[str, str] = {}
        self._short_names: Dict[str, str] = {}
        self._scan()
    
    def _scan(self):
        """扫描模型目录"""
        if not self.models_dir.exists():
            logger.warning(f"模型目录不存在: {self.models_dir}")
            return
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                has_safetensors = any(model_dir.glob("*.safetensors"))
                has_config = (model_dir / "config.json").exists()
                
                if has_safetensors or has_config:
                    name = model_dir.name.lower()
                    self._models[name] = model_dir
                    
                    model_info = detect_model_type(model_dir)
                    self._model_types[name] = model_info
                    
                    try:
                        size_kb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1024
                        size_gb = size_kb / (1024 * 1024)
                        self._model_sizes[name] = size_gb * 1.1
                    except Exception:
                        self._model_sizes[name] = 0.0
                    
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
        
        if name_lower in self._models:
            return self._models[name_lower]
        
        if name_lower in self._aliases:
            return self._models[self._aliases[name_lower]]
        
        for model_name, path in self._models.items():
            if name_lower in model_name or model_name.startswith(name_lower):
                return path
        
        return None
    
    def get_model_type(self, name: str) -> dict:
        """获取模型类型信息"""
        name_lower = name.lower()
        
        if name_lower in self._model_types:
            return self._model_types[name_lower]
        
        if name_lower in self._aliases:
            actual_name = self._aliases[name_lower]
            return self._model_types.get(actual_name, {"is_vl": False, "is_audio": False, "is_moe": False, "arch": "unknown", "capabilities": Capability.TEXT})
        
        for model_name, info in self._model_types.items():
            if name_lower in model_name or model_name.startswith(name_lower):
                return info
        
        return {"is_vl": False, "is_audio": False, "is_moe": False, "arch": "unknown", "capabilities": Capability.TEXT}
    
    def get_model_size(self, name: str) -> float:
        """获取模型内存占用估算 (GB)"""
        name_lower = name.lower()
        
        if name_lower in self._model_sizes:
            return self._model_sizes[name_lower]
        
        if name_lower in self._aliases:
            actual_name = self._aliases[name_lower]
            return self._model_sizes.get(actual_name, 0.0)
        
        for model_name, size in self._model_sizes.items():
            if name_lower in model_name or model_name.startswith(name_lower):
                return size
        
        return 0.0
    
    def list_models(self) -> list:
        """列出所有可用模型"""
        result = []
        for name, path in self._models.items():
            model_type = self._model_types.get(name, {})
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