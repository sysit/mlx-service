#!/usr/bin/env python3
"""MLX Service - Model Type Detector"""
import json
from pathlib import Path
from loguru import logger

from mlx_service.capabilities import Capability


def _check_vision_weights(model_path: Path) -> bool:
    """检查模型权重是否包含 vision tower"""
    try:
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            return any("vision" in k.lower() for k in weight_map.keys())
        
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
            "is_vl": bool,
            "is_audio": bool,
            "is_moe": bool,
            "arch": str,
            "capabilities": Capability,
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
        
        is_audio = model_type.lower() == "whisper" or "whisper" in model_type.lower()
        
        is_vl = False
        if config.get("image_token_id") is not None:
            has_vision_weights = _check_vision_weights(model_path)
            if has_vision_weights:
                is_vl = True
        elif config.get("vision_start_token_id") is not None:
            has_vision_weights = _check_vision_weights(model_path)
            if has_vision_weights:
                is_vl = True
        elif "ForConditionalGeneration" in arch:
            vision_config = config.get("vision_config", {})
            if vision_config and vision_config.get("in_channels", 0) > 0:
                is_vl = True
        if not is_vl and "vl" in model_type.lower():
            is_vl = True
        
        text_config = config.get("text_config", {})
        num_experts = config.get("num_experts", text_config.get("num_experts", 0))
        is_moe = num_experts > 0 or "moe" in model_type.lower()
        
        capabilities = Capability.TEXT
        if is_vl:
            capabilities |= Capability.VISION
        if is_audio:
            capabilities |= Capability.AUDIO
        
        return {
            "is_vl": is_vl,
            "is_audio": is_audio,
            "is_moe": is_moe,
            "arch": arch,
            "capabilities": capabilities,
        }
    except Exception as e:
        logger.warning(f"读取模型配置失败 {model_path}: {e}")
        return {"is_vl": False, "is_audio": False, "is_moe": False, "arch": "unknown"}