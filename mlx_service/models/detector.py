#!/usr/bin/env python3
"""MLX Service - Model Type Detector

参考 vLLM 和 omlx 的判断逻辑：

vLLM 方式：
- 架构名注册表 (_MULTIMODAL_MODELS)
- ClassVar 标志 (supports_multimodal)

omlx 方式：
- model_type 列表 + vision_config 存在

我们采用组合方案：
1. architectures 匹配 VLM_ARCHITECTURES（最准确）
2. model_type 匹配 VLM_MODEL_TYPES + vision_config 存在
3. 权重检测（兜底）
"""
import json
from pathlib import Path
from loguru import logger

from mlx_service.capabilities import Capability


# ============ 魔法数字常量 ============
# 有效 vision_config 至少包含的字段数（典型字段：in_channels, image_size, hidden_size, etc.）
MIN_VISION_CONFIG_FIELDS = 5


# ============ VLM 注册表（单一数据源）============
# 架构 → model_type + capabilities 映射
# 参考 vLLM 的 _MULTIMODAL_MODELS 和 omlx 的 model_type 列表
VLM_REGISTRY = {
    # -------- Qwen 系列 --------
    "QwenVLForConditionalGeneration": {"model_type": "qwen_vl", "supports_vision": True},
    "Qwen2VLForConditionalGeneration": {"model_type": "qwen2_vl", "supports_vision": True},
    "Qwen2_5_VLForConditionalGeneration": {"model_type": "qwen2_5_vl", "supports_vision": True},
    "Qwen3VLForConditionalGeneration": {"model_type": "qwen3_vl", "supports_vision": True},
    "Qwen3VLMoeForConditionalGeneration": {"model_type": "qwen3_vl_moe", "supports_vision": True},
    "Qwen3_5ForConditionalGeneration": {"model_type": "qwen3_5", "supports_vision": True},  # 关键！
    "Qwen3_5MoeForConditionalGeneration": {"model_type": "qwen3_5_moe", "supports_vision": True},  # 关键！

    # -------- Gemma 系列 --------
    "Gemma3ForConditionalGeneration": {"model_type": "gemma3", "supports_vision": True},
    "Gemma3nForConditionalGeneration": {"model_type": "gemma3n", "supports_vision": True},

    # -------- LLaVA 系列 --------
    "LlavaForConditionalGeneration": {"model_type": "llava", "supports_vision": True},
    "LlavaNextForConditionalGeneration": {"model_type": "llava_next", "supports_vision": True},
    "LlavaOnevisionForConditionalGeneration": {"model_type": "multi_modality", "supports_vision": True},

    # -------- Mistral 系列 --------
    "Mistral3ForConditionalGeneration": {"model_type": "mistral3", "supports_vision": True},

    # -------- 其他 VLM --------
    "InternVLChatModel": {"model_type": "internvl_chat", "supports_vision": True},
    "Idefics3ForConditionalGeneration": {"model_type": "idefics3", "supports_vision": True},
    "PaliGemmaForConditionalGeneration": {"model_type": "paligemma", "supports_vision": True},
    "Phi3VForCausalLM": {"model_type": "phi3_v", "supports_vision": True},
    "PixtralForConditionalGeneration": {"model_type": "pixtral", "supports_vision": True},
    "MolmoForCausalLM": {"model_type": "molmo", "supports_vision": True},
    "Florence2ForConditionalGeneration": {"model_type": "florence2", "supports_vision": True},

    # -------- 额外 model_types（无对应架构名或使用别名）--------
    # 这些 model_type 在 VLM_MODEL_TYPES 中但无对应架构名
    "_EXTRA_MLLAMA_": {"model_type": "mllama", "supports_vision": True},
    "_EXTRA_BUNNY_LLAMA_": {"model_type": "bunny_llama", "supports_vision": True},
    "_EXTRA_DEEPSEEKOCR_": {"model_type": "deepseekocr", "supports_vision": True},
    "_EXTRA_DEEPSEEKOCR2_": {"model_type": "deepseekocr_2", "supports_vision": True},
    "_EXTRA_DOTS_OCR_": {"model_type": "dots_ocr", "supports_vision": True},
    "_EXTRA_GLM_OCR_": {"model_type": "glm_ocr", "supports_vision": True},
    "_EXTRA_MINICPMV_": {"model_type": "minicpmv", "supports_vision": True},
    "_EXTRA_PHI4_SIGLIP_": {"model_type": "phi4_siglip", "supports_vision": True},
    "_EXTRA_PHI4MM_": {"model_type": "phi4mm", "supports_vision": True},
}

# 从注册表推导（不再手动维护）
VLM_ARCHITECTURES = {k for k in VLM_REGISTRY if not k.startswith("_EXTRA_")}
VLM_MODEL_TYPES = {v["model_type"] for v in VLM_REGISTRY.values()}

# ============ 已知的音频 model types ============
AUDIO_STT_MODEL_TYPES = {
    "whisper",
    "qwen3_asr",
    "parakeet",
}

AUDIO_STT_ARCHITECTURES = {
    "WhisperForConditionalGeneration",
    "Qwen3ASRForConditionalGeneration",
    "ParakeetForCTC",
}


def _check_vision_weights(model_path: Path) -> bool:
    """检查模型权重是否包含视觉键"""
    try:
        safetensors = list(model_path.glob("*.safetensors"))
        for sf in safetensors:
            from safetensors import safe_open
            with safe_open(sf, framework="mlx") as f:
                for key in f.keys():
                    if 'vision' in key.lower() or 'visual' in key.lower():
                        return True
    except Exception as e:
        logger.warning(f"Failed to check vision weights: {e}")
    
    return False


def detect_model_type(model_path: Path) -> dict:
    """
    从 config.json 检测模型类型
    
    参考 vLLM 和 omlx 的判断逻辑：
    
    优先级：
    1. architectures 匹配 VLM_ARCHITECTURES（最准确，类似 vLLM）
    2. model_type 匹配 VLM_MODEL_TYPES + vision_config 存在（类似 omlx）
    3. 权重检测（兜底）
    
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
        model_type = config.get("model_type", "").lower().replace("-", "_")
        
        # ====== 1. 音频模型判断 ======
        # 1.1 architectures 匹配
        if arch in AUDIO_STT_ARCHITECTURES:
            capabilities = Capability.TEXT | Capability.AUDIO
            return {
                "is_vl": False,
                "is_audio": True,
                "is_moe": False,
                "arch": arch,
                "capabilities": capabilities,
            }
        
        # 1.2 model_type 匹配
        if model_type in AUDIO_STT_MODEL_TYPES:
            capabilities = Capability.TEXT | Capability.AUDIO
            return {
                "is_vl": False,
                "is_audio": True,
                "is_moe": False,
                "arch": arch,
                "capabilities": capabilities,
            }
        
        # ====== 2. VLM 判断 ======
        is_vl = False
        
        # 2.1 architectures 匹配（优先级最高，参考 vLLM）
        # 但需要二次验证：必须有 vision_config 或 视觉权重
        if arch in VLM_ARCHITECTURES:
            # 检查 vision_config 是否存在且有效
            vision_config = config.get("vision_config")
            has_vision_config = vision_config is not None and isinstance(vision_config, dict)
            
            if has_vision_config:
                is_vl = True
                logger.debug(f"VLM detected: architecture={arch} + vision_config exists")
            else:
                # 没有 vision_config，检查权重文件
                if _check_vision_weights(model_path):
                    is_vl = True
                    logger.debug(f"VLM detected: architecture={arch} + vision weights in safetensors")
                else:
                    # 文本版变体
                    logger.debug(f"Text-only variant: architecture={arch} but no vision_config or weights")
        
        # 2.2 model_type 匹配 + vision_config 存在（参考 omlx）
        if not is_vl and model_type in VLM_MODEL_TYPES:
            if "vision_config" in config:
                is_vl = True
                logger.debug(f"VLM detected: model_type={model_type} + vision_config exists")
            else:
                logger.debug(f"Text-only variant: model_type={model_type} but no vision_config")
        
        # 2.3 兜底：检查 vision_config 是否有效
        if not is_vl and "vision_config" in config:
            vision_config = config.get("vision_config")
            if vision_config and isinstance(vision_config, dict) and len(vision_config) >= MIN_VISION_CONFIG_FIELDS:
                is_vl = True
                logger.debug(f"VLM detected: vision_config has {len(vision_config)} fields")
        
        # 2.4 兜底：检查权重文件
        if not is_vl and _check_vision_weights(model_path):
            is_vl = True
            logger.debug(f"VLM detected: has vision weights in safetensors")
        
        # ====== 3. MoE 判断 ======
        text_config = config.get("text_config", {})
        num_experts = config.get("num_experts", text_config.get("num_experts", 0))
        is_moe = num_experts > 0 or "moe" in model_type
        
        # ====== 4. 计算能力标志 ======
        capabilities = Capability.TEXT
        if is_vl:
            capabilities |= Capability.VISION
        
        return {
            "is_vl": is_vl,
            "is_audio": False,
            "is_moe": is_moe,
            "arch": arch,
            "capabilities": capabilities,
        }
    except Exception as e:
        logger.warning(f"读取模型配置失败 {model_path}: {e}")
        return {"is_vl": False, "is_audio": False, "is_moe": False, "arch": "unknown"}