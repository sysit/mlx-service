#!/usr/bin/env python3
"""
MLX Service - Model Capabilities

使用 Flag 枚举表示模型能力，支持位运算组合。
"""
from enum import Flag, auto


class Capability(Flag):
    """模型能力标志"""
    TEXT = auto()        # 文本生成
    VISION = auto()      # 图像理解 (VL 模型)
    AUDIO = auto()       # 音频处理 (Whisper 等)
    EMBEDDING = auto()   # 向量嵌入
    
    # 常用组合
    TEXT_ONLY = TEXT
    MULTIMODAL = TEXT | VISION
    AUDIO_MODEL = TEXT | AUDIO