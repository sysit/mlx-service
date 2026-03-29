#!/usr/bin/env python3
"""
MLX Service - 公共工具函数

抽取公共函数，避免代码重复。
"""
import mlx.core as mx
from loguru import logger


def build_prompt(tokenizer, messages: list) -> str:
    """构建 prompt，禁用 thinking（共享函数）
    
    Args:
        tokenizer: 分词器
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
    
    Returns:
        构建好的 prompt 字符串
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            # 不支持 enable_thinking 参数的旧版本
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])


def cleanup_on_error(model_name: str = None):
    """错误后清理资源
    
    Args:
        model_name: 模型名称（可选，仅用于日志）
    """
    try:
        mx.clear_cache()
        logger.debug(f"GPU cache cleared after error" + (f" for model {model_name}" if model_name else ""))
    except Exception as e:
        logger.warning(f"Failed to clear GPU cache: {e}")