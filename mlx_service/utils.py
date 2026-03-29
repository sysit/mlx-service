#!/usr/bin/env python3
"""
MLX Service - 公共工具函数

抽取公共函数，避免代码重复。

包含：
- Prompt 构建函数（纯文本 + VL）
- 资源清理函数
- SSE chunk 生成
- 消息格式转换
"""
import json
import time
import mlx.core as mx
from typing import List, Dict, Any, Optional
from loguru import logger


# ============================================================================
# Prompt 构建函数（纯文本）
# ============================================================================

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


# ============================================================================
# Prompt 构建函数（VL 多模态）
# ============================================================================

def build_prompt_vl_manual(messages: list) -> str:
    """VL 模型手动构建 prompt（Qwen 格式）
    
    用于不支持 apply_chat_template 的 VL 模型。
    
    Args:
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
    
    Returns:
        构建好的 prompt 字符串
    """
    prompt_parts = []
    for msg in messages:
        prompt_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)


def build_text_content(messages: list, image_token: str = "<|image_pad|>") -> list:
    """构建文本内容，为图片位置插入 image_token
    
    处理多模态消息，将图片 token 插入到文本中。
    
    Args:
        messages: 消息列表，每条消息可能是字符串或多模态列表
        image_token: 图片占位符 token
    
    Returns:
        转换后的消息列表
    """
    result = []
    for msg in messages:
        if isinstance(msg.get('content'), str):
            result.append({"role": msg['role'], "content": msg['content']})
        elif isinstance(msg.get('content'), list):
            # 处理多模态内容
            text_parts = []
            has_image = False
            for item in msg['content']:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") in ("image_url", "image"):
                        has_image = True
                        text_parts.insert(0, image_token)  # 图片 token 放在前面
            
            content = "\n".join(text_parts) if text_parts else image_token if has_image else ""
            result.append({"role": msg['role'], "content": content})
        else:
            result.append({"role": msg['role'], "content": str(msg.get('content', ''))})
    return result


def build_messages(request_messages: list) -> list:
    """转换消息格式（纯文本）
    
    将消息统一转换为 {"role": "...", "content": "..."} 格式。
    
    Args:
        request_messages: 原始消息列表
    
    Returns:
        转换后的消息列表
    """
    return [
        {"role": msg.get('role', 'user'), "content": msg.get('content') if isinstance(msg.get('content'), str) else str(msg.get('content', ''))}
        for msg in request_messages
    ]


def build_prompt_vl(processor, messages: list, image_token: str, model_config: dict = None) -> str:
    """构建 VL 模型的 prompt
    
    使用 mlx_vlm 的 apply_chat_template，支持 enable_thinking 参数。
    
    Args:
        processor: VL 处理器
        messages: 消息列表
        image_token: 图片占位符 token
        model_config: 模型配置（可选）
    
    Returns:
        构建好的 prompt 字符串
    """
    from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
    
    # 构建带 image_token 的文本内容
    text_messages = build_text_content(messages, image_token)
    
    # 使用 mlx_vlm 的 apply_chat_template（支持 enable_thinking）
    try:
        return vlm_apply_chat_template(
            processor,
            model_config or {},
            text_messages,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception as e:
        logger.debug(f"VL prompt template fallback: {e}")
    
    # 尝试使用 processor 的 chat template（fallback）
    if hasattr(processor, 'apply_chat_template'):
        try:
            return processor.apply_chat_template(
                text_messages, tokenize=False, add_generation_prompt=True
            )
        except (TypeError, ValueError) as e:
            logger.debug(f"Processor chat template fallback: {e}")
    
    # 手动构建 prompt（最终 fallback）
    return build_prompt_vl_manual(text_messages)


# ============================================================================
# SSE Chunk 生成
# ============================================================================

def make_chunk(completion_id: str, created: int, model: str, delta: dict, finish_reason: Optional[str] = None) -> str:
    """生成 SSE chunk（OpenAI 格式）
    
    Args:
        completion_id: completion ID
        created: 创建时间戳
        model: 模型名称
        delta: delta 内容
        finish_reason: 结束原因（可选）
    
    Returns:
        SSE 格式的字符串
    """
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ============================================================================
# 图片提取
# ============================================================================

def extract_images(messages: list) -> List[str]:
    """从消息中提取图片 URL
    
    支持 OpenAI 格式和替代格式。
    
    Args:
        messages: 消息列表
    
    Returns:
        图片 URL 列表
    """
    images = []
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # OpenAI format: {"type": "image_url", "image_url": {"url": "..."}}
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            images.append(url)
                    # Alternative format: {"type": "image", "image": "..."}
                    elif item.get("type") == "image":
                        img = item.get("image", "")
                        if img:
                            images.append(img)
    return images


# ============================================================================
# Token 编码（安全处理 VL processor）
# ============================================================================

def encode_tokens(tokenizer, text: str) -> list:
    """安全编码文本为 tokens，处理 VL processor
    
    VL processor 可能没有 encode() 方法，需要使用底层 tokenizer。
    
    Args:
        tokenizer: 分词器或处理器
        text: 要编码的文本
    
    Returns:
        token ID 列表
    """
    # VL processors don't have encode() method, use the underlying tokenizer
    if hasattr(tokenizer, 'encode'):
        return tokenizer.encode(text, add_special_tokens=False)
    elif hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'encode'):
        return tokenizer.tokenizer.encode(text, add_special_tokens=False)
    else:
        # Fallback: return empty list if we can't encode
        logger.warning(f"Tokenizer {type(tokenizer).__name__} has no encode method")
        return []


# ============================================================================
# 资源清理
# ============================================================================

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