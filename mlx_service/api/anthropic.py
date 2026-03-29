#!/usr/bin/env python3
"""
MLX Service - Anthropic Messages API

Provides Anthropic-compatible /v1/messages endpoint.
Converts between Anthropic and OpenAI formats, uses GenerationService.
"""

import json
import uuid
import time
import asyncio
from typing import List, Optional, Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from mlx_service.models import ModelManager
from mlx_service.generation import GenerationService
from mlx_service.config import config
from mlx_service.utils import build_prompt, encode_tokens, cleanup_on_error


router = APIRouter()


# ============ Pydantic Models ============

class AnthropicContentBlock(BaseModel):
    """Content block in Anthropic message."""
    type: str  # "text", "image", "tool_use", "tool_result"
    text: Optional[str] = None
    # tool_use
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict] = None
    # tool_result
    tool_use_id: Optional[str] = None
    content: Optional[str | List] = None
    is_error: Optional[bool] = None
    # image
    source: Optional[dict] = None


class AnthropicMessage(BaseModel):
    """Message in Anthropic conversation."""
    role: str  # "user" | "assistant"
    content: str | List[AnthropicContentBlock]


class AnthropicToolDef(BaseModel):
    """Tool definition in Anthropic format."""
    name: str
    description: Optional[str] = None
    input_schema: Optional[dict] = None


class AnthropicRequest(BaseModel):
    """Anthropic Messages API request."""
    model: str
    messages: List[AnthropicMessage]
    system: Optional[str | List[dict]] = None
    max_tokens: int = 4096  # Required in Anthropic API
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[AnthropicToolDef]] = None
    tool_choice: Optional[dict] = None


class AnthropicUsage(BaseModel):
    """Token usage for Anthropic response."""
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponseContentBlock(BaseModel):
    """Content block in Anthropic response."""
    type: str = "text"
    text: Optional[str] = None
    # tool_use
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Any] = None


class AnthropicResponse(BaseModel):
    """Anthropic Messages API response."""
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: str = "message"
    role: str = "assistant"
    model: str
    content: List[AnthropicResponseContentBlock]
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# ============ Adapter Functions ============

def anthropic_to_openai_messages(request: AnthropicRequest) -> list:
    """Convert Anthropic messages to OpenAI format."""
    messages = []
    
    # System message
    if request.system:
        if isinstance(request.system, str):
            messages.append({"role": "system", "content": request.system})
        elif isinstance(request.system, list):
            parts = []
            for block in request.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            if parts:
                messages.append({"role": "system", "content": "\n".join(parts)})
    
    # Convert messages
    for msg in request.messages:
        converted = _convert_message(msg)
        messages.extend(converted)
    
    return messages


def _convert_message(msg: AnthropicMessage) -> list:
    """Convert single Anthropic message to OpenAI format."""
    if isinstance(msg.content, str):
        return [{"role": msg.role, "content": msg.content}]
    
    messages = []
    text_parts = []
    tool_calls = []
    tool_results = []
    
    for block in msg.content:
        if block.type == "text":
            text_parts.append(block.text or "")
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": block.name or "",
                    "arguments": json.dumps(block.input or {}),
                }
            })
        elif block.type == "tool_result":
            result_content = block.content
            if isinstance(result_content, list):
                parts = []
                for item in result_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                result_content = "\n".join(parts)
            tool_results.append({
                "role": "tool",
                "content": str(result_content) if result_content else "",
                "tool_call_id": block.tool_use_id or "",
            })
    
    # Build messages
    if msg.role == "assistant":
        content = "\n".join(text_parts) if text_parts else ""
        if tool_calls:
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        else:
            messages.append({"role": "assistant", "content": content})
    elif msg.role == "user":
        if text_parts:
            messages.append({"role": "user", "content": "\n".join(text_parts)})
        messages.extend(tool_results)
        if not text_parts and not tool_results:
            messages.append({"role": "user", "content": ""})
    
    return messages


def openai_to_anthropic_response(response: dict, model: str) -> AnthropicResponse:
    """Convert OpenAI response to Anthropic format."""
    content = []
    choice = response.get("choices", [{}])[0]
    
    # Text content
    text = choice.get("message", {}).get("content", "")
    if text:
        content.append(AnthropicResponseContentBlock(type="text", text=text))
    
    # Tool calls
    tool_calls = choice.get("message", {}).get("tool_calls", [])
    for tc in tool_calls:
        try:
            tool_input = json.loads(tc.get("function", {}).get("arguments", "{}"))
        except json.JSONDecodeError:
            tool_input = {}
        content.append(AnthropicResponseContentBlock(
            type="tool_use",
            id=tc.get("id", ""),
            name=tc.get("function", {}).get("name", ""),
            input=tool_input,
        ))
    
    if not content:
        content.append(AnthropicResponseContentBlock(type="text", text=""))
    
    # Stop reason
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
    }.get(finish_reason, "end_turn")
    
    usage = response.get("usage", {})
    return AnthropicResponse(
        model=model,
        content=content,
        stop_reason=stop_reason,
        usage=AnthropicUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        ),
    )


# ============ 依赖注入函数 ============

def get_generation_service(request: Request) -> GenerationService:
    """获取 GenerationService 实例"""
    return request.app.state.generation_service


def get_model_manager(request: Request) -> ModelManager:
    """获取 ModelManager 实例"""
    return request.app.state.model_manager


# ============ 向后兼容：全局 model_manager ============
model_manager: ModelManager = None


def set_model_manager(mgr: ModelManager):
    """向后兼容：设置全局 model_manager"""
    global model_manager
    model_manager = mgr


# ============ API Endpoint ============

@router.post("/v1/messages")
async def create_message(request: AnthropicRequest, req: Request):
    """Anthropic Messages API endpoint."""
    gen = get_generation_service(req)
    
    # Convert to OpenAI format
    messages = anthropic_to_openai_messages(request)
    max_tokens = request.max_tokens or 4096
    temperature = request.temperature if request.temperature is not None else 0.7
    
    if request.stream:
        return StreamingResponse(
            stream_anthropic(gen, messages, max_tokens, temperature, request.model),
            media_type="text/event-stream",
        )
    else:
        return await generate_anthropic(gen, messages, max_tokens, temperature, request.model)


# ============ Generation Functions ============

async def generate_anthropic(gen: GenerationService, messages: list, max_tokens: int, temperature: float, model_name: str) -> AnthropicResponse:
    """Non-streaming generation - 使用 GenerationService."""
    try:
        # 调用 GenerationService 生成
        result = await gen.generate(
            model_id=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        
        # 转换为 Anthropic 格式
        return openai_to_anthropic_response(result, model_name)
    
    except Exception as e:
        cleanup_on_error(model_name)
        logger.exception(f"Generation failed: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")


async def stream_anthropic(gen: GenerationService, messages: list, max_tokens: int, temperature: float, model_name: str) -> AsyncGenerator[str, None]:
    """Streaming generation in Anthropic SSE format - 使用 GenerationService."""
    from mlx_lm import stream_generate as mlx_stream
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import make_prompt_cache
    
    # 获取模型和 tokenizer（GenerationService 内部处理）
    try:
        model, tokenizer = gen.model_manager.get(model_name)
    except Exception as e:
        raise HTTPException(400, f"Model not found: {model_name}")
    
    sampler = make_sampler(temp=temperature) if temperature > 0 else None
    prompt = build_prompt(tokenizer, messages)
    prompt_cache = make_prompt_cache(model)
    
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    start_time = time.time()
    
    try:
        # Send message_start event
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None}})}\n\n"
        
        # Send content_block_start
        block_id = f"blk_{uuid.uuid4().hex[:24]}"
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # Stream content
        for chunk in mlx_stream(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler, prompt_cache=prompt_cache):
            # 检查超时
            if time.time() - start_time > config.GENERATION_TIMEOUT:
                logger.error(f"Anthropic stream generation timeout after {config.GENERATION_TIMEOUT}s")
                timeout_msg = "\n[Generation timeout]"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': timeout_msg}})}\n\n"
                break
            
            if chunk.text:
                delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": chunk.text}}
                yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
            await asyncio.sleep(0)
        
        # Send content_block_stop
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        
        # Send message_delta with stop_reason
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 0}})}\n\n"
        
        # Send message_stop
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    
    except Exception as e:
        cleanup_on_error(model_name)
        logger.exception(f"Stream generation failed: {e}")
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"