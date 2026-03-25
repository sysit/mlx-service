#!/usr/bin/env python3
"""
MLX Service - OpenAI Compatible API
"""
import time
import uuid
import asyncio
import json
from typing import List, Optional, Dict, Any, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from models import ModelManager


router = APIRouter()


# ============ Request Models ============

class ChatMessage(BaseModel):
    role: str
    content: str | List[Dict[str, Any]]
    

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"


# ============ Helper Functions ============

def build_messages(request_messages: List[ChatMessage]) -> list:
    """转换消息格式"""
    return [
        {"role": msg.role, "content": msg.content if isinstance(msg.content, str) else str(msg.content)}
        for msg in request_messages
    ]


def make_chunk(completion_id: str, created: int, model: str, delta: dict, finish_reason: Optional[str] = None) -> str:
    """生成 SSE chunk"""
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ============ Model Manager ============
model_manager: ModelManager = None


def set_model_manager(mgr: ModelManager):
    global model_manager
    model_manager = mgr


# ============ API Endpoints ============

@router.get("/v1/models")
async def list_models():
    if not model_manager:
        raise HTTPException(500, "Model manager not initialized")
    
    models = [ModelInfo(id=m["name"]) for m in model_manager.list_loaded()]
    models += [ModelInfo(id=m["name"]) for m in model_manager.registry.list_models() if not model_manager.is_loaded(m["name"])]
    
    return {"object": "list", "data": [m.model_dump() for m in models]}


@router.get("/v1/models/loaded")
async def list_loaded_models():
    if not model_manager:
        raise HTTPException(500, "Model manager not initialized")
    return {"models": model_manager.list_loaded()}


@router.post("/v1/models/{model_name}/load")
async def load_model(model_name: str):
    if not model_manager:
        raise HTTPException(500, "Model manager not initialized")
    try:
        model_manager.get(model_name)
        return {"status": "ok", "model": model_name}
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/v1/models/{model_name}/unload")
async def unload_model(model_name: str):
    if not model_manager:
        raise HTTPException(500, "Model manager not initialized")
    return {"status": "ok" if model_manager.unload(model_name) else "not_loaded", "model": model_name}


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if not model_manager:
        raise HTTPException(500, "Model manager not initialized")
    
    try:
        model, tokenizer = model_manager.get(request.model)
    except Exception as e:
        raise HTTPException(400, f"Model not found: {request.model}")
    
    messages = build_messages(request.messages)
    max_tokens = request.max_tokens or 8192
    
    if request.stream:
        return StreamingResponse(
            stream_generate(model, tokenizer, messages, max_tokens, request.temperature, request.model),
            media_type="text/event-stream",
        )
    else:
        return await generate_sync(model, tokenizer, messages, max_tokens, request.temperature, request.model)


# ============ Generation Functions ============

def build_prompt(tokenizer, messages: list) -> str:
    """构建 prompt，禁用 thinking"""
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])


async def generate_sync(model, tokenizer, messages: list, max_tokens: int, temperature: float, model_name: str) -> dict:
    """同步生成"""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    
    sampler = make_sampler(temp=temperature) if temperature > 0 else None
    prompt = build_prompt(tokenizer, messages)
    
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)
    
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    completion_tokens = len(tokenizer.encode(response, add_special_tokens=False))
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }


async def stream_generate(model, tokenizer, messages: list, max_tokens: int, temperature: float, model_name: str) -> AsyncGenerator[str, None]:
    """流式生成"""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    
    sampler = make_sampler(temp=temperature) if temperature > 0 else None
    prompt = build_prompt(tokenizer, messages)
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
    first = True
    
    for chunk in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler):
        # chunk.text 是增量文本
        if chunk.text:
            delta = {"content": chunk.text}
            if first:
                delta["role"] = "assistant"
                first = False
            yield make_chunk(completion_id, created, model_name, delta)
        await asyncio.sleep(0)
    
    yield make_chunk(completion_id, created, model_name, {}, "stop")
    yield "data: [DONE]\n\n"