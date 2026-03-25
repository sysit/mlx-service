#!/usr/bin/env python3
"""
MLX Service - Ollama Compatible API
"""
import time
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from models import ModelManager
from api.openai import set_model_manager, chat_completions, ChatRequest, ChatMessage


router = APIRouter()


# ============ Request Models ============

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False
    options: Optional[Dict[str, Any]] = None


# ============ API Endpoints ============

@router.get("/api/tags")
async def ollama_tags():
    """Ollama 兼容：列出模型"""
    from api.openai import model_manager
    
    if not model_manager:
        return {"models": []}
    
    models = []
    
    # 已加载的模型
    for m in model_manager.list_loaded():
        models.append({
            "name": m["name"],
            "model": m["name"],
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(m["loaded_at"])),
            "size": 0,
        })
    
    # 可用但未加载的模型
    for m in model_manager.registry.list_models():
        if not model_manager.is_loaded(m["name"]):
            models.append({
                "name": m["name"],
                "model": m["name"],
                "modified_at": "2026-01-01T00:00:00Z",
                "size": 0,
            })
    
    return {"models": models}


@router.post("/api/chat")
async def ollama_chat(request: OllamaChatRequest):
    """Ollama 兼容：聊天"""
    from api.openai import model_manager, build_prompt
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    
    if not model_manager:
        return JSONResponse(status_code=503, content={"error": "Model manager not initialized"})
    
    try:
        model, tokenizer = model_manager.get(request.model)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    
    options = request.options or {}
    max_tokens = options.get("num_predict", 8192)
    temperature = options.get("temperature", 0.7)
    
    # 构建 prompt（禁用 thinking）
    messages = [{"role": m["role"], "content": m["content"]} for m in request.messages]
    prompt = build_prompt(tokenizer, messages)
    sampler = make_sampler(temp=temperature) if temperature > 0 else None
    
    response = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)
    
    return {
        "model": request.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": response},
        "done": True,
    }


@router.post("/api/generate")
async def ollama_generate(request: OllamaGenerateRequest):
    """Ollama 兼容：生成"""
    from api.openai import model_manager, build_prompt
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    
    if not model_manager:
        return JSONResponse(status_code=503, content={"error": "Model manager not initialized"})
    
    try:
        model, tokenizer = model_manager.get(request.model)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    
    options = request.options or {}
    max_tokens = options.get("num_predict", 8192)
    temperature = options.get("temperature", 0.7)
    
    # 构建 prompt（禁用 thinking）
    prompt = build_prompt(tokenizer, [{"role": "user", "content": request.prompt}])
    sampler = make_sampler(temp=temperature) if temperature > 0 else None
    
    response = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)
    
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    completion_tokens = len(tokenizer.encode(response, add_special_tokens=False))
    
    return {
        "model": request.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": response,
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    }


@router.post("/api/show")
async def ollama_show(request: Request):
    """Ollama 兼容：显示模型信息"""
    data = await request.json()
    model_name = data.get("name", "unknown")
    
    return {
        "name": model_name,
        "modified_at": "2026-01-01T00:00:00Z",
        "details": {
            "format": "mlx",
            "family": "qwen",
        },
    }