#!/usr/bin/env python3
"""
MLX Service - Ollama Compatible API

Ollama 兼容 API 端点，使用 GenerationService。
"""
import asyncio
import time
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from mlx_service.models import ModelManager
from mlx_service.generation import GenerationService
from mlx_service.config import config
from mlx_service.utils import build_prompt, build_prompt_vl_manual, cleanup_on_error


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


# ============ 依赖注入函数 ============

def get_generation_service(request: Request) -> GenerationService:
    """获取 GenerationService 实例"""
    return request.app.state.generation_service


def get_model_manager(request: Request) -> ModelManager:
    """获取 ModelManager 实例"""
    return request.app.state.model_manager


# ============ API Endpoints ============

@router.get("/api/tags")
async def ollama_tags(request: Request):
    """Ollama 兼容：列出模型"""
    mgr = get_model_manager(request)
    
    models = []
    
    # 已加载的模型
    loaded_info = mgr.list_loaded()
    for m in loaded_info.get("models", []):
        models.append({
            "name": m["name"],
            "model": m["name"],
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(m["loaded_at"])),
            "size": 0,
        })
    
    # 可用但未加载的模型
    for m in mgr.registry.list_models():
        if not mgr.is_loaded(m["name"]):
            models.append({
                "name": m["name"],
                "model": m["name"],
                "modified_at": "2026-01-01T00:00:00Z",
                "size": 0,
            })
    
    return {"models": models}


@router.post("/api/chat")
async def ollama_chat(request: OllamaChatRequest, req: Request):
    """Ollama 兼容：聊天 - 使用 GenerationService"""
    gen = get_generation_service(req)
    
    # 构建 messages
    messages = [{"role": m["role"], "content": m["content"]} for m in request.messages]
    
    options = request.options or {}
    max_tokens = options.get("num_predict", 8192)
    temperature = options.get("temperature", 0.7)
    
    try:
        # 调用 GenerationService 生成
        result = await gen.generate(
            model_id=request.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        
        # 转换为 Ollama 格式
        response = result["choices"][0]["message"]["content"]
        
        return {
            "model": request.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": response},
            "done": True,
        }
    
    except Exception as e:
        logger.exception(f"Ollama chat failed for model {request.model}: {e}")
        cleanup_on_error(request.model)
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/generate")
async def ollama_generate(request: OllamaGenerateRequest, req: Request):
    """Ollama 兼容：生成 - 使用 GenerationService"""
    gen = get_generation_service(req)
    
    # 构建 messages（单轮对话）
    messages = [{"role": "user", "content": request.prompt}]
    
    options = request.options or {}
    max_tokens = options.get("num_predict", 8192)
    temperature = options.get("temperature", 0.7)
    
    try:
        # 调用 GenerationService 生成
        result = await gen.generate(
            model_id=request.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        
        # 转换为 Ollama 格式
        response = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        
        return {
            "model": request.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": response,
            "done": True,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }
    
    except Exception as e:
        logger.exception(f"Ollama generate failed for model {request.model}: {e}")
        cleanup_on_error(request.model)
        return JSONResponse(status_code=500, content={"error": str(e)})


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