#!/usr/bin/env python3
"""
MLX Service - OpenAI Compatible API

API 模块只做格式转换和路由，生成逻辑由 GenerationService 提供。
"""
import time
import uuid
import asyncio
import tempfile
import os
from typing import List, Optional, Dict, Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from mlx_service.models import ModelManager, Capability
from mlx_service.generation import GenerationService
from mlx_service.config import config
from mlx_service.utils import (
    build_prompt, build_prompt_vl, build_messages,
    extract_images, make_chunk, encode_tokens, cleanup_on_error
)


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


# ============ API Endpoints ============

@router.get("/v1/models")
async def list_models(request: Request):
    mgr = get_model_manager(request)
    
    loaded_info = mgr.list_loaded()
    loaded_models = loaded_info.get("models", [])
    models = [ModelInfo(id=m["name"]) for m in loaded_models]
    # 使用短别名作为主 ID
    models += [ModelInfo(id=m["name"]) for m in mgr.registry.list_models() if not mgr.is_loaded(m.get("full_name", m["name"]))]
    
    return {"object": "list", "data": [m.model_dump() for m in models]}


@router.get("/v1/models/loaded")
async def list_loaded_models(request: Request):
    mgr = get_model_manager(request)
    return mgr.list_loaded()


@router.post("/v1/models/{model_name}/load")
async def load_model(model_name: str, request: Request):
    mgr = get_model_manager(request)
    try:
        mgr.get(model_name)
        return {"status": "ok", "model": model_name}
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/v1/models/{model_name}/unload")
async def unload_model(model_name: str, request: Request):
    mgr = get_model_manager(request)
    return {"status": "ok" if mgr.unload(model_name) else "not_loaded", "model": model_name}


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, req: Request):
    """聊天完成端点 - 使用 GenerationService"""
    gen = get_generation_service(req)
    
    # 提取图片
    images = extract_images([m.model_dump() for m in request.messages])
    
    max_tokens = request.max_tokens or 8192
    temperature = request.temperature or 0.7
    
    # 调用统一生成
    if request.stream:
        # 流式：generate(stream=True) 返回异步生成器
        # 需要先 await 获取生成器，再迭代
        async def stream_response():
            stream_gen = await gen.generate(
                model_id=request.model,
                messages=[m.model_dump() for m in request.messages],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                images=images
            )
            async for chunk in stream_gen:
                yield chunk
        
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
        )
    else:
        result = await gen.generate(
            model_id=request.model,
            messages=[m.model_dump() for m in request.messages],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            images=images
        )
        return result


# ============ 向后兼容：生成函数 wrapper ============
# 这些函数保留用于过渡期，直接调用 GenerationService 的内部方法

async def generate_sync(model, tokenizer, messages: list, max_tokens: int, temperature: float, model_name: str) -> dict:
    """向后兼容 wrapper"""
    from mlx_service.cache import get_cache
    from mlx_service.generation import GenerationService
    from mlx_service.config import config
    
    # 创建临时 GenerationService（用于向后兼容）
    gen = GenerationService(model_manager, get_cache(), config)
    return await gen._generate(model, tokenizer, messages, max_tokens, temperature, model_name)


async def stream_generate(model, tokenizer, messages: list, max_tokens: int, temperature: float, model_name: str) -> AsyncGenerator[str, None]:
    """向后兼容 wrapper"""
    from mlx_service.cache import get_cache
    from mlx_service.generation import GenerationService
    from mlx_service.config import config
    
    gen = GenerationService(model_manager, get_cache(), config)
    async for chunk in gen._generate_stream(model, tokenizer, messages, max_tokens, temperature, model_name):
        yield chunk


async def generate_sync_vl(model, processor, messages: List[ChatMessage], images: List[str], max_tokens: int, temperature: float, model_name: str) -> dict:
    """向后兼容 wrapper"""
    from mlx_service.cache import get_cache
    from mlx_service.generation import GenerationService
    from mlx_service.config import config
    
    gen = GenerationService(model_manager, get_cache(), config)
    return await gen._generate_vl(model, processor, [m.model_dump() for m in messages], images, max_tokens, temperature, model_name)


async def stream_generate_vl(model, processor, messages: List[ChatMessage], images: List[str], max_tokens: int, temperature: float, model_name: str) -> AsyncGenerator[str, None]:
    """向后兼容 wrapper"""
    from mlx_service.cache import get_cache
    from mlx_service.generation import GenerationService
    from mlx_service.config import config
    
    gen = GenerationService(model_manager, get_cache(), config)
    async for chunk in gen._generate_vl_stream(model, processor, [m.model_dump() for m in messages], images, max_tokens, temperature, model_name):
        yield chunk


async def transcribe_audio(model, audio_path: str, model_name: str) -> dict:
    """向后兼容 wrapper"""
    from mlx_service.cache import get_cache
    from mlx_service.generation import GenerationService
    from mlx_service.config import config
    
    gen = GenerationService(model_manager, get_cache(), config)
    return await gen.transcribe_audio(model_name, audio_path)


# ============ Audio Transcription API ============

# 支持的音频格式
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    req: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
):
    """
    音频转录 API (OpenAI 兼容格式)
    
    支持 WAV, MP3, M4A, FLAC, OGG, WEBM 格式
    """
    gen = get_generation_service(req)
    
    # 检查文件格式
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            400, 
            f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )
    
    # 保存上传的音频到临时文件
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        logger.info(f"🎤 转录音频: {file.filename} ({len(content)} bytes) using {model}")
        
        # 执行转录（使用 GenerationService）
        result = await gen.transcribe_audio(model, tmp_path)
        
        logger.info(f"✅ 转录完成: {result['text'][:50]}...")
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Audio transcription failed: {e}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")