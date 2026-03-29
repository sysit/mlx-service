#!/usr/bin/env python3
"""
MLX Service - OpenAI Compatible API
"""
import time
import uuid
import asyncio
import tempfile
import os
from typing import List, Optional, Dict, Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from mlx_service.models import ModelManager
from mlx_service.cache import get_cache
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
    
    loaded_info = model_manager.list_loaded()
    loaded_models = loaded_info.get("models", [])
    models = [ModelInfo(id=m["name"]) for m in loaded_models]
    # 使用短别名作为主 ID
    models += [ModelInfo(id=m["name"]) for m in model_manager.registry.list_models() if not model_manager.is_loaded(m.get("full_name", m["name"]))]
    
    return {"object": "list", "data": [m.model_dump() for m in models]}


@router.get("/v1/models/loaded")
async def list_loaded_models():
    if not model_manager:
        raise HTTPException(500, "Model manager not initialized")
    return model_manager.list_loaded()


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
        model, processor = model_manager.get(request.model)
        is_vl = model_manager.is_vl(request.model)
    except Exception as e:
        raise HTTPException(400, f"Model not found: {request.model}")
    
    # 提取图片
    images = extract_images([m.model_dump() for m in request.messages])
    
    # 检查 VL 模型是否收到图片
    if images and not is_vl:
        logger.warning(f"Non-VL model {request.model} received images, ignoring")
        images = []
    
    max_tokens = request.max_tokens or 8192
    
    # VL 模型使用专门的生成函数
    if is_vl:
        if request.stream:
            return StreamingResponse(
                stream_generate_vl(model, processor, request.messages, images, max_tokens, request.temperature, request.model),
                media_type="text/event-stream",
            )
        else:
            return await generate_sync_vl(model, processor, request.messages, images, max_tokens, request.temperature, request.model)
    
    # 纯文本模型
    messages = build_messages([m.model_dump() for m in request.messages])
    if request.stream:
        return StreamingResponse(
            stream_generate(model, processor, messages, max_tokens, request.temperature, request.model),
            media_type="text/event-stream",
        )
    else:
        return await generate_sync(model, processor, messages, max_tokens, request.temperature, request.model)


# ============ Generation Functions ============

async def generate_sync(model, tokenizer, messages: list, max_tokens: int, temperature: float, model_name: str) -> dict:
    """同步生成（支持超时和错误恢复）"""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import make_prompt_cache
    
    cache = get_cache()
    sampler = make_sampler(temp=temperature) if temperature > 0 else None
    prompt = build_prompt(tokenizer, messages)
    tokens = encode_tokens(tokenizer, prompt)
    
    try:
        # 检查缓存
        prompt_cache = None
        if cache:
            prompt_cache, remaining = cache.lookup(tokens, id(model))
            if prompt_cache:
                logger.debug(f"Cache hit: {len(tokens)} tokens")
                prompt_cache = None
        
        # 创建或复用 prompt_cache
        if prompt_cache is None:
            prompt_cache = make_prompt_cache(model)
        
        # 使用 asyncio.to_thread 在线程中运行阻塞的 generate
        loop = asyncio.get_event_loop()
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: generate(
                        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                        sampler=sampler, verbose=False, prompt_cache=prompt_cache
                    )
                ),
                timeout=config.GENERATION_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"Generation timeout after {config.GENERATION_TIMEOUT}s for model {model_name}")
            cleanup_on_error(model_name)
            raise HTTPException(504, f"Generation timeout after {config.GENERATION_TIMEOUT}s")
        
        # 存储缓存
        if cache:
            cache.store(tokens, prompt_cache, id(model))
        
        prompt_tokens = len(tokens)
        completion_tokens = len(encode_tokens(tokenizer, response))
        
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Generation failed for model {model_name}: {e}")
        cleanup_on_error(model_name)
        raise HTTPException(500, f"Generation failed: {str(e)}")


async def stream_generate(model, tokenizer, messages: list, max_tokens: int, temperature: float, model_name: str) -> AsyncGenerator[str, None]:
    """流式生成（支持超时和错误恢复）"""
    from mlx_lm import stream_generate as mlx_stream
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import make_prompt_cache
    
    cache = get_cache()
    sampler = make_sampler(temp=temperature) if temperature > 0 else None
    prompt = build_prompt(tokenizer, messages)
    tokens = encode_tokens(tokenizer, prompt)
    
    # 创建 prompt_cache
    prompt_cache = make_prompt_cache(model)
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
    first = True
    start_time = time.time()
    
    try:
        for chunk in mlx_stream(
            model, tokenizer, prompt, max_tokens=max_tokens,
            sampler=sampler, prompt_cache=prompt_cache
        ):
            # 检查超时
            if time.time() - start_time > config.GENERATION_TIMEOUT:
                logger.error(f"Stream generation timeout after {config.GENERATION_TIMEOUT}s")
                yield make_chunk(completion_id, created, model_name, {"content": "\n[Generation timeout]"}, "stop")
                break
            
            if chunk.text:
                delta = {"content": chunk.text}
                if first:
                    delta["role"] = "assistant"
                    first = False
                yield make_chunk(completion_id, created, model_name, delta)
            await asyncio.sleep(0)
        
        # 生成完成后存储缓存
        if cache:
            cache.store(tokens, prompt_cache, id(model))
        
        yield make_chunk(completion_id, created, model_name, {}, "stop")
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        logger.exception(f"Stream generation failed for model {model_name}: {e}")
        cleanup_on_error(model_name)
        yield make_chunk(completion_id, created, model_name, {"content": f"\n[Error: {str(e)}]"}, "stop")
        yield "data: [DONE]\n\n"


# ============ VL Generation Functions ============

async def generate_sync_vl(model, processor, messages: List[ChatMessage], images: List[str], max_tokens: int, temperature: float, model_name: str) -> dict:
    """VL 模型同步生成（支持超时和错误恢复）"""
    from mlx_vlm import generate
    from mlx_lm.sample_utils import make_sampler
    
    sampler = make_sampler(temp=temperature) if temperature > 0 else None
    image_token = getattr(processor, 'image_token', '<|image_pad|>')
    model_config = getattr(model, 'config', {})
    prompt = build_prompt_vl(processor, [m.model_dump() for m in messages], image_token, model_config)
    
    # 使用第一张图片
    image = images[0] if images else None
    
    try:
        # 使用 asyncio.to_thread 在线程中运行阻塞的 generate
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: generate(
                        model, processor,
                        prompt=prompt,
                        image=image,
                        max_tokens=max_tokens,
                        sampler=sampler,
                        verbose=False
                    )
                ),
                timeout=config.GENERATION_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"VL generation timeout after {config.GENERATION_TIMEOUT}s for model {model_name}")
            cleanup_on_error(model_name)
            raise HTTPException(504, f"Generation timeout after {config.GENERATION_TIMEOUT}s")
        
        response = result.text if hasattr(result, 'text') else str(result)
        
        prompt_tokens = len(encode_tokens(processor, prompt))
        completion_tokens = len(encode_tokens(processor, response))
        
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"VL generation failed for model {model_name}: {e}")
        cleanup_on_error(model_name)
        raise HTTPException(500, f"Generation failed: {str(e)}")


async def stream_generate_vl(model, processor, messages: List[ChatMessage], images: List[str], max_tokens: int, temperature: float, model_name: str) -> AsyncGenerator[str, None]:
    """VL 模型流式生成（支持超时和错误恢复）"""
    from mlx_vlm import stream_generate as vlm_stream
    from mlx_lm.sample_utils import make_sampler
    
    sampler = make_sampler(temp=temperature) if temperature > 0 else None
    image_token = getattr(processor, 'image_token', '<|image_pad|>')
    model_config = getattr(model, 'config', {})
    prompt = build_prompt_vl(processor, [m.model_dump() for m in messages], image_token, model_config)
    
    # 使用第一张图片
    image = images[0] if images else None
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
    first = True
    start_time = time.time()
    
    try:
        for chunk in vlm_stream(
            model, processor,
            prompt=prompt,
            image=image,
            max_tokens=max_tokens,
            sampler=sampler
        ):
            # 检查超时
            if time.time() - start_time > config.GENERATION_TIMEOUT:
                logger.error(f"VL stream generation timeout after {config.GENERATION_TIMEOUT}s")
                yield make_chunk(completion_id, created, model_name, {"content": "\n[Generation timeout]"}, "stop")
                break
            
            if chunk.text:
                delta = {"content": chunk.text}
                if first:
                    delta["role"] = "assistant"
                    first = False
                yield make_chunk(completion_id, created, model_name, delta)
            await asyncio.sleep(0)
        
        yield make_chunk(completion_id, created, model_name, {}, "stop")
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        logger.exception(f"VL stream generation failed for model {model_name}: {e}")
        cleanup_on_error(model_name)
        yield make_chunk(completion_id, created, model_name, {"content": f"\n[Error: {str(e)}]"}, "stop")
        yield "data: [DONE]\n\n"


# ============ Audio Transcription API ============

# 支持的音频格式
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


async def transcribe_audio(model, audio_path: str, model_name: str) -> dict:
    """音频转录（支持超时和错误恢复）"""
    
    try:
        # 使用 asyncio.to_thread 在线程中运行阻塞的 transcription
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: model.generate(audio_path, language="auto")
                ),
                timeout=config.GENERATION_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"Audio transcription timeout after {config.GENERATION_TIMEOUT}s for model {model_name}")
            cleanup_on_error(model_name)
            raise HTTPException(504, f"Transcription timeout after {config.GENERATION_TIMEOUT}s")
        
        text = result.text if hasattr(result, 'text') else str(result)
        
        return {
            "text": text,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Audio transcription failed for model {model_name}: {e}")
        cleanup_on_error(model_name)
        raise HTTPException(500, f"Transcription failed: {str(e)}")


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
):
    """
    音频转录 API (OpenAI 兼容格式)
    
    支持 WAV, MP3, M4A, FLAC, OGG, WEBM 格式
    """
    if not model_manager:
        raise HTTPException(500, "Model manager not initialized")
    
    # 检查文件格式
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            400, 
            f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )
    
    try:
        # 获取模型
        audio_model, _ = model_manager.get(model)
        is_audio = model_manager.is_audio(model)
        
        if not is_audio:
            raise HTTPException(400, f"Model {model} is not an audio model")
    except ValueError as e:
        raise HTTPException(400, str(e))
    
    # 保存上传的音频到临时文件
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        logger.info(f"🎤 转录音频: {file.filename} ({len(content)} bytes) using {model}")
        
        # 执行转录
        result = await transcribe_audio(audio_model, tmp_path, model)
        
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