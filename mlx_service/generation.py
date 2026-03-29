#!/usr/bin/env python3
"""
MLX Service - Generation Service

统一生成逻辑，使用依赖注入替代全局变量。

功能：
1. 纯文本生成（同步 + 流式）
2. VL 多模态生成（同步 + 流式）
3. 音频转录
4. 缓存命中检查
5. 超时控制
6. 错误恢复

抽取自 openai.py，API 模块只做格式转换和路由。
"""
import time
import uuid
import asyncio
from typing import Union, AsyncGenerator, List, Dict, Any, Optional

from loguru import logger
from fastapi import HTTPException

from mlx_service.models import ModelManager, Capability
from mlx_service.cache import TieredCache, get_cache
from mlx_service.config import Config
from mlx_service.utils import (
    build_prompt, build_prompt_vl, build_messages,
    extract_images, make_chunk, encode_tokens, cleanup_on_error
)


class GenerationService:
    """统一生成服务"""
    
    def __init__(
        self,
        model_manager: ModelManager,
        cache: Optional[TieredCache],
        config: Config
    ):
        self.model_manager = model_manager
        self.cache = cache
        self.config = config
    
    async def generate(
        self,
        model_id: str,
        messages: list,
        max_tokens: int = None,
        temperature: float = 0.7,
        stream: bool = False,
        images: List[str] = None,
        **kwargs
    ) -> Union[dict, AsyncGenerator[str, None]]:
        """统一生成入口
        
        Args:
            model_id: 模型标识
            messages: 消息列表
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            stream: 是否流式生成
            images: 图片 URL 列表（VL 模型）
            **kwargs: 其他参数
            
        Returns:
            dict（非流式）或 AsyncGenerator（流式）
        """
        try:
            model, tokenizer = self.model_manager.get(model_id)
            is_vl = self.model_manager.has_capability(model_id, Capability.VISION)
        except Exception as e:
            raise HTTPException(400, f"Model not found: {model_id}")
        
        max_tokens = max_tokens or self.config.MAX_TOKENS
        
        # VL 模型检查图片
        if images and not is_vl:
            logger.warning(f"Non-VL model {model_id} received images, ignoring")
            images = []
        
        # VL 模型生成
        if is_vl and images:
            if stream:
                return self._generate_vl_stream(
                    model, tokenizer, messages, images, max_tokens, temperature, model_id
                )
            else:
                return await self._generate_vl(
                    model, tokenizer, messages, images, max_tokens, temperature, model_id
                )
        
        # 纯文本生成
        text_messages = build_messages(messages)
        if stream:
            return self._generate_stream(
                model, tokenizer, text_messages, max_tokens, temperature, model_id
            )
        else:
            return await self._generate(
                model, tokenizer, text_messages, max_tokens, temperature, model_id
            )
    
    # ========================================================================
    # 纯文本生成
    # ========================================================================
    
    async def _generate(
        self,
        model,
        tokenizer,
        messages: list,
        max_tokens: int,
        temperature: float,
        model_name: str
    ) -> dict:
        """同步文本生成（支持缓存、超时和错误恢复）"""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.models.cache import make_prompt_cache
        
        sampler = make_sampler(temp=temperature) if temperature > 0 else None
        prompt = build_prompt(tokenizer, messages)
        tokens = encode_tokens(tokenizer, prompt)
        
        try:
            # 检查缓存
            prompt_cache = None
            if self.cache:
                prompt_cache, remaining = self.cache.lookup(tokens, id(model))
                if prompt_cache:
                    logger.debug(f"Cache hit: {len(tokens)} tokens")
                    prompt_cache = None  # 目前暂不使用缓存命中
            
            # 创建或复用 prompt_cache
            if prompt_cache is None:
                prompt_cache = make_prompt_cache(model)
            
            # 异步执行生成
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
                    timeout=self.config.GENERATION_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"Generation timeout after {self.config.GENERATION_TIMEOUT}s for model {model_name}")
                cleanup_on_error(model_name)
                raise HTTPException(504, f"Generation timeout after {self.config.GENERATION_TIMEOUT}s")
            
            # 存储缓存
            if self.cache:
                self.cache.store(tokens, prompt_cache, id(model))
            
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
    
    async def _generate_stream(
        self,
        model,
        tokenizer,
        messages: list,
        max_tokens: int,
        temperature: float,
        model_name: str
    ) -> AsyncGenerator[str, None]:
        """流式文本生成（支持缓存、超时和错误恢复）"""
        from mlx_lm import stream_generate as mlx_stream
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.models.cache import make_prompt_cache
        
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
                if time.time() - start_time > self.config.GENERATION_TIMEOUT:
                    logger.error(f"Stream generation timeout after {self.config.GENERATION_TIMEOUT}s")
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
            if self.cache:
                self.cache.store(tokens, prompt_cache, id(model))
            
            yield make_chunk(completion_id, created, model_name, {}, "stop")
            yield "data: [DONE]\n\n"
        
        except Exception as e:
            logger.exception(f"Stream generation failed for model {model_name}: {e}")
            cleanup_on_error(model_name)
            yield make_chunk(completion_id, created, model_name, {"content": f"\n[Error: {str(e)}]"}, "stop")
            yield "data: [DONE]\n\n"
    
    # ========================================================================
    # VL 多模态生成
    # ========================================================================
    
    async def _generate_vl(
        self,
        model,
        processor,
        messages: list,
        images: List[str],
        max_tokens: int,
        temperature: float,
        model_name: str
    ) -> dict:
        """VL 模型同步生成（支持超时和错误恢复）"""
        from mlx_vlm import generate
        from mlx_lm.sample_utils import make_sampler
        
        sampler = make_sampler(temp=temperature) if temperature > 0 else None
        image_token = getattr(processor, 'image_token', '<|image_pad|>')
        model_config = getattr(model, 'config', {})
        prompt = build_prompt_vl(processor, messages, image_token, model_config)
        
        # 使用第一张图片
        image = images[0] if images else None
        
        try:
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
                    timeout=self.config.GENERATION_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"VL generation timeout after {self.config.GENERATION_TIMEOUT}s for model {model_name}")
                cleanup_on_error(model_name)
                raise HTTPException(504, f"Generation timeout after {self.config.GENERATION_TIMEOUT}s")
            
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
    
    async def _generate_vl_stream(
        self,
        model,
        processor,
        messages: list,
        images: List[str],
        max_tokens: int,
        temperature: float,
        model_name: str
    ) -> AsyncGenerator[str, None]:
        """VL 模型流式生成（支持超时和错误恢复）"""
        from mlx_vlm import stream_generate as vlm_stream
        from mlx_lm.sample_utils import make_sampler
        
        sampler = make_sampler(temp=temperature) if temperature > 0 else None
        image_token = getattr(processor, 'image_token', '<|image_pad|>')
        model_config = getattr(model, 'config', {})
        prompt = build_prompt_vl(processor, messages, image_token, model_config)
        
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
                if time.time() - start_time > self.config.GENERATION_TIMEOUT:
                    logger.error(f"VL stream generation timeout after {self.config.GENERATION_TIMEOUT}s")
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
    
    # ========================================================================
    # 音频转录
    # ========================================================================
    
    async def transcribe_audio(
        self,
        model_id: str,
        audio_path: str
    ) -> dict:
        """音频转录
        
        Args:
            model_id: 音频模型标识
            audio_path: 音频文件路径
            
        Returns:
            {"text": "转录文本"}
        """
        try:
            model, _ = self.model_manager.get(model_id)
            is_audio = self.model_manager.has_capability(model_id, Capability.AUDIO)
            
            if not is_audio:
                raise HTTPException(400, f"Model {model_id} is not an audio model")
        except ValueError as e:
            raise HTTPException(400, str(e))
        
        try:
            loop = asyncio.get_event_loop()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: model.generate(audio_path, language="auto")
                    ),
                    timeout=self.config.GENERATION_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"Audio transcription timeout after {self.config.GENERATION_TIMEOUT}s for model {model_id}")
                cleanup_on_error(model_id)
                raise HTTPException(504, f"Transcription timeout after {self.config.GENERATION_TIMEOUT}s")
            
            text = result.text if hasattr(result, 'text') else str(result)
            
            return {"text": text}
        
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Audio transcription failed for model {model_id}: {e}")
            cleanup_on_error(model_id)
            raise HTTPException(500, f"Transcription failed: {str(e)}")