#!/usr/bin/env python3
"""
MLX Service v3.0
"""
import time
from contextlib import asynccontextmanager

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from mlx_service.config import config
from mlx_service.models import ModelRegistry, ModelManager
from mlx_service.cache import init_cache
from mlx_service.api import openai, ollama, anthropic


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期"""
    logger.info("🚀 MLX Service v3.0 启动中...")
    
    # 初始化缓存
    init_cache(config)
    
    # 初始化模型管理器
    registry = ModelRegistry(config.MODELS_DIR)
    model_manager = ModelManager(
        registry=registry,
        max_loaded=config.MAX_LOADED_MODELS,
        idle_timeout=config.MODEL_IDLE_TIMEOUT_SEC,
        max_memory_gb=config.MAX_MEMORY_GB,
    )
    openai.set_model_manager(model_manager)
    anthropic.set_model_manager(model_manager)
    
    logger.success(f"✅ 服务就绪: http://{config.HOST}:{config.PORT}")
    logger.info(f"📁 模型目录: {config.MODELS_DIR}")
    logger.info(f"🤖 可用模型: {[m['name'] for m in registry.list_models()]}")
    logger.info(f"💾 内存预算: {config.MAX_MEMORY_GB}GB")
    
    yield
    
    # 清理
    logger.info("🛑 服务关闭中...")
    model_manager.shutdown()
    mx.clear_cache()


app = FastAPI(title="MLX Service", version="3.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai.router)
app.include_router(ollama.router)
app.include_router(anthropic.router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "mlx-service", "version": "3.1.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/v1/cache/stats")
async def cache_stats():
    from mlx_service.cache import get_cache
    cache = get_cache()
    return cache.get_stats() if cache else {"enabled": False}


@app.post("/v1/cache/clear")
async def cache_clear():
    from mlx_service.cache import get_cache
    cache = get_cache()
    if cache:
        cache.clear()
    return {"status": "ok"}


@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    """访问日志中间件"""
    start_time = time.time()
    
    # 处理请求
    response = await call_next(request)
    
    # 记录访问日志
    duration = (time.time() - start_time) * 1000
    status = response.status_code
    method = request.method
    path = request.url.path
    
    # 状态码颜色
    if status < 400:
        status_str = f"{status}"
    elif status < 500:
        status_str = f"{status}"
    else:
        status_str = f"{status}"
    
    logger.info(f"{method} {path} → {status_str} ({duration:.1f}ms)")
    
    return response


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # 只读接口免认证
    public_paths = [
        "/",
        "/health",
        "/v1/health",
        "/v1/models",
        "/v1/models/loaded",
        "/api/tags",
        "/v1/cache/stats",
    ]
    if request.url.path in public_paths:
        return await call_next(request)
    
    # 匹配 /v1/models/{name}/load 等动态路径的前缀
    if request.url.path.startswith("/v1/models/") and request.method == "GET":
        return await call_next(request)
    
    if config.API_KEYS:
        auth = request.headers.get("Authorization", "")
        key = auth.replace("Bearer ", "") if auth.startswith("Bearer ") else None
        if not key or key not in config.API_KEYS:
            return JSONResponse(status_code=401, content={"error": "Invalid API key"})
    
    return await call_next(request)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器，避免泄露内部信息"""
    import uuid
    error_id = str(uuid.uuid4())[:8]
    logger.bind(error=True, error_id=error_id).exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500, 
        content={"error": "Internal server error", "error_id": error_id}
    )


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)