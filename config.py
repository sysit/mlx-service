#!/usr/bin/env python3
"""
MLX Service Configuration
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Set
from loguru import logger


@dataclass
class Config:
    """全局配置"""
    
    # 路径
    SERVICE_DIR: Path = field(default_factory=lambda: Path(__file__).parent)
    MODELS_DIR: Path = field(default_factory=lambda: Path.home() / "models" / "mlx-community")
    LOGS_DIR: Path = field(init=False)
    CACHE_DIR: Path = field(init=False)
    
    # 模型
    DEFAULT_MODEL: str = "qwen3.5-122b"
    MAX_LOADED_MODELS: int = 2
    MODEL_IDLE_TIMEOUT_SEC: int = 1800
    MAX_MEMORY_GB: float = 120.0  # 内存预算上限，留 8GB 给系统
    
    # 服务
    HOST: str = "0.0.0.0"
    PORT: int = 11434
    
    # 生成
    MAX_TOKENS: int = 16384
    TEMPERATURE: float = 0.7
    GENERATION_TIMEOUT: int = 300  # 生成超时（秒）
    
    # Cache
    ENABLE_PREFIX_CACHE: bool = True
    CACHE_MAX_ENTRIES: int = 20
    CACHE_MAX_MEMORY_GB: float = 30.0
    ENABLE_CACHE_PERSISTENCE: bool = True
    
    # 安全
    API_KEYS: Set[str] = field(default_factory=set)
    
    # 日志
    LOG_LEVEL: str = "INFO"
    
    def __post_init__(self):
        self.LOGS_DIR = self.SERVICE_DIR / "logs"
        self.CACHE_DIR = self.SERVICE_DIR / "cache"
        
        # 环境变量覆盖
        if os.getenv("MODELS_DIR"):
            self.MODELS_DIR = Path(os.getenv("MODELS_DIR"))
        if os.getenv("DEFAULT_MODEL"):
            self.DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
        if os.getenv("PORT"):
            self.PORT = int(os.getenv("PORT"))
        if os.getenv("MAX_TOKENS"):
            self.MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
        if os.getenv("GENERATION_TIMEOUT"):
            self.GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT"))
        if os.getenv("API_KEYS"):
            self.API_KEYS = {k.strip() for k in os.getenv("API_KEYS").split(",") if k.strip()}
        
        # 创建目录
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 配置日志
        self._setup_logging()
    
    def _setup_logging(self):
        logger.remove()
        
        # 控制台输出
        logger.add(sys.stdout, level=self.LOG_LEVEL)
        
        # 服务日志（启动 + 访问）
        logger.add(
            self.LOGS_DIR / "service_{time:YYYY-MM-DD}.log",
            rotation="10 MB",
            retention="7 days",
            level=self.LOG_LEVEL,
            filter=lambda record: not record["extra"].get("error", False),
        )
        
        # 错误日志（单独文件）
        logger.add(
            self.LOGS_DIR / "error_{time:YYYY-MM-DD}.log",
            rotation="10 MB",
            retention="30 days",
            level="ERROR",
            filter=lambda record: record["extra"].get("error", False) or record["level"].no >= 40,
        )


config = Config()