#!/usr/bin/env python3
"""测试配置"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_service.config import config


def test_config():
    """测试配置"""
    print(f"模型目录: {config.MODELS_DIR}")
    print(f"默认模型: {config.DEFAULT_MODEL}")
    print(f"端口: {config.PORT}")
    print(f"Cache: {'启用' if config.ENABLE_PREFIX_CACHE else '禁用'}")
    print("\n✅ 配置正常")


if __name__ == "__main__":
    test_config()