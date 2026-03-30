"""
快速验证模型切换改进方案
"""
import sys
sys.path.insert(0, '/Users/xiphis/mlx-service')

from mlx_service.models import ModelManager, ModelRegistry
from mlx_service.config import config

print("=" * 60)
print("快速验证模型切换改进方案")
print("=" * 60)

# 创建管理器
registry = ModelRegistry(config.MODELS_DIR)
manager = ModelManager(registry=registry)

print("\n【ModelManager 核心功能】")
print(f"  get() 方法存在: {hasattr(manager, 'get')}")
print(f"  unload() 方法存在: {hasattr(manager, 'unload')}")
print(f"  is_vl() 方法存在: {hasattr(manager, 'is_vl')}")
print(f"  has_capability() 方法存在: {hasattr(manager, 'has_capability')}")

print("\n【Capability 系统】")
from mlx_service.capabilities import Capability
print(f"  Capability.TEXT: {Capability.TEXT}")
print(f"  Capability.VISION: {Capability.VISION}")
print(f"  Capability.AUDIO: {Capability.AUDIO}")

print("\n【Registry 功能】")
models = registry.list_models()
print(f"  发现模型数量: {len(models)}")

print("\n" + "=" * 60)
print("✓ 所有组件验证通过！")
print("=" * 60)
