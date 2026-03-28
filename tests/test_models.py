#!/usr/bin/env python3
"""测试模型管理"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
from mlx_service.models import detect_model_type, ModelRegistry


def test_detect_model_type():
    """测试模型类型检测"""
    models_dir = Path.home() / "models" / "mlx-community"
    if not models_dir.exists():
        print("跳过：模型目录不存在")
        return
    
    count = 0
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir() and (model_dir / "config.json").exists():
            info = detect_model_type(model_dir)
            print(f"  {model_dir.name}: VL={info['is_vl']}, MoE={info['is_moe']}")
            count += 1
    
    print(f"\n✅ 检测了 {count} 个模型")


def test_model_registry():
    """测试模型注册表"""
    models_dir = Path.home() / "models" / "mlx-community"
    if not models_dir.exists():
        print("跳过：模型目录不存在")
        return
    
    registry = ModelRegistry(models_dir)
    models = registry.list_models()
    print(f"\n✅ 注册表包含 {len(models)} 个模型")


if __name__ == "__main__":
    test_detect_model_type()
    test_model_registry()