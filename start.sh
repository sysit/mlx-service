#!/bin/bash
# MLX Service 启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 环境变量
export MODELS_DIR="${MODELS_DIR:-$HOME/models/mlx-community}"
export DEFAULT_MODEL="${DEFAULT_MODEL:-qwen3.5-122b}"
export MAX_TOKENS="${MAX_TOKENS:-16384}"
export PORT="${PORT:-11434}"
export HOST="${HOST:-0.0.0.0}"

# Cache 配置
export ENABLE_PREFIX_CACHE="${ENABLE_PREFIX_CACHE:-true}"
export CACHE_MAX_ENTRIES="${CACHE_MAX_ENTRIES:-20}"
export CACHE_MAX_MEMORY_GB="${CACHE_MAX_MEMORY_GB:-30}"

# API Keys (可选)
export API_KEYS="sk-xxx,sk-yyy,sk-prod-xxx"

# VRAM 限制
export VRAM_LIMIT_GB="${VRAM_LIMIT_GB:-110}"

echo "🚀 MLX Service v3.1"
echo "   模型目录: $MODELS_DIR"
echo "   默认模型: $DEFAULT_MODEL"
echo "   端口: $PORT"
echo "   日志目录: $SCRIPT_DIR/logs"
echo "   Cache目录: $SCRIPT_DIR/cache"
echo ""

# 启动服务
python -m mlx_service.main "$@"
