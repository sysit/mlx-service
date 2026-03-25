# MLX Service v3.0

本地 MLX 模型服务，支持 OpenAI 和 Ollama API。

## 特性

- 模块化架构
- 按需加载模型
- LRU 淘汰策略
- 空闲自动卸载
- 基于 config.json 识别模型类型
- Prefix Cache 支持

## 目录结构

```
mlx-service/
├── main.py          # 入口
├── config.py        # 配置管理
├── models.py        # 模型管理
├── cache.py         # KV Cache 管理
├── api/
│   ├── openai.py    # OpenAI API
│   └── ollama.py    # Ollama API
├── tests/
│   ├── test_models.py
│   ├── test_api.py
│   └── test_config.py
├── start.sh         # 启动脚本
└── requirements.txt
```

## 安装

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 启动

```bash
./start.sh
# 或
PORT=11434 python main.py
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MODELS_DIR | ~/models/mlx-community | 模型目录 |
| DEFAULT_MODEL | qwen3.5-122b | 默认模型 |
| MAX_TOKENS | 16384 | 最大生成 tokens |
| PORT | 11434 | 服务端口 |
| ENABLE_PREFIX_CACHE | true | 启用缓存 |

## API

### OpenAI 兼容

```bash
# 列出模型
curl http://localhost:11434/v1/models

# 聊天补全
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-122b", "messages": [{"role": "user", "content": "你好"}]}'
```

### Ollama 兼容

```bash
# 列出模型
curl http://localhost:11434/api/tags
```

## 测试

```bash
cd tests
python test_models.py
python test_api.py
```