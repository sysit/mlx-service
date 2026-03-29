# MLX Service 架构文档

## 概述

MLX Service 是一个本地 MLX 模型推理服务，支持 OpenAI/Ollama/Anthropic API 格式，具备按需加载、内存预算控制、分层缓存等特性。

---

## 模块架构

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                    (FastAPI 入口 + 生命周期)                  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  api/openai   │    │  api/ollama   │    │ api/anthropic │
│  OpenAI API   │    │  Ollama API   │    │ Anthropic API │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│    models     │    │     cache     │    │    config     │
│  模型管理      │    │   分层缓存    │    │   全局配置    │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## 模块职责

### 1. main.py — 入口层

**职责**：
- FastAPI 应用创建与配置
- 生命周期管理（启动/关闭）
- 路由注册
- 中间件（CORS、认证、日志、异常处理）

**公开接口**：
- `app: FastAPI` — FastAPI 应用实例
- `lifespan()` — 异步生命周期管理

**依赖**：
- `mlx_service.config`
- `mlx_service.models`
- `mlx_service.cache`
- `mlx_service.api.*`

---

### 2. config.py — 配置层

**职责**：
- 全局配置定义
- 环境变量覆盖
- 日志配置

**公开接口**：
- `config: Config` — 全局配置实例

**配置项**：
| 配置 | 说明 | 默认值 |
|------|------|--------|
| `HOST` | 服务监听地址 | `0.0.0.0` |
| `PORT` | 服务监听端口 | `8000` |
| `MODELS_DIR` | 模型目录 | `~/models/mlx-community` |
| `DEFAULT_MODEL` | 默认模型 | `qwen3.5-122b` |
| `MAX_LOADED_MODELS` | 最大加载模型数 | `2` |
| `MODEL_IDLE_TIMEOUT_SEC` | 模型空闲超时 | `1800` |
| `MAX_MEMORY_GB` | 内存预算上限 | `120.0` |
| `MAX_TOKENS` | 默认最大生成 Token 数 | `4096` |
| `GENERATION_TIMEOUT` | 生成超时 | `300` |
| `API_KEYS` | API 密钥列表（逗号分隔） | `` (空，不启用认证) |
| `ENABLE_PREFIX_CACHE` | 启用前缀缓存 | `True` |
| `CACHE_MAX_ENTRIES` | 缓存最大条目数 | `100` |
| `CACHE_MAX_MEMORY_GB` | 缓存最大内存占用 | `10.0` |

**依赖**：无

---

### 3. models.py — 模型管理层

**职责**：
- 模型发现与注册（`ModelRegistry`）
- 按需加载与 LRU 淘汰（`ModelManager`）
- 内存预算控制
- 模型类型检测（VL/Audio/MoE）

**公开接口**：
- `ModelRegistry` — 模型注册表
  - `resolve(name)` — 解析模型名称到路径
  - `list_models()` — 列出所有可用模型
  - `get_model_type(name)` — 获取模型类型信息
  - `get_model_size(name)` — 获取模型内存估算

- `ModelManager` — 模型管理器
  - `get(name)` — 获取模型（按需加载）
  - `unload(name)` — 卸载模型
  - `list_loaded()` — 列出已加载模型
  - `is_loaded(name)` — 检查是否已加载
  - `shutdown()` — 关闭管理器

**依赖**：
- `mlx_service.config`（配置读取）
- `mlx_lm`, `mlx_vlm`（模型加载）
- `mlx.core`（缓存清理）

---

### 4. cache.py — 缓存层

**职责**：
- 分层缓存（Hot 内存 + Cold SSD）
- 异步 SSD 写入
- 前缀缓存命中

**公开接口**：
- `init_cache(config)` — 初始化全局缓存
- `get_cache()` — 获取全局缓存实例
- `TieredCache`：
  - `lookup(tokens, model_id)` — 查找缓存
  - `store(tokens, prompt_cache, model_id)` — 存储缓存
  - `clear()` — 清空缓存
  - `get_stats()` — 获取统计信息

**依赖**：
- `mlx_service.config`（配置读取）
- `mlx_lm.models.cache`（缓存序列化）

---

### 5. api/openai.py — OpenAI 兼容 API

**职责**：
- OpenAI 格式 API 实现
- 文本/VL/Audio 模型推理
- 流式/同步生成

**公开接口**：
- `router: APIRouter` — FastAPI 路由
- `set_model_manager(mgr)` — 设置模型管理器

**端点**：
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/models` | 列出模型 |
| GET | `/v1/models/loaded` | 列出已加载模型 |
| POST | `/v1/models/{name}/load` | 加载模型 |
| POST | `/v1/models/{name}/unload` | 卸载模型 |
| POST | `/v1/chat/completions` | 聊天补全 |
| POST | `/v1/audio/transcriptions` | 音频转录 |

**依赖**：
- `mlx_service.models`（模型管理）
- `mlx_service.cache`（缓存）
- `mlx_service.config`（配置）

---

### 6. api/ollama.py — Ollama 兼容 API

**职责**：
- Ollama 格式 API 实现
- 复用 OpenAI API 生成逻辑

**公开接口**：
- `router: APIRouter` — FastAPI 路由

**端点**：
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/tags` | 列出模型 |
| POST | `/api/chat` | 聊天 |
| POST | `/api/generate` | 生成 |
| POST | `/api/show` | 显示模型信息 |

**`/api/show` 返回格式**：
```json
{
  "name": "qwen3.5-122b",
  "size": 68719476736,
  "digest": "abc123",
  "details": {
    "format": "mlx",
    "family": "qwen",
    "parameter_size": "122B",
    "quantization_level": "f16"
  },
  "model_info": {
    "context_length": 32768,
    "embedding_length": 4096,
    "num_layers": 80
  }
}
```
> 注：`digest` 字段为模型文件哈希，`details` 来自模型 `config.json` 元数据。

**依赖**：
- `mlx_service.api.openai`（共享生成函数）
- `mlx_service.models`（模型管理）
- `mlx_service.config`（配置）

---

### 7. api/anthropic.py — Anthropic Messages API

**职责**：
- Anthropic 格式 API 实现
- 格式转换（Anthropic ↔ OpenAI）

**公开接口**：
- `router: APIRouter` — FastAPI 路由
- `set_model_manager(mgr)` — 设置模型管理器

**端点**：
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/messages` | Anthropic Messages API |

**依赖**：
- `mlx_service.models`（模型管理）
- `mlx_service.config`（配置）

---

## 依赖关系

```
main.py
    ├── config.py (无依赖)
    ├── models.py → config
    ├── cache.py → config
    └── api/
        ├── openai.py → config, models, cache
        ├── ollama.py → config, models, openai
        └── anthropic.py → config, models
```

**依赖方向**：API 层 → 模型层/缓存层 → 配置层

---

## 扩展点

### 1. 新增 API 格式

1. 在 `mlx_service/api/` 创建新模块
2. 实现 `router: APIRouter`
3. 在 `main.py` 中注册路由：`app.include_router(new_api.router)`

**示例代码**：
```python
# mlx_service/api/custom.py
from fastapi import APIRouter
from mlx_service.models import get_model_manager

router = APIRouter(prefix="/custom", tags=["custom"])

@router.post("/generate")
async def generate(request: dict):
    mgr = get_model_manager()
    model = await mgr.get(request.get("model", config.DEFAULT_MODEL))
    # ... 生成逻辑
    return {"result": "..."}
```

```python
# main.py 中注册
from mlx_service.api import custom
app.include_router(custom.router)
```

### 2. 新增模型类型

1. 在 `models.py` 的 `detect_model_type()` 中添加检测逻辑
2. 在 `ModelManager.get()` 中添加加载逻辑
3. 在对应 API 中添加生成逻辑

**示例代码**：
```python
# models.py
def detect_model_type(model_path: Path) -> str:
    config = json.loads((model_path / "config.json").read_text())
    arch = config.get("architectures", [None])[0]
    
    if arch == "MyNewModelForCausalLM":
        return "my_new_type"
    # ... 现有逻辑
```

```python
# ModelManager.get() 中添加
async def get(self, name: str):
    # ...
    model_type = self.registry.get_model_type(name)
    if model_type == "my_new_type":
        from my_new_loader import load_model
        return load_model(model_path)
    # ... 现有逻辑
```

### 3. 新增缓存策略

1. 在 `cache.py` 中扩展 `TieredCache`
2. 或创建新的缓存管理器

**示例代码**：
```python
# cache.py
class RedisCache:
    """分布式缓存后端（可选扩展）"""
    
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)
    
    async def lookup(self, tokens: list, model_id: str) -> Optional[bytes]:
        key = f"cache:{model_id}:{hash(tuple(tokens))}"
        return await self.redis.get(key)
    
    async def store(self, tokens: list, data: bytes, model_id: str):
        key = f"cache:{model_id}:{hash(tuple(tokens))}"
        await self.redis.set(key, data, ex=86400)  # 24h TTL
```

---

## 约束

### 模块边界

- **config.py** — 只负责配置，不包含业务逻辑
- **models.py** — 不依赖 API 层，不处理请求格式
- **cache.py** — 不处理业务逻辑，只负责缓存
- **api/** — 不直接操作模型文件，通过 `ModelManager` 访问

### 接口兼容性

- API 端点签名变更需要版本号升级
- `ModelManager` 接口变更需要评估影响范围
- 配置项变更需要提供默认值

### 资源管理

- 模型加载后必须能卸载
- 缓存必须能清理
- GPU 缓存使用后必须 `mx.clear_cache()`

---

## 异常处理规范

### cleanup_on_error() 规范

所有可能失败的资源操作必须使用 `cleanup_on_error()` 确保资源释放：

```python
# models.py
import mlx.core as mx

def cleanup_on_error():
    """GPU 缓存清理函数，在异常时调用"""
    mx.clear_cache()
    logger.debug("GPU cache cleared after error")

# 使用示例
async def load_model_safe(name: str):
    try:
        model = await load_model(name)
        return model
    except Exception as e:
        cleanup_on_error()
        raise ModelLoadError(f"Failed to load {name}: {e}") from e
```

### 异常类型定义

```python
# exceptions.py
class MLXServiceError(Exception):
    """基础异常类"""
    pass

class ModelNotFoundError(MLXServiceError):
    """模型不存在"""
    pass

class ModelLoadError(MLXServiceError):
    """模型加载失败"""
    pass

class MemoryBudgetExceeded(MLXServiceError):
    """内存预算超限"""
    pass

class GenerationTimeout(MLXServiceError):
    """生成超时"""
    pass

class AuthenticationError(MLXServiceError):
    """API 认证失败"""
    pass
```

### 错误处理流程

1. **捕获异常** → 记录日志
2. **调用 `cleanup_on_error()`** → 清理 GPU 缓存
3. **返回错误响应** → HTTP 状态码 + 错误信息
4. **可选** → 卸载问题模型

---

## 开发规范

### 添加新功能

1. 在对应模块添加实现
2. 更新此文档的模块职责和公开接口
3. 添加单元测试到 `tests/`
4. 运行回归测试：`pytest tests/`

### 修改现有功能

1. 评估影响范围（依赖模块、公开接口）
2. 确保向后兼容或升级版本
3. 更新文档和测试

### 代码风格

- 使用 loguru 日志
- 异步函数使用 `async/await`
- 超时控制使用 `asyncio.wait_for`
- 错误后调用 `cleanup_on_error()` 清理 GPU 缓存

---

## 版本策略

### 语义化版本

遵循 [SemVer 2.0](https://semver.org/)：

- **MAJOR** — 不兼容的 API 变更
- **MINOR** — 向后兼容的功能新增
- **PATCH** — 向后兼容的问题修复

### API 版本控制

| 变更类型 | 版本号变更 | 说明 |
|----------|------------|------|
| 新增端点 | MINOR | 扩展功能 |
| 新增可选参数 | MINOR | 向后兼容 |
| 新增必需参数 | MAJOR | 破坏兼容性 |
| 删除端点 | MAJOR | 破坏兼容性 |
| 响应格式变更 | MAJOR | 破坏兼容性 |
| Bug 修复 | PATCH | 不影响 API |

### 模型兼容性

- 模型名称变更：在新版本中保留别名映射
- 模型格式升级：支持旧格式加载，警告提示迁移
- 模型参数变更：默认值保持向后兼容

### 配置迁移

配置项变更时提供迁移指南：

```python
# config.py
DEPRECATED_CONFIGS = {
    "OLD_CACHE_SIZE": "CACHE_MAX_MEMORY_GB",
}

def migrate_config(config: dict) -> dict:
    for old_key, new_key in DEPRECATED_CONFIGS.items():
        if old_key in config:
            logger.warning(f"Config '{old_key}' is deprecated, use '{new_key}'")
            config[new_key] = config.pop(old_key)
    return config
```

---

---

## 文档信息

- **创建者**: Quistis 🏛️ (architect)
- **创建日期**: 2026-03-29
- **遵守者**: Selphie 💻 (developer)
- **审核者**: Rinoa 🛡️ (sentinel)

_Selphie 开发 mlx-service 时必须遵守此架构文档。修改架构需经过 Quistis 审核。_