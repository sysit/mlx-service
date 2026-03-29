# MLX Service 全面测试用例框架

> 版本: 1.0 | 日期: 2026-03-29 | 状态: 🟡 设计稿

---

## 目录

1. [概述](#1-概述)
2. [架构层测试](#2-架构层测试)
3. [代码层测试](#3-代码层测试)
4. [日志层测试](#4-日志层测试)
5. [性能层测试](#5-性能层测试)
6. [稳定性层测试](#6-稳定性层测试)
7. [安全层测试](#7-安全层测试)
8. [测试脚本框架设计](#8-测试脚本框架设计)
9. [每日测试执行计划](#9-每日测试执行计划)

---

## 1. 概述

### 1.1 目标

建立系统性测试框架，确保覆盖 **架构、代码、日志、性能、稳定性、安全** 六个维度。

### 1.2 测试矩阵

| 维度 | 测试类型 | 覆盖率目标 | 优先级 |
|------|----------|------------|--------|
| 架构层 | 模块依赖、API 兼容、模型加载、缓存 | 90% | P0 |
| 代码层 | 单元测试、边界条件、错误处理、输入验证 | 85% | P0 |
| 日志层 | 错误日志、访问日志、日志轮转 | 80% | P1 |
| 性能层 | 延迟、并发、内存、GPU 资源 | 90% | P1 |
| 稳定性层 | 长时间运行、崩溃恢复、资源耗尽 | 85% | P1 |
| 安全层 | 认证、CORS、输入注入 | 100% | P0 |

---

## 2. 架构层测试

### 2.1 模块依赖测试

| 用例 ID | 测试用例 | 验证方法 | 预期结果 |
|---------|----------|----------|----------|
| ARCH-001 | main.py 导入所有必需模块 | 静态导入检查 | 无 ImportError |
| ARCH-002 | config.py 提供全局配置实例 | `config` 对象存在性 | 非 None |
| ARCH-003 | ModelRegistry 正确发现模型 | 调用 `list_models()` | 返回模型列表 |
| ARCH-004 | ModelManager 生命周期管理 | 启动/关闭流程 | 无资源泄漏 |
| ARCH-005 | 缓存系统初始化 | `init_cache()` 调用 | 缓存目录创建成功 |

### 2.2 API 兼容性测试

#### 2.2.1 OpenAI API 兼容

| 用例 ID | 测试用例 | 端点 | 预期状态码 |
|---------|----------|------|------------|
| OAI-001 | 健康检查 | GET /health | 200 |
| OAI-002 | 模型列表 | GET /v1/models | 200 |
| OAI-003 | 聊天补全 | POST /v1/chat/completions | 200 |
| OAI-004 | 流式聊天补全 | POST /v1/chat/completions (stream) | 200 |
| OAI-005 | 嵌入生成 | POST /v1/embeddings | 200 |
| OAI-006 | 模型卸载 | POST /v1/models/unload | 200 |
| OAI-007 | 无效模型请求 | POST /v1/chat/completions (不存在的模型) | 404 |
| OAI-008 | 缺少必需字段 | POST /v1/chat/completions (无 messages) | 422 |

#### 2.2.2 Ollama API 兼容

| 用例 ID | 测试用例 | 端点 | 预期状态码 |
|---------|----------|------|------------|
| OLL-001 | 模型标签列表 | GET /api/tags | 200 |
| OLL-002 | 生成 | POST /api/generate | 200 |
| OLL-003 | 聊天 | POST /api/chat | 200 |
| OLL-004 | 嵌入 | POST /api/embed | 200 |
| OLL-005 | 模型信息 | GET /api/show | 200 |

#### 2.2.3 Anthropic API 兼容

| 用例 ID | 测试用例 | 端点 | 预期状态码 |
|---------|----------|------|------------|
| ANT-001 | 消息创建 | POST /v1/messages | 200 |
| ANT-002 | 流式消息 | POST /v1/messages (stream) | 200 |
| ANT-003 | 模型列表 | GET /v1/models | 200 |

### 2.3 模型加载/卸载测试

| 用例 ID | 测试用例 | 验证方法 | 预期结果 |
|---------|----------|----------|----------|
| ML-001 | 首次模型加载 | 首次请求触发加载 | 加载成功，状态为 LOADED |
| ML-002 | 模型重复加载 | 同一模型请求两次 | 直接使用缓存，无重复加载 |
| ML-003 | 模型卸载 | 调用卸载 API | 模型状态转为 UNLOADED |
| ML-004 | 多模型切换 | 依次请求不同模型 | 按 LRU 策略卸载旧模型 |
| ML-005 | 空闲模型自动卸载 | 等待超时 | 空闲模型被自动卸载 |
| ML-006 | 内存不足时模型加载 | 模拟内存限制 | 卸载低优先级模型后加载 |
| ML-007 | 损坏模型处理 | 加载损坏的模型文件 | 返回错误，不崩溃 |

### 2.4 缓存系统测试

| 用例 ID | 测试用例 | 验证方法 | 预期结果 |
|---------|----------|----------|----------|
| CACHE-001 | 前缀缓存命中 | 连续相同前缀的请求 | 第二次请求显著更快 |
| CACHE-002 | 缓存驱逐策略 | 超过最大条目数 | LRU 驱逐最久未使用项 |
| CACHE-003 | 缓存持久化 | 重启服务 | 缓存数据恢复 |
| CACHE-004 | 缓存内存限制 | 设置内存上限 | 超过阈值时驱逐 |

---

## 3. 代码层测试

### 3.1 单元测试（核心函数）

| 用例 ID | 测试目标 | 输入 | 预期输出 |
|---------|----------|------|----------|
| UNIT-001 | `ModelRegistry.list_models()` | - | 返回模型列表（字典数组） |
| UNIT-002 | `ModelRegistry.get_model_info(name)` | 有效模型名 | 返回模型元信息 |
| UNIT-003 | `ModelRegistry.get_model_info(name)` | 无效模型名 | 抛出 KeyError |
| UNIT-004 | `Config.validate()` | 有效配置 | 返回 True |
| UNIT-005 | `Config.validate()` | 无效配置（如负数超时） | 返回 False |
| UNIT-006 | `ModelManager.load_model(name)` | 有效模型名 | 返回加载结果 |
| UNIT-007 | `ModelManager.unload_model(name)` | 已加载模型 | 卸载成功 |
| UNIT-008 | `Cache.get(key)` | 存在 key | 返回缓存值 |
| UNIT-009 | `Cache.get(key)` | 不存在 key | 返回 None |
| UNIT-010 | `Cache.set(key, value)` | 有效 kv | 设置成功 |

### 3.2 边界条件测试

| 用例 ID | 测试场景 | 边界值 | 预期行为 |
|---------|----------|--------|----------|
| BOUND-001 | 最大 token 超限 | `max_tokens=0` | 正确处理，返回错误 |
| BOUND-002 | 最大 token 超限 | `max_tokens=-1` | 正确处理，返回错误 |
| BOUND-003 | 超长输入 | `messages` 超过 100KB | 返回 413 或截断 |
| BOUND-004 | 空消息列表 | `messages=[]` | 返回 422 |
| BOUND-005 | 温度参数边界 | `temperature=0` | 使用确定性输出 |
| BOUND-006 | 温度参数边界 | `temperature=2.0` | 超过范围，返回错误 |
| BOUND-007 | 超多模型数 | 请求加载超过 MAX_LOADED_MODELS | 按 LRU 策略卸载 |
| BOUND-008 | 超长模型名 | 超过 255 字符 | 返回错误 |
| BOUND-009 | 特殊字符模型名 | 包含 `/` 或 `-` | 正确处理 |

### 3.3 错误处理测试

| 用例 ID | 测试场景 | 预期行为 |
|---------|----------|----------|
| ERR-001 | 模型文件不存在 | 抛出 FileNotFoundError，优雅返回 500 |
| ERR-002 | 模型加载超时 | 返回 504，超时错误信息 |
| ERR-003 | 生成被中断 | 返回部分结果，标记中断 |
| ERR-004 | 内存不足 | 触发 OOM 处理，返回 503 |
| ERR-005 | GPU 不可用 | 降级或返回错误 |
| ERR-006 | 配置文件损坏 | 使用默认值，警告日志 |
| ERR-007 | 缓存写入失败 | 降级运行，日志警告 |

### 3.4 输入验证测试

| 用例 ID | 测试场景 | 预期行为 |
|---------|----------|----------|
| VAL-001 | 无效 JSON | 返回 400 |
| VAL-002 | 缺少 required 字段 | 返回 422 |
| VAL-003 | 字段类型错误 | 返回 422 |
| VAL-004 | 枚举值越界 | 返回 422 |
| VAL-005 | 超长字符串 | 截断或返回错误 |
| VAL-006 | SQL/NoSQL 注入尝试 | 净化处理，不执行 |

---

## 4. 日志层测试

### 4.1 错误日志完整性

| 用例 ID | 测试场景 | 验证点 |
|---------|----------|--------|
| LOG-ERR-001 | API 错误响应 | 错误码、错误信息、时间戳、请求 ID |
| LOG-ERR-002 | 模型加载失败 | 堆栈跟踪、模型名、内存状态 |
| LOG-ERR-003 | 内存警告 | 内存使用量、触发阈值 |
| LOG-ERR-004 | 认证失败 | IP、API Key（脱敏）、时间 |
| LOG-ERR-005 | 异常堆栈 | 完整的 Python 堆栈，无截断 |

### 4.2 访问日志正确性

| 用例 ID | 验证点 |
|---------|--------|
| LOG-ACCESS-001 | 每个请求有唯一 request_id |
| LOG-ACCESS-002 | 包含请求方法、路径、状态码 |
| LOG-ACCESS-003 | 包含响应时间 (ms) |
| LOG-ACCESS-004 | 包含客户端 IP |
| LOG-ACCESS-005 | 流式响应正确记录结束时间 |
| LOG-ACCESS-006 | 认证请求记录 API Key 前缀 |

### 4.3 日志轮转测试

| 用例 ID | 测试场景 | 预期行为 |
|---------|----------|----------|
| LOG-ROT-001 | 日志文件大小达到阈值 | 自动轮转，生成新文件 |
| LOG-RET-002 | 保留天数超过设定 | 删除旧日志文件 |
| LOG-ROT-003 | 服务重启 | 继续写入正确文件 |
| LOG-ROT-004 | 磁盘空间不足 | 警告日志，降级运行 |

---

## 5. 性能层测试

### 5.1 模型切换延迟

| 用例 ID | 测试场景 | 指标 |
|---------|----------|------|
| PERF-001 | 冷启动（首次加载模型） | < 60s (取决于模型大小) |
| PERF-002 | 热切换（已加载模型间切换） | < 5s |
| PERF-003 | 模型预热（首次推理） | < 10s |
| PERF-004 | 前缀缓存命中 | 延迟降低 > 50% |

### 5.2 并发请求处理

| 用例 ID | 测试场景 | 指标 |
|---------|----------|------|
| PERF-010 | 5 并发请求 | 全部成功，无超时 |
| PERF-011 | 10 并发请求 | 全部成功，无超时 |
| PERF-012 | 20 并发请求 | 队列管理正确 |
| PERF-013 | 并发模型切换 | 无竞态条件 |
| PERF-014 | 并发 + 流式 | 流式输出正确 |

### 5.3 内存使用测试

| 用例 ID | 测试场景 | 指标 |
|---------|----------|------|
| PERF-020 | 空闲状态内存 | < 2GB |
| PERF-021 | 单模型加载后 | 符合预期增量 |
| PERF-022 | 双模型加载后 | 符合预期增量 |
| PERF-023 | 内存增长监控 | 无内存泄漏 |
| PERF-024 | 内存限制触发 | 正确卸载模型 |

### 5.4 GPU 资源管理

| 用例 ID | 测试场景 | 指标 |
|---------|----------|------|
| PERF-030 | GPU 内存占用 | 模型大小 + 20% 缓冲 |
| PERF-031 | 多模型 GPU 分配 | 按配置比例分配 |
| PERF-032 | GPU 显存不足处理 | 正确报错，卸载模型 |
| PERF-033 | GPU 利用率监控 | 峰值 > 80% |

---

## 6. 稳定性层测试

### 6.1 长时间运行测试

| 用例 ID | 测试场景 | 时长 |
|---------|----------|------|
| STAB-001 | 持续请求（无模型切换） | 1 小时 |
| STAB-002 | 持续请求（频繁模型切换） | 30 分钟 |
| STAB-003 | 空闲后恢复 | 空闲 30min 后请求 |
| STAB-004 | 跨日运行 | 24+ 小时 |

### 6.2 崩溃恢复测试

| 用例 ID | 测试场景 | 预期行为 |
|---------|----------|----------|
| STAB-010 | 服务异常退出 | 重启后正常加载 |
| STAB-011 | 强制 kill -9 | 再次启动无状态损坏 |
| STAB-012 | OOM 触发 | 服务恢复后正常 |
| STAB-013 | 模型加载中断 | 干净回滚，无僵尸进程 |
| STAB-014 | 网络中断恢复 | 重试成功 |

### 6.3 资源耗尽测试

| 用例 ID | 测试场景 | 预期行为 |
|---------|----------|----------|
| STAB-020 | 内存耗尽 | 卸载模型，返回 503 |
| STAB-021 | 磁盘空间耗尽 | 警告，缓存降级 |
| STAB-022 | 文件句柄耗尽 | 正确报错 |
| STAB-023 | 模型文件损坏 | 跳过，加载其他模型 |

---

## 7. 安全层测试

### 7.1 认证测试

| 用例 ID | 测试场景 | 预期行为 |
|---------|----------|----------|
| SEC-001 | 无 API Key | 返回 401 |
| SEC-002 | 无效 API Key | 返回 401 |
| SEC-003 | 过期 API Key | 返回 401 |
| SEC-004 | 有效 API Key | 正常访问 |
| SEC-005 | 缺失 Bearer 前缀 | 返回 401 |
| SEC-006 | 管理员 Key 权限 | 可访问管理端点 |

### 7.2 CORS 测试

| 用例 ID | 测试场景 | 预期行为 |
|---------|----------|----------|
| CORS-001 | 同源请求 | 正常响应 |
| CORS-002 | 跨域请求（允许的域） | 返回正确 CORS 头 |
| CORS-003 | 跨域请求（不允许的域） | 拒绝请求 |
| CORS-004 | 预检请求 (OPTIONS) | 返回正确 CORS 头 |
| CORS-005 | 带凭证的跨域 | 正确处理 |

### 7.3 输入注入测试

| 用例 ID | 测试场景 | 预期行为 |
|---------|----------|----------|
| INJ-001 | Prompt 注入尝试 | 净化处理，不执行 |
| INJ-002 | JSON 注入 | 解析失败，返回错误 |
| INJ-003 | 路径遍历尝试 | 拒绝访问 |
| INJ-004 | 命令注入尝试 | 净化处理 |
| INJ-005 | XSS 尝试（通过 messages） | 净化处理 |

---

## 8. 测试脚本框架设计

### 8.1 目录结构

```
tests/
├── conftest.py              # pytest 配置与 fixtures
├── requirements-test.txt    # 测试依赖
├── unit/                    # 单元测试
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_cache.py
│   └── test_utils.py
├── integration/             # 集成测试
│   ├── __init__.py
│   ├── test_openai_api.py
│   ├── test_ollama_api.py
│   ├── test_anthropic_api.py
│   └── test_model_lifecycle.py
├── performance/             # 性能测试
│   ├── __init__.py
│   ├── test_latency.py
│   ├── test_concurrency.py
│   ├── test_memory.py
│   └── test_gpu.py
├── stability/               # 稳定性测试
│   ├── __init__.py
│   ├── test_long_running.py
│   ├── test_crash_recovery.py
│   └── test_resource_exhaustion.py
├── security/                # 安全测试
│   ├── __init__.py
│   ├── test_auth.py
│   ├── test_cors.py
│   └── test_injection.py
├── logs/                    # 日志测试
│   ├── __init__.py
│   ├── test_error_logging.py
│   ├── test_access_logging.py
│   └── test_log_rotation.py
├── framework/               # 测试框架基础设施
│   ├── __init__.py
│   ├── client.py            # HTTP 客户端封装
│   ├── fixtures.py          # 共享 fixtures
│   ├── decorators.py        # 测试装饰器
│   ├── reporters.py         # 报告生成
│   └── constants.py         # 常量定义
└── scripts/                 # 执行脚本
    ├── run_all.py           # 运行所有测试
    ├── run_by_tag.py        # 按标签运行
    ├── run_daily.py         # 每日测试
    └── generate_report.py   # 生成报告
```

### 8.2 核心框架设计

```python
# tests/framework/client.py
"""API 客户端封装"""
import aiohttp
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class APIClient:
    """统一 API 客户端"""
    base_url: str
    api_key: str
    timeout: int = 300
    
    async def request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        url = f"{self.base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method, url, json=json, params=params,
                headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if stream:
                    return {"status": resp.status, "stream": resp.content}
                return {"status": resp.status, "data": await resp.json()}
    
    # 便捷方法
    async def chat(self, model: str, messages: list, **kwargs):
        return await self.request("POST", "/v1/chat/completions", 
                                   json={"model": model, "messages": messages, **kwargs})
    
    async def list_models(self):
        return await self.request("GET", "/v1/models")
    
    async def health(self):
        return await self.request("GET", "/health")
```

```python
# tests/framework/fixtures.py
"""共享 fixtures"""
import pytest
import asyncio
from typing import Generator
from tests.framework.client import APIClient


@pytest.fixture(scope="session")
def event_loop():
    """事件循环 fixture"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def api_client() -> APIClient:
    """API 客户端 fixture"""
    import os
    return APIClient(
        base_url=os.getenv("BASE_URL", "http://localhost:11434"),
        api_key=os.getenv("API_KEY", "sk-test-key")
    )


@pytest.fixture(scope="function")
async def clean_cache(api_client: APIClient):
    """清理缓存 fixture"""
    yield
    # 测试后清理


@pytest.fixture(scope="session")
def test_models():
    """测试用模型列表"""
    return ["qwen3.5-122b", "qwen2.5-7b"]


@pytest.fixture
def mock_model_manager():
    """模拟模型管理器（用于单元测试）"""
    from unittest.mock import MagicMock
    return MagicMock()
```

```python
# tests/framework/decorators.py
"""测试装饰器"""
import pytest
import asyncio
import time
from functools import wraps
from typing import Callable


def async_test(func: Callable) -> Callable:
    """异步测试装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


def timed_test(threshold_ms: float = 1000):
    """性能阈值装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = asyncio.run(func(*args, **kwargs))
            elapsed_ms = (time.time() - start) * 1000
            assert elapsed_ms < threshold_ms, f"耗时 {elapsed_ms}ms 超过阈值 {threshold_ms}ms"
            return result
        return wrapper
    return decorator


def retry_on_fail(max_retries: int = 3, delay: float = 1.0):
    """失败重试装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return asyncio.run(func(*args, **kwargs))
                except Exception as e:
                    if i == max_retries - 1:
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
```

```python
# tests/framework/reporters.py
"""测试报告生成"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class TestReporter:
    """测试报告生成器"""
    
    def __init__(self, output_dir: str = "tests/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
    
    def add_result(self, test_name: str, status: str, duration: float, 
                   message: str = "", details: Dict = None):
        """添加测试结果"""
        self.results.append({
            "test_name": test_name,
            "status": status,  # pass, fail, skip
            "duration_ms": duration * 1000,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def generate_report(self, format: str = "markdown") -> str:
        """生成报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            report_path = self.output_dir / f"report_{timestamp}.json"
            with open(report_path, "w") as f:
                json.dump({
                    "generated_at": timestamp,
                    "total": len(self.results),
                    "passed": sum(1 for r in self.results if r["status"] == "pass"),
                    "failed": sum(1 for r in self.results if r["status"] == "fail"),
                    "skipped": sum(1 for r in self.results if r["status"] == "skip"),
                    "results": self.results
                }, f, indent=2)
            return str(report_path)
        
        # Markdown 格式
        report_path = self.output_dir / f"report_{timestamp}.md"
        passed = sum(1 for r in self.results if r["status"] == "pass")
        failed = sum(1 for r in self.results if r["status"] == "fail")
        
        content = f"""# 测试报告

生成时间: {timestamp}
总测试数: {len(self.results)}
通过: {passed} ✅ | 失败: {failed} ❌ | 跳过: {len(self.results) - passed - failed} ⏭️

## 详细结果

| 测试用例 | 状态 | 耗时 (ms) | 说明 |
|----------|------|-----------|------|
"""
        for r in self.results:
            status_icon = {"pass": "✅", "fail": "❌", "skip": "⏭️"}[r["status"]]
            content += f"| {r['test_name']} | {status_icon} {r['status']} | {r['duration_ms']:.2f} | {r['message']} |\n"
        
        with open(report_path, "w") as f:
            f.write(content)
        
        return str(report_path)
```

### 8.3 测试用例示例

```python
# tests/integration/test_openai_api.py
"""OpenAI API 集成测试"""
import pytest
import asyncio
from tests.framework.client import APIClient
from tests.framework.decorators import async_test, timed_test


class TestOpenAIAPI:
    """OpenAI API 兼容性测试"""
    
    @pytest.mark.asyncio
    @async_test
    async def test_health(self, api_client: APIClient):
        """OAI-001: 健康检查"""
        result = await api_client.health()
        assert result["status"] == 200
    
    @pytest.mark.asyncio
    async def test_list_models(self, api_client: APIClient):
        """OAI-002: 模型列表"""
        result = await api_client.request("GET", "/v1/models")
        assert result["status"] == 200
        assert "data" in result["data"]
    
    @pytest.mark.asyncio
    @timed_test(threshold_ms=10000)
    async def test_chat_completions(self, api_client: APIClient):
        """OAI-003: 聊天补全"""
        result = await api_client.chat(
            model="qwen3.5-122b",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=20
        )
        assert result["status"] == 200
        assert "choices" in result["data"]
    
    @pytest.mark.asyncio
    async def test_invalid_model(self, api_client: APIClient):
        """OAI-007: 无效模型请求"""
        result = await api_client.chat(
            model="nonexistent-model-12345",
            messages=[{"role": "user", "content": "test"}]
        )
        assert result["status"] == 404
```

```python
# tests/security/test_auth.py
"""认证测试"""
import pytest
from tests.framework.client import APIClient


class TestAuthentication:
    """认证安全测试"""
    
    @pytest.mark.asyncio
    async def test_no_api_key(self):
        """SEC-001: 无 API Key"""
        client = APIClient(base_url="http://localhost:11434", api_key="")
        result = await client.list_models()
        assert result["status"] == 401
    
    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """SEC-002: 无效 API Key"""
        client = APIClient(base_url="http://localhost:11434", api_key="invalid-key")
        result = await client.list_models()
        assert result["status"] == 401
    
    @pytest.mark.asyncio
    async def test_valid_api_key(self, api_client: APIClient):
        """SEC-004: 有效 API Key"""
        result = await api_client.list_models()
        assert result["status"] == 200
```

---

## 9. 每日测试执行计划

### 9.1 测试执行时间表

| 时间 | 测试类型 | 执行内容 | 时长 |
|------|----------|----------|------|
| 08:00 | 冒烟测试 | 核心 API 健康检查 | 5 min |
| 09:00 | 单元测试 | 代码层测试 | 10 min |
| 10:00 | 集成测试 | API 兼容性 | 15 min |
| 12:00 | 性能测试 | 延迟/并发 | 20 min |
| 14:00 | 安全测试 | 认证/CORS | 10 min |
| 16:00 | 稳定性 | 长时间运行（持续） | - |
| 18:00 | 日报告 | 汇总报告生成 | 5 min |

### 9.2 测试执行命令

```bash
# 冒烟测试 (快速验证)
pytest tests/integration/test_openai_api.py::TestOpenAIAPI::test_health -v

# 完整单元测试
pytest tests/unit/ -v --tb=short

# 按标签运行
pytest tests/ -m "not slow" -v
pytest tests/ -m "security" -v

# 性能测试
pytest tests/performance/ -v --benchmark-only

# 稳定性测试 (需要长时间)
pytest tests/stability/test_long_running.py -v -s

# 生成报告
pytest tests/ --html=tests/reports/report.html --self-contained-html

# 完整测试套件
python tests/scripts/run_all.py
```

### 9.3 CI/CD 集成

```yaml
# .github/workflows/test.yml (示例)
name: Test Suite

on:
  push:
    branches: [main, develop]
  schedule:
    - cron: '0 8 * * *'  # 每日 8:00

jobs:
  test:
    runs-on: macos
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Start service
        run: |
          source venv/bin/activate
          ./start.sh &
          sleep 30
      
      - name: Run tests
        run: |
          pytest tests/ -v --tb=short --junit-xml=tests/reports/junit.xml
      
      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: test-reports
          path: tests/reports/
```

### 9.4 测试优先级矩阵

| 优先级 | 维度 | 测试用例数 | 执行频率 |
|--------|------|------------|----------|
| P0 | 架构层 - API 兼容 | 20+ | 每次提交 |
| P0 | 安全层 - 认证 | 10+ | 每次提交 |
| P0 | 代码层 - 单元测试 | 30+ | 每次提交 |
| P1 | 架构层 - 模型生命周期 | 15+ | 每日 |
| P1 | 性能层 - 延迟/并发 | 15+ | 每日 |
| P1 | 日志层 - 错误日志 | 10+ | 每日 |
| P2 | 稳定性层 - 长时间运行 | 10+ | 每周 |

---

## 附录

### A. 测试环境要求

| 资源 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 内存 | 32GB | 64GB+ |
| GPU | 24GB | 48GB+ (M3 Max) |
| 磁盘 | 100GB | 500GB+ |
| Python | 3.10+ | 3.11+ |

### B. 依赖包

```
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-html>=3.2.0
pytest-cov>=4.1.0
aiohttp>=3.8.0
psutil>=5.9.0
```

### C. 维护计划

- **周维护**: 更新测试用例，覆盖新功能
- **月维护**: 审查测试覆盖率，调整阈值
- **季度维护**: 全面重构测试框架

---

_质量是设计出来的，测试是质量的守门人。_ 🛡️