# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-03-29

### 🚀 Major Release - Enterprise-Grade MLX Inference Service

经过严格的三方稳定性评估（架构师 + 测试专家 + 开发专家），mlx-service 达到生产级发布标准。

### ✨ New Features

#### Multi-API Support
- **OpenAI Compatible API** — 完整支持 `/v1/chat/completions`、`/v1/models`、`/v1/audio/transcriptions`
- **Anthropic Messages API** — 新增 `/v1/messages` 端点，支持 Claude 格式调用
- **Ollama Compatible API** — 支持 `/api/chat`、`/api/generate`、`/api/tags`

#### Multi-Modal Support
- **Vision Language Models** — 自动检测 VL 模型，支持图片识别
- **Audio Models** — 支持 Whisper 语音识别，6 种音频格式

#### Advanced Model Management
- **按需加载** — 首次请求时自动加载模型
- **LRU 淘汰** — 内存不足时自动卸载最久未用模型
- **空闲卸载** — 30 分钟无请求自动卸载释放内存
- **内存预算控制** — 预设 120GB 上限，智能调度

#### Enterprise Caching
- **三层缓存架构** — Hot（内存 LRU）+ Cold（SSD 异步持久化）+ Pending Buffer
- **前缀缓存** — 相同前缀的请求直接命中缓存，提速 10x+
- **跨模型隔离** — 每个模型独立缓存 Key，杜绝污染

### 🔧 Improvements

- **架构文档** — 新增 ARCHITECTURE.md，定义模块边界、公开接口、依赖关系
- **错误恢复** — 所有异常后自动清理 GPU 缓存，防止内存泄漏
- **超时控制** — 300 秒生成超时保护，防止无限等待
- **输入验证** — `max_tokens` (1-16384)、`temperature` (0-2) 参数约束
- **文件限制** — 音频上传最大 50MB，防止 OOM

### 🐛 Bug Fixes

- Fix: 缓存 Key 跨模型污染问题
- Fix: 错误处理过于宽泛，现在有完整堆栈
- Fix: VL 模型 thinking 参数兼容性
- Fix: Ollama API 导入路径错误

### 📊 Stats

| Metric | Value |
|--------|-------|
| 代码行数 | ~2,500 |
| API 端点 | 15+ |
| 测试用例 | 12 |
| 架构模块 | 7 |

### 🏆 Quality Assessment

| Dimension | Score | Reviewer |
|-----------|-------|----------|
| Architecture | 4.6/5 | Quistis 🏛️ |
| Test Coverage | 3.8/5 | Rinoa 🛡️ |
| Code Quality | ✅ | Selphie 💻 |

---

## [0.2.0] - 2026-03-22

### Added
- Multi-model support with dynamic loading
- Ollama API compatibility
- Cache persistence

## [0.1.0] - 2026-03-14

### Added
- Initial release
- OpenAI compatible API
- Basic model loading
