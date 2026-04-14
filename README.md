# CodeSage

CodeSage 是一个面向研发场景的多智能体代码助手，围绕代码审查、代码库问答、文档检索、会话记忆和受控代码修改构建统一工作流。项目使用 FastAPI 提供服务接口，使用 LangGraph / LangChain 编排 Agent 流程，使用 Milvus 承担向量检索与长期记忆能力，通过检索降低大模型在代码修改和PR审查时的幻觉。

它适合这几类场景：

- 审查 GitHub Pull Request 或直接分析 diff
- 为本地仓库建立索引后进行实现定位、调用链追踪和架构问答
- 上传产品文档、接口文档后统一检索
- 用自然语言触发代码修改，并对高风险变更先预览、后确认

## 核心能力

- 智能路由：Supervisor Agent 会将请求路由到 `review`、`rag`、`modify`、`index` 或 `none`
- PR 审查：支持 GitHub Webhook、PR 链接和原始 diff 文本，输出结构化审查意见
- 增强问答：Enhanced RAG Agent 支持 Self-RAG、查询改写、step-back 扩展、重排、证据摘要和 grounding 校验
- 代码修改：当前版本提供受控修改入口、预览区与确认链路，面向 `collect_context -> analyze -> plan -> execute -> verify -> output` 工作流持续演进
- 安全确认：高风险修改不会直接落盘，而是先写入预览区，等待 `/modify/confirm` 确认
- 会话记忆：支持短期上下文窗口和基于 Milvus 的长期记忆注入
- Skill 系统：支持项目级和用户级 Skill，既可显式指定，也可自动命中
- 运行观测：支持查看运行记录、事件流、工具调用和活跃任务
- 对话取消：支持取消正在执行的流式对话请求

## 新增功能

- `/chat/cancel`：取消正在执行的聊天请求
- `/runs`、`/runs/active`、`/runs/{run_id}`、`/runs/{run_id}/events`、`/runs/{run_id}/tool_calls`：运行观测接口
- Fork 子任务执行与父子 run 关系追踪
- Skill 自动选择与显式 `/skill:<name> <task>` 调用
- grounding 校验、unsupported claims、missing evidence 等增强 RAG 能力
- Embedding 后端切换，支持 `hash`、`defaultembeddingfunction`、`bgem3`

## 架构概览

### 核心组件

- `codesage/api/main.py`：FastAPI 入口，暴露 Web UI、REST API 和 SSE 对话流
- `codesage/agents/routing/supervisor_agent.py`：统一路由层，决定请求走审查、问答、改码还是索引帮助
- `codesage/agents/review/pr_agent.py`：PR 审查编排器，串联审查相关子能力
- `codesage/agents/rag/enhanced_rag_agent.py`：增强检索问答链路
- `codesage/agents/modify/code_modify_agent.py`：受控代码修改工作流与预览确认机制
- `codesage/indexing/ingestion.py`：本地仓库索引入库
- `codesage/document_processor/`：PDF、Word、PPT、Excel、HTML、CSV 等文档处理与入库
- `codesage/memory/service.py`：会话记忆、上下文策略与长期记忆抽取
- `codesage/skills/discovery.py`：Skill 发现、选择、加载和渲染入口
- `codesage/core/observability.py`：运行状态、事件流和工具调用观测

### 技术栈

| 类别 | 技术 |
| --- | --- |
| Web 框架 | FastAPI + Uvicorn |
| Agent 编排 | LangGraph + LangChain |
| LLM 接入 | OpenAI 兼容接口 |
| 向量存储 | Milvus |
| Embedding | 内置 `hash`，可切换 `defaultembeddingfunction` / `bgem3` |
| 前端 | 模板页 + Vue 3 + SSE |
| 文档解析 | `pypdf`、`python-docx`、`python-pptx`、`openpyxl`、`pandas`、`beautifulsoup4` |

## 目录结构

```text
codesage/
├── agents/
│   ├── fork/
│   ├── framework/
│   ├── modify/
│   ├── rag/
│   ├── review/
│   ├── routing/
│   └── runtimes/
├── api/
│   ├── main.py
│   └── templates/
├── core/
├── document_processor/
├── evals/
├── indexing/
├── memory/
├── skills/
├── tools/
└── __init__.py

pyproject.toml                # Python 包定义
.env.example                  # 环境变量模板
```

## 快速开始

### 1. 环境要求

- Python 3.10 及以上
- 一个可用的 OpenAI 兼容模型接口
- Milvus

说明：

- 没有 Milvus 时，`/health` 仍可用，但索引、RAG、文档检索和长期记忆相关能力会受限
- 没有 `LLM_API_KEY` 或 `LLM_MODEL` 时，问答、审查和代码修改能力无法正常工作

### 2. 安装依赖

项目使用 `pyproject.toml` 管理依赖，推荐直接安装为可编辑模式。

Windows PowerShell:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .[dev]
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

如果你只需要运行而不需要开发依赖：

```bash
pip install -e .
```

### 3. 配置环境变量

将 `.env.example` 复制为 `.env`：

```powershell
Copy-Item .env.example .env
```

Linux / macOS:

```bash
cp .env.example .env
```

关键配置项如下：

| 变量 | 作用 | 是否必需 |
| --- | --- | --- |
| `LLM_API_KEY` | LLM API Key | 是 |
| `LLM_BASE_URL` | OpenAI 兼容接口地址 | 否 |
| `LLM_MODEL` | 对话模型名称 | 是 |
| `GITHUB_TOKEN` | GitHub PR 拉取 diff 与回写评论 | 审查 GitHub PR 时需要 |
| `MILVUS_HOST` / `MILVUS_PORT` | Milvus 连接配置 | 检索与索引时需要 |
| `COLLECTION_NAME` | 代码索引集合名 | 否 |
| `DOCS_COLLECTION_NAME` | 文档索引集合名 | 否 |
| `MEMORY_COLLECTION_NAME` | 长期记忆集合名 | 否 |
| `EMBEDDING_BACKEND` | `hash`、`defaultembeddingfunction` 或 `bgem3` | 否 |
| `MEMORY_ENABLED` | 是否启用记忆 | 否 |

说明：为了兼容旧配置，仍支持 `MINIMAX_API_KEY`、`MINIMAX_CHAT_BASE_URL`、`MINIMAX_CHAT_MODEL` 这组历史别名。

默认 `.env.example`：

```bash
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=

GITHUB_TOKEN=
GITHUB_WEBHOOK_SECRET=
GITHUB_HTTP_TIMEOUT_SECONDS=10.0
PR_REVIEW_MAX_DIFF_BYTES=262144

MILVUS_HOST=localhost
MILVUS_PORT=19530
COLLECTION_NAME=codesage
DOCS_COLLECTION_NAME=codesage_documents
MEMORY_COLLECTION_NAME=codesage_memory

EMBEDDING_BACKEND=hash
EMBEDDING_DIM=512
BGEM3_MODEL_NAME=BAAI/bge-m3
BGEM3_DEVICE=cpu
BGEM3_BATCH_SIZE=16
BGEM3_USE_FP16=false

MEMORY_ENABLED=true
MEMORY_SHORT_WINDOW_TURNS=8
MEMORY_SHORT_TTL_MINUTES=120
MEMORY_LONG_TOP_K=3
MEMORY_WRITE_MIN_CONFIDENCE=0.75

CHAT_TIMEOUT_SECONDS=45
ASK_TIMEOUT_SECONDS=30
```

### 4. 启动服务

开发模式：

```bash
uvicorn codesage.api.main:app --reload --port 8000
```

直接运行模块：

```bash
python -m codesage.api.main
```

安装为命令后也可以直接执行：

```bash
codesage
```

启动后可访问：

- 首页：http://localhost:8000/
- OpenAPI 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health
- 就绪检查：http://localhost:8000/ready

## 推荐使用路径

如果你第一次接触这个项目，最省事的流程是：

1. 配好 `.env` 并启动 Milvus
2. 启动服务
3. 调用 `/index` 为本地仓库建索引，或调用 `/index_docs` 上传文档
4. 通过 `/chat` 直接提问、审查或发起修改
5. 如遇高风险修改，使用 `/modify/confirm` 决定是否落盘
6. 如需排查执行过程，可通过 `/runs*` 系列接口查看运行详情

## API 概览

### 系统接口

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `/` | GET | 前端页面 |
| `/health` | GET | 存活探针 |
| `/ready` | GET | 依赖就绪状态 |

### 审查接口

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `/webhook` | POST | GitHub PR Webhook 入口，仅处理 `opened` / `synchronize` |
| `/review` | POST | 直接提交 diff 文本进行审查 |

### 检索与索引

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `/ask` | POST | 轻量代码问答接口，基于代码索引返回答案 |
| `/index` | POST | 为本地仓库建立代码索引 |
| `/index_docs` | POST | 上传文档并写入文档索引 |

### 对话与改码

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `/chat` | POST | 统一多智能体入口，返回 `text/event-stream` |
| `/chat/cancel` | POST | 取消正在执行的聊天请求 |
| `/modify` | POST | 直接调用代码修改 Agent |
| `/modify/confirm` | POST | 确认或取消高风险修改预览 |

### 运行观测

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `/runs` | GET | 列出已观测的 Agent 运行记录 |
| `/runs/active` | GET | 列出活跃运行记录 |
| `/runs/{run_id}` | GET | 获取运行详情 |
| `/runs/{run_id}/events` | GET | 获取运行事件 |
| `/runs/{run_id}/tool_calls` | GET | 获取运行工具调用 |

## 使用示例

### 1. 为本地仓库建立索引

`/index` 当前只支持本地路径，不支持直接传 GitHub URL。

```powershell
curl.exe -X POST http://localhost:8000/index `
  -H "Content-Type: application/json" `
  -d '{"repo_path":"D:/projects/demo-repo"}'
```

### 2. 轻量代码问答

`/ask` 会查询代码索引并返回简短答案与来源文件。

```powershell
curl.exe -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"build_readiness_report 在哪里定义？"}'
```

### 3. 上传文档入库

支持的扩展名包括 `pdf`、`txt`、`md`、`docx`、`xlsx`、`xls`、`pptx`、`html`、`csv`。

```powershell
curl.exe -X POST http://localhost:8000/index_docs `
  -F "file=@D:/docs/spec.pdf"
```

### 4. 使用统一对话入口

`/chat` 是最推荐的入口。它会自动决定这是问答、审查、改码还是索引帮助请求。

```powershell
curl.exe -N -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -d '{"message":"SupervisorAgent 的路由逻辑在哪里定义？","thread_id":"demo-thread"}'
```

`/chat` 会返回 SSE 事件流，常见事件包括：

- `step`：路由、上下文、Skill、执行阶段等进度事件
- `message`：最终助手文本
- `confirmation_required`：高风险修改进入确认态
- `error`：执行失败或超时
- `done`：本轮流结束

### 5. 取消正在运行的对话

```powershell
curl.exe -X POST http://localhost:8000/chat/cancel `
  -H "Content-Type: application/json" `
  -d '{"run_id":"<run-id>"}'
```

### 6. 直接提交 diff 进行审查

```powershell
curl.exe -X POST http://localhost:8000/review `
  -H "Content-Type: application/json" `
  -d '{"repo":"owner/repo","pr_number":123,"diff_text":"--- a/file.py\n+++ b/file.py\n@@ -1,2 +1,3 @@\n+print(\"hello\")"}'
```

### 7. 发起代码修改

`approval_mode` 支持：

- `off`：直接返回结果
- `high_risk`：仅高风险修改需要确认
- `always`：所有修改都走确认

```powershell
curl.exe -X POST http://localhost:8000/modify `
  -H "Content-Type: application/json" `
  -d '{"instruction":"重构 supervisor 路由逻辑并保持现有接口不变","working_dir":".","approval_mode":"high_risk"}'
```

如果返回 `awaiting_confirmation`，再调用：

```powershell
curl.exe -X POST http://localhost:8000/modify/confirm `
  -H "Content-Type: application/json" `
  -d '{"preview_id":"<preview-id>","decision":"approve"}'
```

## Skill 系统

当前 Skill 主要服务于 `rag` 和 `modify` 路由。

- 项目级 Skill 目录：`.agents/skills/<skill-name>/SKILL.md`
- 用户级 Skill 目录：`~/.agents/skills/<skill-name>/SKILL.md`
- 同名时项目级 Skill 会覆盖用户级 Skill
- 显式调用格式：`/skill:<skill-name> 你的请求`

示例：

```text
/skill:python-refactor 帮我重构认证模块并保持现有行为不变
```

## 常见问题

### `/ready` 返回 503

优先检查这几项：

- `LLM_API_KEY` 和 `LLM_MODEL` 是否已配置
- Milvus 是否已启动且 `MILVUS_HOST` / `MILVUS_PORT` 正确
- `langchain-openai` 是否已随依赖正确安装
- embedding 后端是否能正常初始化

### `/index` 提示不支持 GitHub URL

当前实现只接受本地路径。请先将仓库克隆到本地，再把本地目录传给 `/index`。

### `/ask` 与 `/chat` 的区别

- `/ask`：轻量、直接，默认查询代码索引
- `/chat`：统一入口，包含路由、记忆、Skill、增强 RAG 和流式事件

## License

MIT
