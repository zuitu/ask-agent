# Agent 工程化 Demo（LangGraph）

分层解耦、配置驱动、可测试、可独立部署的企业级 Agent 工程骨架。
功能：RAG 知识库检索 + 计算器工具调用 + ReAct 智能体 + 断点持久化。

> **版本说明**：本项目适配 **langchain 1.x**（requirements.txt 已锁定当前环境版本）。
> langchain 1.x 移除了旧版 `AgentExecutor` / `create_openai_tools_agent`，
> 官方标准写法是 `langchain.agents.create_agent`（底层基于 LangGraph）。
> 网上的旧教程（0.2/0.3 时代）写法与此不同，请注意甄别。

## 目录结构

```
agent_project/
├── .env.example          # 环境变量模板（复制为 .env 使用）
├── requirements.txt      # 依赖
├── cli.py                # 控制台调试入口
├── docker-compose.yml    # pgvector + redis + agent 服务一键拉起
├── src/
│   ├── config/           # 全局配置（pydantic-settings，读 .env）
│   ├── agent/            # Agent 编排层
│   │   ├── state.py      #   全局状态 AgentState
│   │   ├── graph_builder.py  # 图组装工厂（核心）
│   │   ├── checkpointer.py   # 断点存储工厂（memory/sqlite/redis）
│   │   ├── nodes/        #   图节点（llm_node）
│   │   ├── tools/        #   工具（calculator、policy_retriever）
│   │   └── prompts/      #   提示词模板（yaml）
│   ├── rag/              # RAG 模块（与编排解耦）
│   │   ├── loader.py     #   文档加载
│   │   ├── splitter.py   #   文本切分
│   │   ├── vector_store.py  # 向量库工厂（chroma/pgvector 切换）
│   │   └── retriever.py  #   检索器
│   ├── api/              # FastAPI 接入层
│   │   ├── main.py       #   应用入口
│   │   ├── router.py     #   /agent/chat 路由
│   │   └── schema.py     #   请求/响应模型
├── tests/                # 单元测试
└── README.md
```

## 快速开始

```bash
cd agent_project
cp .env.example .env          # 填入 OPENAI_API_KEY
pip install -r requirements.txt

# 方式一：控制台直接跑
python cli.py

# 方式二：HTTP 服务
uvicorn src.api.main:app --reload --port 8000
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "s1", "query": "入职满2年有几天年假？"}'
```

## 测试

```bash
pytest tests/ -v
```

## 配置切换示例

| 配置项 | 值 | 说明 |
|---|---|---|
| `VECTOR_STORE_TYPE` | `chroma` | 本地 Demo（默认） |
| `VECTOR_STORE_TYPE` | `pgvector` | 生产（需配 `PGVECTOR_CONN_STR`） |
| `CHECKPOINTER_TYPE` | `memory` | 进程内，仅测试 |
| `CHECKPOINTER_TYPE` | `sqlite` | 本地文件持久化（默认） |
| `CHECKPOINTER_TYPE` | `redis` | 分布式会话（需自行接入） |

## 工程化要点

1. **关注点分离**：RAG、工具、状态、编排、API 互相独立，改 RAG 不动编排。
2. **可测试**：工具、检索器、节点均可单独单测，无需启动整个 Agent。
3. **配置驱动**：模型、向量库、断点存储由 `.env` 控制，dev/test/prod 隔离。
4. **可替换**：向量库 Chroma → pgvector 只改配置，代码零改动。
5. **生产扩展**：限流/鉴权在 API 层；长任务入队（Redis）异步执行；Langfuse 统一观测。

## 安全提醒

- `calculator` 中的 `eval` 仅 Demo 使用，生产请替换为 `sympy` 安全求值。
- Chroma 只适合 Demo，生产使用 pgvector / Milvus。
