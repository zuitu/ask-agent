"""FastAPI 入口。

启动：
    uvicorn src.api.main:app --reload --port 8000
"""
from fastapi import FastAPI

from src.api.router import router

app = FastAPI(
    title="Agent 工程化 Demo",
    description="LangGraph + RAG + 工具调用 的企业级 Agent 服务",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
