"""Agent 路由：接收 HTTP 请求，调用图执行，返回 JSON。

工程要点：
- 限流、鉴权、请求幂等性可在此层实现，不污染 Agent 核心
- 长耗时任务可改为入队（Redis）异步执行，避免 HTTP 超时
"""
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from src.agent.graph_builder import build_agent_graph
from src.api.schema import AgentRequest, AgentResponse

router = APIRouter()

# 图实例全局复用（编译一次，多次调用）
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph


@router.post("/agent/chat", response_model=AgentResponse)
async def agent_chat(req: AgentRequest):
    graph = get_graph()
    config = {"configurable": {"thread_id": req.thread_id}}
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=req.query)]},
            config=config,
        )
    except Exception as e:  # 生产按异常类型细化处理
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {e}") from e

    answer = result["messages"][-1].content
    return AgentResponse(answer=answer, thread_id=req.thread_id)
