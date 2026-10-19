"""API 请求/响应模型。"""
from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    thread_id: str = Field(..., description="会话ID，用于断点恢复与多轮记忆")
    query: str = Field(..., description="用户问题")


class AgentResponse(BaseModel):
    answer: str = Field(..., description="Agent 最终回答")
    thread_id: str = Field(..., description="回显会话ID")
