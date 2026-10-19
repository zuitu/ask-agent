"""知识库检索工具：把 RAG 检索器包装成 Agent 可用工具。

langchain 1.x：create_retriever_tool 位于 langchain_core.tools。
"""
from langchain_core.tools import create_retriever_tool

from src.rag.retriever import get_retriever


def get_policy_retriever_tool():
    """返回公司政策检索工具（RAG 能力以工具形式暴露给 Agent）。"""
    retriever = get_retriever()
    return create_retriever_tool(
        retriever,
        name="company_policy_search",
        description="用于查询公司人事、年假、加班、报销相关内部政策文档，用户问到公司制度时使用。",
    )
