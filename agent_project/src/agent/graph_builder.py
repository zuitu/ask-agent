"""图组装工厂：读取配置、加载节点与工具、定义分支、编译 LangGraph。

核心思想：编排层只负责定义状态流转；
业务能力（工具、RAG、LLM）全部下沉到独立模块，可单独测试、可替换。
"""
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.nodes.llm_node import build_llm_node
from src.agent.tools.calculator_tool import calculator
from src.agent.tools.policy_retriever import get_policy_retriever_tool
from src.agent.checkpointer import get_checkpointer


def should_continue(state: AgentState):
    """条件分支：模型输出 tool_calls 则进入工具节点，否则结束。"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "tools"
    return END


def build_agent_graph():
    """组装并编译 Agent 状态图。"""
    # 1. 加载所有工具
    policy_tool = get_policy_retriever_tool()
    tools = [calculator, policy_tool]

    # 2. 构建节点
    agent_node = build_llm_node(tools)
    tool_node = ToolNode(tools)

    # 3. 组装图
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    # 4. 断点存储（配置驱动：memory / sqlite / redis）
    checkpointer = get_checkpointer()
    graph = workflow.compile(checkpointer=checkpointer)
    return graph
