"""LLM 思考节点工厂。

节点 = 输入 state、输出 state 更新的函数/闭包。
节点之间不直接互相调用，只通过状态机流转。

LLM 提供商由配置驱动：ollama（本地，默认）或 openai。
"""
from src.config.settings import settings
from src.agent.state import AgentState


def _build_llm(tools):
    """根据 settings.llm_provider 构建绑定工具的大模型。"""
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.llm_model,
            temperature=settings.agent_temperature,
            base_url=settings.ollama_base_url,
        )
    elif settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.agent_temperature,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    else:
        raise ValueError(f"未知的 LLM 提供商: {settings.llm_provider}")

    return llm.bind_tools(tools)


def build_llm_node(tools):
    """构建 LLM 思考节点。

    Args:
        tools: 绑定给 LLM 的工具列表，模型据此决定是否发起工具调用。

    Returns:
        可被 StateGraph.add_node 使用的节点函数。
    """
    llm_with_tools = _build_llm(tools)

    def llm_node(state: AgentState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return llm_node
