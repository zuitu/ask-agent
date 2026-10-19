"""
LangGraph 版 Demo（等价功能：RAG知识库 + 计算器工具 Agent）
===============================================================
和 simple_agent_demo.py（LangChain 版）功能完全一致：
1. 查询公司政策知识库（Chroma RAG）
2. 数学计算器工具
3. 自主判断是否调用工具 / 直接回答

新增能力（LangGraph 标志性能力）：
- 显式状态 State
- 图节点 Node + 条件分支
- Checkpoint 断点（thread_id 会话隔离、状态保存）

运行方式：
    python langgraph_agent_demo.py

依赖安装：
    pip install langchain langchain-openai langchain-community chroma langgraph python-dotenv

环境变量（.env 文件，与脚本同目录）：
    OPENAI_API_KEY=sk-xxx
    LANGCHAIN_API_KEY=ls-xxx            # 可选，用于 LangSmith 追踪
    LANGCHAIN_TRACING_V2=true           # 可选

注意：
- 生产环境严禁直接 eval 表达式（有安全风险），可用 sympy 安全计算
- Chroma 只适合本地 Demo，生产要换成 pgvector / Milvus
- 内存 Checkpointer 只适合测试；生产使用 sqlite / Redis checkpoint
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_ollama import ChatOllama, OllamaEmbeddings  # Ollama 原生接口（官方包）
from langchain_chroma import Chroma  # 独立向量库包（langchain-community 已 sunset，勿用旧导入）
from langchain_text_splitters import RecursiveCharacterTextSplitter
# langchain 1.x：create_retriever_tool 位于 langchain_core.tools
from langchain_core.tools import create_retriever_tool, tool
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# ====================== 1. LLM、Embedding、知识库、工具 ======================
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
    base_url="http://localhost:11434",
)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)

# 私有文档
raw_docs_text = """
公司政策：
1. 员工年假：入职满1年享有5天年假，满3年享有10天年假。
2. 加班：工作日加班1.5倍薪资，周末加班2倍薪资。
3. 报销：差旅费报销单需要在出差结束后7天内提交。
"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = text_splitter.split_text(raw_docs_text)

# 初始化 Chroma 向量库（Demo 用内存模式即可，重启重建；生产换 pgvector/Milvus）
vectorstore = Chroma.from_texts(texts=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

retrieve_policy_tool = create_retriever_tool(
    retriever,
    name="company_policy_search",
    description="用于查询公司人事、年假、加班、报销相关内部政策文档，用户问到公司制度时使用。",
)


@tool
def calculator(expression: str) -> str:
    """
    数学计算器工具，用来计算加减乘除。
    参数expression：数学表达式字符串，例如 "100 + 2 * 5"
    """
    # 注意：仅Demo使用，生产环境严禁直接eval表达式，有安全风险
    return str(eval(expression))


tools = [retrieve_policy_tool, calculator]
# 把工具绑定到LLM，让模型输出工具调用
llm_with_tools = llm.bind_tools(tools)

# ====================== 2. 【LangGraph核心】定义全局状态 State ======================
# State 保存整个对话所有消息，所有节点共享、修改这个状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# ====================== 3. 定义图节点（Node） ======================
# 节点1：模型思考节点
def agent_node(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 节点2：工具执行节点（LangGraph内置ToolNode，自动执行工具，返回ToolMessage）
tool_node = ToolNode(tools)

# ====================== 4. 条件分支：判断下一步走工具还是结束 ======================
def should_continue(state: AgentState):
    last_msg = state["messages"][-1]
    # 如果模型输出tool_calls，就去调用工具；否则任务结束
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "tools"
    return END

# ====================== 5. 构建状态图 ======================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# 设置边：
workflow.set_entry_point("agent")  # 入口先进入agent思考节点
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)
# 工具执行完，回到agent节点继续思考
workflow.add_edge("tools", "agent")

# 编译图，开启Checkpoint断点（内存存储，重启丢失；生产可换Redis/SQL持久化）
graph = workflow.compile(checkpointer=InMemorySaver())

# ====================== 6. 运行 ======================
if __name__ == "__main__":
    print("=== LangGraph Agent 启动，输入 exit 退出 ===")
    # thread_id 会话ID，用来保存状态、断点恢复
    config = {"configurable": {"thread_id": "demo_001"}}
    while True:
        user_q = input("\n请输入问题：")
        if user_q.strip() == "exit":
            break
        # 输入用户消息，写入状态
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_q)]},
            config=config,
        )
        # 取出最后一条AI回答
        final_answer = result["messages"][-1].content
        print("\n✅ 最终回答：", final_answer)
