"""
极简 Agent Demo（方案A：快速原型）
====================================
技术栈：Python + LangChain 1.x + LangGraph + Chroma + LLM API

功能：
1. 加载本地文档到 Chroma 向量库（RAG 知识库）
2. Agent 可以查知识库（company_policy_search 工具）
3. Agent 可以调用计算器工具（calculator 工具）
4. ReAct 智能体：模型自主判断是查文档、调用工具还是直接回答

【版本说明（重要）】
本代码适配 langchain 1.x（当前环境 langchain==1.4.6 / langgraph==1.2.16）。
langchain 1.x 已移除旧版 AgentExecutor / create_openai_tools_agent，
官方标准做法是 langgraph.prebuilt.create_react_agent。
旧版（langchain 0.2/0.3）的写法与此不同，网上教程请注意版本。

运行方式：
    python simple_agent_demo.py

依赖安装（已按 1.x 锁定版本）：
    pip install -r requirements.txt

环境变量（.env 文件，与脚本同目录）：
    OPENAI_API_KEY=sk-xxx

注意：
- 生产环境严禁直接 eval 表达式（有安全风险），可用 sympy 安全计算
- Chroma 只适合本地 Demo，生产要换成 pgvector / Milvus
"""

from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings  # Ollama 原生接口（官方包）
from langchain_chroma import Chroma  # 独立向量库包（langchain-community 已 sunset，勿用旧导入）
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool, tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent  # langchain 1.x 官方 Agent 入口
from langgraph.checkpoint.memory import InMemorySaver

# ========== 加载环境变量 ==========
# .env 文件写入：
# OPENAI_API_KEY=sk-xxx
load_dotenv()

# ========== 1. 初始化LLM、Embedding（本地 Ollama） ==========
# ChatOllama / OllamaEmbeddings 走 Ollama 原生 API，本地无需任何 key
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
    base_url="http://localhost:11434",
)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)

# ========== 2. 构造测试知识库，存入Chroma ==========
# 模拟私有文档
raw_docs_text = """
公司政策：
1. 员工年假：入职满1年享有5天年假，满3年享有10天年假。
2. 加班：工作日加班1.5倍薪资，周末加班2倍薪资。
3. 报销：差旅费报销单需要在出差结束后7天内提交。
"""

# 文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = text_splitter.split_text(raw_docs_text)

# 初始化 Chroma 向量库（Demo 用内存模式即可，重启重建；生产换 pgvector/Milvus）
vectorstore = Chroma.from_texts(texts=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 把向量检索包装成Agent可用工具
retrieve_policy_tool = create_retriever_tool(
    retriever,
    name="company_policy_search",
    description="用于查询公司人事、年假、加班、报销相关内部政策文档，用户问到公司制度时使用。",
)

# ========== 3. 自定义工具：计算器 ==========
@tool
def calculator(expression: str) -> str:
    """
    数学计算器工具，用来计算加减乘除。
    参数expression：数学表达式字符串，例如 "100 + 2 * 5"
    """
    # 注意：仅Demo使用，生产环境严禁直接eval表达式，有安全风险
    return str(eval(expression))

# 全部工具列表
tools = [retrieve_policy_tool, calculator]

# ========== 4. 构建 Agent（langchain 1.x 标准写法：create_agent） ==========
system_prompt = (
    "你是企业内部助手，可以查询公司政策，也可以做数学计算。"
    "遇到不知道的内容优先查询知识库，不要编造。"
)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),  # 会话级断点，thread_id 隔离多轮对话
)

# ========== 5. 运行Agent ==========
if __name__ == "__main__":
    print("=== 极简Agent启动（langchain 1.x + create_react_agent），输入 exit 退出 ===")
    config = {"configurable": {"thread_id": "demo_001"}}
    while True:
        user_q = input("\n请输入问题：")
        if user_q.strip() == "exit":
            break
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_q)]},
            config=config,
        )
        # 打印完整链路：展示模型思考 + 工具调用过程
        for m in result["messages"]:
            if m.type == "ai" and m.content:
                print(f"🤖 AI: {m.content}")
            elif m.type == "tool":
                print(f"🔧 工具[{m.name}]: {m.content}")
        print("\n✅ 最终回答：", result["messages"][-1].content)
