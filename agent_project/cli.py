"""控制台交互入口：不经过 HTTP，直接调用编译好的图，方便本地调试。

运行：
    python cli.py
"""
from langchain_core.messages import HumanMessage

from src.agent.graph_builder import build_agent_graph


def main():
    graph = build_agent_graph()
    config = {"configurable": {"thread_id": "cli_session"}}
    print("=== 工程化 Agent CLI（LangGraph）启动，输入 exit 退出 ===")
    while True:
        user_q = input("\n请输入问题：")
        if user_q.strip() == "exit":
            break
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_q)]},
            config=config,
        )
        print("\n✅ 最终回答：", result["messages"][-1].content)


if __name__ == "__main__":
    main()
