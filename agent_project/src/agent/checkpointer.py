"""断点存储工厂：配置驱动，开发用内存，生产用 sqlite / redis。"""
from src.config.settings import settings

# 生产可替换为：
# from langgraph.checkpoint.redis import RedisSaver
# from langgraph.checkpoint.postgres import PostgresSaver


def get_checkpointer():
    """根据配置返回 checkpointer 实例。

    - memory: 进程内存储，重启丢失（仅测试）
    - sqlite: 本地文件持久化，适合单机生产
    - redis: 分布式会话持久化，适合多实例生产（需自行安装 langgraph-checkpoint-redis）
    """
    cp_type = settings.checkpointer_type

    if cp_type == "memory":
        # 不传 checkpointer，或使用 InMemorySaver
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()

    if cp_type == "sqlite":
        # 注意：langgraph-checkpoint-sqlite 3.x 中 from_conn_string 返回上下文管理器，
        # 需要自己持有 sqlite3 连接来构造 SqliteSaver（连接保持打开，进程退出自动关闭）
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect("./checkpoints/agent.db", check_same_thread=False)
        return SqliteSaver(conn)

    if cp_type == "redis":
        # 需要安装 langgraph-checkpoint-redis 并配置连接
        raise NotImplementedError(
            "Redis checkpointer 需要安装 langgraph-checkpoint-redis 并配置连接串，"
            "当前示例未内置，请自行接入。"
        )

    raise ValueError(f"未知的 checkpointer 类型: {cp_type}")
