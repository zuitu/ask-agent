"""检索器封装：面向业务提供统一 get_retriever 入口。"""
from src.config.settings import settings
from src.rag.vector_store import get_vector_store


def get_retriever():
    """返回语义检索器。

    首次调用时若向量库为空，会自动灌入 Demo 文档（真实工程由离线索引任务负责）。
    """
    store, _ = get_vector_store()

    # 惰性初始化：为空时灌入 Demo 数据
    if store._collection.count() == 0:
        from src.rag.loader import load_demo_docs
        from src.rag.splitter import split_text

        docs = split_text(load_demo_docs())
        store.add_texts(docs)

    return store.as_retriever(search_kwargs={"k": settings.rag_top_k})
