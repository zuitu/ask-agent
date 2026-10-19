"""向量库工厂：配置驱动，切换 Chroma / pgvector 只改配置，业务代码零改动。

Embedding 提供商由 settings.llm_provider 驱动（ollama 本地 / openai 云端）。
"""
from src.config.settings import settings


def _build_embeddings():
    """根据配置构建 embedding 模型。"""
    if settings.llm_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
    if settings.llm_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
    raise ValueError(f"未知的 LLM 提供商: {settings.llm_provider}")


def get_vector_store():
    """根据配置返回向量库实例。

    - chroma: 嵌入式向量库，本地 Demo 首选
    - pgvector: 复用已有 PostgreSQL，中小规模生产首选
    - milvus / weaviate: 大规模生产可在此扩展
    """
    store_type = settings.vector_store_type
    embeddings = _build_embeddings()

    if store_type == "chroma":
        # langchain-community 已 sunset，Chroma 使用官方独立包 langchain-chroma
        from langchain_chroma import Chroma

        store = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db",
        )
        return store, embeddings

    if store_type == "pgvector":
        # 官方独立包 langchain-postgres（需安装 psycopg）
        from langchain_postgres import PGVector

        store = PGVector(
            embeddings=embeddings,
            connection=settings.pgvector_conn_str,
            collection_name="company_policy",
        )
        return store, embeddings

    raise ValueError(f"未知的向量库类型: {store_type}")
