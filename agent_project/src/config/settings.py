"""全局配置：从 .env / 环境变量读取，配置驱动组件行为。"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM 提供商：openai | ollama（本地）
    llm_provider: str = "ollama"
    # OpenAI（llm_provider=openai 时使用）
    llm_model: str = "gpt-3.5-turbo"
    openai_api_key: str = "sk-xxx"
    openai_base_url: str = "https://api.openai.com/v1"
    # Ollama 本地（llm_provider=ollama 时使用）
    ollama_base_url: str = "http://localhost:11434"

    # Embedding
    embedding_model: str = "nomic-embed-text"

    # 向量库：chroma | pgvector
    vector_store_type: str = "chroma"
    pgvector_conn_str: str = ""

    # RAG
    rag_top_k: int = 2
    rag_chunk_size: int = 200
    rag_chunk_overlap: int = 20

    # Agent
    agent_temperature: float = 0.0
    agent_max_iterations: int = 10

    # 断点存储：memory | sqlite | redis
    checkpointer_type: str = "sqlite"


settings = Settings()
