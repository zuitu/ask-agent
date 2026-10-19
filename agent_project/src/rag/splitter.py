"""文本切分器：长文档切成小块再向量化，保证检索命中粒度。"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )


def split_text(text: str) -> list[str]:
    """把原始文档切分为 chunk 列表。"""
    return get_splitter().split_text(text)
