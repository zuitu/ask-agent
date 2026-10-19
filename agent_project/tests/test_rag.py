"""RAG 模块测试：验证切分与检索器初始化。"""
import pytest

from src.rag.loader import load_demo_docs
from src.rag.splitter import split_text


def test_demo_docs_split():
    """验证示例文档能正常切分且产生 chunk。"""
    text = load_demo_docs()
    chunks = split_text(text)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c.strip() for c in chunks)


def test_splitter_config_from_settings():
    """验证切分参数来自配置。"""
    from src.config.settings import settings
    from src.rag.splitter import get_splitter

    splitter = get_splitter()
    assert splitter._chunk_size == settings.rag_chunk_size
