"""文档加载器：把本地文档读入内存（支持 txt / md / pdf 等，可按需扩展）。"""
from pathlib import Path


def load_documents(doc_path: str) -> str:
    """读取单个文本类文档，返回原始字符串。"""
    p = Path(doc_path)
    if not p.exists():
        raise FileNotFoundError(f"文档不存在: {doc_path}")
    return p.read_text(encoding="utf-8")


def load_demo_docs() -> str:
    """Demo 内置示例文档（真实工程中由知识库文档目录驱动）。"""
    return """
公司政策：
1. 员工年假：入职满1年享有5天年假，满3年享有10天年假。
2. 加班：工作日加班1.5倍薪资，周末加班2倍薪资。
3. 报销：差旅费报销单需要在出差结束后7天内提交。
"""
