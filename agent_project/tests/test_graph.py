"""图组装与分支逻辑测试（不调用真实 LLM，仅验证图结构可编译）。"""
import pytest

from src.agent.graph_builder import build_agent_graph


def test_graph_compiles():
    """验证状态图能成功编译。"""
    graph = build_agent_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    """验证图包含 agent / tools 两个节点。"""
    graph = build_agent_graph()
    nodes = graph.get_graph().nodes
    assert "agent" in nodes
    assert "tools" in nodes
