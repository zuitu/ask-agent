"""工具单元测试：不启动整个 Agent，单独验证工具行为。"""
import pytest

from src.agent.tools.calculator_tool import calculator


def test_calculator_add():
    assert calculator.invoke({"expression": "1 + 2"}) == "3"


def test_calculator_complex():
    assert calculator.invoke({"expression": "100 * 2 + 30"}) == "230"
