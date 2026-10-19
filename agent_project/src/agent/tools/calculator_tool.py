"""计算器工具。独立封装，不依赖 LangGraph，可单独单元测试。

安全提示：生产环境禁止使用 eval 执行用户输入，存在任意代码执行风险。
建议使用 sympy 安全求值：
    from sympy import sympify
    result = sympify(expression).evalf()
"""
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """数学计算器工具，用来计算加减乘除。

    参数 expression：数学表达式字符串，例如 "100 + 2 * 5"
    """
    # 仅 Demo 使用；生产请替换为 sympy 安全求值
    return str(eval(expression))
