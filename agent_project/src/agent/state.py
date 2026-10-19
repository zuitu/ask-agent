"""AgentState：全局状态定义。纯数据结构，无业务逻辑。"""
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """所有图节点共享的状态。

    messages 使用 operator.add 作为 reducer，
    每次节点返回的新消息会追加到已有消息列表，而不是覆盖。
    """

    messages: Annotated[Sequence[BaseMessage], operator.add]
