"""Deep Research 智能体的图状态定义与数据结构。"""

import operator
from typing import Annotated, Optional, List

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# 结构化输出模型 (Structured Outputs)
###################
class ClarifyWithUser(BaseModel):
    """用于向用户请求澄清调研需求的数据模型。"""

    need_clarification: bool = Field(
        description="是否需要向用户提出澄清问题以明确调研范围。",
    )
    question: str = Field(
        description="向用户提出的澄清问题，用于明确报告范围与具体要求。",
    )
    verification: str = Field(
        description="确认消息，告知用户在提供必要信息后将正式开始调研。",
    )

class ResearchQuestion(BaseModel):
    """用于指导后续调研的具体研究问题与调研简报。"""

    research_brief: str = Field(
        description="用于指导整个调研过程的研究问题与任务简报。",
    )

class ConductResearch(BaseModel):
    """调用此工具以对特定子主题展开深入调研。"""
    research_topic: str = Field(
        description="待调研的具体子主题。必须是单一主题，且需包含高度详尽的描述（至少一段话）。",
    )

class ResearchComplete(BaseModel):
    """调用此工具表示所有调研工作已完成。"""

class Summary(BaseModel):
    """包含核心发现的调研总结。"""

    summary: str
    key_excerpts: str


###################
# 状态定义 (State Definitions)
###################

def override_reducer(current_value, new_value):
    """自定义状态归约器（Reducer），支持列表追加（Add）或直接覆盖（Override）状态值。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)

class AgentInputState(MessagesState):
    """系统外部输入状态，仅包含用户初始输入的对话消息列表（messages）。"""

class AgentState(MessagesState):
    """全局主图状态，包含完整的对话历史及全流程的调研数据。"""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]   # 需求澄清后生成的标准调研提纲/简报需求澄清后生成的标准调研提纲/简报
    raw_notes: Annotated[list[str], override_reducer] = []  # 原始调研记录，包含所有子调研员的原始输出
    notes: Annotated[list[str], override_reducer] = []  # 经过清洗、提炼后的高质量事实笔记
    final_report: str   # 最终交付给用户的 Markdown 长文研报

class SupervisorState(TypedDict):
    """主管智能体（Supervisor）的专用状态，负责管理和派发调研任务。"""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]   # Supervisor 自身的思考链与工具调用记录，与全局用户的 messages 隔离，避免 Supervisor 的内部决策污染外层对话。
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0    # 循环安全锁。记录 Supervisor 已经派发了多少轮调研，达到上限时强制终止，防止死循环耗尽 API 额度。
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """独立子调研员（Researcher）的私有执行状态。"""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """子调研员执行完毕后，回传给主管智能体或全局状态的输出数据结构。"""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []