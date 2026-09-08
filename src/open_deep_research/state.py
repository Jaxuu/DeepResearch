"""Deep Research 智能体的图状态定义与数据结构。"""
import operator
from typing import Annotated, Optional, List, Union, Any, Dict, Literal

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
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
    # 技能清单，移除写死的工具枚举，要求依赖 search_tools_catalog 的结果
    required_tools: List[str] = Field(
        description="The specific tools this agent should have access to. For standard web search, use ['web_search', 'fetch_webpage']. For specialized tasks (e.g., RAG queries, DB queries, Git repos, internal docs), you MUST use the exact tool names returned by `search_tools_catalog`.",
        default=["web_search", "fetch_webpage"]
    )
    # 技能分配字段
    required_skills: List[str] = Field(
        description="分配给该研究员的专业技能。目前可用选项: 'quantitative_analysis', 'long_doc_mining', 'data_visualization'",
        default=[])

class ResearchComplete(BaseModel):
    """调用此工具表示所有调研工作已完成。"""

class Fact(BaseModel):
    """从检索内容中提纯的最小知识单元"""
    entity: str = Field(description="事实主体（如：特定设备部件、工艺参数、机构名称）")
    claim: str = Field(description="具体的事实断言、数据或机理描述")
    source: str = Field(description="来源出处（具体的 URL 或本地文档 ID）")

class FactBoard(BaseModel):
    """当前子主题的结构化事实看板"""
    topic: str = Field(default="综合调研", description="当前总结的子主题或查询意图")
    facts: List[Fact] = Field(default_factory=list, description="提取出的高度提纯的结构化事实列表")

def add_facts_reducer(current_facts: List[Fact], new_facts: List[Fact]) -> List[Fact]:
    """事实去重归约器：基于断言内容进行基础去重，防止循环检索造成的事实冗余"""
    if not current_facts:
        return new_facts
    existing_claims = {f.claim for f in current_facts}
    unique_new = [f for f in new_facts if f.claim not in existing_claims]
    return current_facts + unique_new

# -------- 新增：核查器使用的结构化输出模型 --------
class CitationCheckResult(BaseModel):
    """单条断言的核查结果"""
    claim: str = Field(description="从报告中抽取的具体陈述或数据")
    citation_index: Union[int, List[int], None] = Field(default=None,
        description="The index or list of indices of the citation(s) supporting the claim.")
    is_supported: bool = Field(description="该陈述是否完全被 FactBoard 中对应的 Source 或事实完全支持")
    reason: str = Field(description="判定支持或不支持的具体理由，指出是否存在虚假篡改或夸大")

class VerificationReport(BaseModel):
    """对抗核查汇总结果"""
    detailed_checks: List[CitationCheckResult] = Field(description="逐句核查的过程记录。必须提取报告中所有带有引用的断言进行一对一核对。")
    has_hallucinations: bool = Field(description="报告是否存在未被 FactBoard 支持的幻觉或错误引用")
    citation_precision_score: float = Field(description="引用准确度评分，范围 0.0 到 1.0")
    hallucinated_claims: List[str] = Field(default_factory=list, description="被判定位幻觉的具体语句列表")
    feedback: str = Field(description="给重写模型的修改指导意见，清晰指出哪一段需要删除或纠正")

class SectionOutline(BaseModel):
    """大纲的单一章节结构"""
    section_title: str = Field(description="大纲章节标题")
    description: str = Field(description="该章节需要覆盖的核心要点、逻辑描述及预期结论")
    relevant_fact_indices: List[int] = Field(description="分配给该章节的 Fact ID 列表（例如 [0, 3, 12]）。必须覆盖所有提供的事实，不要遗漏。")

class ReportOutline(BaseModel):
    """生成的全局报告大纲"""
    sections: List[SectionOutline] = Field(description="报告的章节列表（按正文逻辑顺序排列）")

class SectionDraft(BaseModel):
    """并发子节点生成的章节草稿"""
    section_title: str = Field(description="章节标题")
    content: str = Field(description="章节内容")

class TaskRouting(BaseModel):
    """决定任务类型的路由模型"""
    task_type: Literal["research", "direct_answer"] = Field(
        description="如果是需要联网查资料的查询，返回'research'。如果是纯逻辑推理、数学计算、概率题等不需要联网即可解答的问题，返回'direct_answer'。"
    )


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
    structured_facts: Annotated[list[Fact], add_facts_reducer] = []  # 经过清洗、提炼后的高质量事实笔记
    consecutive_low_gain_rounds: int = 0

    report_outline: Optional[List[SectionOutline]] = None
    section_drafts: Annotated[list[SectionDraft], override_reducer] = []
    final_report: str   # 最终交付给用户的 Markdown 长文研报

    verification_feedback: Optional[str] = None
    verification_retries: int = 0

class SupervisorState(TypedDict):
    """主管智能体（Supervisor）的专用状态，负责管理和派发调研任务。"""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]   # Supervisor 自身的思考链与工具调用记录，与全局用户的 messages 隔离，避免 Supervisor 的内部决策污染外层对话。
    research_brief: str
    structured_facts: Annotated[list[Fact], add_facts_reducer] = [] # 经过清洗、提炼后的高质量事实笔记
    staged_facts: Annotated[list[Fact], override_reducer] = []  # 暂存最新的事实列表，用来和structured_facts对比实现熔断
    research_iterations: int = 0    # 循环安全锁。记录 Supervisor 已经派发了多少轮调研，达到上限时强制终止，防止死循环耗尽 API 额度。
    consecutive_low_gain_rounds: int = 0

class WriteSectionState(TypedDict):
    """并发写手节点的独立状态"""
    section_title: str
    section_description: str
    assigned_facts: List[Dict[str, Any]]
    research_brief: str

class ResearcherState(TypedDict):
    """独立子调研员（Researcher）的私有执行状态。"""

    researcher_messages: Annotated[list[MessageLikeRepresentation], add_messages]
    research_topic: str
    compressed_research: str
    tool_call_iterations: int = 0
    required_tools: list[str]
    required_skills: list[str]
    tool_call_id: str

class ResearcherOutputState(BaseModel):
    """子调研员执行完毕后，回传给主管智能体或全局状态的输出数据结构。"""

    staged_facts: list[Fact]
    supervisor_messages: list[Any] # 用于将压缩完成的信号转化为 ToolMessage

