"""Deep Research 智能体的 LangGraph 主实现。"""

import asyncio
import json
import re
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    RemoveMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
    report_verifier_prompt,
    rewrite_report_prompt,
    generate_outline_prompt,
    write_section_prompt,
    memory_folding_prompt,
    task_routing_prompt,
    direct_answering_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
    FactBoard,
    Fact,
    VerificationReport,
    SectionOutline,
    ReportOutline,
    SectionDraft,
    WriteSectionState,
    TaskRouting,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
    quantitative_analysis_skill,
    long_doc_mining_skill,
    data_visualization_skill,
    get_active_skills,
    search_tools_catalog
)

# 初始化一个可配置的模型，将在整个智能体中使用
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "model_kwargs"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["route_task", "__end__"]]:
    """分析用户消息，如果研究范围不清晰则提出澄清问题。

    此函数判断用户的请求在继续研究之前是否需要澄清。
    如果澄清功能被禁用或不需要，则直接进入研究阶段。

    Args:
        state: 当前智能体状态，包含用户消息
        config: 运行时配置，包含模型设置和偏好

    Returns:
        Command: 要么以澄清问题结束，要么继续生成研究简报
    """
    # 第1步：检查配置中是否启用了澄清功能
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # 跳过澄清步骤，直接进入研究简报生成
        return Command(goto="route_task")

    # 第2步：准备模型进行结构化澄清分析
    messages = state["messages"]
    model_config = {
        "model": configurable.supervisor_model,
        "max_tokens": configurable.supervisor_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.supervisor_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 配置模型：结构化输出 + 重试逻辑
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser,method="json_mode")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    # 第3步：分析是否需要澄清
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    # 第4步：根据澄清分析结果路由
    if response.need_clarification:
        # 以向用户提出的澄清问题结束
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # 继续进入研究阶段，并附带确认消息
        return Command(
            goto="route_task",
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def route_task(state: AgentState, config: RunnableConfig) -> Command[
    Literal["write_research_brief", "direct_answering"]]:
    """智能路由节点：判断是走检索流水线，还是直接走逻辑推理。"""
    configurable = Configuration.from_runnable_config(config)
    model = configurable_model.with_structured_output(TaskRouting, method="json_mode").with_config({
        "model": configurable.supervisor_model,
        "max_tokens": configurable.supervisor_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.supervisor_model, config),
        "tags": ["langsmith:nostream"]
    })

    prompt = task_routing_prompt.format(messages=get_buffer_string(state.get("messages", [])))
    decision = await model.ainvoke([HumanMessage(content=prompt)])

    if decision.task_type == "direct_answer":
        print("\n[🔀 智能路由] 识别为逻辑/计算题，启动思维链短路推理，跳过检索流程。")
        return Command(goto="direct_answering")
    else:
        print("\n[🔀 智能路由] 识别为调研任务，进入多智能体深度检索流水线。")
        return Command(goto="write_research_brief")


async def direct_answering(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """短路推理节点：用于回答纯逻辑和数学题。"""
    configurable = Configuration.from_runnable_config(config)
    model = configurable_model.with_config({
        "model": configurable.logical_reasoning_model,
        "max_tokens": configurable.logical_reasoning_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.logical_reasoning_model, config),
    })

    prompt = direct_answering_prompt.format(problem=get_buffer_string(state.get("messages", [])))

    response = await model.ainvoke([HumanMessage(content=prompt)])

    return Command(
        goto=END,
        update={
            "final_report": response.content,
            "messages": [response]
        }
    )

async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """将用户消息转换为结构化的研究概要，并初始化主管智能体。

    此函数分析用户消息，生成一个聚焦的研究概要来指导研究主管，
    并设置初始的主管上下文，包含适当的提示词和指令。

    Args:
        state: 当前智能体状态，包含用户消息
        config: 运行时配置，包含模型设置

    Returns:
        Command: 继续进入研究主管节点，并附带初始化后的上下文
    """
    # 第1步：设置用于结构化输出的研究模型
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.supervisor_model,
        "max_tokens": configurable.supervisor_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.supervisor_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 配置模型：结构化输出 + 重试逻辑
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion,method="json_mode")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # 第2步：从用户消息生成结构化的研究简报
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])

    # 第3步：使用研究简报和指令初始化主管智能体
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """首席研究主管，负责规划研究策略并将任务委派给研究员。

    主管分析研究简报，决定如何将研究分解为可管理的任务。
    可以使用 think_tool 进行战略规划，使用 ConductResearch 将任务委派给子研究员，
    或在收集到足够发现后使用 ResearchComplete 结束研究。

    Args:
        state: 当前主管状态，包含消息和研究上下文
        config: 运行时配置，包含模型设置

    Returns:
        Command: 继续进入主管工具执行节点
    """
    # 第1步：配置主管模型及可用工具
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.supervisor_model,
        "max_tokens": configurable.supervisor_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.supervisor_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 可用工具：研究委派、完成信号、战略思考
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool, search_tools_catalog]

    # 配置模型：绑定工具 + 重试逻辑 + 模型设置
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # 第2步：根据当前上下文生成主管响应
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)

    # 第3步：更新状态并继续进入工具执行
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """执行主管调用的工具，包括研究委派和战略思考。

    此函数处理三种类型的主管工具调用：
    1. think_tool - 战略反思，继续对话
    2. ConductResearch - 将研究任务委派给子研究员
    3. ResearchComplete - 标记研究阶段完成

    Args:
        state: 当前主管状态，包含消息和迭代计数
        config: 运行时配置，包含研究限制和模型设置

    Returns:
        Command: 要么继续主管循环，要么结束研究阶段
    """
    # 第1步：提取当前状态并检查退出条件
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # 定义研究阶段的退出条件
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls

    # 提取派发任务
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    # 只有在【没有派发新任务】的前提下，完成信号才生效
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls
    )
    should_end_research = research_complete_called and not conduct_research_calls

    # 如果满足任一终止条件则退出
    if exceeded_allowed_iterations or no_tool_calls or should_end_research:
        return Command(
            goto=END,
            update={"structured_facts": state.get("structured_facts", [])}
        )

    # 第2步：同时处理所有工具调用（包括 think_tool 和 ConductResearch）
    update_payload = {"supervisor_messages": []}
    send_actions = []

    # 处理 think_tool 调用（战略反思）
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        update_payload["supervisor_messages"].append(ToolMessage(
            content=f"Reflections recorded: {tool_call['args'].get('reflection', '')}",
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ))

    # 处理 search_tools_catalog 调用
    catalog_tool_calls = [
        tc for tc in most_recent_message.tool_calls
        if tc["name"] == "search_tools_catalog"
    ]
    for tc in catalog_tool_calls:
        # 直接调用工具并返回观察结果
        observation = await search_tools_catalog.ainvoke(tc["args"], config)
        update_payload["supervisor_messages"].append(ToolMessage(
            content=observation,
            name=tc["name"],
            tool_call_id=tc["id"]
        ))

    # 处理 ConductResearch 调用（研究委派）
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        # 限制并发研究单元数量，防止资源耗尽
        allowed = conduct_research_calls[:configurable.max_concurrent_research_units]
        overflow = conduct_research_calls[configurable.max_concurrent_research_units:]

        # 生成动态图分支
        for tool_call in allowed:
            send_actions.append(Send("researcher_subgraph", {
                "researcher_messages": [HumanMessage(content=tool_call["args"]["research_topic"])],
                "research_topic": tool_call["args"]["research_topic"],
                "required_tools": tool_call["args"].get("required_tools", ["web_search", "fetch_webpage"]),
                "required_skills": tool_call["args"].get("required_skills", []),
                "tool_call_id": tool_call["id"]  # 注入溯源 ID
            }))

        # 处理溢出拒绝
        for tool_call in overflow:
            update_payload["supervisor_messages"].append(ToolMessage(
                content=f"Error: Exceeded max concurrent research units ({configurable.max_concurrent_research_units}).",
                name="ConductResearch",
                tool_call_id=tool_call["id"]
            ))

        # 如果有下发的调研任务，图状态流转至评估节点；否则流转回主管
    if send_actions:
        return Command(goto=send_actions, update=update_payload)

    return Command(goto="supervisor", update=update_payload)

async def evaluate_research(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """在所有子图并发执行完毕后，统一评估收集到的事实，执行剪枝与熔断。"""
    staged_facts = state.get("staged_facts", [])
    existing_facts = state.get("structured_facts", [])
    existing_claims = {getattr(f, 'claim', '') for f in existing_facts}

    new_unique_facts_count = 0
    for fact in staged_facts:
        claim = getattr(fact, 'claim', '')
        if claim and claim not in existing_claims:
            new_unique_facts_count += 1
            existing_claims.add(claim)

    low_gain_rounds = state.get("consecutive_low_gain_rounds", 0)

    if new_unique_facts_count == 0:
        low_gain_rounds += 1
        print(f"\n[✂️ 动态剪枝] 本轮并行检索产生 0 条全新事实。当前连续停滞轮数: {low_gain_rounds}")
    else:
        low_gain_rounds = 0
        print(f"\n[📈 信息增益] 本轮并行新增 {new_unique_facts_count} 条独立高优事实。")

    # 更新结构化事实，并严格清空暂存区，防止脏数据污染下一轮
    update_payload = {
        "consecutive_low_gain_rounds": low_gain_rounds,
        "structured_facts": staged_facts,
        "staged_facts": {"type": "override", "value": []}
    }

    if low_gain_rounds >= 2:
        print("\n[🛑 强制熔断] 知识图谱已饱和，提早终止 Supervisor，进入成文阶段。")
        return Command(goto=END, update=update_payload)

    # 软警告回流
    if new_unique_facts_count == 0 and low_gain_rounds == 1:
        print("\n[⚠️ 系统警告] 提醒 Supervisor 改变策略。")
        warning_msg = HumanMessage(
            content="[SYSTEM WARNING] Your last concurrent delegation yielded ZERO new unique facts. The search space is saturating. You MUST drastically change your strategy or call 'ResearchComplete'."
        )
        update_payload["supervisor_messages"] = [warning_msg]

    return Command(goto="fold_memory", update=update_payload)

async def fold_memory(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor"]]:
    """折叠主管的历史记忆，防止 Token 爆炸。"""
    messages = state.get("supervisor_messages", [])

    # 设定阈值：当消息数量超过 10 条（初始2条 + 至少4轮交互）时触发折叠
    if len(messages) <= 10:
        return Command(goto="supervisor")

    # 核心安全逻辑：寻找最后一个 AIMessage 的索引
    # 必须保留最后一次决策及其对应的 ToolMessage，防止破坏 LangGraph 的工具校验
    last_ai_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            last_ai_idx = i
            break

    # 如果历史太短，或者找不到分割点，则放弃折叠
    if last_ai_idx <= 2:
        return Command(goto="supervisor")

    # 截取需要被压缩的中间历史
    # messages[0] 和 [1] 是初始的 SystemMessage 和 HumanMessage
    history_to_fold = messages[2:last_ai_idx]
    retained_latest = messages[last_ai_idx:]

    history_str = get_buffer_string(history_to_fold)

    configurable = Configuration.from_runnable_config(config)
    # 调用价格低廉/速度快的压缩模型（例如 GPT-4o-mini 或同等模型）
    model_config = {
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    }

    model = configurable_model.with_config(model_config)
    prompt = memory_folding_prompt.format(history=history_str)

    # 执行压缩
    try:
        summary_msg = await model.ainvoke([HumanMessage(content=prompt)])
        folded_memory_text = summary_msg.content
    except Exception as e:
        print(f"\n[⚠️ 记忆折叠失败] 跳过本次压缩: {e}")
        return Command(goto="supervisor")

    # 构造新的记忆胶囊
    folded_memory = SystemMessage(
        content=f"[SYSTEM: LONG-TERM MEMORY FROM PREVIOUS STEPS]\n{folded_memory_text}"
    )

    # 拼接全新的状态（初始设定 + 浓缩记忆 + 当前执行现场）
    new_messages = messages[:2] + [folded_memory] + retained_latest

    print(f"\n[🧠 记忆折叠] 主管上下文已压缩，消息数: {len(messages)} -> {len(new_messages)}")

    # 利用 override_reducer 直接覆写 Supervisor 的消息流
    return Command(
        goto="supervisor",
        update={"supervisor_messages": {"type": "override", "value": new_messages}}
    )

async def human_review(state: AgentState, config: RunnableConfig) -> Command[
    Literal["research_supervisor", "generate_outline"]]:
    """人在回路 (HITL) 节点：挂起工作流，等待人类审核事实或追加干预指令。"""

    # ===== 自动化测试旁路开关 =====
    if config.get("configurable", {}).get("simulate_human_approval", False):
        print("\n[🤖 自动化测试] 检测到测试环境，跳过人类审查，直接放行进入大纲生成阶段。")
        return Command(goto="generate_outline")
    # ====================================

    facts = state.get("structured_facts", [])

    # 兼容 Pydantic v2 (model_dump) 和 v1 (dict)
    serialized_facts = []
    for f in facts:
        if hasattr(f, 'model_dump'):
            serialized_facts.append(f.model_dump())
        elif hasattr(f, 'dict'):
            serialized_facts.append(f.dict())
        else:
            serialized_facts.append(f)

    # 触发中断，挂起当前节点。给前端/API返回明确的提示信息
    user_response = interrupt({
        "action_required": "review_facts",
        "message": "请审核收集到的事实。如有意见请输入修改指令并返回；若满意，请直接留空返回。",
        "facts": serialized_facts
    })

    feedback_text = ""

    if isinstance(user_response, str):
        feedback_text = user_response.strip()
        # 兼容旧版序列化 JSON
        if feedback_text.startswith("{") and feedback_text.endswith("}"):
            try:
                parsed = json.loads(feedback_text)
                feedback_text = parsed.get("feedback", "").strip() if parsed.get("action") == "feedback" else ""
            except:
                pass
    elif isinstance(user_response, dict):
        feedback_text = user_response.get("feedback", "").strip() if user_response.get("action") == "feedback" else ""
    elif user_response is not None:
        feedback_text = str(user_response).strip()

    if not feedback_text:
        # 如果 feedback_text 为空字符串，说明用户没有输入意见，直接放行
        print("\n[✅ 人类干预] 审核通过，进入大纲与分层成文阶段。")
        return Command(goto="generate_outline")

        # 如果有内容，说明用户不满意，将输入内容作为指令打回
    print(f"\n[🔄 人类干预] 收到打回指令: {feedback_text}")
    intervention_msg = HumanMessage(
        content=f"[HUMAN INTERVENTION] The structured facts were reviewed and rejected. You MUST conduct further research based on this directive: {feedback_text}"
    )
    return Command(
        goto="research_supervisor",
        update={
            "supervisor_messages": [intervention_msg],
            "consecutive_low_gain_rounds": 0  # 重置熔断计数器，防止二次熔断
        }
    )

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """独立研究员，负责对特定主题进行聚焦研究。

    该研究员接收主管分配的特定研究主题，使用可用工具（搜索、think_tool、MCP 工具）
    收集全面信息。可以在多次搜索之间使用 think_tool 进行战略规划。

    Args:
        state: 当前研究员状态，包含消息和主题上下文
        config: 运行时配置，包含模型设置和工具可用性

    Returns:
        Command: 继续进入研究员工具执行节点
    """
    # 第1步：加载配置并验证工具可用性
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    # 获取所有可用的研究工具（搜索、MCP、think_tool）
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "未找到可用于研究的工具：请配置您的搜索 API 或在配置中添加 MCP 工具。"
        )

    # 精准挂载分配的工具
    allowed_tool_names = state.get("required_tools", ["web_search", "fetch_webpage"])
    active_tools = []

    tool_names = []
    for tool in tools:
        tool_name = tool.name if hasattr(tool, "name") else tool.get("name", "web_search")
        # 永远保留思考节点，并挂载 Supervisor 准许的业务工具
        if tool_name == "think_tool" or tool_name in allowed_tool_names:
            tool_names.append(tool_name)
            active_tools.append(tool)

    # if tool_names:
        # print(f"\n[🔧 工具挂载] 激活专属工具: {tool_names}")

    # 精准挂载分配的技能
    assigned_skills = state.get("required_skills", [])
    loaded_skills = get_active_skills(assigned_skills)

    if loaded_skills:
        active_tools.extend(loaded_skills)
        loaded_skill_names = [s.name for s in loaded_skills]
        # print(f"\n[🔧 技能挂载] 激活专属技能: {loaded_skill_names}")

    active_tool_names = [t.name if hasattr(t, "name") else t.get("name") for t in active_tools]
    # print(f"\n[🎯 精准挂载] 任务: {state.get('research_topic', 'Unknown')[:15]}... | 武器: {active_tool_names}")

    # 第2步：配置研究员模型及工具
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config)
    }

    # 准备系统提示词，如果可用则包含 MCP 上下文
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "",
        date=get_today_str()
    )

    # 配置模型：绑定工具 + 重试逻辑 + 设置
    research_model = (
        configurable_model
        .bind_tools(active_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # 第3步：使用系统上下文生成研究员响应
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    # 第4步：更新状态并继续进入工具执行
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# 工具执行辅助函数
async def execute_tool_safely(tool, args, config):
    """安全执行工具，带有错误处理。"""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"执行工具时出错：{str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """执行研究员调用的工具，包括搜索工具和战略思考。

    此函数处理多种类型的研究员工具调用：
    1. think_tool - 战略反思，继续研究对话
    2. 搜索工具（tavily_search, web_search）- 信息收集
    3. MCP 工具 - 外部工具集成
    4. ResearchComplete - 标记单个研究任务完成

    Args:
        state: 当前研究员状态，包含消息和迭代计数
        config: 运行时配置，包含研究限制和工具设置

    Returns:
        Command: 要么继续研究循环，要么进入压缩阶段
    """
    # 第1步：提取当前状态并检查提前退出条件
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    # 如果没有工具调用则提前退出（包括原生网络搜索）
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or
        anthropic_websearch_called(most_recent_message)
    )

    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")

    # 第2步：处理工具调用（搜索、MCP 工具、skills等）
    tools = await get_all_tools(config)
    from open_deep_research.utils import _mcp_client
    if _mcp_client is not None:
        try:
            mcp_tools = await _mcp_client.get_tools()
            tools.extend(mcp_tools)
        except Exception:
            pass
    # 动态合并分配到的所有技能
    assigned_skills = state.get("required_skills", [])
    tools.extend(get_active_skills(assigned_skills))

    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }

    # 并行执行所有工具调用
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # 从执行结果创建工具消息
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # 第3步：检查延迟退出条件（处理完工具后）
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or research_complete_called:
        # 结束研究并进入压缩阶段
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )

    # 使用工具结果继续研究循环
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """将研究发现压缩整合为简洁、结构化的摘要。

    此函数接收研究员的全部研究发现、工具输出和 AI 消息，
    将其提炼为干净、全面的摘要，同时保留所有重要信息和发现。

    Args:
        state: 当前研究员状态，包含累积的研究消息
        config: 运行时配置，包含压缩模型设置

    Returns:
        Dictionary: 包含压缩后的研究摘要和原始笔记
    """
    # 第1步：配置压缩模型
    configurable = Configuration.from_runnable_config(config)
    model_config = {
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    }

    structured_synthesizer_model = (
        configurable_model
        .with_structured_output(FactBoard, method="json_mode")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    # 第2步：准备压缩用的消息
    researcher_messages = state.get("researcher_messages", [])
    compression_prompt = compress_research_system_prompt.format(date=get_today_str())
    # 添加指令，从研究模式切换到压缩模式
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    messages = [SystemMessage(content=compression_prompt)] + researcher_messages

    # 第3步：使用重试逻辑进行压缩，处理 token 限制问题，执行结构化提取
    synthesis_attempts = 0
    max_attempts = 3

    while synthesis_attempts < max_attempts:
        try:
            # 执行压缩， 强制挂载 FactBoard 结构化输出
            response = await structured_synthesizer_model.ainvoke(messages)

            # 用 Python 物理提取图表，绝对不依赖大模型的自觉性
            for msg in state.get("researcher_messages", []):
                # 找到可视化工具的输出消息
                if getattr(msg, "name", None) == "data_visualization_skill":
                    # 正则匹配 Markdown 图片链接 ![xxx](yyy)
                    chart_match = re.search(r'(!\[.*?\]\(.*?\))', msg.content)
                    if chart_match:
                        # 强制构造一个 Fact 对象并追加到大模型的输出结果中
                        # 注意：需要确保你的上下文中能访问到 Fact 类
                        chart_fact = Fact(
                            entity="Data Visualization Chart",
                            claim=chart_match.group(1),
                            source="data_visualization_skill"
                        )
                        # 如果 response.facts 已经存在，则追加进去
                        if hasattr(response, 'facts'):
                            response.facts.append(chart_fact)

            # 提取当前子图中所有消息的 ID (包含庞大的 ToolMessage)
            delete_messages = [
                RemoveMessage(id=m.id)
                for m in state.get("researcher_messages", [])
                if getattr(m, 'id', None) is not None
            ]

            # 返回标准 TypedDict，LangGraph 会自动将其归约到 Supervisor 状态
            return {
                "staged_facts": response.facts,
                "supervisor_messages": [ToolMessage(
                    content=f"Successfully extracted {len(response.facts)} structured facts.",
                    name="ConductResearch",
                    tool_call_id=state.get("tool_call_id", "")
                )],
                # 下发销毁指令给底层 Checkpointer
                "researcher_messages": delete_messages
            }

        except Exception as e:
            synthesis_attempts += 1
            # 处理 token 超限，通过移除较旧的消息
            if is_token_limit_exceeded(e, configurable.compression_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            # 其他错误，继续重试
            continue

    # 第4步：如果所有尝试都失败，返回错误结果,同样执行垃圾回收
    delete_messages = [
        RemoveMessage(id=m.id)
        for m in state.get("researcher_messages", [])
        if getattr(m, 'id', None) is not None
    ]

    return {
        "staged_facts": [],
        "supervisor_messages": [ToolMessage(
            content="Failed to extract structured facts.",
            name="ConductResearch",
            tool_call_id=state.get("tool_call_id", "")
        )],
        "researcher_messages": delete_messages
    }

# 研究员子图构建
# 创建独立研究员工作流，用于对特定主题进行聚焦研究
researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState,
    config_schema=Configuration
)

# 添加研究员节点：研究执行和压缩
researcher_builder.add_node("researcher", researcher)                 # 研究员主逻辑
researcher_builder.add_node("researcher_tools", researcher_tools)     # 工具执行处理器
researcher_builder.add_node("compress_research", compress_research)   # 研究压缩

# 定义研究员工作流的边
researcher_builder.add_edge(START, "researcher")           # 研究员入口点
researcher_builder.add_edge("compress_research", END)      # 压缩后退出

# 编译研究员子图，供主管并行调用
researcher_subgraph = researcher_builder.compile()


# 主管子图构建
# 创建管理工作委派和协调的研究主管工作流
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# 添加研究管理相关的主管节点
supervisor_builder.add_node("supervisor", supervisor)           # 主管主逻辑
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # 工具执行处理器
supervisor_builder.add_node("researcher_subgraph", researcher_subgraph)
supervisor_builder.add_node("evaluate_research", evaluate_research)
supervisor_builder.add_node("fold_memory", fold_memory)

# 定义主管工作流的边
supervisor_builder.add_edge(START, "supervisor")  # 主管入口点
supervisor_builder.add_edge("researcher_subgraph", "evaluate_research")

# 编译主管子图，供主工作流使用
supervisor_subgraph = supervisor_builder.compile()


async def generate_outline(state: AgentState, config: RunnableConfig) -> Command[Literal["write_section"]]:
    """生成大纲并完成数据路由，下发切片后的事实给并发节点。"""
    configurable = Configuration.from_runnable_config(config)
    structured_facts = state.get("structured_facts", [])

    # 极简模式：只给模型看 Entity 和截断的 Claim，并附带明确的 ID (索引)
    formatted_findings = "\n".join([
        f"Fact ID [{i}]: Entity: {getattr(f, 'entity', 'Unknown')} | Claim: {str(getattr(f, 'claim', ''))[:100]}..."
        for i, f in enumerate(structured_facts)
    ])

    prompt = generate_outline_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=formatted_findings,
        date=get_today_str()
    )

    outline_model = configurable_model.with_structured_output(ReportOutline,method="json_mode").with_config({
        "model": configurable.research_model,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    })

    try:
        response: ReportOutline = await outline_model.ainvoke([HumanMessage(content=prompt)])
    except Exception as e:
        # 极简模式下几乎不会超限，若出错直接走兜底
        print(f"大纲生成失败，启用兜底: {e}")
        response = ReportOutline(sections=[
            SectionOutline(
                section_title="核心调研发现",
                description="综合归纳所有搜集到的事实。",
                relevant_fact_indices=list(range(len(structured_facts)))  # 兜底时全部塞给单一章节
            )
        ])

    # 核心拦截器1：查找被大模型遗漏的事实 ID（防止幻觉导致数据丢失）
    assigned_indices = set()
    for sec in response.sections:
        assigned_indices.update(sec.relevant_fact_indices)

    unassigned_indices = [i for i in range(len(structured_facts)) if i not in assigned_indices]

    if unassigned_indices:
        # 如果有被遗漏的事实，自动追加一个“补充发现”章节兜底
        response.sections.append(SectionOutline(
            section_title="补充调研发现",
            description="其他重要的数据与事实补充。",
            relevant_fact_indices=unassigned_indices
        ))

    send_actions = []
    valid_sections = []  # 记录真正派发成功的有效章节

    for sec in response.sections:
        # 精准切片：只提取当前章节被分配到的事实
        assigned_facts_with_ids = []
        for idx in sec.relevant_fact_indices:
            if 0 <= idx < len(structured_facts):  # 防止幻觉越界
                fact = structured_facts[idx]
                # 将 Fact 转换为字典，并强制注入与 assemble_report 对应的全局 ID (idx + 1)
                assigned_facts_with_ids.append({
                    "global_id": idx + 1,
                    "entity": getattr(fact, 'entity', ''),
                    "claim": getattr(fact, 'claim', ''),
                    "source": getattr(fact, 'source', '')
                })

        # 核心拦截器 2：如果该章节没有分到任何事实，直接从大纲中剔除，不予派发
        if not assigned_facts_with_ids:
            continue

        valid_sections.append(sec)

        send_actions.append(Send("write_section", {
            "section_title": sec.section_title,
            "section_description": sec.description,
            "assigned_facts": assigned_facts_with_ids,  # 下发包含 global_id 的字典
            "research_brief": state.get("research_brief", "")
        }))

    return Command(
        goto=send_actions,
        update={"report_outline": valid_sections,
                "section_drafts": {"type": "override", "value": []}}
    )


async def write_section(state: WriteSectionState, config: RunnableConfig):
    """并发写手节点：利用分配到的极少量事实撰写局部章节。"""
    configurable = Configuration.from_runnable_config(config)

    # 提前把分配给这个章节的图表链接提取出来备用
    chart_links = []
    for f in state["assigned_facts"]:
        claim = getattr(f, 'claim', '')
        # 寻找形如 ![alt](url) 的 Markdown 图片
        match = re.search(r'(!\[.*?\]\(.*?\))', claim)
        if match:
            chart_links.append(match.group(1))

    # 使用 global_id 作为事实的明显标识
    formatted_findings = "\n".join([
        f"Fact [{f['global_id']}]: Entity: {f.get('entity', '')} | Claim: {f.get('claim', '')} | Source: {f.get('source', '')}"
        for f in state["assigned_facts"]
    ])

    prompt = write_section_prompt.format(
        research_brief=state["research_brief"],
        section_title=state["section_title"],
        section_description=state["section_description"],
        findings=formatted_findings,
        date=get_today_str()
    )

    writer_model = configurable_model.with_config({
        "model": configurable.writer_model,
        "max_tokens": 4000,
        "api_key": get_api_key_for_model(configurable.writer_model, config)
    })

    section_content = await writer_model.ainvoke([HumanMessage(content=prompt)])
    final_text = section_content.content

    # 检查大模型是否漏掉了图表或篡改了链接
    for chart in chart_links:
        if chart not in final_text:
            # 如果原封不动的图表链接不在正文里，说明大模型犯病了，我们强行补在章节最后
            final_text += f"\n\n{chart}\n\n"

    return {
        "section_drafts": [SectionDraft(
            section_title=state["section_title"],
            content=final_text
        )]
    }

async def assemble_report(state: AgentState, config: RunnableConfig) -> Command[Literal["report_verifier"]]:
    """组装节点 (Reduce)：将并发生成的章节按大纲顺序拼合，并生成统一引用。"""
    outline = state.get("report_outline", [])
    drafts = state.get("section_drafts", [])
    structured_facts = state.get("structured_facts", [])

    # 建立草稿字典以实现无序到有序的映射 O(1) 查找
    draft_map = {draft.section_title: draft.content for draft in drafts}

    # 严格按照大纲顺序拼接
    assembled_parts = []
    for sec in outline:
        content = draft_map.get(sec.section_title, f"## {sec.section_title}\n[该章节内容生成失败]")
        assembled_parts.append(content)

    # 生成全局统一的参考文献列表
    sources_section = ["\n\n### 参考文献 (Sources)"]
    for i, fact in enumerate(structured_facts):
        sources_section.append(f"- [{i + 1}] Source: {getattr(fact, 'source', 'Unknown')}")

    full_report = "\n\n".join(assembled_parts) + "\n".join(sources_section)

    return Command(
        goto="report_verifier",
        update={"final_report": full_report, "messages": [AIMessage(content=full_report)]}
    )

async def report_verifier(state: AgentState, config: RunnableConfig) -> Command[Literal["rewrite_report", "__end__"]]:
    """核查节点：使用小模型对抗式核验成文报告的引用与断言准确率。"""
    configurable = Configuration.from_runnable_config(config)
    structured_facts = state.get("structured_facts", [])
    current_report = state.get("final_report", "")
    retries = state.get("verification_retries", 0)

    # 1. 格式化 FactBoard
    formatted_facts = "\n".join([
        f"- Entity: {getattr(f, 'entity', 'Unknown')} | Claim: {getattr(f, 'claim', '')} | Source: {getattr(f, 'source', '')}"
        for f in structured_facts
    ])

    # 2. 配置核查模型（选用推理能力强、成本适度的小/中模型）
    verifier_model = (configurable_model
      .with_structured_output(VerificationReport, method="json_mode")
      .with_config({
        "model": configurable.verifier_model,
        "max_tokens": configurable.verifier_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.verifier_model, config),
        "tags": ["langsmith:nostream"]
    }).with_retry(stop_after_attempt=configurable.max_structured_output_retries)    # 挂载重试机制，捕获 Pydantic 解析异常并让 LLM 自动纠正
    )

    prompt = report_verifier_prompt.format(
        date=get_today_str(),
        findings=formatted_facts,
        report=current_report
    )

    verification: VerificationReport = await verifier_model.ainvoke([HumanMessage(content=prompt)])

    # 3. 判定路由逻辑：若无幻觉，或重试达到上限（防止死循环），则放行结束
    if not verification.has_hallucinations or retries >= configurable.max_verification_retries:
        return Command(
            goto=END,
            update={"verification_feedback": None}
        )

    # 存在幻觉且仍有重试预算，流转到定向重写节点
    return Command(
        goto="rewrite_report",
        update={
            "verification_feedback": verification.feedback,
            "verification_retries": retries + 1
        }
    )

async def rewrite_report(state: AgentState, config: RunnableConfig) -> Command[Literal["report_verifier"]]:
    """定向修正节点：根据核查反馈剔除幻觉，重写报告。"""
    configurable = Configuration.from_runnable_config(config)
    structured_facts = state.get("structured_facts", [])
    current_report = state.get("final_report", "")
    feedback = state.get("verification_feedback", "")

    existing_charts = re.findall(r'(!\[.*?\]\(.*?\))', current_report)

    formatted_facts = "\n".join([
        f"- Entity: {getattr(f, 'entity', 'Unknown')} | Claim: {getattr(f, 'claim', '')} | Source: {getattr(f, 'source', '')}"
        for f in structured_facts
    ])

    rewrite_prompt = rewrite_report_prompt.format(
        date=get_today_str(),
        findings=formatted_facts,
        report=current_report,
        feedback=feedback
    )

    writer_model_config = {
        "model": configurable.writer_model,
        "max_tokens": configurable.writer_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.writer_model, config),
        "tags": ["langsmith:nostream"]
    }

    revised_report = await configurable_model.with_config(writer_model_config).ainvoke([
        HumanMessage(content=rewrite_prompt)
    ])

    final_text = revised_report.content

    # 检查大模型在重写时是否删掉了图表
    for chart in existing_charts:
        if chart not in final_text:
            # 如果大模型误删了图表，我们强行把它插回正文和参考文献之间
            if "参考文献 (Sources)" in final_text:
                final_text = final_text.replace("参考文献 (Sources)", f"\n\n{chart}\n\n参考文献 (Sources)")
            elif "## Sources" in final_text:
                final_text = final_text.replace("## Sources", f"\n\n{chart}\n\n## Sources")
            else:
                final_text += f"\n\n{chart}\n\n"

    # 同步更新 Message 对象里的内容，保持状态一致
    revised_report.content = final_text

    return Command(
        goto="report_verifier",
        update={
            "final_report": revised_report.content,
            "messages": [revised_report]
        }
    )


# 主 Deep Researcher 图构建
# 创建从用户输入到最终报告的完整深度研究工作流
deep_researcher_builder = StateGraph(
    AgentState,
    input=AgentInputState,
    config_schema=Configuration
)

# 添加主工作流节点：完整研究过程
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # 用户澄清阶段
deep_researcher_builder.add_node("route_task", route_task)                         # 任务路由节点
deep_researcher_builder.add_node("direct_answering", direct_answering)             # 直接回答节点
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # 研究规划阶段
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # 研究执行阶段
deep_researcher_builder.add_node("human_review", human_review)                     # HITL审核节点
deep_researcher_builder.add_node("generate_outline", generate_outline)             # 生成大纲节点
deep_researcher_builder.add_node("write_section", write_section)                   # 生成章节节点
deep_researcher_builder.add_node("assemble_report", assemble_report)               # 组装报告节点
deep_researcher_builder.add_node("report_verifier", report_verifier)               # 核查节点
deep_researcher_builder.add_node("rewrite_report", rewrite_report)                 # 重写节点

# 定义主工作流边：顺序执行
deep_researcher_builder.add_edge(START, "clarify_with_user")                                # 入口点
deep_researcher_builder.add_edge("research_supervisor", "human_review") # 研究到报告
deep_researcher_builder.add_edge("write_section", "assemble_report") # 报告到核查

# 编译完整的深度研究工作流
deep_researcher = deep_researcher_builder.compile()