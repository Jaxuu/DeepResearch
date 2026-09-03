"""Deep Research 智能体的 LangGraph 主实现。"""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
    report_verifier_prompt,
    rewrite_report_prompt
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
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# 初始化一个可配置的模型，将在整个智能体中使用
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "model_kwargs"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
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
        return Command(goto="write_research_brief")

    # 第2步：准备模型进行结构化澄清分析
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
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
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """将用户消息转换为结构化的研究简报，并初始化主管智能体。

    此函数分析用户消息，生成一个聚焦的研究简报来指导研究主管，
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
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 配置模型：结构化输出 + 重试逻辑
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
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
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 可用工具：研究委派、完成信号、战略思考
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

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
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # 如果满足任一终止条件则退出
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "structured_facts": state.get("structured_facts", []),
                "research_brief": state.get("research_brief", "")
            }
        )

    # 第2步：同时处理所有工具调用（包括 think_tool 和 ConductResearch）
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # 处理 think_tool 调用（战略反思）
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think0_tool"
    ]

    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflections have been recorded：{reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    # 处理 ConductResearch 调用（研究委派）
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        try:
            # 限制并发研究单元数量，防止资源耗尽
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]

            # 并行执行研究任务
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config)
                for tool_call in allowed_conduct_research_calls
            ]

            tool_results = await asyncio.gather(*research_tasks)

            # 使用研究结果创建工具消息
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error compressing research report: Exceeded max retries."),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))

            # 为超出限制的研究调用返回错误消息
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Exceeded max concurrent research units ({configurable.max_concurrent_research_units}). Please try submitting with fewer units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))

            # 汇总所有研究结果中的原始笔记
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", []))
                for observation in tool_results
            ])

            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

            # 收集并汇总结构化事实
            all_structured_facts = []
            for observation in tool_results:
                facts = observation.get("structured_facts", [])
                if facts:
                    all_structured_facts.extend(facts)

            if all_structured_facts:
                update_payload["structured_facts"] = all_structured_facts

            # 基于信息增益的动态剪枝
            existing_facts = state.get("structured_facts", [])
            # 提取已有的所有断言，用于查重
            existing_claims = {getattr(f, 'claim', '') for f in existing_facts}

            new_unique_facts_count = 0
            for fact in all_structured_facts:
                claim = getattr(fact, 'claim', '')
                if claim and claim not in existing_claims:
                    new_unique_facts_count += 1
                    existing_claims.add(claim)

            # 获取当前的连续低增益轮数
            low_gain_rounds = state.get("consecutive_low_gain_rounds", 0)

            if new_unique_facts_count == 0:
                low_gain_rounds += 1
                print(f"\n[✂️ 动态剪枝追踪] 本轮检索产生 0 条全新事实。当前连续停滞轮数: {low_gain_rounds}")
            else:
                low_gain_rounds = 0  # 只要有新发现，重置计数器
                print(f"\n[📈 信息增益检测] 本轮新增 {new_unique_facts_count} 条独立高优事实。")

            update_payload["consecutive_low_gain_rounds"] = low_gain_rounds

            # 触发强制熔断条件（连续 2 轮未获取新事实，或大模型主动停止）
            if low_gain_rounds >= 2:
                print("\n[🛑 强制熔断触发] 知识图谱已饱和，提早终止 Supervisor 盲目派发，进入成文阶段。")
                return Command(
                    goto=END,
                    update=update_payload  # 携带最新的状态强制退出
                )

            # 如果处于低增益状态但还没熔断，给 Supervisor 智能体发一条严重警告
            if new_unique_facts_count == 0 and low_gain_rounds == 1:
                print("\n[⚠️ 发送系统警告] 提醒 Supervisor 改变策略或提早结束。")
                # 安全做法：直接把警告追加到刚刚执行完的最后一个工具调用的返回内容里
                if all_tool_messages:
                    all_tool_messages[-1].content += (
                        "\n\n[SYSTEM WARNING: CRITICAL ALERT]\n"
                        "Your last delegation yielded ZERO new unique facts. The search space is saturating. "
                        "You MUST drastically change your search strategy (use completely different keywords) "
                        "OR call 'ResearchComplete' immediately to avoid wasting resources."
                    )

        except Exception as e:
            # 处理研究执行错误
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token 超限或其他错误 - 结束研究阶段
                return Command(
                    goto=END,
                    update={
                        "structured_facts": state.get("structured_facts", []),
                        "research_brief": state.get("research_brief", "")
                    }
                )

    # 第3步：返回包含所有工具结果的 Command
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    )

# 主管子图构建
# 创建管理工作委派和协调的研究主管工作流
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# 添加研究管理相关的主管节点
supervisor_builder.add_node("supervisor", supervisor)           # 主管主逻辑
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # 工具执行处理器

# 定义主管工作流的边
supervisor_builder.add_edge(START, "supervisor")  # 主管入口点

# 编译主管子图，供主工作流使用
supervisor_subgraph = supervisor_builder.compile()

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

    # 第2步：配置研究员模型及工具
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 准备系统提示词，如果可用则包含 MCP 上下文
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "",
        date=get_today_str()
    )

    # 配置模型：绑定工具 + 重试逻辑 + 设置
    research_model = (
        configurable_model
        .bind_tools(tools)
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

    # 第2步：处理其他工具调用（搜索、MCP 工具等）
    tools = await get_all_tools(config)
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

            # 提取所有工具消息和 AI 消息中的原始笔记
            raw_notes_content = "\n".join([
                str(message.content)
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])

            # 返回成功的压缩结果
            return {
                "raw_notes": [raw_notes_content],
                "structured_facts": response.facts,
                "compressed_research": f"Successfully extracted {len(response.facts)} structured facts."
            }

        except Exception as e:
            synthesis_attempts += 1

            # 处理 token 超限，通过移除较旧的消息
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue

            # 其他错误，继续重试
            continue

    # 第4步：如果所有尝试都失败，返回错误结果
    raw_notes_content = "\n".join([
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])

    return {
        "raw_notes": [raw_notes_content],
        "structured_facts": [],
        "compressed_research": f"Failed to extract structured facts."
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

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """生成最终的综合研究报告，带有 token 限制的重试逻辑。

    此函数接收所有收集的研究发现，并使用配置的报告生成模型
    将其合成为结构良好、全面的最终报告。

    Args:
        state: 智能体状态，包含研究发现和上下文
        config: 运行时配置，包含模型设置和 API 密钥

    Returns:
        Dictionary: 包含最终报告和清空的状态
    """
    # 第1步：提取结构化事实看板数据
    structured_facts = state.get("structured_facts", [])
    facts_to_use = structured_facts.copy()  # 用于后续可能的动态截断

    # 第2步：配置最终报告生成模型
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }

    # 第3步：使用 token 限制重试逻辑尝试生成报告
    max_retries = 3
    current_retry = 0

    while current_retry <= max_retries:
        try:
            # 将 Pydantic 对象列表格式化为高信噪比的纯文本上下文
            formatted_findings = []
            for i, fact in enumerate(facts_to_use):
                # 兼容可能是 dict 或 Pydantic BaseModel 的情况
                entity = fact.entity if hasattr(fact, 'entity') else fact.get('entity', 'Unknown')
                claim = fact.claim if hasattr(fact, 'claim') else fact.get('claim', '')
                source = fact.source if hasattr(fact, 'source') else fact.get('source', '')

                formatted_findings.append(
                    f"Fact [{i + 1}]:\n"
                    f" - Entity: {entity}\n"
                    f" - Claim: {claim}\n"
                    f" - Source: {source}"
                )

            findings_text = "\n\n".join(formatted_findings)

            # 创建包含所有研究上下文的综合提示词
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings_text,
                date=get_today_str()
            )

            # 生成最终报告
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])

            # 返回成功的报告生成结果
            return {
                "final_report": final_report.content,
                "messages": [final_report],
            }

        except Exception as e:
            # 处理 token 超限错误：优雅地丢弃最末尾的 10% 事实，而非截断半句话
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1

                if len(facts_to_use) > 0:
                    # 每次重试保留前 90% 的事实记录
                    keep_count = max(1, int(len(facts_to_use) * 0.9))
                    facts_to_use = facts_to_use[:keep_count]
                continue
            else:
                # 非 token 超限错误：立即返回错误
                return {
                    "final_report": f"生成最终报告时出错：{e}",
                    "messages": [AIMessage(content="因错误导致报告生成失败")]
                }

    # 第4步：如果所有重试都已耗尽
    return {
        "final_report": "生成最终报告时出错：超过最大重试次数，上下文仍然过长。",
        "messages": [AIMessage(content="超过最大重试次数后报告生成失败")]
    }

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
    verifier_model = (configurable_model.with_config({
        "model": configurable.verifier_model,
        "max_tokens": configurable.verifier_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.verifier_model, config),
        "tags": ["langsmith:nostream"]
    }).with_structured_output(VerificationReport, method="function_calling")        # 显式指定使用 function_calling 模式
      .with_retry(stop_after_attempt=configurable.max_structured_output_retries)    # 挂载重试机制，捕获 Pydantic 解析异常并让 LLM 自动纠正
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
        "model": configurable.rewrite_model,
        "max_tokens": configurable.rewrite_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.rewrite_model, config),
        "tags": ["langsmith:nostream"]
    }

    revised_report = await configurable_model.with_config(writer_model_config).ainvoke([
        HumanMessage(content=rewrite_prompt)
    ])

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
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # 研究规划阶段
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # 研究执行阶段
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # 报告生成阶段
deep_researcher_builder.add_node("report_verifier", report_verifier)                 # 核查节点
deep_researcher_builder.add_node("rewrite_report", rewrite_report)                 # 重写节点

# 定义主工作流边：顺序执行
deep_researcher_builder.add_edge(START, "clarify_with_user")                                # 入口点
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # 研究到报告
deep_researcher_builder.add_edge("final_report_generation", "report_verifier")     # 报告到核查


# 编译完整的深度研究工作流
deep_researcher = deep_researcher_builder.compile()