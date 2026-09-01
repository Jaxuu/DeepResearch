"""Deep Research 智能体的实用工具与辅助函数。"""

import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.state import ResearchComplete, Summary

##########################
# Tavily 搜索工具组件
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from Tavily search API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        topic: Topic filter for search results (general, news, or finance)
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing summarized search results
    """
    # 步骤 1：异步执行搜索查询
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )

    # 步骤 2：按 URL 进行去重，避免重复处理相同内容
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}

    # 步骤 3：根据配置初始化摘要模型
    configurable = Configuration.from_runnable_config(config)

    # 字符数限制以保持在模型 Token 限制内（可配置）
    max_char_to_include = configurable.max_content_length

    # 初始化带重试逻辑的摘要模型
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary, method="function_calling").with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )

    # 步骤 4：创建摘要任务（跳过空内容）
    async def noop():
        """针对无原始正文结果的空操作函数。"""
        return None

    summarization_tasks = [
        noop() if not result.get("raw_content")
        else summarize_webpage(
            summarization_model,
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]

    # 步骤 5：并行执行所有摘要任务
    summaries = await asyncio.gather(*summarization_tasks)

    # 步骤 6：将搜索结果与生成的摘要进行合并
    summarized_results = {
        url: {
            'title': result['title'],
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(),
            unique_results.values(),
            summaries
        )
    }

    # 步骤 7：格式化最终输出
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    # 增加单次 ToolMessage 输出总长度防御性截断（最多保留 25000 字符，约 7000 tokens）
    MAX_TOOL_OUTPUT_CHARS = 25000
    if len(formatted_output) > MAX_TOOL_OUTPUT_CHARS:
        formatted_output = formatted_output[:MAX_TOOL_OUTPUT_CHARS] + "\n\n[...搜索结果总量达到上限，已安全截断...]"

    return formatted_output

async def tavily_search_async(
    search_queries,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    config: RunnableConfig = None
):
    """异步并发执行多个 Tavily 搜索查询。

    参数:
        search_queries: 待执行的搜索查询字符串列表
        max_results: 每个查询返回的最大结果数
        topic: 过滤结果的主题类别
        include_raw_content: 是否包含完整的网页原始正文
        config: 用于获取 API 密钥的运行时配置

    返回:
        来自 Tavily API 的搜索结果字典列表
    """
    # 使用来自配置的 API 密钥初始化 Tavily 客户端
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))

    # 创建用于并行执行的搜索任务
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]

    # 并行执行所有搜索查询并返回结果
    search_results = await asyncio.gather(*search_tasks)
    return search_results

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """使用 AI 模型对网页正文进行摘要，并带有超时保护。

    参数:
        model: 配置好的用于生成摘要的 Chat 模型
        webpage_content: 待摘要的网页原始正文

    返回:
        格式化后的包含核心摘录的摘要，若摘要失败则返回截断后的原始内容
    """
    try:
        # 创建包含当前日期上下文的提示词
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content,
            date=get_today_str()
        )

        # 执行带超时的摘要生成，防止任务挂起
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0  # 摘要生成设置 60 秒超时
        )

        # 将摘要格式化为结构化区块
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary


    except asyncio.TimeoutError:
        # 摘要生成超时 - 返回截断后的原始内容
        logging.warning("Summarization timed out after 60 seconds, returning original content limit 1500 characters")
        return webpage_content[:1500]
    except Exception as e:
        # 摘要生成遇到其他错误 - 记录日志并返回截断后的原始内容
        logging.warning(f"Summarization failed with error: {str(e)}, returning original content limit 1500 characters")
        return webpage_content[:1500]

##########################
# 策略反思工具组件
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

##########################
# MCP 工具组件
##########################

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """使用 OAuth Token 交换机制，将 Supabase Token 换取为 MCP 访问 Token。

    参数:
        supabase_token: 有效的 Supabase 身份验证 Token
        base_mcp_url: MCP 服务器的基础 URL

    返回:
        若成功返回包含 Token 数据的字典，失败则返回 None
    """
    try:
        # 准备 OAuth Token 交换的请求数据
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }

        # 执行 Token 交换请求
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # 成功获取 Token
                    token_data = await response.json()
                    return token_data
                else:
                    # 记录错误详情以便调试
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")

    except Exception as e:
        logging.error(f"Error during token exchange: {e}")

    return None

async def get_tokens(config: RunnableConfig):
    """从存储中获取带有过期验证的身份验证 Token。

    参数:
        config: 包含线程和用户标识符的运行时配置

    返回:
        若有效且未过期返回 Token 字典，否则返回 None
    """
    store = get_store()

    # 从配置中提取所需的标识符
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None

    # 获取已存储的 Token
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None

    # 检查 Token 是否过期
    expires_in = tokens.value.get("expires_in")  # 距离过期的秒数
    created_at = tokens.created_at  # Token 创建时间
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)

    if current_time > expiration_time:
        # Token 已过期，清理并返回 None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """将身份验证 Token 保存到配置存储中。

    参数:
        config: 包含线程和用户标识符的运行时配置
        tokens: 要存储的 Token 字典
    """
    store = get_store()

    # 从配置中提取所需的标识符
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return

    # 存储 Token
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """获取并刷新 MCP Token，必要时重新获取新 Token。

    参数:
        config: 包含身份验证详情的运行时配置

    返回:
        有效的 Token 字典，若无法获取则返回 None
    """
    # 优先尝试获取现有有效的 Token
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens

    # 提取 Supabase Token 用于新的 Token 交换
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None

    # 提取 MCP 配置
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None

    # 使用 Supabase Token 换取 MCP Token
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # 存储新 Token 并返回
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """为 MCP 工具包装完善的身份验证与错误处理逻辑。

    参数:
        tool: 要包装的 MCP 结构化工具

    返回:
        增强了身份验证错误处理的工具
    """
    original_coroutine = tool.coroutine

    async def authentication_wrapper(**kwargs):
        """增强的协程函数，包含 MCP 错误处理和用户友好的提示信息。"""

        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """在异常链中递归查找 MCP 相关错误。"""
            if isinstance(exc, McpError):
                return exc

            # 通过检查属性处理 ExceptionGroup（Python 3.11+）
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None

        try:
            # 执行原始工具功能
            return await original_coroutine(**kwargs)

        except BaseException as original_error:
            # 在异常链中搜索 MCP 特定的错误
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # 非 MCP 错误，重新抛出原始异常
                raise original_error

            # 处理 MCP 特定的错误情况
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}

            # 检查是否为需要身份验证/交互的错误
            if error_code == -32003:  # 需要交互的错误代码
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"

                # 提取用户友好的提示信息（若存在）
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message

                # 若提供 URL 则追加供用户参考
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"

                raise ToolException(error_message) from original_error

            # 对于其他 MCP 错误，重新抛出原始异常
            raise original_error

    # 将工具的原协程替换为增强版本
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """加载并配置带有身份验证的 MCP（模型上下文协议）工具。

    参数:
        config: 包含 MCP 服务器详情的运行时配置
        existing_tool_names: 已使用的工具名称集合，用于避免名称冲突

    返回:
        已配置且可直接使用的 MCP 工具列表
    """
    configurable = Configuration.from_runnable_config(config)

    # 步骤 1：若需要则处理身份验证
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None

    # 步骤 2：校验配置要求
    config_valid = (
        configurable.mcp_config and
        configurable.mcp_config.url and
        configurable.mcp_config.tools and
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )

    if not config_valid:
        return []

    # 步骤 3：建立 MCP 服务器连接
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"

    # 若 Token 可用则配置身份验证请求头
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}

    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO: 当 Multi-MCP Server 支持合并到 OAP 后更新此代码

    # 步骤 4：从 MCP 服务器加载工具
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # 若 MCP 服务器连接失败，返回空列表
        return []

    # 步骤 5：过滤并配置工具
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # 跳过名称冲突的工具
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue

        # 仅包含配置中显式指定的工具
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue

        # 为工具包装身份验证处理并添加到列表
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)

    return configured_tools


##########################
# 基础工具组件
##########################

async def get_search_tool(search_api: SearchAPI):
    """根据指定的 API 提供商配置并返回搜索工具。

    参数:
        search_api: 搜索 API 提供商（Anthropic、OpenAI、Tavily 或 None）

    返回:
        指定提供商的已配置搜索工具对象列表
    """
    if search_api == SearchAPI.ANTHROPIC:
        # 带使用次数限制的 Anthropic 原生网络搜索
        return [{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5
        }]

    elif search_api == SearchAPI.OPENAI:
        # OpenAI 原生网络搜索预览功能
        return [{"type": "web_search_preview"}]

    elif search_api == SearchAPI.TAVILY:
        # 配置带元数据的 Tavily 搜索工具
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}),
            "type": "search",
            "name": "web_search"
        }
        return [search_tool]

    elif search_api == SearchAPI.NONE:
        # 未配置任何搜索功能
        return []

    # 未知搜索 API 类型的默认兜底处理
    return []

async def get_all_tools(config: RunnableConfig):
    """组装完整工具包，包含研究、搜索和 MCP 工具。

    参数:
        config: 指定搜索 API 和 MCP 设置的运行时配置

    返回:
        用于研究操作的所有已配置且可用的工具列表
    """
    # 首先添加核心研究工具
    tools = [tool(ResearchComplete), think_tool]

    # 添加配置的搜索工具
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)

    # 记录已有工具名称以防冲突
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search")
        for tool in tools
    }

    # 若有配置则添加 MCP 工具
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)

    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """从工具调用消息中提取笔记内容。"""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# 模型供应商原生网络搜索组件
##########################

def anthropic_websearch_called(response):
    """检测响应中是否调用了 Anthropic 的原生网络搜索。

    参数:
        response: 来自 Anthropic API 的响应对象

    返回:
        若调用了网络搜索返回 True，否则返回 False
    """
    try:
        # 遍历解析响应元数据结构
        usage = response.response_metadata.get("usage")
        if not usage:
            return False

        # 检查服务端工具使用信息
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False

        # 查询网络搜索请求计数
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False

        # 若发起了任何网络搜索请求则返回 True
        return web_search_requests > 0

    except (AttributeError, TypeError):
        # 处理响应结构不符合预期的情况
        return False

def openai_websearch_called(response):
    """检测响应中是否使用了 OpenAI 的网络搜索功能。

    参数:
        response: 来自 OpenAI API 的响应对象

    返回:
        若调用了网络搜索返回 True，否则返回 False
    """
    # 检查响应元数据中的工具输出
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False

    # 检查工具输出中是否存在网络搜索调用
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True

    return False


##########################
# Token 上限超限检测组件
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """判断异常是否表明超出了 Token/上下文限制。

    参数:
        exception: 待分析的异常对象
        model_name: 可选的模型名称，用于优化供应商判断

    返回:
        若异常指示 Token 限制超限返回 True，否则返回 False
    """
    error_str = str(exception).lower()

    # 步骤 1：若提供了模型名称，先判断供应商类型
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('qwen') or 'qwen' in model_str:
            provider = 'qwen'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'

    # 步骤 2：检查特定供应商的 Token 超限模式
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'qwen':
        return _check_qwen_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)

    # 步骤 3：若供应商未知，逐一检查所有供应商的超限特征
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_qwen_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )


def _check_qwen_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表明千问 (Qwen/DashScope) 的 Token/上下文长度超限。"""
    exception_type = str(type(exception)).lower()
    class_name = exception.__class__.__name__

    # 1. 判定是否为 DashScope SDK 或通过 OpenAI 兼容端点调用的异常
    is_qwen_or_dashscope = (
            'dashscope' in exception_type or
            'qwen' in exception_type or
            class_name in ['BadRequestError', 'InvalidRequestError', 'ModelCallFailedException']
    )

    if is_qwen_or_dashscope:
        # 阿里云 DashScope / OpenAI 兼容接口常见的超长错误关键字
        qwen_token_keywords = [
            'range of input length should be',
            'maximum context length',
            'context_length_exceeded',
            'token limit',
            'input data exceeds maximum length',
            'tokens exceeded',
            'prompt too long',
            'length of input'
        ]
        if any(keyword in error_str for keyword in qwen_token_keywords):
            return True

    # 2. 检查特定错误码
    if hasattr(exception, 'code'):
        error_code = str(getattr(exception, 'code', '')).lower()
        if error_code in ['context_length_exceeded', 'invalidparameter.length']:
            return True

    return False

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表明 OpenAI 的 Token 超限。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # 检查是否为 OpenAI 异常
    is_openai_exception = (
        'openai' in exception_type.lower() or
        'openai' in module_name.lower()
    )

    # 检查典型的 OpenAI Token 超限错误类型
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']

    if is_openai_exception and is_request_error:
        # 在错误信息中查找 Token 相关的关键字
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True

    # 检查特定的 OpenAI 错误码
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')

        if (error_code == 'context_length_exceeded' or
            error_type == 'invalid_request_error'):
            return True

    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表明 Anthropic 的 Token 超限。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # 检查是否为 Anthropic 异常
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or
        'anthropic' in module_name.lower()
    )

    # 检查 Anthropic 特定的错误模式
    is_bad_request = class_name == 'BadRequestError'

    if is_anthropic_exception and is_bad_request:
        # Anthropic 在 Token 超限时使用特定的错误提示信息
        if 'prompt is too long' in error_str:
            return True

    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表明 Google/Gemini 的 Token 超限。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # 检查是否为 Google/Gemini 异常
    is_google_exception = (
        'google' in exception_type.lower() or
        'google' in module_name.lower()
    )

    # 检查 Google 特有的资源耗尽（Resource Exhaustion）错误
    is_resource_exhausted = class_name in [
        'ResourceExhausted',
        'GoogleGenerativeAIFetchError'
    ]

    if is_google_exception and is_resource_exhausted:
        return True

    # 检查特定的 Google API 资源耗尽模式
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True

    return False

# 注意：以下配置可能已过时或不适用于您的模型，请根据需要进行更新。
MODEL_TOKEN_LIMITS = {
    "qwen": 98304,
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}

def get_model_token_limit(model_string):
    """查询特定模型的 Token 上限。

    参数:
        model_string: 待查询的模型标识符字符串

    返回:
        若在表中找到则返回整数类型的 Token 上限，若未找到则返回 None
    """
    # 遍历已知的模型 Token 上限配置
    if not model_string:
        return None

    model_lower = str(model_string).lower()
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key.lower() in model_lower:
            return token_limit

    # 查找表中未找到该模型
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """通过移除直到最后一条 AI 消息的内容来截断消息历史。

    用于在发生 Token 超限错误时丢弃最近的上下文以进行回滚重试。

    参数:
        messages: 待截断的消息对象列表

    返回:
        截断后的消息列表（截取到最后一条 AI 消息之前，不包含该 AI 消息）
    """
    # 从后向前反向遍历消息，查找最后一条 AI 消息
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # 返回最后一条 AI 消息之前的所有内容（不包含该 AI 消息）
            return messages[:i]

    # 未找到任何 AI 消息，返回原始列表
    return messages

##########################
# 其他杂项工具
##########################

def get_today_str() -> str:
    """获取当前格式化日期，用于在提示词和输出中显示。

    返回:
        类似 'Mon Jan 15, 2024' 格式的人类可读日期字符串
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """从配置中提取值，处理枚举类型（Enum）与 None 值。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """从环境变量或配置中获取指定模型的 API 密钥。"""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("qwen"):
            return api_keys.get("DASHSCOPE_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"):
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("qwen"):
            return os.getenv("DASHSCOPE_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    """从环境变量或配置中获取 Tavily API 密钥。"""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")