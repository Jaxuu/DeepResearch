"""Deep Research 智能体的实用工具与辅助函数。"""

import asyncio
import json
import logging
import os
import httpx
import warnings
import re
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional

import urllib.parse
from tavily import AsyncTavilyClient
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_experimental.utilities import PythonREPL

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.state import ResearchComplete

##########################
# Tavily 搜索工具组件
##########################
@tool(description="Search the web for news, facts, and public data. Returns titles, URLs, and concise snippets.")
async def web_search(
        queries: List[str],
        max_results: Annotated[int, InjectedToolArg] = 5,
        topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
        config: RunnableConfig = None
) -> str:
    """仅检索网页元数据与摘要片段，不下载全文。"""
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))

    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=False,  # 关闭全量下载
            topic=topic
        )
        for query in queries
    ]
    search_results = await asyncio.gather(*search_tasks)

    # 简单去重
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result

    if not unique_results:
        return "No valid search results found."

    formatted_output = "Search Results (Snippets only):\n\n"
    for i, (url, res) in enumerate(unique_results.items()):
        formatted_output += f"[{i + 1}] Title: {res.get('title', 'N/A')}\n"
        formatted_output += f"    URL: {url}\n"
        formatted_output += f"    Snippet: {res.get('content', '')}\n\n"

    return formatted_output


@tool(
    description="Fetch and parse the full text of a specific URL into clean Markdown. Use this when snippets are insufficient.")
async def fetch_webpage(url: str) -> str:
    """使用远端 Jina Reader 服务直接提取网页正文 Markdown。"""
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "X-Timeout": "15"  # 15秒超时设置
    }

    # 若在环境变量中配置了 JINA_API_KEY，则自动携带；没有配置也能直接免密访问
    jina_key = os.getenv("JINA_API_KEY")
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(jina_url, headers=headers)
            if response.status_code == 200:
                content = response.text
                # 安全截断，防止目标网页过长（保留约 15000 字符，足够提取长篇专业数据）
                if len(content) > 15000:
                    content = content[:15000] + "\n\n[...正文超长，已安全截断...]"
                return content
            else:
                return f"Failed to fetch content from {url}. Status code: {response.status_code}"
    except Exception as e:
        return f"Error reading URL {url}: {str(e)}"


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
# 模拟企业级架构中的“连接池单例”
# 保证跨越 Supervisor 和 Sub-agent 多个节点时，网络通道持久存活
_mcp_client = None

async def load_mcp_tools(
        config,
        existing_tool_names: set[str],
):
    """加载标准的远端 MCP 工具 (基于官方 MultiServerMCPClient 封装)"""
    global _mcp_client

    # 混合 MCP 配置字典
    mcp_config = {
        "industrial_rag": {
            "transport": "sse",
            "url": "http://127.0.0.1:8080/sse"
        },
        "sqlite_db": {
            "transport": "sse",
            "url": "http://127.0.0.1:8001/sse"
        }
    }

    try:
        # 如果是第一次请求，初始化官方高阶客户端
        if _mcp_client is None:
            _mcp_client = MultiServerMCPClient(mcp_config)
            # 对于 0.1.0 版本的 langchain-mcp-adapters，直接使用其内部 session

        # 一行代码拉取所有可用工具
        available_tools = await _mcp_client.get_tools()

        # 调试信息：确保拿到工具
        print(f"\n[🔌 MCP 挂载成功 (SSE 模式)] 获取私有库工具: {[t.name for t in available_tools]}")

    except Exception as e:
        print(f"\n[⚠️ MCP 连接失败] 请检查 Server 是否在 8080 端口启动: {e}")
        return []

    # 过滤重复工具
    configured_tools = []
    for mcp_tool in available_tools:
        if mcp_tool.name not in existing_tool_names:
            configured_tools.append(mcp_tool)

    return configured_tools

##########################
# Skill工具组件
##########################

# 初始化 REPL 单例
_python_repl = PythonREPL()

@tool(
    description="A highly capable quantitative analysis skill. Pass in raw data (like financial tables or stats) and a specific calculation goal. It will autonomously write, execute, and debug code to find the answer.")
async def quantitative_analysis_skill(data_context: str, calculation_goal: str, config: RunnableConfig = None) -> str:
    """
    Skill: 动态量化分析沙箱。
    向内部大模型隐藏复杂的代码生成与执行逻辑，对外仅暴露自然语言接口。

    Args:
        data_context: 包含所有计算所需原始数值的文本片段（如财报段落、表格摘录）。
        calculation_goal: 明确的计算指令（例如："Calculate the profit margin and convert it to a percentage"）。
    """
    # 动态获取配置中的模型资源，为 Skill 内部的“隐形工头”提供算力
    from open_deep_research.configuration import Configuration
    from open_deep_research.deep_researcher import configurable_model

    configurable = Configuration.from_runnable_config(config)
    skill_model = configurable_model.with_config({
        "model": configurable.compression_model,  # 使用速度快、成本低的小模型写代码即可
        "max_tokens": 1000,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
    })

    system_prompt = f"""You are a Python Data Analyst. Your job is to achieve the calculation goal based on the data.
                        Data Context:
                        {data_context}
                        
                        Goal: {calculation_goal}
                        
                        Write a python script to compute this. Print the exact final numerical result clearly.
                        Return ONLY valid python code wrapped in ```python```. Do not explain."""

    max_retries = 5
    current_prompt = system_prompt

    # SOP 闭环：生成 -> 执行 -> 校验验 -> 自愈
    for attempt in range(max_retries):
        try:
            # 1. 内部生成代码
            response = await skill_model.ainvoke([HumanMessage(content=current_prompt)])

            # 2. 提取代码块
            code_match = re.search(r"```python\n(.*?)\n```", response.content, re.DOTALL)
            code = code_match.group(1) if code_match else response.content.replace("```python", "").replace("```", "")

            # 3. 沙箱执行
            output = _python_repl.run(code)

            # 4. 结果校验
            if "Error" in output or "Exception" in output or "Traceback" in output:
                # 触发自愈逻辑，将错误喂回给模型
                current_prompt += f"\n\nPrevious attempt failed with error:\n{output}\nPlease fix the code and try again."
                continue

            if not output.strip():
                current_prompt += f"\n\nPrevious attempt ran successfully but printed nothing. You MUST use print() to output the final result."
                continue

            # 成功则直接将结果抛给外层的 Researcher
            return f"[✅ Skill Verified Calculation] Successfully executed quantitative analysis.\nResult details:\n{output.strip()}"

        except Exception as e:
            current_prompt += f"\n\nSystem error occurred: {str(e)}\nFix the issue and rewrite."

    return "[❌ Skill Failed] Unable to compute the requested data after multiple attempts. You may need to rely on the raw text."


@tool(
    description="""A Sub-RAG skill for deep mining of EXTREMELY LONG documents (e.g., annual reports, SEC filings, PDFs, long academic papers). 
    Use this when 'fetch_webpage' is not enough due to length limits. 
    Pass the specific 'url' and a highly detailed 'extraction_query'. It will read the entire document in the background and extract the exact answer."""
)
async def long_doc_mining_skill(url: str, extraction_query: str, config: RunnableConfig = None) -> str:
    """
    长文/PDF 深度挖掘技能。
    突破 15000 字符限制，使用内存级 BM25 检索目标块，交由小模型精准提纯。
    """
    # 1. 全量获取文档 (无字符截断)
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "X-Timeout": "30"  # 延长超时时间以应对巨型 PDF
    }

    # 自动携带 Jina Key（如果有配置）
    jina_key = os.getenv("JINA_API_KEY")
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"

    try:
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
            response = await client.get(jina_url, headers=headers)
            if response.status_code != 200:
                return f"[❌ Skill Failed] Unable to fetch {url}. Status: {response.status_code}"
            full_content = response.text
    except Exception as e:
        return f"[❌ Skill Failed] Network error fetching {url}: {str(e)}"

    # 2. 内存级快速文本切片
    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=300)
    chunks = splitter.split_text(full_content)

    # 如果文档并不长，无需 RAG，直接取全部
    if len(chunks) <= 3:
        context = full_content
    else:
        # 3. 本地 BM25 召回最相关的 Top-5 块 (不耗费任何 API Token)
        docs = [Document(page_content=chunk) for chunk in chunks]
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = 5
        top_docs = retriever.invoke(extraction_query)
        context = "\n\n---\n\n".join([d.page_content for d in top_docs])

    # 4. 唤醒便宜的小模型执行阅读理解与提纯
    from open_deep_research.configuration import Configuration
    configurable = Configuration.from_runnable_config(config)

    # 就地初始化模型，防止与 deep_researcher 产生循环引用
    skill_model = init_chat_model(
        model=configurable.compression_model,
        api_key=get_api_key_for_model(configurable.compression_model, config),
        max_tokens=1500
    )

    prompt = f"""You are an elite Data Extraction Analyst.
                    <Source URL>
                    {url}
                    </Source URL>
                    
                    <Your Task>
                    Extract precise information to answer this query: {extraction_query}
                    </Your Task>
                    
                    <Retrieved Context>
                    {context}
                    </Retrieved Context>
                    
                    <Instructions>
                    1. Synthesize the answer ONLY using the <Retrieved Context>. Do not hallucinate.
                    2. If the context does not contain the answer, explicitly state: "The requested information could not be found in the document."
                    3. Include specific numbers, dates, and metrics if present.
                    </Instructions>
                """

    try:
        res = await skill_model.ainvoke([HumanMessage(content=prompt)])
        return f"[✅ Long-Doc Mining Success]\nExtracted from {url}:\n{res.content}"
    except Exception as e:
        return f"[❌ Skill Failed] LLM extraction error: {str(e)}"


@tool(
    description="""A specialized Data Visualization skill. 
    Use this when you need to create charts, graphs, or diagrams (e.g., comparing revenue, visualizing market share, showing timelines).
    Pass the raw numerical data or relationships as 'data_context', and state exactly what kind of chart you want in 'visualization_goal'."""
)




@tool
async def data_visualization_skill(data_context: str, visualization_goal: str, config: RunnableConfig = None) -> str:
    """
    调用外部工业级图表 QuickChart API 生成真实图片链接
    """
    from open_deep_research.configuration import Configuration
    from langchain.chat_models import init_chat_model

    configurable = Configuration.from_runnable_config(config)
    skill_model = init_chat_model(
        model=configurable.compression_model,
        temperature=0.1,
        max_tokens=1500
    )

    # 工业界做法：让模型只输出标准的 Chart.js JSON 配置，这种 JSON 模型极难出错
    prompt = f"""You are a Data Visualization API expert. Create a Chart.js JSON configuration for the following goal.
                <Data Context>
                {data_context}
                </Data Context>
                <Goal>
                {visualization_goal}
                </Goal>
                
                <Strict Rules>
                1. Output ONLY a valid JSON object representing a Chart.js configuration.
                2. NO markdown formatting, NO backticks, NO explanations.
                3. Example of valid output:
                {{
                  "type": "bar",
                  "data": {{
                    "labels": ["Apple", "Microsoft"],
                    "datasets": [{{ "label": "Revenue", "data": [383, 211] }}]
                  }}
                }}
                </Strict Rules>
            """

    for attempt in range(3):
        try:
            res = await skill_model.ainvoke([HumanMessage(content=prompt)])
            raw_content = res.content

            # 1. 兼容 LangChain 内容块列表解析 (处理 [{"type":"text", "text":"..."}] 的情况)
            chart_text = ""
            if isinstance(raw_content, list):
                for block in raw_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        chart_text += block.get("text", "")
            else:
                chart_text = str(raw_content)

            # 2. 精准提取 JSON (寻找第一对 { 和最后一对 } 之间的内容，过滤大模型的废话)
            match = re.search(r'(\{.*\})', chart_text, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in the model response.")
            clean_json_str = match.group(1)

            # 验证 JSON 是否合法
            chart_json = json.loads(clean_json_str)

            # 3. 工业级做法：POST 到 QuickChart 获取短链接
            req_data = json.dumps({"chart": chart_json}).encode('utf-8')
            req = urllib.request.Request(
                "https://quickchart.io/chart/create",
                data=req_data,
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req) as response:
                res_body = json.loads(response.read().decode('utf-8'))
                if res_body.get("success"):
                    short_url = res_body.get("url")
                    # 返回干净清爽的 Markdown 图片链接
                    return f"[✅ Visualization Generated]\n![{visualization_goal}]({short_url})"
                else:
                    raise Exception("QuickChart API failed to generate short URL.")

        except Exception as e:
            if attempt == 2:
                return f"[❌ Skill Failed] {str(e)}"
            prompt += "\nError: Output must be pure JSON."

    return "[❌ Skill Failed]"

# 在这里注册所有可用的复合技能
AVAILABLE_SKILLS = {
    "quantitative_analysis": quantitative_analysis_skill,
    "long_doc_mining": long_doc_mining_skill,
    "data_visualization": data_visualization_skill
}

def get_active_skills(assigned_skill_names: List[str]) -> List[Any]:
    """
    根据状态中传入的技能名称列表，动态映射并返回对应的技能实体函数。
    """
    active_skills = []
    if not assigned_skill_names:
        return active_skills

    for skill_name in assigned_skill_names:
        skill_func = AVAILABLE_SKILLS.get(skill_name)
        if skill_func:
            active_skills.append(skill_func)
        else:
            print(f"\n[⚠️ 技能告警] 未找到名为 '{skill_name}' 的技能配置，跳过挂载。")

    return active_skills

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
        search_tool = web_search
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

    # 若选择 Tavily，则挂载新改造的 web_search
    if search_api == SearchAPI.TAVILY:
        tools.append(web_search)
        tools.append(fetch_webpage)
    else:
        search_tools = await get_search_tool(search_api)
        tools.extend(search_tools)

    # 记录已有工具名称以防冲突
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search")
        for tool in tools
    }

    # # 若有配置则添加 MCP 工具
    # mcp_tools = await load_mcp_tools(config, existing_tool_names)
    # tools.extend(mcp_tools)

    return tools

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
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
        else:
            provider = 'qwen'

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