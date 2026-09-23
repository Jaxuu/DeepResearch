"""Deep Research 智能体的实用工具与辅助函数。"""
import asyncio
import os
import re
import sys
import httpx
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional
from pydantic import create_model
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from langchain_core.messages import AIMessage, HumanMessage, MessageLikeRepresentation
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

from open_deep_research.state import ResearchComplete

##########################
# 思考工具
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


@tool(
    description="""Fetch and parse the full text of a specific URL into clean Markdown. 
    RULE: ONLY call this on 1-2 high-authority URLs when snippets lack depth (e.g., financial tables, detailed specs). NEVER fetch every URL."""
)
async def fetch_webpage(url: str) -> str:
    """使用远端 Jina Reader 服务直接提取网页正文 Markdown。"""
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "X-Timeout": "15",  # 15秒超时设置
        "X-Return-Format": "markdown"  # 明确要求标准 Markdown
    }

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(jina_url, headers=headers)
            if response.status_code == 200:
                content = response.text
                # 安全截断，防止目标网页过长
                if len(content) > 50000:
                    content = content[:50000] + "\n\n[...正文超长，已安全截断...]"
                return content
            else:
                return f"Failed to fetch content from {url}. Status code: {response.status_code}"
    except Exception as e:
        return f"Error reading URL {url}: {str(e)}"

##########################
# MCP 客户端管理
##########################

# --- stdio server：手动长连接 ---
_stdio_sessions = {}          # {server_name: {"session": ..., "exit_stack": ...}}
_stdio_sessions_lock = asyncio.Lock()

# --- 远程 server（SSE / streamable_http）：继续用 MultiServerMCPClient ---
_mcp_client = None
_mcp_client_lock = asyncio.Lock()

# --- 工具缓存 ---
_cached_mcp_tools = None
_mcp_tools_lock = asyncio.Lock()

def _build_stdio_env() -> dict:
    """构建传给 stdio 子进程的环境变量（白名单注入）。"""
    env = os.environ.copy()
    load_dotenv(override=True)
    env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    env["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "")
    env["SUPERVISOR_MODEL"] = os.getenv("SUPERVISOR_MODEL", "qwen-plus")
    env["VISUAL_MODEL"] = os.getenv("VISUAL_MODEL", "qwen-vl-plus")
    env["PYTHONIOENCODING"] = "utf-8"
    return env

async def _start_stdio_session(name: str, script_path: str, env: dict):
    """启动一个 stdio 子进程并建立长连接 session。"""
    print(f"[MCP] 启动 stdio 子进程: {name}", file=sys.stderr, flush=True)
    exit_stack = AsyncExitStack()
    try:
        read, write = await exit_stack.enter_async_context(
            stdio_client(StdioServerParameters(
                command=sys.executable,
                args=[script_path],
                env=env,
            ))
        )
        session = await exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        print(f"[MCP] {name} 握手成功", file=sys.stderr, flush=True)
        _stdio_sessions[name] = {"session": session, "exit_stack": exit_stack}
    except Exception as e:
        print(f"[MCP] {name} 启动失败: {e!r}", file=sys.stderr, flush=True)
        await exit_stack.aclose()
        raise

async def _ensure_stdio_sessions():
    """确保两个 stdio 子进程都已启动（只启动一次）。"""
    if _stdio_sessions:
        return
    async with _stdio_sessions_lock:
        if _stdio_sessions:
            return
        env = _build_stdio_env()
        await _start_stdio_session(
            "basic_tools",
            "D:/Project/llm-project/open_deep_research/mcp_servers/mcp_basic_tools.py",
            env,
        )
        await _start_stdio_session(
            "advanced_tools",
            "D:/Project/llm-project/open_deep_research/mcp_servers/mcp_advanced_tools.py",
            env,
        )

# JSON Schema 类型 → Python 类型
_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

def _json_schema_to_pydantic(schema: dict, model_name: str):
    """把 MCP 的 JSON Schema 转成 Pydantic 模型类。"""
    if not isinstance(schema, dict):
        schema = {}

    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])

    fields = {}
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            prop_schema = {}

        json_type = prop_schema.get("type", "string")
        # 处理 type 是 list 的情况（如 ["string", "null"]）
        if isinstance(json_type, list):
            json_type = next((t for t in json_type if t != "null"), "string")

        py_type = _JSON_TYPE_MAP.get(json_type, Any)

        if prop_name in required:
            fields[prop_name] = (py_type, ...)      # 必填
        else:
            fields[prop_name] = (py_type, None)     # 可选

    # 如果没有字段，生成一个空模型（避免 StructuredTool 报错）
    if not fields:
        fields["_dummy"] = (Optional[str], None)

    return create_model(model_name, **fields)


def _make_langchain_tool_from_session(server_name: str, tool_def):
    """把一个 MCP tool 定义包装成 LangChain Tool（通过长连接 session 调用）。"""
    tool_name = tool_def.name
    description = tool_def.description or ""
    input_schema = getattr(tool_def, "inputSchema", None) or {}

    # 动态生成 Pydantic 模型
    try:
        args_model = _json_schema_to_pydantic(input_schema, f"{tool_name}_Args")
    except Exception as e:
        print(f"[MCP] {tool_name} schema 转换失败: {e!r}", file=sys.stderr, flush=True)
        raise

    async def _ainvoke(**kwargs):
        # 去掉 _dummy
        kwargs.pop("_dummy", None)
        session = _stdio_sessions[server_name]["session"]
        result = await session.call_tool(tool_name, arguments=kwargs)
        texts = []
        for block in result.content:
            if getattr(block, "type", None) == "text":
                texts.append(block.text)
        return "\n".join(texts) if texts else str(result)

    return StructuredTool(
        name=tool_name,
        description=description,
        args_schema=args_model,
        coroutine=_ainvoke,
    )


async def _get_stdio_tools():
    """从两个长连接 session 拉取工具列表。"""
    await _ensure_stdio_sessions()
    tools = []
    for server_name, info in _stdio_sessions.items():
        listed = await info["session"].list_tools()
        for t in listed.tools:
            tools.append(_make_langchain_tool_from_session(server_name, t))
    return tools


async def cleanup_mcp_sessions():
    """程序退出时调用，关闭所有 stdio 长连接。"""
    for name, info in list(_stdio_sessions.items()):
        try:
            await info["exit_stack"].aclose()
            print(f"[MCP] {name} 已关闭", file=sys.stderr, flush=True)
        except Exception:
            pass
    _stdio_sessions.clear()


async def get_or_create_mcp_client():
    """获取或初始化全局 MCP 客户端单例（只负责远程 server）。"""
    global _mcp_client
    if _mcp_client is not None:
        return _mcp_client
    async with _mcp_client_lock:
        if _mcp_client is not None:
            return _mcp_client

        load_dotenv(override=True)
        # ⚠️ 这里只放远程 server。两个 stdio server 已经挪到长连接管理。
        mcp_config = {
            # "industrial_rag": {
            #     "transport": "sse",
            #     "url": "http://127.0.0.1:8080/sse"
            # },
            # "sqlite_db": {
            #     "transport": "sse",
            #     "url": "http://127.0.0.1:8001/sse"
            # },
            "tavily_remote_search": {
                "transport": "streamable_http",
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY', '')}"
            }
        }
        _mcp_client = MultiServerMCPClient(mcp_config)
    return _mcp_client


async def get_mcp_tools_cached():
    """获取 MCP 工具（带缓存，只拉一次）——合并 stdio 长连接工具和远程工具。"""
    global _cached_mcp_tools
    if _cached_mcp_tools is not None:
        return _cached_mcp_tools
    async with _mcp_tools_lock:
        if _cached_mcp_tools is not None:
            return _cached_mcp_tools

        all_tools = []

        # 1. stdio 长连接工具
        try:
            stdio_tools = await _get_stdio_tools()
            print(f"[MCP] stdio 工具加载完成: {len(stdio_tools)} 个", file=sys.stderr, flush=True)
            all_tools.extend(stdio_tools)
        except Exception as e:
            print(f"[MCP] stdio 工具加载失败: {e!r}", file=sys.stderr, flush=True)

        # 2. 远程 MCP 工具
        try:
            client = await get_or_create_mcp_client()
            print("[MCP] 拉取远程工具...", file=sys.stderr, flush=True)
            remote_tools = await asyncio.wait_for(client.get_tools(), timeout=60.0)
            print(f"[MCP] 远程工具加载完成: {len(remote_tools)} 个", file=sys.stderr, flush=True)
            all_tools.extend(remote_tools)
        except Exception as e:
            print(f"[MCP] 远程工具加载失败: {e!r}", file=sys.stderr, flush=True)

        _cached_mcp_tools = all_tools
    return _cached_mcp_tools




##########################
# 动态感知Tools
##########################

# 1. MCP 远程工具白名单
ALLOWED_MCP_TOOLS = {
    "tavily_search",               # Tavily 提供的标准搜索
    "search_equipment_knowledge",  # 你的 RAG 检索
    "query_erp_database",          # 你的 ERP 查询
    "generate_chart_image",        # 生成图表图片
    "calculate_with_python",       # Python 计算
    "extract_from_long_document",  # 从长文档中提取信息
    "analyze_webpage_visual_layout",  # 分析网页视觉布局
}

# 2. 本地原生工具注册表 (Native Tools Registry)
NATIVE_TOOLS = [
    fetch_webpage
]

RESEARCH_TOOLS = [
    think_tool,
    tool(ResearchComplete)
]
def _tool_name(t):
    return getattr(t, "name", getattr(t, "__name__", str(t)))
RESEARCH_TOOL_NAMES = {_tool_name(t) for t in RESEARCH_TOOLS}


async def get_all_tools(config: RunnableConfig):
    """主程序拉取可用工具的唯一入口"""
    tools = []

    # 1. 挂载研究控制工具
    tools.extend(RESEARCH_TOOLS)

    # 2. 挂载本地原生研究工具
    tools.extend(NATIVE_TOOLS)

    # 3. 动态拉取并过滤远端 MCP 工具
    try:
        mcp_tools = await get_mcp_tools_cached()
        # 白名单物理拦截，剔除所有未授权的工具
        filtered_mcp_tools = [t for t in mcp_tools if t.name in ALLOWED_MCP_TOOLS]
        tools.extend(filtered_mcp_tools)
    except Exception as e:
        # 将 ExceptionGroup 里的真实子异常剥离出来打印
        print(f"\n[⚠️ MCP 告警] 获取扩展工具失败: {str(e)}")
        if hasattr(e, 'exceptions'):
            for sub_e in e.exceptions:
                print(f"   -> 具体崩溃原因: {repr(sub_e)}")
    return tools

async def get_tool_catalog_for_supervisor() -> str:
    """动态工具发现。生成系统和远程 MCP 工具的说明字符串，供直接注入提示词。"""
    catalog_info = "Available Capabilities in System & Remote MCP Registry:\n\n"

    # 1：加载本地原生能力
    for t in NATIVE_TOOLS:
        catalog_info += f"- Tool Name: `{t.name}`\n  Description: {t.description}\n\n"

    # 2：加载远程 MCP 能力
    try:
        available_tools = await get_mcp_tools_cached()

        # 通过白名单过滤工具名录
        filtered_tools = [t for t in available_tools if t.name in ALLOWED_MCP_TOOLS]
        if filtered_tools:
            for t in filtered_tools:
                catalog_info += f"- Tool Name: `{t.name}`\n  Description: {t.description}\n\n"
        else:
            catalog_info += "(No MCP integrations currently active)\n\n"
    except Exception as e:
        catalog_info += f"(Failed to load MCP integrations: {str(e)})\n\n"

    return catalog_info

##########################
# 动态读取Skills
##########################
"""
Agent Skill 管理模块。
负责从文件系统中加载 Markdown 格式的 SOP 操作手册，并向大模型注入上下文。
"""
def _find_skills_dir() -> str:
    """从当前文件向上查找，定位项目根目录下的 skills 文件夹。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        candidate = os.path.join(here, "skills")
        if os.path.isdir(candidate):
            return candidate
        here = os.path.dirname(here)
    # 兜底：按原相对路径猜测
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills")

SKILLS_DIR = _find_skills_dir()

def _load_all_skills() -> Dict[str, str]:
    """
    读取 skills 目录下每个子文件夹中的 SKILL.md。
    返回 {skill_id: 完整文件内容}，skill_id 取文件夹名。
    """
    skills: Dict[str, str] = {}
    if not os.path.exists(SKILLS_DIR):
        return skills

    for entry in os.listdir(SKILLS_DIR):
        skill_dir = os.path.join(SKILLS_DIR, entry)
        skill_file = os.path.join(skill_dir, "SKILL.md")
        if os.path.isdir(skill_dir) and os.path.exists(skill_file):
            with open(skill_file, "r", encoding="utf-8") as f:
                skills[entry] = f.read()
    return skills


def _parse_frontmatter(content: str) -> Dict[str, str]:
    """
    轻量解析 SKILL.md 顶部的 YAML frontmatter、真实的 H1 标题和依赖的底层工具。
    只提取 name / description / title / tools，不引入 pyyaml 依赖。
    """
    meta: Dict[str, str] = {}

    # 1. 提取 YAML frontmatter (提取 name 和 description)
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if match:
        current_key = None
        buffer: List[str] = []
        for line in match.group(1).split("\n"):
            kv = re.match(r'^([A-Za-z0-9_-]+)\s*:\s*(.*)$', line)
            if kv:
                if current_key is not None:
                    meta[current_key] = " ".join(buffer).strip()
                current_key = kv.group(1).strip()
                buffer = [kv.group(2).strip()]
            elif current_key is not None:
                buffer.append(line.strip())
        if current_key is not None:
            meta[current_key] = " ".join(buffer).strip()

        # 去掉包裹引号
        for k, v in meta.items():
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                meta[k] = v[1:-1]

    # 2. 提取真正的 Markdown 标题 (匹配第一个 # 开头的行)
    h1_match = re.search(r'^#\s+(.*?)$', content, re.MULTILINE)
    if h1_match:
        meta['title'] = h1_match.group(1).strip()

    # 3. 精准提取 "依赖的底层工具" 内容 (提取到下一个 ## 标题为止)
    tools_match = re.search(r'##\s*依赖的底层工具.*?\n(.*?)(?=\n##\s|$)', content, re.DOTALL | re.IGNORECASE)
    if tools_match:
        meta['required_tools_text'] = tools_match.group(1).strip()
    else:
        meta['required_tools_text'] = "None explicitly listed."

    return meta


def get_skill_catalog_for_supervisor() -> str:
    """
    为 Supervisor 生成技能目录 (Catalog)。
    从 frontmatter 和 H1 提取信息，并将强制绑定的底层工具暴露给主管。
    """
    skills = _load_all_skills()
    if not skills:
        return "No advanced skills available."

    catalog = "Available Standard Operating Procedures (Skills):\n\n"

    for skill_id, content in skills.items():
        meta = _parse_frontmatter(content)

        # 优先使用解析出的 H1 标题，如果没有则降级使用 name 或 skill_id
        real_title = meta.get("title", meta.get("name", skill_id))
        purpose = meta.get("description", "No description provided.")
        req_tools = meta.get("required_tools_text", "")

        catalog += f"- **Skill ID**: `{skill_id}`\n"
        catalog += f"  - **Title**: {real_title}\n"
        catalog += f"  - **Description**: {purpose}\n"

        # 将技能需要的工具列表加入提示词
        if req_tools and req_tools != "None explicitly listed.":
            # 增加缩进对齐，使排版更美观
            tools_indented = "\n    ".join(req_tools.split('\n'))
            catalog += f"  - **[CRITICAL] Required Tools for this skill**:\n    {tools_indented}\n\n"
        else:
            catalog += "\n"

    return catalog

def get_skill_instructions_for_researcher(required_skills: List[str]) -> str:
    """
    为 Researcher 生成技能执行指南。
    根据 Supervisor 派发的技能列表，注入完整 SOP（已剥离 frontmatter）。
    """
    if not required_skills:
        return ""

    all_skills = _load_all_skills()
    instructions = "\n\n" + "=" * 40 + "\n"
    instructions += "🚨 MANDATORY STANDARD OPERATING PROCEDURES (SOP) 🚨\n"
    instructions += "You have been assigned specific skills for this task. You MUST strictly follow the operating procedures below:\n\n"

    loaded_count = 0
    for skill_id in required_skills:
        if skill_id in all_skills:
            body = all_skills[skill_id]
            instructions += f"--- START OF SKILL: {skill_id} ---\n"
            instructions += body
            instructions += "\n--- END OF SKILL ---\n\n"
            loaded_count += 1

    if loaded_count == 0:
        return ""

    return instructions

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
        # if model_name.startswith("openai:deepseek"):
        #     return api_keys.get("DEEPSEEK_API_KEY")
        elif model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("qwen"):
            return api_keys.get("DASHSCOPE_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        # if model_name.startswith("openai:deepseek"):
        #     return os.getenv("DEEPSEEK_API_KEY")
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