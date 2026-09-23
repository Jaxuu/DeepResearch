import asyncio
import os
import sys
import traceback
from dotenv import load_dotenv
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# 获取当前环境，确保子进程能拿到 API Keys
env = os.environ.copy()
load_dotenv(override=True)
env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
env["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
env["SUPERVISOR_MODEL"] = os.getenv("SUPERVISOR_MODEL", "qwen-plus")
env["VISUAL_MODEL"] = os.getenv("VISUAL_MODEL", "qwen-vl-plus")

BASIC_SERVER = "D:/Project/llm-project/open_deep_research/mcp_servers/mcp_basic_tools.py"
ADVANCED_SERVER = "D:/Project/llm-project/open_deep_research/mcp_servers/mcp_advanced_tools.py"


async def call_with_timeout(session, name, arguments, timeout=90.0):
    """给 call_tool 加超时，避免无限卡死。"""
    print(f"\n--- 调用 {name} ---")
    try:
        res = await asyncio.wait_for(
            session.call_tool(name, arguments=arguments),
            timeout=timeout,
        )
        print("🎉 成功返回:\n", res)
        return res
    except asyncio.TimeoutError:
        print(f"❌ {name} 超时 ({timeout}s)，服务器内部可能卡住。")
    except Exception as e:
        print(f"❌ {name} 失败:", repr(e))
        traceback.print_exc()
    return None


async def test_basic_tools():
    basic_server_params = StdioServerParameters(
        command=sys.executable,
        args=[BASIC_SERVER],
        env=env,
    )

    print("\n" + "=" * 40)
    print("🟢 正在拉起子进程: mcp_basic_tools.py")
    try:
        async with stdio_client(basic_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ 握手成功！开始调用 generate_chart_image...")

                await call_with_timeout(
                    session,
                    "generate_chart_image",
                    arguments={
                        "data_context": "Q1: 150k, Q2: 200k, Q3: 180k, Q4: 250k",
                        "visualization_goal": "Create a line chart showing quarterly revenue",
                    },
                    timeout=90.0,
                )
    except Exception as e:
        print("❌ Basic Tools 子进程通信/执行失败:", repr(e))
        traceback.print_exc()


async def test_advanced_tools():
    advanced_server_params = StdioServerParameters(
        command=sys.executable,
        args=[ADVANCED_SERVER],
        env=env,
    )

    print("\n" + "=" * 40)
    print("🔵 正在拉起子进程: mcp_advanced_tools.py")
    try:
        async with stdio_client(advanced_server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ 握手成功！准备执行高级工具...")

                # 2.1 测试 Python 计算
                await call_with_timeout(
                    session,
                    "calculate_with_python",
                    arguments={
                        "data_context": "BYD's 2023 revenue: 602,320,000,000 CNY; Exchange rate: 6.773455; Tesla's 2023 revenue: $96,773,000,000",
                        "calculation_goal": "Convert BYD's 2023 revenue to USD, compute percentage Tesla exceeds BYD.",
                    },
                    timeout=90.0,
                )

                # 2.2 测试 PDF 提取
                await call_with_timeout(
                    session,
                    "extract_from_long_document",
                    arguments={
                        "url": "https://example.com",
                        "extraction_query": "What is the main purpose of this domain?",
                    },
                    timeout=90.0,
                )

                # 2.3 测试视觉分析
                await call_with_timeout(
                    session,
                    "analyze_webpage_visual_layout",
                    arguments={
                        "url": "https://example.com",
                        "specific_question": "What is the background color of this webpage?",
                    },
                    timeout=90.0,
                )
    except Exception as e:
        print("❌ Advanced Tools 子进程通信崩溃:", repr(e))
        traceback.print_exc()


async def main():
    print("🚀 开始进行真实的 MCP 子进程通信测试...")
    # 先只测 basic，定位清楚再放开 advanced
    await test_basic_tools()
    await test_advanced_tools()


if __name__ == "__main__":
    asyncio.run(main())