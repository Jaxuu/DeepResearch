import asyncio
import os
import sys
import traceback
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    print("🚀 [1] 准备环境变量与配置...")
    env = os.environ.copy()
    load_dotenv(override=True)

    env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    env["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "")
    env["SUPERVISOR_MODEL"] = os.getenv("SUPERVISOR_MODEL", "qwen-plus")
    env["VISUAL_MODEL"] = os.getenv("VISUAL_MODEL", "qwen-vl-plus")
    env["PYTHONIOENCODING"] = "utf-8"

    mcp_config = {
        "basic_tools": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["D:/Project/llm-project/open_deep_research/mcp_servers/mcp_basic_tools.py"],
            "env": env
        },
        "advanced_tools": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["D:/Project/llm-project/open_deep_research/mcp_servers/mcp_advanced_tools.py"],
            "env": env
        }
    }

    print("🚀 [2] 实例化 MultiServerMCPClient (根据 v0.1.0 规范，此处不再强制握手)...")
    client = MultiServerMCPClient(mcp_config)

    print("\n🔍 [3] 测试: 调用 get_tools() (此时底层会自动拉起子进程并连接)...")
    try:
        # 这个操作会隐式建立所有子进程连接，并返回 LangChain Tool 对象
        tools = await asyncio.wait_for(client.get_tools(), timeout=30.0)
        print(f"✅ 成功获取到 {len(tools)} 个工具: {[t.name for t in tools]}")
    except Exception as e:
        print("❌ 获取工具失败:")
        traceback.print_exc()
        return

    print("\n⚙️ [4] 测试: 工具直接调用 (calculate_with_python)...")
    # 从返回的工具列表中找到我们要测的那个工具
    calc_tool = next((t for t in tools if t.name == "calculate_with_python"), None)

    if calc_tool:
        try:
            res = await asyncio.wait_for(
                calc_tool.ainvoke({
                    "data_context": "BYD's 2023 revenue: 602,320,000,000 CNY",
                    "calculation_goal": "Divide by 10000"
                }),
                timeout=45.0
            )
            print("🎉 工具调用成功:\n", res)
        except asyncio.TimeoutError:
            print("❌ 工具调用超时卡死！")
        except Exception as e:
            print("❌ 工具调用报错:")
            traceback.print_exc()
    else:
        print("❌ 未在工具列表中找到 calculate_with_python")

    print("\n✅ 测试结束。")


if __name__ == "__main__":
    asyncio.run(main())