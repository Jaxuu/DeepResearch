import sys
import os
import re
import json
import asyncio
import httpx

from mcp.server.fastmcp import FastMCP
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


def _trace(msg: str):
    """同时写 stderr 和文件，确保主流程卡死时也能看到。"""
    line = f"[TRACE] {msg}\n"
    # 写 stderr（如果父进程转发了就能看到）
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except Exception:
        pass
    # 写文件（永远能看到）
    try:
        with open("D:/mcp_trace.log", "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


_trace("========== 子进程启动 ==========")
_trace(f"cwd={os.getcwd()}")
_trace(f"OPENAI_API_KEY set? {bool(os.getenv('OPENAI_API_KEY'))}")
_trace(f"OPENAI_BASE_URL = {os.getenv('OPENAI_BASE_URL')}")
_trace(f"SUPERVISOR_MODEL = {os.getenv('SUPERVISOR_MODEL')}")

# 初始化 FastMCP 服务器
mcp = FastMCP("ChartGenerationTool")


@mcp.tool()
async def generate_chart_image(data_context: str, visualization_goal: str) -> str:
    """
    Generate an industrial-grade chart image (e.g., bar, line, pie, scatter) via QuickChart API.
    Use this tool to visualize raw data. Pass the data context and your specific charting goal.
    Returns a Markdown formatted image link.
    """
    _trace(">>> generate_chart_image ENTER")

    model_name = os.getenv("SUPERVISOR_MODEL", "qwen-plus")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    _trace(f"model_name={model_name}, base_url={base_url}, api_key_set={bool(api_key)}")

    _trace(">>> before init_chat_model")
    try:
        skill_model = init_chat_model(
            model=model_name,
            model_provider="openai",
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,
            max_tokens=1500,
        )
    except Exception as e:
        _trace(f"!!! init_chat_model FAILED: {e!r}")
        return f"[❌ Tool Failed] init_chat_model error: {e}"
    _trace(">>> after init_chat_model")

    prompt = f"""You are a Data Visualization API expert. Create a Chart.js JSON configuration for the following goal.
                <Data Context>
                {data_context}
                </Data Context>
                <Goal>
                {visualization_goal}
                </Goal>

                <Strict Rules>
                1. Output ONLY a valid JSON object representing a Chart.js configuration.
                2. Choose the most appropriate chart type based on the <Goal>.
                3. NO markdown formatting, NO backticks, NO explanations.
                </Strict Rules>
            """

    for attempt in range(3):
        _trace(f">>> attempt {attempt + 1}: calling ainvoke")
        try:
            res = await asyncio.wait_for(
                skill_model.ainvoke([HumanMessage(content=prompt)]),
                timeout=60.0,
            )
            _trace(">>> ainvoke returned")
            raw_content = res.content

            chart_text = ""
            if isinstance(raw_content, list):
                for block in raw_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        chart_text += block.get("text", "")
            else:
                chart_text = str(raw_content)

            match = re.search(r'(\{.*\})', chart_text, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in the model response.")

            chart_json = json.loads(match.group(1))
            _trace(">>> chart JSON parsed, calling QuickChart")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://quickchart.io/chart/create",
                    json={"chart": chart_json},
                )
                response.raise_for_status()
                res_body = response.json()

                if res_body.get("success"):
                    short_url = res_body.get("url")
                    _trace(f">>> SUCCESS: {short_url}")
                    return f"[✅ Visualization Generated]\n![Data_Visualization_Chart]({short_url})"
                else:
                    raise Exception("QuickChart API failed to generate short URL.")

        except asyncio.TimeoutError:
            _trace(f"!!! attempt {attempt + 1} TIMEOUT (ainvoke > 60s)")
            if attempt == 2:
                return "[❌ Tool Failed] Model call timed out after 60s."
            prompt += "\nError: Output must be valid JSON."
        except Exception as e:
            _trace(f"!!! attempt {attempt + 1} ERROR: {e!r}")
            if attempt == 2:
                return f"[❌ Tool Failed] Could not generate chart: {str(e)}"
            prompt += "\nError: Output must be valid JSON."

    return "[❌ Tool Failed] Exceeded max retries."


if __name__ == "__main__":
    _trace(">>> basic starting mcp.run(transport='stdio')")
    try:
        mcp.run(transport="stdio")
    except BaseException as e:
        _trace(f"!!! mcp.run CRASHED: {type(e).__name__}: {e}")
        import traceback

        _trace(traceback.format_exc())
        raise