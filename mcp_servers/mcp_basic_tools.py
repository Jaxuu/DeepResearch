import sys
import os
import re
import json
import asyncio
import httpx

from mcp.server.fastmcp import FastMCP
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

mcp = FastMCP("ChartGenerationTool")


@mcp.tool()
async def generate_chart_image(data_context: str, visualization_goal: str) -> str:
    """
    Generate an industrial-grade chart image (e.g., bar, line, pie, scatter) via QuickChart API.
    Use this tool to visualize raw data. Pass the data context and your specific charting goal.
    Returns a Markdown formatted image link.
    """
    model_name = os.getenv("SUPERVISOR_MODEL", "qwen-plus")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    skill_model = init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        max_tokens=1500,
    )

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
        try:
            res = await asyncio.wait_for(
                skill_model.ainvoke([HumanMessage(content=prompt)]),
                timeout=60.0,
            )
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

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://quickchart.io/chart/create",
                    json={"chart": chart_json},
                )
                response.raise_for_status()
                res_body = response.json()

                if res_body.get("success"):
                    short_url = res_body.get("url")
                    return f"[✅ Visualization Generated]\n![Data_Visualization_Chart]({short_url})"
                else:
                    raise Exception("QuickChart API failed to generate short URL.")

        except asyncio.TimeoutError:
            if attempt == 2:
                return "[❌ Tool Failed] Model call timed out after 60s."
            prompt += "\nError: Output must be valid JSON."
        except Exception as e:
            if attempt == 2:
                return f"[❌ Tool Failed] Could not generate chart: {str(e)}"
            prompt += "\nError: Output must be valid JSON."

    return "[❌ Tool Failed] Exceeded max retries."


if __name__ == "__main__":
    mcp.run(transport="stdio")