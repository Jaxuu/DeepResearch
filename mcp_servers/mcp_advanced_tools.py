import subprocess
import sys
import os
import re
import io
import base64
import tempfile
import asyncio
import textwrap
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import httpx

from mcp.server.fastmcp import FastMCP
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from playwright.async_api import async_playwright

mcp = FastMCP("AdvancedResearchTools")


def _safe_exec(code: str, timeout: float = 15.0):
    """在独立线程里执行代码，带超时。返回 (stdout, error)。"""
    buf = io.StringIO()
    result = {"output": "", "error": None}

    def _run():
        try:
            with redirect_stdout(buf):
                exec(code, {"__builtins__": __builtins__}, {})
            result["output"] = buf.getvalue()
        except BaseException as e:
            result["output"] = buf.getvalue()
            result["error"] = f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            future.result(timeout=timeout)
        except FuturesTimeoutError:
            result["error"] = f"TimeoutError: 执行超过 {timeout}s（可能是死循环）"
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"

    return result["output"], result["error"]


@mcp.tool()
async def calculate_with_python(data_context: str, calculation_goal: str) -> str:
    """
    A quantitative analysis sandbox. Use this to perform complex math, statistics, or unit conversions.
    """
    model_name = os.getenv("SUPERVISOR_MODEL", "qwen-plus")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        return "[❌ Tool Failed] Missing OPENAI_API_KEY environment variable."

    skill_model = init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        max_tokens=1000,
    )

    system_prompt = f"""You are a Python Data Analyst. Your job is to achieve the calculation goal based on the data.
                        Data Context:
                        {data_context}
                        Goal: {calculation_goal}
                        Write a python script to compute this. Print the exact final numerical result clearly.
                        Return ONLY valid python code wrapped in ```python```. Do not explain."""

    current_prompt = system_prompt
    for attempt in range(5):
        try:
            response = await skill_model.ainvoke([HumanMessage(content=current_prompt)])
            code_match = re.search(r"```python\n(.*?)\n```", response.content, re.DOTALL)
            code = code_match.group(1) if code_match else response.content.replace("```python", "").replace("```", "")

            code = textwrap.dedent(code).strip()
            output, err = await asyncio.to_thread(_safe_exec, code, 15.0)
            if err:
                output = output + f"\nTraceback Exception: {err}"

            if "Error" in output or "Exception" in output or "Traceback" in output:
                current_prompt += f"\n\nPrevious attempt failed with error:\n{output}\nPlease fix the code and try again."
                continue
            if not output.strip():
                current_prompt += "\n\nPrevious attempt ran successfully but printed nothing. You MUST use print()."
                continue

            return f"[✅ Calculation Success]\nResult details:\n{output.strip()}"
        except Exception as e:
            current_prompt += f"\n\nSystem error occurred: {str(e)}\nFix the issue and rewrite."

    return "[❌ Tool Failed] Unable to compute the requested data."


@mcp.tool()
async def extract_from_long_document(url: str, extraction_query: str) -> str:
    """
    Extract specific information from extremely long documents (PDFs, massive HTML pages).
    Pass the URL and a highly optimized, keyword-rich extraction query.
    """
    full_content = ""

    if url.lower().endswith(".pdf") or "pdf" in url.lower():
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(response.content)
                temp_pdf_path = temp_file.name

            loader = PyPDFLoader(temp_pdf_path)
            full_content = "\n\n".join([p.page_content for p in loader.load()])
            os.remove(temp_pdf_path)
        except Exception as e:
            return f"[❌ Tool Failed] Native PDF extraction failed: {str(e)}"
    else:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {"User-Agent": "Mozilla/5.0", "X-Timeout": "30"}
        jina_key = os.getenv("JINA_API_KEY")
        if jina_key:
            headers["Authorization"] = f"Bearer {jina_key}"
        try:
            with httpx.Client(timeout=45.0, follow_redirects=True) as client:
                resp = client.get(jina_url, headers=headers)
            if resp.status_code != 200:
                return f"[❌ Tool Failed] Status: {resp.status_code}"
            full_content = resp.text
        except Exception as e:
            return f"[❌ Tool Failed] Network error: {str(e)}"

    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=300)
    chunks = splitter.split_text(full_content)

    if len(chunks) <= 3:
        context = full_content
    else:
        retriever = BM25Retriever.from_documents([Document(page_content=c) for c in chunks])
        retriever.k = 10
        context = "\n\n---\n\n".join([d.page_content for d in retriever.invoke(extraction_query)])

    model_name = os.getenv("SUPERVISOR_MODEL", "qwen-plus")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    skill_model = init_chat_model(
        model=model_name,
        model_provider="openai",
        base_url=base_url,
        api_key=api_key,
        max_tokens=1500,
    )

    prompt = f"""You are an elite Data Extraction Analyst. Synthesize the answer ONLY using the retrieved context.
                <Source URL>{url}</Source URL>
                <Query>{extraction_query}</Query>
                <Retrieved Context>\n{context}\n</Retrieved Context>"""
    try:
        res = await skill_model.ainvoke([HumanMessage(content=prompt)])
        return f"[✅ Long-Doc Mining Success]\nExtracted from {url}:\n{res.content}"
    except Exception as e:
        return f"[❌ Tool Failed] LLM extraction error: {str(e)}"


@mcp.tool()
async def analyze_webpage_visual_layout(url: str, specific_question: str) -> str:
    """
    Analyze the physical layout, colors, or typography of a webpage visually using a headless browser and Vision LLM.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30000)
            screenshot_bytes = await page.screenshot(full_page=True)
            await browser.close()
    except Exception as e:
        return f"Failed to capture webpage: {str(e)}"

    base64_image = base64.b64encode(screenshot_bytes).decode("utf-8")

    model_name = os.getenv("VISUAL_MODEL", "qwen-vl-plus")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    vision_llm = init_chat_model(
        model=model_name,
        model_provider="openai",
        base_url=base_url,
        api_key=api_key,
    )
    message = HumanMessage(
        content=[
            {"type": "text",
             "text": f"You are a visual layout expert. Analyze this webpage screenshot and answer: {specific_question}. Pay strict attention to CSS styling, indentations, and spatial arrangement."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]
    )
    response = await vision_llm.ainvoke([message])
    return f"Visual Analysis Result for {url}:\n{response.content}"


@mcp.tool()
async def execute_python_code(code: str) -> str:
    """
    Executes a given string of Python code in a local sandbox environment.
    Use this tool to parse files (Excel, Word, PPTX), perform complex math,
    or run algorithms. Provide the full, runnable Python script.
    Use absolute file paths if reading local files.
    Always print() the final result so it can be captured in stdout.
    """
    # 去掉代码的统一缩进，防止模型生成的代码带缩进导致 IndentationError
    code = textwrap.dedent(code).strip()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_script:
        temp_script.write(code)
        temp_script_path = temp_script.name

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, temp_script_path],
            text=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
            cwd=os.getcwd(),
        )

        output = result.stdout
        if result.stderr:
            output += f"\n--- STDERR ---\n{result.stderr}"
        if not output.strip():
            output = "Code executed successfully but returned no output. Did you forget to print()?"

        return output

    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 60 seconds."
    except Exception as e:
        return f"Execution failed: {str(e)}"
    finally:
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)


if __name__ == "__main__":
    mcp.run(transport="stdio")