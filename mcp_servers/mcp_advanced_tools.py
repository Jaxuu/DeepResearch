import pathlib
import sys
import os
import re
import io
import cv2
import base64
import pandas as pd
import openpyxl
import json
import docx
import pptx
import pymupdf
import tempfile
import asyncio
import textwrap
import contextlib
import subprocess
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import httpx
import zipfile
import tempfile
import shutil

import openai
import subprocess
import yt_dlp
from mcp.server.fastmcp import FastMCP
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
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
        temp_script.write("# -*- coding: utf-8 -*-\n" + code)
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


@mcp.tool()
async def search_exact_url(query: str) -> str:
    """
    Search the web specifically to find exact URLs for videos, articles, or resources.
    Returns ONLY a clean list of Titles and URLs (no summaries) to force you to visit the actual link.
    Use this when you need a source URL before analyzing a video or document.
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "[❌ Tool Failed] Missing TAVILY_API_KEY environment variable."

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"query": query, "api_key": api_key, "search_depth": "basic", "max_results": 5}
            )
            response.raise_for_status()
            data = response.json()

        results_str = "[✅ Search Success] Found the following URLs:\n\n"
        for res in data.get("results", []):
            results_str += f"- Title: {res.get('title')}\n  URL: {res.get('url')}\n\n"

        if not data.get("results"):
            return "[❌ Tool Failed] No relevant URLs found."

        return results_str.strip()
    except Exception as e:
        return f"[❌ Tool Failed] Search error: {str(e)}"


@mcp.tool()
async def transcribe_audio_file(file_path: str, extraction_query: str) -> str:
    """
    Transcribe a local audio file (e.g., .mp3, .wav) using faster-whisper (fully offline),
    and extract specific information based on the query.
    """
    if not os.path.exists(file_path):
        return f"[❌ Tool Failed] Audio file not found at absolute path: {file_path}"

    try:
        # 在子线程中执行，避免阻塞事件循环
        def _run_asr_sync():
            from faster_whisper import WhisperModel
            model = WhisperModel("small", device="cpu", compute_type="int8")
            segments, _info = model.transcribe(file_path, beam_size=5)
            return " ".join(seg.text.strip() for seg in segments).strip()

        full_text = await asyncio.to_thread(_run_asr_sync)

        if not full_text.strip():
            return "[❌ Tool Failed] Audio transcribed but no speech found."

        # 以下 qwen-plus 提取逻辑保持不变
        model_name = os.getenv("SUPERVISOR_MODEL", "qwen-plus")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        skill_model = init_chat_model(
            model=model_name,
            model_provider="openai",
            api_key=api_key,
            base_url=base_url,
            max_tokens=1000,
        )

        prompt = f"""You are an Audio Extraction Analyst. 
                    Listen to the transcribed text and answer the query accurately.

                    CRITICAL EXTRACTION RULES:
                    1. PRESERVE EXACT PHRASING: When extracting entities, ingredients, or specific items, you MUST preserve the exact phrasing used in the transcript.
                    2. DO NOT STRIP ADJECTIVES: NEVER remove descriptive adjectives, modifiers, or specific details (e.g., MUST extract "ripe strawberries" or "pure vanilla extract", NOT just "strawberries" or "vanilla") unless the query explicitly commands you to do so.
                    3. Do not summarize or paraphrase specific nouns.

                    <Audio Transcript>
                    {full_text}
                    </Audio Transcript>
                    <Query>
                    {extraction_query}
                    </Query>
                    Return the precise answer based strictly on the rules above. If not found, say so."""

        response = await skill_model.ainvoke([HumanMessage(content=prompt)])
        return f"[✅ Audio Transcription & Extraction Success]\nExtracted Answer:\n{response.content}"

    except Exception as e:
        return f"[❌ Tool Failed] Failed to process audio file: {str(e)}"


@mcp.tool()
async def fetch_video_transcript(url: str, extraction_query: str) -> str:
    """
    Fetch the transcript/subtitles of a video and extract specific information.
    Note: This only extracts spoken text. It cannot see the video visually.
    """
    try:
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'subtitlesformat': 'vtt',
            'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'noprogress': True,
            'logger': None,
        }

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_id = info['id']

        sub_file = None
        temp_dir = tempfile.gettempdir()
        for f in os.listdir(temp_dir):
            if f.startswith(video_id) and f.endswith('.vtt'):
                sub_file = os.path.join(temp_dir, f)
                break

        if not sub_file:
            return "[❌ Tool Failed] No subtitles could be found or downloaded for this video."

        with open(sub_file, 'r', encoding='utf-8') as f:
            vtt_lines = f.readlines()

        os.remove(sub_file)

        text_lines = []
        for line in vtt_lines:
            line = line.strip()
            if line and not line.startswith(('WEBVTT', 'Kind:', 'Language:', '00:')):
                clean_line = re.sub(r'<[^>]+>', '', line)
                if clean_line and clean_line not in text_lines:
                    text_lines.append(clean_line)

        full_text = " ".join(text_lines)

        if not full_text.strip():
            return "[❌ Tool Failed] Subtitles were empty after extraction."

        model_name = os.getenv("SUPERVISOR_MODEL", "qwen-plus")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        skill_model = init_chat_model(
            model=model_name, model_provider="openai", api_key=api_key, base_url=base_url, max_tokens=1000,
        )

        # 【核心修正】：增加 ASR 容错提示词，防止大模型死扣人名拼写
        prompt = f"""You are a Video Transcript Analyst. 
                    Based on the spoken content of the video, answer the query accurately.

                    CRITICAL WARNING ABOUT TRANSCRIPTS:
                    This text is auto-generated by YouTube. It often contains phonetic typos and misheard words (e.g., character names like "Teal'c" might be transcribed as "tea oak" or similar gibberish). 
                    DO NOT be overly literal. Use phonetic similarity and surrounding context clues to find the answer.

                    <Video Transcript>
                    {full_text}
                    </Video Transcript>
                    <Query>
                    {extraction_query}
                    </Query>"""

        response = await skill_model.ainvoke([HumanMessage(content=prompt)])
        return f"[✅ YouTube Transcript Extraction Success]\nExtracted Answer:\n{response.content}"

    except Exception as e:
        return f"[❌ Tool Failed] Failed to fetch or process YouTube transcript: {str(e)}"


@mcp.tool()
async def analyze_video_visually(url: str, query: str) -> str:
    """
    Download a video, extract evenly spaced key frames, and analyze them VISUALLY.
    Use this ONLY when you need to see what is on the screen (e.g., counting objects, describing scenes).
    """
    try:
        ydl_opts = {
            'format': 'worstvideo[ext=mp4]',
            'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'noprogress': True,
            'logger': None,
        }

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=True)
                    video_path = ydl.prepare_filename(info_dict)

        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            # 【核心修正】：提升到 12 帧以增加命中率
            step = max(1, frame_count // 12)
            for i in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (512, 288))
                    # 压缩率从 80 降到 60，减小图片传输体积，防超时
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    frames.append(base64.b64encode(buffer).decode('utf-8'))
                if len(frames) >= 12:
                    break
        cap.release()
        os.remove(video_path)

        if not frames:
            return "[❌ Tool Failed] Could not extract any frames from the video."

        model_name = os.getenv("VISUAL_MODEL", "qwen-vl-plus")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        vision_llm = init_chat_model(
            model=model_name, model_provider="openai", base_url=base_url, api_key=api_key
        )

        prompt_text = (
            f"You are a highly observant Video AI Analyst. I am providing you with {len(frames)} evenly sampled frames from a video in chronological order. "
            f"Carefully inspect EVERY frame. Pay close attention to subtle details, distinct entities, text, and visual changes over time.\n"
            f"Based on your comprehensive observation across all frames, answer the following query accurately: {query}\n"
            f"If the question asks for a count, maximum, or specific detail, ensure you analyze all frames to find the exact moment that answers the query."
        )
        content = [{"type": "text", "text": prompt_text}]
        for b64 in frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        message = HumanMessage(content=content)
        response = await vision_llm.ainvoke([message])
        return f"[✅ Video Visual Analysis Success]\nExtracted Answer:\n{response.content}"

    except Exception as e:
        return f"[❌ Tool Failed] Failed to analyze video visually: {str(e)}"


@mcp.tool()
async def analyze_local_image(file_path: str, question: str) -> str:
    """
    Analyze a local image file (e.g., .png, .jpg, .jpeg) visually to answer a specific question.
    Use this for OCR, reading charts, or analyzing chess boards/diagrams.
    """
    if not os.path.exists(file_path):
        return f"[❌ Tool Failed] Image file not found at absolute path: {file_path}"

    try:
        # 在子线程中使用 PIL 对超大图片进行压缩，防止 Base64 字符串撑爆网络
        def _compress_image():
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"OpenCV cannot read image at: {file_path}")

            # 等比例缩放，限制最大边长为 2048 像素
            h, w = img.shape[:2]
            max_dim = 2048
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            # 存为高压缩率的 JPEG 字节流
            success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise ValueError("Cannot encode image to JPEG.")

            return base64.b64encode(buffer).decode("utf-8")

        base64_image = await asyncio.to_thread(_compress_image)

        model_name = os.getenv("VISUAL_MODEL", "qwen-vl-max")
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
                 "text": f"You are an elite Vision Analyst. Analyze this image and answer the question accurately: {question}\n\nCRITICAL SYSTEM INSTRUCTION: If the user asks you to extract a list of items (e.g., text, fractions, objects), you MUST NOT deduplicate them. You MUST extract and list EVERY SINGLE OCCURRENCE exactly as it appears in the image, reading strictly from top-left to bottom-right. If an item appears 5 times, it must be listed 5 times."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        )

        # 内部包裹 60 秒硬熔断，防止单次视觉请求挂死引发全局崩溃
        response = await asyncio.wait_for(
            vision_llm.ainvoke([message]),
            timeout=60.0
        )

        return f"[✅ Local Image Analysis Success]\nExtracted Answer:\n{response.content}"

    except asyncio.TimeoutError:
        return "[❌ Tool Failed] Visual analysis timed out (>60s). The API might be unresponsive or the image is too complex."
    except Exception as e:
        return f"[❌ Tool Failed] Failed to analyze image: {str(e)}"


@mcp.tool()
async def inspect_structured_data(file_path: str) -> str:
    """
    Inspect the sheet names, column headers, and preview first 3 rows of tabular data (.xlsx, .csv, .json).
    CRITICAL RESTRICTION: This tool ONLY extracts text/numerical data. It CANNOT detect cell formatting,
    fill colors, highlights, or styles.
    If the question involves cell colors (e.g., green/blue plots), map visualization, or cell formatting,
    DO NOT use this tool — use `execute_python_code` with `openpyxl` directly.
    """
    if not os.path.exists(file_path):
        return f"[❌ Error] File not found: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    output = []

    try:
        if ext in ['.xlsx', '.xls']:
            xls = pd.ExcelFile(file_path)
            output.append(f"📊 Excel File Detected. Sheets found: {xls.sheet_names}\n")
            for sheet in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet, nrows=3)
                output.append(f"--- Sheet: '{sheet}' ---")
                output.append(f"Columns: {list(df.columns)}")
                output.append("Preview (first 3 rows):")
                output.append(df.to_string(index=False) + "\n")

        elif ext == '.csv':
            df = pd.read_csv(file_path, nrows=3)
            output.append("📄 CSV File Detected.\n")
            output.append(f"Columns: {list(df.columns)}")
            output.append("Preview (first 3 rows):")
            output.append(df.to_string(index=False) + "\n")

        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            output.append("📦 JSON File Detected. Structure preview:")
            output.append(json.dumps(data, indent=2)[:1500] + "\n... (truncated)")

        else:
            return f"[❌ Error] Unsupported file type for quick inspection: {ext}. Use python sandbox instead."

        return "\n".join(output)
    except Exception as e:
        return f"[❌ Error] Failed to inspect file: {str(e)}"


@mcp.tool()
async def read_pdf_by_page(file_path: str, start_page: int, end_page: int) -> str:
    """
    Read specific pages from a long PDF document (1-indexed).
    Use this to extract text precisely when you know the rough page range (e.g., from reading the Table of Contents).
    Never read more than 15 pages at a time to avoid overwhelming your context.
    """
    if not os.path.exists(file_path):
        return f"[❌ Error] PDF not found: {file_path}"

    if end_page - start_page > 15:
        return "[❌ Error] Page range too large. Please request 15 pages or fewer at a time."

    try:
        # PyMuPDF 索引从 0 开始，用户输入从 1 开始
        doc = pymupdf.open(file_path)
        total_pages = len(doc)

        start_idx = max(0, start_page - 1)
        end_idx = min(total_pages, end_page)

        output = [f"📖 PDF File: {os.path.basename(file_path)} (Total Pages: {total_pages})"]
        output.append(f"--- Extracting Pages {start_page} to {end_idx} ---\n")

        for i in range(start_idx, end_idx):
            page = doc.load_page(i)
            text = page.get_text("text")
            output.append(f"[[ PAGE {i + 1} ]]\n{text.strip()}\n")

        doc.close()
        return "\n".join(output)
    except Exception as e:
        return f"[❌ Error] Failed to read PDF: {str(e)}"


@mcp.tool()
async def execute_local_python_script(file_path: str) -> str:
    """
    Execute a local .py script file directly and return its standard output.
    Use this ONLY when the user explicitly asks you to run or get the output of an attached Python file.
    """
    if not os.path.exists(file_path):
        return f"[❌ Error] File not found: {file_path}"
    if not file_path.lower().endswith('.py'):
        return "[❌ Error] Not a Python file."

    try:
        def _run_script():
            return subprocess.run(
                ["python", file_path],
                capture_output=True,
                text=True,
                timeout=120
            )

        result = await asyncio.to_thread(_run_script)

        output = []
        if result.stdout:
            output.append(f"[STDOUT]\n{result.stdout.strip()}")
        if result.stderr:
            output.append(f"[STDERR]\n{result.stderr.strip()}")

        if not output:
            return "[✅ Execution Success] Script ran successfully but produced no output."

        return "\n\n".join(output)
    except subprocess.TimeoutExpired:
        return "[❌ Error] Script execution timed out (>60s). It might be an infinite loop."
    except Exception as e:
        return f"[❌ Error] Failed to execute script: {str(e)}"


@mcp.tool()
async def unzip_and_list_directory(zip_path: str) -> str:
    """
    Unzip a local .zip archive and list its complete folder structure and files.
    Always use this when the user attaches a .zip file for algorithmic or data tasks.
    It extracts to a safe temporary directory and returns the absolute paths of all extracted files.
    """
    if not os.path.exists(zip_path):
        return f"[❌ Error] ZIP file not found: {zip_path}"

    if not zip_path.lower().endswith('.zip'):
        return "[❌ Error] This tool only supports .zip files."

    try:
        # 创建专属的临时解压目录
        extract_dir = tempfile.mkdtemp(prefix="gaia_unzip_")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        output = [f"✅ Successfully extracted ZIP to: {extract_dir}"]
        output.append("--- Directory Tree ---")

        # 遍历生成文件树结构和绝对路径
        for root, dirs, files in os.walk(extract_dir):
            level = root.replace(extract_dir, '').count(os.sep)
            indent = ' ' * 4 * (level)
            output.append(f"{indent}📁 {os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                full_path = os.path.join(root, f)
                output.append(f"{subindent}📄 {f} (Path: {full_path})")

        return "\n".join(output)
    except Exception as e:
        return f"[❌ Error] Failed to unzip and list directory: {str(e)}"


@mcp.tool()
async def read_local_text_file(file_path: str, max_lines: int = 1500) -> str:
    """
    Read the text content of local documents, including .txt, .md, .json, .csv, .docx (Word), and .pptx (PowerPoint).
    For Word/PPT, it extracts paragraphs, tables, and ALL deep text elements (including grouped shapes and notes).
    Always use this tool first to read document contents, count slides, or find text evidence before trying to write Python parsing scripts.
    """
    if not os.path.exists(file_path):
        return f"[❌ Error] File not found: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    lines = []

    try:
        if ext == '.docx':
            doc = docx.Document(file_path)
            for p in doc.paragraphs:
                text = p.text.strip()
                if text:
                    lines.append(text)
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                    if row_text:
                        lines.append(row_text)

        elif ext == '.pptx':
            prs = pptx.Presentation(file_path)

            def extract_text_from_shape(shape):
                texts = []
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
                if shape.has_table:
                    for row in shape.table.rows:
                        for cell in row.cells:
                            if cell.text_frame.text.strip():
                                texts.append(cell.text_frame.text.strip())
                if shape.shape_type == 6:  # Group shape
                    for child in shape.shapes:
                        texts.extend(extract_text_from_shape(child))
                return texts

            for i, slide in enumerate(prs.slides):
                slide_texts = []
                for shape in slide.shapes:
                    slide_texts.extend(extract_text_from_shape(shape))
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        slide_texts.append(f"[Notes]: {notes}")
                if slide_texts:
                    lines.append(f"--- Slide {i + 1} ---")
                    lines.extend(slide_texts)

        else:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    lines.append(line.rstrip("\n"))

        total_lines = len(lines)
        output = f"📄 File: {os.path.basename(file_path)}\n"
        output += f"Total Lines extracted: {total_lines}\n"
        output += f"--- Showing first {min(total_lines, max_lines)} lines ---\n\n"
        output += "\n".join(lines[:max_lines])

        if total_lines > max_lines:
            output += f"\n\n... (File truncated. It has {total_lines - max_lines} more lines)."

        return output
    except Exception as e:
        return f"[❌ Error] Failed to read file: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")