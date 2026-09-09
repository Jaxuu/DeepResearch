# 用户打开页面
#   │
#   ▼
# 第1次全量执行 ──── 未登录 → 显示登录表单 → st.stop()
#   │
#   │ 用户点击「登录」
#   ▼
# 第2次全量执行 ──── 登录成功 → st.rerun()
#   │
#   ▼
# 第3次全量执行 ──── 已登录 → 显示聊天界面，等待输入
#   │
#   │ 用户输入 prompt 并回车
#   ▼
# 第4次全量执行 ──── 进入 stream_agent_run()
#   │                 │
#   │                 └─ for chunk in stream:  ← 阻塞，持续更新 UI
#   │                     收到100个token → 原地刷新33次
#   │                     流结束
#   │
#   │ 如果后端挂起(human_review) → st.rerun()
#   ▼
# 第5次全量执行 ──── 检测到 pending_interrupt → 显示干预表单
#   │
#   │ 用户点击「直接生成报告」
#   ▼
# 第6次全量执行 ──── is_resuming=True → 进入 stream_agent_run()
#                     │
#                     └─ for chunk in stream:  ← 再次阻塞等待
#                         流结束 → 显示最终报告
#
#
# 一次 Streamlit 执行（不 rerun）：
# ┌──────────────────────────────────────────────┐
# │                                              │
# │  status = st.status(...)        ← 创建状态框  │
# │  placeholder = st.empty()       ← 创建占位符  │
# │                                              │
# │  for chunk in stream:          ← 开始阻塞等待 │
# │      ├─ 收到 token 1 → placeholder.markdown() │  ← 原地更新
# │      ├─ 收到 token 2 → placeholder.markdown() │  ← 原地更新
# │      ├─ 收到 token 3 → placeholder.markdown() │  ← 原地更新
# │      └─ ... 直到流结束 ...                    │
# │                                              │
# │  st.rerun()  ← 流结束后，触发下一次全量重跑    │
# └──────────────────────────────────────────────┘


import os
import re
import httpx
import streamlit as st
from supabase import create_client, Client
from langgraph_sdk import get_sync_client
from dotenv import load_dotenv
load_dotenv(override=True)

# ================= 页面与主题配置 =================
st.set_page_config(page_title="Deep Research", layout="wide")

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
        max-width: 1200px !important;
    }
    header {visibility: hidden;}
    button[kind="primary"] {
        background-color: #1565C0 !important;
        color: white !important;
        border-color: #1565C0 !important;
    }
    button[kind="primary"]:hover {
        background-color: #0D47A1 !important;
        border-color: #0D47A1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ================= Supabase 客户端初始化 =================
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    st.error("未找到 SUPABASE_URL 或 SUPABASE_KEY 环境变量，请检查 .env 配置。")
    st.stop()

supabase: Client = create_client(supabase_url, supabase_key)

# ================= 状态初始化 =================
if "user" not in st.session_state:
    st.session_state.user = None
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= 客户端获取工厂 =================
LANGGRAPH_URL = "http://localhost:2024"
ASSISTANT_ID = "Deep Researcher"

def get_new_client():
    """
    每次调用时创建一个新的客户端实例，配置 3600 秒超长超时时间防止深度调研中断。
    """
    timeout_config = httpx.Timeout(3600.0)
    token = st.session_state.get("access_token")

    return get_sync_client(
        url=LANGGRAPH_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout_config
    )

# ================= 工具函数 =================
def get_recent_threads():
    client = get_new_client()
    try:
        return client.threads.search(limit=5)
    except Exception:
        return []

def create_new_thread(title):
    client = get_new_client()
    thread = client.threads.create(metadata={"title": title})
    return thread["thread_id"]

def fetch_thread_messages(thread_id):
    client = get_new_client()
    try:
        state = client.threads.get_state(thread_id)
        if not state or "values" not in state:
            return []

        raw_msgs = state["values"].get("messages", [])
        chat_history = []

        for m in raw_msgs:
            m_type = m.get("type", "")
            content = m.get("content", "")
            if not content and "kwargs" in m:
                content = m["kwargs"].get("content", "")

            if m_type == "human":
                chat_history.append({"role": "user", "content": content})
            elif m_type == "ai" and content:
                chat_history.append({"role": "assistant", "content": content})

        final_report = state["values"].get("final_report", "")
        if final_report and not any(msg["content"] == final_report for msg in chat_history):
            chat_history.append({"role": "assistant", "content": final_report})

        return chat_history
    except Exception:
        return []

# ================= 侧边栏：鉴权与历史会话管理 =================
with st.sidebar:
    st.title("🔐 用户认证")

    # --- 登录阻断逻辑 ---
    if st.session_state.user is None:
        email = st.text_input("邮箱 (Email)")
        password = st.text_input("密码 (Password)", type="password")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("登录", use_container_width=True, type="primary"):
                try:
                    response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user = response.user
                    st.session_state.access_token = response.session.access_token
                    st.rerun()
                except Exception as e:
                    st.error(f"登录失败: {str(e)}")
        with col2:
            if st.button("注册", use_container_width=True):
                try:
                    response = supabase.auth.sign_up({"email": email, "password": password})
                    st.success("注册成功！请直接登录。")
                except Exception as e:
                    st.error(f"注册失败: {str(e)}")

        st.divider()
        st.warning("⚠️ 请先登录以使用深度调研系统并加载您的专属历史记录。")
        st.stop()  # 阻断后续代码执行，强制用户登录

    else:
        # 已登录状态
        st.success(f"已登录: {st.session_state.user.email}")
        if st.button("退出登录 (Logout)", type="secondary"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.session_state.access_token = None
            st.session_state.thread_id = None
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # --- 历史会话管理 ---
    st.title("💬 会话历史")
    if st.button("➕ 新建深度调研", use_container_width=True, type="primary"):
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("最近 5 次会话")

    threads = get_recent_threads()
    for t in threads:
        tid = t["thread_id"]
        title = t.get("metadata", {}).get("title", f"会话 {tid[:8]}")
        is_current = (tid == st.session_state.thread_id)
        button_type = "primary" if is_current else "secondary"
        if st.button(title, key=tid, use_container_width=True, type=button_type):
            if not is_current:
                st.session_state.thread_id = tid
                st.session_state.messages = fetch_thread_messages(tid)
                st.rerun()

# ================= 主页面：聊天交互区 =================
if not st.session_state.messages:
    st.markdown("<h1 style='text-align: center; color: #333; margin-top: 15vh;'>🔍 Deep Research 智能体</h1>",
                unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ================= 状态拦截与恢复控制 =================
current_state = None
if st.session_state.thread_id:
    client = get_new_client()
    try:
        # 直接同步调用
        current_state = client.threads.get_state(st.session_state.thread_id)

        #检测后端流程是否停在了「人工审核」节点
        if current_state and current_state.get("next") and "human_review" in current_state["next"]:
            st.session_state.pending_interrupt = True
            tasks = current_state.get("tasks", [])
            if tasks and tasks[0].get("interrupts"):
                st.session_state.interrupt_data = tasks[0]["interrupts"][0].get("value", {})
        else:
            st.session_state.pending_interrupt = False
    except Exception:
        pass

# 判断当前是否处于「恢复挂起的流程」的状态
is_resuming = st.session_state.get("resume_payload") is not None

# 渲染干预表单
if st.session_state.get("pending_interrupt") and not is_resuming:
    st.warning("⏸️ **流程已挂起**：情报检索阶段完成，等待您的审核。")
    facts = st.session_state.interrupt_data.get("facts", [])
    with st.expander("📊 查看已收集的事实 (FactBoard)", expanded=True):
        for i, fact in enumerate(facts):
            st.markdown(
                f"- **{fact.get('entity', 'N/A')}**: {fact.get('claim', '')} *(来源: {fact.get('source', '')})*")

    feedback = st.text_input("✍️ 补充干预指令（若无需干预请留空）：")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 事实无误，直接生成报告", use_container_width=True, type="primary"):
            st.session_state.resume_payload = {"action": "continue"}
            st.rerun()
    with col2:
        if st.button("🔄 打回节点，继续调研", use_container_width=True):
            if feedback.strip():
                st.session_state.resume_payload = {"action": "feedback", "feedback": feedback}
                st.rerun()

# ================= 底部固定：输入框 =================
prompt = st.chat_input("输入调研需求...") if not is_resuming and not st.session_state.get("pending_interrupt") else None

if prompt or is_resuming:
    client = get_new_client()

    if prompt and not is_resuming:
        if st.session_state.thread_id is None:
            short_title = prompt[:12] + "..." if len(prompt) > 12 else prompt
            new_id = create_new_thread(title=short_title)
            st.session_state.thread_id = new_id

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    with st.chat_message("assistant"):
        status_label = "🔄 收到干预指令，正在恢复流转..." if is_resuming else "🧠 DeepResearch流程已启动，正在分析需求..."
        status = st.status(status_label, expanded=True)     # 状态指示器，在页面上渲染一个可折叠的状态框，显示当前进度标签
        message_placeholder = st.empty()    # 创建一个空的占位符，后续可以动态替换其内容，在调研过程中，通过 message_placeholder.markdown(...) 多次更新这个位置的内容，实现流式打字机效果


        # 启动并实时监听后端 LangGraph 工作流的执行过程
        def stream_agent_run():
            if is_resuming:
                payload = st.session_state.resume_payload
                st.session_state.resume_payload = None
                st.session_state.pending_interrupt = False

                # 启动 LangGraph 后端工作流并建立流式连接的调用
                stream = client.runs.stream(
                    thread_id=st.session_state.thread_id,
                    assistant_id=ASSISTANT_ID,
                    input=None,
                    command={"resume": payload},
                    stream_mode=["values", "messages", "updates"]
                )
            else:
                payload = {"messages": [{"role": "user", "content": prompt}]}
                stream = client.runs.stream(
                    thread_id=st.session_state.thread_id,
                    assistant_id=ASSISTANT_ID,
                    input=payload,
                    stream_mode=["values", "messages", "updates"]
                )

            final_report = ""
            raw_report = ""
            seen_ids = set()    #后端在核查失败时，可能对同一个 verification_feedback 推送多次 values 事件，导致前端重复显示"核查未通过"警告，用 set() 记录已经显示过的反馈 ID，只显示一次
            display_locked_to_final = False #当 final_report 已就绪并显示后，后端的 messages/partial 事件可能还在继续推送，如果不加锁，报告内容会被打字机输出覆盖掉。一旦检测到 final_report 就绪，就锁死显示。
            token_count = 0  # LLM 流式输出时每个 token 都会触发一次 messages/partial 事件，如果每来一个 token 就刷新一次 Streamlit UI，WebSocket 会被高频更新撑爆。改为每 3 个 token 才刷新一次 UI：

            for chunk in stream:
                # 模式	           触发                           事件类型	                            推送内容	                            前端用途
                # "updates"	   节点完成时触发               chunk.event == "updates"	            哪些节点刚执行完毕（节点名列表）	     更新状态栏标题，告诉用户进度（"正在检索..."、"正在撰写..."）
                # "values"	   全局状态变化时触发            chunk.event == "values"	            当前完整的 state 快照	             检测 final_report 是否就绪、检测 verification_feedback 核查失败
                # "messages"   LLM 吐出token 时实时触发     chunk.event == "messages/partial"	    LLM 节点的流式打字机输出（逐 token）	 实时渲染 supervisor 和 rewrite_report 的思考/输出内容
                if chunk.event == "updates":
                    completed_nodes = list(chunk.data.keys())
                    for node in completed_nodes:
                        if node == "clarify_with_user":
                            status.update(label="📝 正在生成结构化调研概要...", expanded=False)     # expanded=False 状态框折叠，只显示标题栏，隐藏内部内容
                        elif node == "write_research_brief":
                            status.update(label="️‍🕵️‍♂️ 主管正在调度研究员与分配子任务...", expanded=False)
                            if not display_locked_to_final: message_placeholder.empty()
                        elif node == "research_supervisor":
                            status.update(label="👨 所有事实正在通过用户审核...", expanded=False)
                        elif node == "human_review":
                            status.update(label="📑 正在根据情报规划报告大纲...", expanded=False)
                        elif node == "generate_outline":
                            status.update(label="⚡ 正在并发撰写各章节内容...", expanded=False)
                            if not display_locked_to_final:
                                message_placeholder.markdown(
                                    "> ⚡ **多智能体并发撰写中**：已将大纲与检索事实切片下发给多个写手节点，正在极速成文中...")
                        elif node == "write_section":
                            status.update(label="📕 报告已完成，正在核验所有事实来源...", expanded=False)


                elif chunk.event == "values":
                    state = chunk.data
                    if "verification_feedback" in state and state["verification_feedback"]:
                        fb_id = str(hash(state["verification_feedback"]))
                        if fb_id not in seen_ids:
                            seen_ids.add(fb_id)
                            status.write("⚠️ **核查未通过**: 发现未证实断言，正在打回重写...")

                    if state.get("final_report") and state.get("next") != ["rewrite_report"]:
                        if not display_locked_to_final:
                            display_locked_to_final = True
                            final_report = state["final_report"]
                            message_placeholder.markdown(final_report)
                            status.update(label="✅ 初稿拼接完成，进行自动化核查...", state="running")

                # TODO 这一块是流式输出，后面可以改输出各个节点的llm输出。
                elif chunk.event == "messages/partial" and not display_locked_to_final:
                    msg_data = chunk.data[0] if isinstance(chunk.data, list) and len(chunk.data) > 0 else chunk.data
                    metadata = chunk.data[1] if isinstance(chunk.data, list) and len(chunk.data) > 1 else {}
                    node_name = metadata.get("langgraph_node", "")

                    if node_name == "rewrite_report":
                        status.update(label="✨ 根据核查意见重写报告...", expanded=False)
                        display_locked_to_final = False

                    if node_name in ["research_supervisor", "rewrite_report"]:
                        chunk_text = ""
                        content_piece = msg_data.get("content", "")
                        tool_calls = msg_data.get("tool_calls", [])

                        if isinstance(content_piece, str) and content_piece:
                            chunk_text = content_piece
                        elif tool_calls:
                            for tc in tool_calls:
                                args = tc.get("args", "") or tc.get("function", {}).get("arguments", "")
                                if isinstance(args, str): chunk_text += args

                        if chunk_text:
                            raw_report += chunk_text
                            token_count += 1

                            # 把 thinking... 部分从显示给用户的文本中剔除掉，只保留最终回复内容
                            if "<think>" in raw_report:
                                clean_text = re.sub(r'<think>.*?(?:</think>|$)', '', raw_report, flags=re.DOTALL)
                            else:
                                clean_text = raw_report

                            if clean_text.strip():
                                # 性能优化：轻微降频刷新UI，防止 WebSocket 被高频打字机拥塞撑爆
                                if token_count % 3 == 0:
                                    message_placeholder.markdown(clean_text.strip() + " ▌")
                                if node_name == "rewrite_report":
                                    final_report = clean_text.strip()

            if final_report:
                message_placeholder.markdown(final_report)
                status.update(label="✅ 深度调研报告已生成", state="complete", expanded=False)
            else:
                final_state = client.threads.get_state(st.session_state.thread_id)
                if final_state and final_state.get("next") and "human_review" in final_state["next"]:
                    status.update(label="⏸️ 检索已就绪，等待您的审批指令...", state="complete", expanded=False)
                    st.session_state.pending_interrupt = True
                else:
                    status.update(label="⚠️ 调研进程被意外中断 (后端异常或超时)", state="error", expanded=False)

            return final_report


        # 直接同步调用执行
        final_rep = stream_agent_run()

        if final_rep:
            st.session_state.messages.append({"role": "assistant", "content": final_rep})

        if st.session_state.get("pending_interrupt"):
            st.rerun()