import re
import httpx
import streamlit as st
from langgraph_sdk import get_sync_client  # 【修复 1】导入同步客户端

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

# ================= 客户端获取工厂 =================
LANGGRAPH_URL = "http://localhost:2024"
ASSISTANT_ID = "Deep Researcher"


def get_new_client():
    """
    每次调用时创建一个新的客户端实例，配置 3600 秒超长超时时间防止深度调研中断。
    """
    timeout_config = httpx.Timeout(3600.0)
    # 【修复 1】使用 get_sync_client 替代 get_client，彻底拥抱 Streamlit 的同步特性
    return get_sync_client(
        url=LANGGRAPH_URL,
        headers={"Authorization": "Bearer local-dev-token-123"},
        timeout=timeout_config
    )


# ================= 工具函数 (移除所有 async/await) =================
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


# ================= 状态初始化 =================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= 侧边栏：历史会话管理 =================
with st.sidebar:
    st.title("💬 会话历史")
    if st.button("➕ 新建深度调研", use_container_width=True, type="primary"):
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("最近 5 次会话")

    # 直接同步调用，无需 asyncio.run
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
        if current_state and current_state.get("next") and "human_review" in current_state["next"]:
            st.session_state.pending_interrupt = True
            tasks = current_state.get("tasks", [])
            if tasks and tasks[0].get("interrupts"):
                st.session_state.interrupt_data = tasks[0]["interrupts"][0].get("value", {})
        else:
            st.session_state.pending_interrupt = False
    except Exception:
        pass

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
            # 【修复 1】直接同步调用
            new_id = create_new_thread(title=short_title)
            st.session_state.thread_id = new_id

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    with st.chat_message("assistant"):
        status_label = "🔄 收到干预指令，正在恢复流转..." if is_resuming else "🧠 主控节点已启动，正在分析需求..."
        status = st.status(status_label, expanded=True)
        message_placeholder = st.empty()


        # 改为常规同步函数，去掉 async
        def stream_agent_run():
            if is_resuming:
                payload = st.session_state.resume_payload
                st.session_state.resume_payload = None
                st.session_state.pending_interrupt = False

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
            seen_ids = set()
            display_locked_to_final = False
            token_count = 0  # 用于降低 UI 刷新频率

            # 去掉 async for，直接使用 for
            for chunk in stream:
                if chunk.event == "updates":
                    completed_nodes = list(chunk.data.keys())
                    for node in completed_nodes:
                        if node == "clarify_with_user":
                            status.update(label="📝 正在生成结构化调研提纲...", expanded=False)
                        elif node == "write_research_brief":
                            status.update(label="🧠 主管已接管，正在规划战略...", expanded=False)
                            if not display_locked_to_final: message_placeholder.empty()
                        elif node == "supervisor":
                            status.update(label="🕵️‍♂️ 主管正在调度研究员与分配子任务...", expanded=False)
                        elif node == "researcher":
                            status.update(label="🔍 子研究员正在全网检索与深度阅读...", expanded=False)
                        elif node == "compress_research":
                            status.update(label="🗜️ 子研究员正在压缩与提炼结构化事实库...", expanded=False)
                        elif node == "gen":
                            status.update(label="📑 正在根据情报规划报告大纲...", expanded=False)
                        elif node == "generate_outline":
                            status.update(label="⚡ 正在并发撰写各章节内容...", expanded=False)
                            if not display_locked_to_final:
                                message_placeholder.markdown(
                                    "> ⚡ **多智能体并发撰写中**：已将大纲与检索事实切片下发给多个写手节点，正在极速成文中...")

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

                elif chunk.event == "messages/partial" and not display_locked_to_final:
                    msg_data = chunk.data[0] if isinstance(chunk.data, list) and len(chunk.data) > 0 else chunk.data
                    metadata = chunk.data[1] if isinstance(chunk.data, list) and len(chunk.data) > 1 else {}
                    node_name = metadata.get("langgraph_node", "")

                    if node_name == "rewrite_report":
                        status.update(label="✨ 根据核查意见重写报告...", expanded=False)
                        display_locked_to_final = False

                    if node_name in ["supervisor", "rewrite_report"]:
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

                            # 性能优化：仅在包含 <think> 时才执行消耗性能的正则操作
                            if "<think>" in raw_report:
                                clean_text = re.sub(r'<think>.*?(?:</think>|$)', '', raw_report, flags=re.DOTALL)
                            else:
                                clean_text = raw_report

                            if clean_text.strip():
                                # 性能优化：轻微降频刷新UI，防止 WebSocket 被高频打字机拥塞撑爆
                                if token_count % 3 == 0 or node_name == "rewrite_report":
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