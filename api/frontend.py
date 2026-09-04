import asyncio
import re
import streamlit as st
from langgraph_sdk import get_client

# ================= 页面与主题配置 =================
st.set_page_config(page_title="Deep Research", layout="wide")

st.markdown("""
<style>
    /* 消除顶部巨大留白 */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
        max-width: 1200px !important;
    }
    header {visibility: hidden;}

    /* 深蓝色系按钮，仿 DeepSeek */
    button[kind="primary"] {
        background-color: #1565C0 !important;
        color: white !important;
        border-color: #1565C0 !important;
    }
    button[kind="primary"]:hover {
        background-color: #0D47A1 !important;
        border-color: #0D47A1 !important;
    }
    section[data-testid="stSidebar"] button p {
        text-align: left !important;
    }
</style>
""", unsafe_allow_html=True)

# ================= 核心修复：客户端获取工厂 =================
LANGGRAPH_URL = "http://localhost:2024"
ASSISTANT_ID = "Deep Researcher"


def get_new_client():
    """
    【核心修复】每次调用时创建一个新的客户端实例。
    彻底解决 Streamlit 刷新导致的 Event loop is closed 报错问题。
    """
    return get_client(url=LANGGRAPH_URL,headers={"Authorization": "Bearer local-dev-token-123"})


# ================= 工具函数 =================
async def get_recent_threads():
    """从数据库获取最近的 5 个历史会话"""
    client = get_new_client()
    try:
        return await client.threads.search(limit=5)
    except Exception:
        return []


async def create_new_thread(title):
    """创建会话并附带标题"""
    client = get_new_client()
    thread = await client.threads.create(metadata={"title": title})
    return thread["thread_id"]


async def fetch_thread_messages(thread_id):
    """深度解析历史消息，用于历史会话的内容回显"""
    client = get_new_client()
    try:
        state = await client.threads.get_state(thread_id)
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

        # 提取图状态中的独立最终报告
        final_report = state["values"].get("final_report", "")
        if final_report and not any(msg["content"] == final_report for msg in chat_history):
            chat_history.append({"role": "assistant", "content": final_report})

        return chat_history
    except Exception as e:
        st.sidebar.error(f"解析历史记录失败: {e}")
        return []


# ================= 状态初始化 =================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= 侧边栏：真·历史会话管理 =================
with st.sidebar:
    st.title("💬 会话历史")

    if st.button("➕ 新建深度调研", use_container_width=True, type="primary"):
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("最近 5 次会话")

    # 获取只包含 5 条数据的列表
    threads = asyncio.run(get_recent_threads())

    for t in threads:
        tid = t["thread_id"]
        title = t.get("metadata", {}).get("title", f"会话 {tid[:8]}")

        is_current = (tid == st.session_state.thread_id)
        button_type = "primary" if is_current else "secondary"

        if st.button(title, key=tid, use_container_width=True, type=button_type):
            if not is_current:
                st.session_state.thread_id = tid
                st.session_state.messages = asyncio.run(fetch_thread_messages(tid))
                st.rerun()

# ================= 主页面：聊天交互区 =================
if not st.session_state.messages:
    st.markdown("<h1 style='text-align: center; color: #333; margin-top: 15vh;'>🔍 Deep Research 智能体</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>在下方输入您的调研需求，系统将自动开始多维深搜。</p>",
                unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ================= 底部固定：输入框 =================
# ================= 底部固定：输入框 =================
if prompt := st.chat_input("输入调研需求（如：对比宁德时代与比亚迪最新财报）..."):

    if st.session_state.thread_id is None:
        short_title = prompt[:12] + "..." if len(prompt) > 12 else prompt
        new_id = asyncio.run(create_new_thread(title=short_title))
        st.session_state.thread_id = new_id

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 【核心改造 1】引入可折叠的进度指示器 (模拟 DeepSeek 的 <think> 标签)
        status = st.status("🧠 主控节点已启动，正在规划调研路径...", expanded=True)
        message_placeholder = st.empty()


        async def stream_agent_run():
            client = get_new_client()
            payload = {"messages": [{"role": "user", "content": prompt}]}

            stream = client.runs.stream(
                thread_id=st.session_state.thread_id,
                assistant_id=ASSISTANT_ID,
                input=payload,
                stream_mode=["values", "messages"]
            )

            final_report = ""
            raw_report = ""  # 用于存放包含 <think> 的原始字符串
            current_streaming_node = ""  # 用于追踪当前是哪个节点在输出
            seen_ids = set()

            async for chunk in stream:
                # ==========================================
                # 通道 A：解析状态流 (折叠面板)
                # ==========================================
                if chunk.event == "values":
                    state = chunk.data

                    # 侦测核查器是否发现了幻觉
                    if "verification_feedback" in state and state["verification_feedback"]:
                        # 确保只播报一次
                        fb_id = str(hash(state["verification_feedback"]))
                        if fb_id not in seen_ids:
                            seen_ids.add(fb_id)
                            status.write("⚠️ **核查未通过**: 发现未证实断言，正在打回重写...")

                    if "messages" in state:
                        for msg in state["messages"]:
                            m_id = msg.get("id", str(hash(str(msg))))
                            if m_id not in seen_ids:
                                seen_ids.add(m_id)
                                m_type = msg.get("type", "")

                                if m_type == "ai" and msg.get("tool_calls"):
                                    tools = [tc["name"] for tc in msg["tool_calls"]]
                                    if "ConductResearch" in tools:
                                        status.write("🚀 **调度集群**: 派发子探员进行多路并行深度检索...")
                                    elif "think_tool" in tools:
                                        status.write("🤔 **战略反思**: 主管正在评估当前情报的知识边界...")
                                elif m_type == "tool":
                                    tool_name = msg.get("name", "")
                                    if tool_name == "ConductResearch":
                                        status.write("✅ **情报汇聚**: 子探员完成检索，提炼出结构化事实。")

                    if state.get("consecutive_low_gain_rounds", 0) >= 2:
                        status.write("🛑 **知识饱和**: 剪枝机制触发，停止搜寻，开始成文...")

                # ==========================================
                # 通道 B：解析 Token 流 (正文打字机与覆盖机制)
                # ==========================================
                elif chunk.event == "messages/partial":
                    msg_data, metadata = chunk.data
                    node_name = metadata.get("langgraph_node", "")

                    # 仅拦截成文与重写节点的文本流
                    if node_name in ["final_report_generation", "rewrite_report"]:

                        # 【核心修复 1】：一旦切换输出节点，立刻清空缓存，实现“重写覆盖”！
                        if current_streaming_node != node_name:
                            current_streaming_node = node_name
                            raw_report = ""
                            final_report = ""

                            # 动态更新面板标题
                            action = "初稿撰写中" if node_name == "final_report_generation" else "根据核查意见重写中"
                            if status.state == "running":
                                status.update(label=f"✨ {action}...", state="complete", expanded=False)
                            else:
                                # 如果之前已经折叠，重新展开提醒用户正在重写
                                status.update(label=f"✨ {action}...", expanded=False)

                        # 安全提取 Token 文本
                        content_piece = msg_data.get("content", "")
                        if isinstance(content_piece, str):
                            raw_report += content_piece
                        elif isinstance(content_piece, list):
                            for block in content_piece:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    raw_report += block.get("text", "")

                        # 【核心修复 2】：使用正则实时过滤 <think> 块，包括未闭合的 <think>
                        clean_text = re.sub(r'<think>.*?(?:</think>|$)', '', raw_report, flags=re.DOTALL)

                        # 只有当非 think 内容产生时，才更新屏幕
                        if clean_text.strip():
                            final_report = clean_text.strip()
                            message_placeholder.markdown(final_report + " ▌")

            if final_report:
                message_placeholder.markdown(final_report)
            else:
                status.update(label="⚠️ 调研进程被意外中断", state="error", expanded=False)

            return final_report