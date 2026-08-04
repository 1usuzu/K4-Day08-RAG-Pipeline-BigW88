"""Giao diện Streamlit cho E-commerce Support RAG Chatbot.

Chạy ứng dụng:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Trợ lý chính sách Shopee",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SUGGESTIONS = [
    "Thời hạn yêu cầu trả hàng hoặc hoàn tiền là bao lâu?",
    "Tôi cần chuẩn bị bằng chứng gì để yêu cầu hoàn tiền?",
    "Đơn hàng của tôi chưa cập nhật trạng thái, tôi cần làm gì?",
    "Quy định đăng bán sản phẩm dành cho người bán là gì?",
]


def initialise_state() -> None:
    """Khởi tạo lịch sử hội thoại và câu hỏi từ nút gợi ý."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_query", None)


def show_sources(sources: list[dict]) -> None:
    """Hiển thị tài liệu tham khảo mà không làm rối luồng hội thoại."""
    if not sources:
        return

    with st.expander(f"Nguồn tham khảo · {len(sources)} đoạn", expanded=False):
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})
            source_name = metadata.get("source", "Tài liệu không rõ tên")
            document_type = metadata.get("type", "document")
            score = source.get("score", 0)
            st.markdown(
                f"**{index}. {source_name}**<br>`Loại: {document_type}` · `Điểm: {score:.3f}`"
            )
            st.caption(source.get("content", "")[:360].strip() + "…")
            if index < len(sources):
                st.divider()


def render_message(message: dict) -> None:
    """Render một message theo đúng Streamlit chat UI."""
    role = message["role"]
    avatar = "💬" if role == "assistant" else "🙂"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])
        if role == "assistant":
            show_sources(message.get("sources", []))


def answer_question(query: str, top_k: int) -> tuple[str, list[dict]]:
    """Gọi Task 10 và trả về lỗi thân thiện nếu pipeline chưa sẵn sàng."""
    try:
        from src.task10_generation import generate_with_citation

        response = generate_with_citation(query, top_k=top_k)
        return response.get("answer", "Tôi chưa tìm được câu trả lời phù hợp."), response.get(
            "sources", []
        )
    except Exception:
        return (
            "Mình chưa thể kết nối dịch vụ trả lời ngay lúc này. "
            "Bạn thử lại sau ít phút nhé.",
            [],
        )


initialise_state()

st.markdown(
    """
    <style>
      .stApp {
        background: linear-gradient(180deg, #ffffff 0%, #ffffff 22%, #eaf9ff 42%, #f7fdff 100%);
        min-height: 100vh;
      }
      .stApp::before, .stApp::after {
        content: ""; position: fixed; z-index: 0; pointer-events: none; opacity: .9;
      }
      .stApp::before {
        width: 42vw; height: 42vw; right: -20vw; top: 12vh;
        background: linear-gradient(145deg, #ecfbff 28%, #bceeff 28% 58%, #2ba7dc 58%);
        clip-path: polygon(35% 0, 100% 0, 100% 100%, 0 52%);
      }
      .stApp::after {
        width: 35vw; height: 30vw; left: -11vw; bottom: -7vw;
        background: linear-gradient(135deg, #0784c0 0 28%, #57c7ed 28% 55%, #d7f6ff 55%);
        clip-path: polygon(0 0, 100% 100%, 0 100%);
      }
      .block-container { position: relative; z-index: 1; max-width: 910px; padding: 1.15rem 1rem 7rem; }
      [data-testid="stSidebar"] { background: rgba(255, 255, 255, .96); }
      [data-testid="stSidebar"] > div:first-child { border-right: 1px solid #dcecf5; }
      .topbar {
        display: flex; justify-content: space-between; align-items: center;
        padding: .2rem .15rem 1.15rem; border-bottom: 1px solid #edf3f6; margin-bottom: 1.1rem;
      }
      .brand { color: #0c83be; font-size: .9rem; font-weight: 800; letter-spacing: .02em; }
      .brand span { display: block; color: #5b7890; font-size: .65rem; font-weight: 600; }
      .top-actions { color: #4c7895; font-size: .72rem; }
      .top-actions b { background: #e9f8ff; color: #168ac3; border-radius: 20px; padding: .34rem .65rem; margin-left: .35rem; }
      .hero {
        width: min(630px, 92%); margin: 1.7rem auto 1.25rem; padding: 1.55rem 2rem 1.45rem;
        background: rgba(255,255,255,.96); border-radius: 0 0 24px 24px;
        box-shadow: 0 12px 24px rgba(42, 111, 145, .16); border-top: 3px solid #119ed8;
      }
      .hero h1 { color: #0872ae; font-size: clamp(1.4rem, 3vw, 2.05rem); line-height: 1.05; margin: .35rem 0 .65rem; }
      .hero p { color: #39627d; font-size: .92rem; line-height: 1.55; margin: 0; }
      .eyebrow { color: #2c91c2; font-weight: 800; font-size: .68rem; letter-spacing: .13em; }
      .suggestion-label { color: #247da7; font-size: .68rem; font-weight: 800; letter-spacing: .16em; text-align: center; margin: 1.25rem 0 .6rem; }
      .stButton > button {
        background: rgba(255,255,255,.94); border: 1px solid #cceafa; border-radius: 10px;
        color: #1972a3; min-height: 47px; text-align: left; font-size: .78rem; font-weight: 600; transition: .15s ease;
      }
      .stButton > button:hover { border-color: #159bd6; color: #09618e; background: #effaff; transform: translateY(-1px); }
      [data-testid="stChatMessage"] {
        background: rgba(255,255,255,.90); border: 1px solid #d8edf7;
        border-radius: 15px; padding: .35rem .55rem; margin: .75rem auto; box-shadow: 0 5px 14px rgba(60, 130, 160, .06);
      }
      [data-testid="stChatInput"] textarea { border-radius: 9px; border-color: #91cfe9; }
      [data-testid="stExpander"] { background: rgba(255,255,255,.75); border-radius: 10px; }
      @media (max-width: 700px) {
        .hero { width: 96%; padding: 1.3rem; } .top-actions { display: none; }
        .block-container { padding-left: .65rem; padding-right: .65rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ Tuỳ chỉnh tìm kiếm")
    st.caption("Điều chỉnh cách chatbot tra cứu nguồn tham khảo.")
    
    top_k = st.slider(
        "Số chunks retrieval (top_k)",
        min_value=3,
        max_value=10,
        value=5,
        help="Số lượng chunks tài liệu được đưa vào context cho LLM. Tăng top_k → nhiều evidence hơn nhưng dễ 'lost in the middle'; giảm top_k → câu trả lời tập trung hơn nhưng có thể thiếu ngữ cảnh.",
    )
    st.caption("Nhiều nguồn hơn giúp đối chiếu tốt hơn nhưng có thể chậm hơn.")

    st.divider()
    st.subheader("🧭 Kiến trúc hệ thống")
    with st.expander("Supervisor + Workers song song", expanded=False):
        st.markdown(
            """
**Supervisor** (`task9_retrieval_pipeline.retrieve`) điều phối 2 workers chạy song song rồi tổng hợp kết quả:

1. 🔎 **Semantic Worker** — dense vector search (embeddings)
2. 🔤 **Lexical Worker** — BM25 keyword search
Maybe I'm like
**Supervisor** sau đó:
- Merge kết quả 2 workers bằng **RRF** (Reciprocal Rank Fusion)
- **Rerank** lại theo mức độ liên quan
- Nếu điểm cosine gốc (semantic) < ngưỡng → **fallback** sang **PageIndex** (vectorless)
- Trả `top_k` chunks tốt nhất cho **LLM Generation** sinh câu trả lời kèm citation
            """
        )
    st.caption("Hybrid Retrieval (Semantic + BM25) → RRF Rerank → PageIndex Fallback → LLM Generation có Citation")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("🛒 E-commerce Support RAG Chatbot")
st.caption("Hệ thống hỏi đáp chính sách e-commerce và trợ giúp khách hàng")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander(f"📚 Nguồn tham khảo ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "Unknown")
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0)
                    st.markdown(f"**[{i}] {source_name}** `{doc_type}` | score: `{score:.4f}`")
                    st.text(src.get("content", "")[:300] + "...")
                    st.divider()

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn…")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    user_message = {"role": "user", "content": query}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant", avatar="💬"):
        with st.spinner("Mình đang tìm chính sách phù hợp…"):
            answer, sources = answer_question(query, top_k)
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
