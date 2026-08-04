import streamlit as st
import sys
import io

# Force UTF-8 cho console print để tránh UnicodeEncodeError trên Windows
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import os
from dotenv import load_dotenv

# Load env
load_dotenv()

# Cấu hình đường dẫn để import được thư mục src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.task10_generation import generate_with_citation

# Cấu hình giao diện trang web
st.set_page_config(
    page_title="Shopee Support RAG Bot", 
    layout="centered"
)

# Sidebar
with st.sidebar:
    st.header("Thông tin Kiến trúc")
    st.markdown("**Kiến trúc Supervisor RAG**")
    st.markdown("""
    Hệ thống sử dụng mô hình Supervisor để điều phối thông minh các luồng:
    - **Dense Search** (Semantic qua BAAI/bge-m3)
    - **Sparse Search** (Lexical qua BM25)
    - **Reranker** (RRF kết hợp xếp hạng)
    - **Fallback logic** (PageIndex Vectorless)
    """)
    
    st.divider()
    
    # Tạo slider điều chỉnh top_k trong sidebar
    top_k = st.slider("Số lượng chunks (top_k) cần truy xuất:", min_value=3, max_value=10, value=5, step=1)
    
    st.divider()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key.endswith("..."):
        st.error("CẢNH BÁO: Bạn chưa cấu hình OPENROUTER_API_KEY thật trong file .env!")
    else:
        st.success("API Key đã sẵn sàng!")

# Main Chat Interface
st.title("Shopee Support Assistant")
st.markdown("""
Chào mừng bạn đến với Chatbot tư vấn chính sách Shopee. Đây là sản phẩm tổng hợp toàn bộ các kỹ thuật RAG từ Day 08.
""")

# Thêm các nút bấm gợi ý câu hỏi mẫu
st.markdown("### Gợi ý câu hỏi:")
col1, col2, col3 = st.columns(3)
suggestion_clicked = None

if col1.button("Trả hàng mất phí bao nhiêu?"):
    suggestion_clicked = "Trả hàng mất phí bao nhiêu?"
if col2.button("Hóa đơn SPayLater quá hạn thì sao?"):
    suggestion_clicked = "Hóa đơn SPayLater quá hạn thì bị phạt như thế nào?"
if col3.button("Hạn sử dụng thực phẩm quy định sao?"):
    suggestion_clicked = "Quy định về hạn sử dụng thực phẩm đăng bán trên Shopee như thế nào?"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lại tin nhắn cũ bằng khung st.chat_message
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Nguồn trích dẫn"):
                for i, src in enumerate(message["sources"], 1):
                    source_name = src.get('metadata', {}).get('source', 'Unknown')
                    st.markdown(f"**[{i}] File: {source_name}**")
                    st.text(src.get('content', '')[:300] + "...")

# Xử lý tin nhắn mới (từ ô input hoặc tự động lấy từ nút gợi ý)
prompt = st.chat_input("Nhập câu hỏi (vd: Trả hàng mất phí bao nhiêu?)...")

if suggestion_clicked:
    prompt = suggestion_clicked

if prompt:
    # Thêm tin nhắn user vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chạy AI sinh câu trả lời
    with st.chat_message("assistant"):
        with st.spinner(f"Đang lục tìm chính sách (lấy top {top_k} tài liệu)..."):
            try:
                # Nối hàm Task 10 và truyền top_k vào
                result = generate_with_citation(prompt, top_k=top_k)
                answer = result["answer"]
                sources = result["sources"]
                
                # Hiển thị câu trả lời
                st.markdown(answer)
                
                # Xử lý danh sách sources trả về cho UI
                if sources:
                    with st.expander("📚 Nguồn trích dẫn"):
                        for i, src in enumerate(sources, 1):
                            source_name = src.get('metadata', {}).get('source', 'Unknown')
                            st.markdown(f"**[{i}] File: {source_name}**")
                            st.text(src.get('content', '')[:300] + "...")
                
                # Lưu vào lịch sử
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
            # Bắt try/except nếu LLM bị lỗi
            except Exception as e:
                st.error(f"Đã xảy ra lỗi từ LLM hoặc hệ thống: {str(e)}")
                st.markdown("*(Gợi ý: Kiểm tra lại API Key hoặc xem kết nối mạng)*")
