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
    st.header("Thông tin Lab DAY08")
    st.markdown("**Dự án:** E-commerce Support RAG Chatbot (Track 3)")
    st.markdown("""
    Hệ thống RAG nâng cao kết hợp nhiều kỹ thuật tiên tiến được xây dựng trong chuỗi bài tập:
    
    - **Data Pipeline**: Crawl4AI & MarkItDown
    - **Chunking**: RecursiveCharacterTextSplitter
    - **Hybrid Search**: Semantic (BAAI/bge-m3) kết hợp Lexical (BM25)
    - **Vector DB**: ChromaDB Local
    - **Reranking**: Reciprocal Rank Fusion (RRF)
    - **Fallback**: Vectorless RAG qua PageIndex API
    - **Generation**: Áp dụng Lost-in-the-middle reordering
    - **LLM Engine**: gpt-4o-mini (OpenRouter)
    """)
    
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
Hãy thử đặt các câu hỏi hóc búa để kiểm tra khả năng truy xuất, xếp hạng và xử lý ngôn ngữ của Bot nhé!
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lại tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Nguồn trích dẫn"):
                for i, src in enumerate(message["sources"], 1):
                    source_name = src.get('metadata', {}).get('source', 'Unknown')
                    st.markdown(f"**[{i}] File: {source_name}**")
                    st.text(src.get('content', '')[:300] + "...")

# Xử lý tin nhắn mới
if prompt := st.chat_input("Nhập câu hỏi (vd: Trả hàng mất phí bao nhiêu?)..."):
    # Thêm tin nhắn user vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chạy AI sinh câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang lục tìm chính sách Shopee..."):
            try:
                result = generate_with_citation(prompt)
                answer = result["answer"]
                sources = result["sources"]
                
                # Hiển thị câu trả lời
                st.markdown(answer)
                
                # Hiển thị nguồn
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
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {str(e)}")
                st.markdown("*(Gợi ý: Kiểm tra lại các file config và API Key)*")
