"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
from dotenv import load_dotenv

load_dotenv()

def _generate_hypothetical_doc(query: str) -> str:
    """
    Sinh tài liệu giả định (Hypothetical Document) từ câu truy vấn của người dùng.
    Dùng LLM để tạo ra một đoạn văn bản trả lời giả định.
    """
    from openai import OpenAI
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ Không tìm thấy API Key, HyDE sẽ bị vô hiệu hoá. Trả về query gốc.")
        return query
        
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    model = "openai/gpt-4o-mini"
    
    prompt = f"""Bạn là một chuyên gia hỗ trợ khách hàng của Shopee.
Người dùng hỏi: "{query}"
Hãy viết một đoạn văn ngắn (khoảng 2-3 câu) giải đáp câu hỏi này một cách trực tiếp, giống như nội dung được trích xuất từ tài liệu chính thức của Shopee. KHÔNG cần chào hỏi hay giải thích dài dòng."""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        fake_doc = response.choices[0].message.content.strip()
        print(f"  [HyDE] Generated fake doc: {fake_doc[:100]}...")
        return fake_doc
    except Exception as e:
        print(f"  [HyDE] Lỗi khi sinh fake doc: {e}. Trả về query gốc.")
        return query


def semantic_search(query: str, top_k: int = 10, use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.
    Nếu use_hyde = True, sẽ sinh hypothetical document trước khi nhúng.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        use_hyde: Có sử dụng HyDE hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict
        }
    """
    from .task4_chunking_indexing import get_collection, get_embedding_model
    
    # 1. Áp dụng HyDE nếu được yêu cầu
    search_text = query
    if use_hyde:
        print(f"  [HyDE] Áp dụng HyDE cho query: '{query}'")
        hypothetical_doc = _generate_hypothetical_doc(query)
        # Kết hợp query gốc và fake doc để nhúng
        search_text = f"{query}\n{hypothetical_doc}"

    # 2. Nhúng search_text thành vector
    model = get_embedding_model()
    query_vector = model.encode(search_text).tolist()

    # 3. Tìm kiếm trong ChromaDB
    collection = get_collection()
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    # results["documents"] là list of list, ta lấy phần tử đầu tiên
    if not results["documents"] or not results["documents"][0]:
        return output
        
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # ChromaDB dùng cosine distance, similarity = 1 - distance
        score = max(0.0, 1.0 - dist)
        output.append({"content": doc, "score": round(score, 4), "metadata": meta})

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    import sys
    # Fix lỗi không in được tiếng Việt trên Windows
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    # Test
    print("--- Test TÌM KIẾM BÌNH THƯỜNG ---")
    results = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=3, use_hyde=False)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
        
    print("\n--- Test TÌM KIẾM CÓ HYDE ---")
    results_hyde = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=3, use_hyde=True)
    for r in results_hyde:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
