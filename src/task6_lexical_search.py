"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def _tokenize(text: str) -> list[str]:
    """Chuẩn hoá văn bản thành tokens, vẫn giữ được ký tự tiếng Việt."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _load_default_corpus() -> list[dict]:
    """Nạp Markdown đã chuẩn hoá; dùng dữ liệu crawl thô khi Markdown chưa có."""
    corpus: list[dict] = []
    standardized_dir = PROJECT_ROOT / "data" / "standardized"

    for filepath in sorted(standardized_dir.rglob("*.md")):
        content = filepath.read_text(encoding="utf-8").strip()
        if content:
            corpus.append(
                {
                    "content": content,
                    "metadata": {
                        "source": filepath.name,
                        "type": filepath.parent.name,
                        "path": str(filepath.relative_to(PROJECT_ROOT)),
                    },
                }
            )

    if corpus:
        return corpus

    # Task 3 chưa tạo Markdown trong repo hiện tại, nên dùng 15 bài crawl sẵn có
    # để module vẫn có corpus thật và app có thể demo lexical retrieval.
    news_dir = PROJECT_ROOT / "data" / "landing" / "news"
    for filepath in sorted(news_dir.glob("*.json")):
        try:
            article = json.loads(filepath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        content = article.get("content_markdown", "").strip()
        if content:
            corpus.append(
                {
                    "content": content,
                    "metadata": {
                        "source": article.get("title", filepath.name),
                        "type": "news",
                        "url": article.get("url", ""),
                        "category": article.get("category", ""),
                        "path": str(filepath.relative_to(PROJECT_ROOT)),
                    },
                }
            )

    return corpus


CORPUS: list[dict] = _load_default_corpus()


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus:
        return None

    tokenized_corpus = [_tokenize(str(doc.get("content", ""))) for doc in corpus]
    if not any(tokenized_corpus):
        return None

    from rank_bm25 import BM25Okapi

    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if top_k <= 0 or not CORPUS:
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    bm25 = build_bm25_index(CORPUS)
    if bm25 is None:
        return []

    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(
        range(len(CORPUS)), key=lambda index: scores[index], reverse=True
    )

    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        document = CORPUS[index]
        results.append(
            {
                "content": document.get("content", ""),
                "score": score,
                "metadata": document.get("metadata", {}),
            }
        )
        if len(results) == top_k:
            break

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
