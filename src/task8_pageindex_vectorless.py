"""Task 8 — PageIndex vectorless retrieval with a persistent doc-id cache."""

import json
import os
import time
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PROJECT_ROOT = Path(__file__).parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "landing" / "legal"
DOC_ID_CACHE = PROJECT_ROOT / "data" / "pageindex_doc_ids.json"
RETRIEVAL_URL = "https://api.pageindex.ai/retrieval/"
MAX_RETRIEVAL_POLLS = 10
POLL_INTERVAL_SECONDS = 1


def load_doc_ids() -> dict[str, str]:
    """Đọc ánh xạ ``relative_pdf_path -> PageIndex doc_id`` từ cache JSON."""
    if not DOC_ID_CACHE.exists():
        return {}

    try:
        cached = json.loads(DOC_ID_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(cached, dict):
        return {}
    return {
        str(source_path): str(doc_id)
        for source_path, doc_id in cached.items()
        if isinstance(source_path, str) and isinstance(doc_id, str) and doc_id
    }


def save_doc_ids(doc_ids: dict[str, str]) -> None:
    """Lưu cache theo kiểu atomic để tránh JSON dở dang khi chương trình dừng."""
    DOC_ID_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = DOC_ID_CACHE.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(DOC_ID_CACHE)


def _create_client():
    """Khởi tạo Python SDK trễ để import module không cần API key hoặc SDK."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong file .env")

    try:
        from pageindex import PageIndexClient
    except ImportError as error:
        raise RuntimeError("Chưa cài PageIndex: pip install -U pageindex") from error

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents() -> dict[str, str]:
    """Upload PDF chính sách mới và lưu ``doc_id`` để không upload lại lần sau.

    PageIndex Document Processing hiện nhận PDF, vì vậy dùng file gốc ở
    ``data/landing/legal/`` thay vì Markdown đã chuẩn hoá.
    """
    client = _create_client()
    doc_ids = load_doc_ids()

    for pdf_path in sorted(PDF_DIR.rglob("*.pdf")):
        source_path = str(pdf_path.relative_to(PROJECT_ROOT))
        if source_path in doc_ids:
            continue

        response = client.submit_document(str(pdf_path))
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex không trả doc_id cho {pdf_path.name}")

        doc_ids[source_path] = str(doc_id)
        save_doc_ids(doc_ids)

    return doc_ids


def _iter_relevant_contents(value: object) -> Iterator[dict]:
    """Hỗ trợ cả schema legacy list[dict] và list[list[dict]]."""
    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_relevant_contents(item)


def _extract_results(retrieval: dict, source_path: str) -> list[dict]:
    """Chuẩn hoá ``retrieved_nodes`` của PageIndex thành format retrieval chung."""
    extracted: list[dict] = []
    for node in retrieval.get("retrieved_nodes", []):
        if not isinstance(node, dict):
            continue
        for item in _iter_relevant_contents(node.get("relevant_contents", [])):
            content = item.get("relevant_content", "").strip()
            if not content:
                continue
            extracted.append(
                {
                    "content": content,
                    "metadata": {
                        "source": source_path,
                        "section": item.get("section_title") or node.get("title", ""),
                        "node_id": node.get("node_id", ""),
                        "page_index": item.get("page_index"),
                        "doc_id": retrieval.get("doc_id", ""),
                    },
                    "source": "pageindex",
                }
            )

    # Legacy retrieval không có relevance score; điểm theo rank chỉ phục vụ
    # interface chung với các retriever khác.
    for rank, result in enumerate(extracted, start=1):
        result["score"] = 1.0 / rank
    return extracted


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Truy vấn PageIndex cached documents và trả về các section liên quan.

    Nếu cache chưa có doc_id hoặc chưa cấu hình API key, trả về list rỗng để
    retrieval pipeline có thể tiếp tục hoạt động mà không bị lỗi integration.
    """
    if not query.strip() or top_k <= 0:
        return []

    doc_ids = load_doc_ids()
    if not doc_ids or not PAGEINDEX_API_KEY:
        return []

    import requests

    headers = {"api_key": PAGEINDEX_API_KEY}
    results: list[dict] = []

    for source_path, doc_id in doc_ids.items():
        try:
            response = requests.post(
                RETRIEVAL_URL,
                headers=headers,
                json={"doc_id": doc_id, "query": query, "thinking": False},
                timeout=30,
            )
            response.raise_for_status()
            retrieval_id = response.json().get("retrieval_id")
            if not retrieval_id:
                continue

            retrieval = {}
            for _ in range(MAX_RETRIEVAL_POLLS):
                poll = requests.get(
                    f"{RETRIEVAL_URL}{retrieval_id}/", headers=headers, timeout=30
                )
                poll.raise_for_status()
                retrieval = poll.json()
                if retrieval.get("status") == "completed":
                    break
                if retrieval.get("status") in {"failed", "error"}:
                    retrieval = {}
                    break
                time.sleep(POLL_INTERVAL_SECONDS)
        except requests.RequestException:
            continue

        if retrieval.get("status") == "completed":
            results.extend(_extract_results(retrieval, source_path))
        if len(results) >= top_k:
            break

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://dash.pageindex.ai/")
    else:
        doc_ids = upload_documents()
        print(f"✓ Cache {len(doc_ids)} PageIndex document IDs: {DOC_ID_CACHE}")

        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for result in results:
            print(f"[{result['score']:.3f}] {result['content'][:100]}...")
