"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang chính thức của một sàn TMĐT.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai Shopee Vietnam — help.shopee.vn):
    - https://help.shopee.vn/portal/4/article/77251 (Chính sách trả hàng và hoàn tiền)
    - https://help.shopee.vn/portal/4/article/79198 (Phương thức thanh toán)
    - https://help.shopee.vn/portal/4/article/77244 (Chính sách bảo mật)

Gợi ý văn bản (chủ đề chính sách thương mại điện tử):
    - Chính sách đổi trả/hoàn tiền (Returns/Refund Policy)
    - Phương thức thanh toán (Payment Methods)
    - Chính sách bảo mật (Privacy Policy)
    - Quy định đăng bán sản phẩm cho người bán (Seller Listing Regulations)

Nhớ gắn metadata `customer_role` (`buyer`/`seller`/`both`) cho từng tài liệu — yêu cầu riêng
của K4 Variant (kế thừa từ Lab 07), cần thiết để viết benchmark query dùng metadata_filter.

Lưu ý: một số trang help center dùng JavaScript render nội dung (SPA) — crawl về chỉ thấy
tiêu đề mà không có nội dung thật. Đổi sang bài viết khác cùng domain thay vì cố xử lý,
và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from fpdf import FPDF

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# Nguồn: Shopee Vietnam Trung tâm trợ giúp (help.shopee.vn) — trang công khai,
# nội dung render sẵn ở server-side (không phải SPA thuần JS).
DOCUMENTS = [
    # {
    #     "url": "https://help.shopee.vn/portal/4/article/77250",
    #     "filename": "shopee_chinh_sach_van_chuyen.pdf",
    #     "title": "Chinh sach van chuyen Shopee",
    #     "customer_role": "both",
    # },
    # {
    #     "url": "https://help.shopee.vn/portal/4/article/77251",
    #     "filename": "shopee_chinh_sach_tra_hang_hoan_tien.pdf",
    #     "title": "Chinh sach tra hang va hoan tien Shopee",
    #     "customer_role": "both",
    # },
    # {
    #     "url": "https://help.shopee.vn/portal/4/article/77244",
    #     "filename": "shopee_chinh_sach_bao_mat.pdf",
    #     "title": "Chinh sach bao mat Shopee",
    #     "customer_role": "both",
    # },
    # {
    #     "url": "https://help.shopee.vn/portal/4/article/77246",
    #     "filename": "shopee_quy_dinh_dang_ban_san_pham.pdf",
    #     "title": "Quy dinh ve dang ban san pham tren Shopee",
    #     "customer_role": "seller",
    # },
    {
        "url": "https://help.shopee.vn/portal/4/article/77265",
        "filename": "shopee_quy_dinh_giai_quyet_tranh_chap.pdf",
        "title": "Quy dinh ve giai quyet tranh chap va xu ly khieu nai",
        "customer_role": "both",
    },
]

# Font Unicode để fpdf2 render được tiếng Việt có dấu (font core của fpdf2 chỉ
# hỗ trợ Latin-1). Dùng Arial có sẵn trên Windows.
UNICODE_FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def fetch_article_text(url: str) -> tuple[str, str]:
    """Tải trang help center và trích xuất tiêu đề + nội dung văn bản thuần.

    Trả về (title, body_text). Raise nếu không tìm thấy nội dung — dấu hiệu
    trang là SPA JS-render, cần đổi sang bài viết khác cùng domain.
    """
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    article_el = soup.select_one("[class*=article_detail]")
    if article_el is None:
        raise ValueError(
            f"Không tìm thấy nội dung tại {url} — có thể trang này là SPA "
            "JS-render, hãy đổi sang bài viết khác cùng domain."
        )

    title_tag = article_el.select_one("[class*=title]")
    title = title_tag.get_text(strip=True) if title_tag else url

    # ".ssr-key-content" là class cố định (không hash) bọc đúng phần thân bài
    # viết, tránh lẫn widget "Bạn có hài lòng" / bài viết liên quan là các
    # phần tử anh em (sibling) nằm ngoài div này.
    body_el = article_el.select_one(".ssr-key-content")
    if body_el is None or not body_el.get_text(strip=True):
        raise ValueError(
            f"Không tìm thấy nội dung tại {url} — có thể trang này là SPA "
            "JS-render, hãy đổi sang bài viết khác cùng domain."
        )

    # Mỗi thẻ block (p, li, h*) xuống một dòng riêng để giữ cấu trúc văn bản.
    lines = []
    for el in body_el.find_all(["p", "li", "h1", "h2", "h3", "h4"]):
        text = el.get_text(" ", strip=True)
        if text:
            lines.append(text)

    body_text = "\n".join(lines) if lines else body_el.get_text("\n", strip=True)
    return title, body_text


def save_as_pdf(title: str, body_text: str, filepath: Path):
    """Ghi title + body_text thành file PDF, hỗ trợ Unicode tiếng Việt."""
    pdf = FPDF()
    pdf.add_page()

    if UNICODE_FONT_PATH.exists():
        pdf.add_font("Arial", "", str(UNICODE_FONT_PATH))
        pdf.set_font("Arial", size=14)
    else:
        pdf.set_font("Helvetica", size=14)

    pdf.multi_cell(0, 10, title)
    pdf.ln(4)

    font_name = "Arial" if UNICODE_FONT_PATH.exists() else "Helvetica"
    pdf.set_font(font_name, size=11)
    pdf.multi_cell(0, 7, body_text)

    pdf.output(str(filepath))


def download_document(doc: dict):
    """Tải 1 văn bản chính sách: fetch HTML -> extract text -> lưu PDF."""
    print(f"Đang tải: {doc['url']}")
    title, body_text = fetch_article_text(doc["url"])
    filepath = DATA_DIR / doc["filename"]
    save_as_pdf(title, body_text, filepath)
    print(f"  ✓ Đã lưu: {filepath} ({len(body_text)} ký tự)")


def write_manifest():
    """Ghi manifest.json ghi lại nguồn + metadata customer_role cho từng tài liệu.

    Task 4 (chunking) đọc manifest này để gắn metadata `customer_role`
    (`buyer`/`seller`/`both`) cho từng chunk, theo yêu cầu riêng của K4 Variant.
    """
    manifest = [
        {
            "filename": doc["filename"],
            "url": doc["url"],
            "title": doc["title"],
            "customer_role": doc["customer_role"],
        }
        for doc in DOCUMENTS
    ]
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✓ Đã ghi manifest: {manifest_path}")


def collect_all():
    setup_directory()
    for doc in DOCUMENTS:
        download_document(doc)
    write_manifest()
    print(f"\n✓ Hoàn tất: {len(DOCUMENTS)} văn bản trong {DATA_DIR}")


if __name__ == "__main__":
    collect_all()
