import asyncio
import json
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# Mỗi mục là trang danh mục (category) trên help.shopee.vn, chỉ lấy page=1.
# Trang danh mục là SPA (nội dung tải bằng JS) nên phải dùng trình duyệt headless
# (Playwright — engine mà Crawl4AI cũng dùng bên dưới) để lấy link các bài viết con.
CATEGORIES = [
    {
        "name": "Đơn hàng & Vận chuyển",
        "url": "https://help.shopee.vn/portal/4/category/60-%25C4%2590%25C6%25A1n-H%25C3%25A0ng-V%25E1%25BA%25ADn-Chuy%25E1%25BB%2583n/703-%25C4%2590%25C6%25A1n-h%25C3%25A0ng?page=1",
        "customer_role": "buyer",
    },
    {
        "name": "Ví ShopeePay",
        "url": "https://help.shopee.vn/portal/4/category/59-Thanh-To%25C3%25A1n/708-V%25C3%25AD-ShopeePay?page=1",
        "customer_role": "buyer",
    },
    {
        "name": "Chương trình khuyến mãi",
        "url": "https://help.shopee.vn/portal/4/category/58-Khuy%25E1%25BA%25BFn-M%25C3%25A3i-%25C6%25AFu-%25C4%2590%25C3%25A3i/705-Ch%25C6%25B0%25C6%25A1ng-tr%25C3%25ACnh-khuy%25E1%25BA%25BFn-m%25C3%25A3i?page=1",
        "customer_role": "buyer",
    },
    {
        "name": "ShopeeVIP",
        "url": "https://help.shopee.vn/portal/4/category/58-Khuy%25E1%25BA%25BFn-M%25C3%25A3i-%25C6%25AFu-%25C4%2590%25C3%25A3i/17520-ShopeeVIP?page=1",
        "customer_role": "buyer",
    },
    {
        "name": "SPayLater",
        "url": "https://help.shopee.vn/portal/4/category/59-Thanh-To%25C3%25A1n/11438-SPayLater?page=1",
        "customer_role": "buyer",
    },
]

# Số bài viết con tối đa lấy từ mỗi danh mục (5 danh mục x 3 = 15 bài, đủ dư so
# với yêu cầu tối thiểu 5 bài).
MAX_ARTICLES_PER_CATEGORY = 3


async def get_article_links(category_url: str) -> list[str]:
    """Mở trang danh mục (SPA) bằng Playwright, chờ JS render, rồi lấy các
    link bài viết con (dạng /portal/4/article/...).

    Trang danh mục còn chứa vài link footer chung (điều khoản, chính sách...)
    dạng /portal/article/... (thiếu "4" — không phải bài hướng dẫn con của
    danh mục) nên chỉ giữ link đúng dạng /portal/4/article/.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(category_url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(3000)  # chờ React render xong danh sách bài viết
        hrefs = await page.eval_on_selector_all(
            "a[href]", "els => els.map(e => e.getAttribute('href'))"
        )
        await browser.close()

    links = []
    seen = set()
    for href in hrefs:
        if not href or "/portal/4/article/" not in href:
            continue
        url = href if href.startswith("http") else f"https://help.shopee.vn{href}"
        url = url.split("?")[0]  # bỏ query string (vd. ?previousPage=...)
        if url not in seen:
            seen.add(url)
            links.append(url)
    return links


def crawl_article(url: str, category: str, customer_role: str) -> dict:
    """
    Crawl một bài viết (trang chi tiết bài viết được server-side render) và
    trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "category": str,
            "customer_role": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
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

    lines = []
    for el in body_el.find_all(["p", "li", "h1", "h2", "h3", "h4"]):
        text = el.get_text(" ", strip=True)
        if text:
            lines.append(text)
    body_text = "\n".join(lines) if lines else body_el.get_text("\n", strip=True)

    content_markdown = f"# {title}\n\n{body_text}"

    return {
        "url": url,
        "title": title,
        "category": category,
        "customer_role": customer_role,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


async def crawl_all():
    """Duyệt từng danh mục -> lấy link bài viết con -> crawl từng bài -> lưu JSON."""
    setup_directory()

    count = 0
    for cat in CATEGORIES:
        print(f"\n--- Danh mục: {cat['name']} ---")
        print(f"Đang lấy danh sách bài viết: {cat['url']}")
        links = await get_article_links(cat["url"])
        print(f"  Tìm thấy {len(links)} bài viết, lấy tối đa {MAX_ARTICLES_PER_CATEGORY}")

        for url in links[:MAX_ARTICLES_PER_CATEGORY]:
            count += 1
            print(f"[{count}] Crawling: {url}")
            article = crawl_article(url, cat["name"], cat["customer_role"])

            filename = f"article_{count:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  ✓ Saved: {filepath}")

    print(f"\n✓ Hoàn tất: {count} bài viết trong {DATA_DIR}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
