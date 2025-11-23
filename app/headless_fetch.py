# app/headless_fetch.py
from typing import Optional

from playwright.sync_api import sync_playwright

from .config import settings


def fetch_html_headless(url: str, user_agent: Optional[str] = None) -> str:
    """
    Dùng Playwright (Chromium headless) để tải HTML của 1 trang.
    Chỉ nên dùng như fallback khi requests bị chặn / HTML thiếu nội dung.
    """
    timeout_ms = int(getattr(settings, "HEADLESS_TIMEOUT_MS", 15000))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=user_agent
            or (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        )
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        html = page.content()
        browser.close()
    return html
