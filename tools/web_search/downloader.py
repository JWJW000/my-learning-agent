"""网页内容下载器 (downloader)。

使用 httpx 并行/异步获取指定 URL 列表的 HTML 源码内容，并加入并发与时间超时控制。
"""

from __future__ import annotations

import asyncio
import logging
import httpx

logger = logging.getLogger(__name__)


class Downloader:
    """提供健壮的异步/并行网页内容下载服务。"""

    def __init__(self, timeout_sec: float = 3.0, max_concurrent: int = 5):
        """
        Args:
            timeout_sec: 每一个网页下载的硬性超时限制（秒）。
            max_concurrent: 最大并发爬取线程/协程数。
        """
        self.timeout_sec = timeout_sec
        self.max_concurrent = max_concurrent

    async def _download_one(self, client: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore) -> tuple[str, str]:
        """下载单个网页的异步辅助函数。"""
        async with semaphore:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            try:
                response = await client.get(url, headers=headers, timeout=self.timeout_sec, follow_redirects=True)
                if response.status_code == 200:
                    return url, response.text
                else:
                    logger.warning("Failed to download page: %s, Status Code: %d", url, response.status_code)
                    return url, ""
            except Exception as e:
                logger.debug("Exception downloading page %s: %s", url, e)
                return url, ""

    async def download_pages_async(self, urls: list[str]) -> dict[str, str]:
        """并行下载多个网页（异步）。

        Returns:
            dict: 包含 {url: raw_html_content} 的字典映射。
        """
        if not urls:
            return {}

        semaphore = asyncio.Semaphore(self.max_concurrent)
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

        async with httpx.AsyncClient(limits=limits) as client:
            tasks = [self._download_one(client, url, semaphore) for url in urls]
            results = await asyncio.gather(*tasks)
            return {url: html for url, html in results if html}

    def download_pages(self, urls: list[str]) -> dict[str, str]:
        """同步包装器，在同步上下文里同步调用异步的 download_pages_async 并安全获取结果。"""
        if not urls:
            return {}
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 当前线程无事件循环，直接运行新循环
            return asyncio.run(self.download_pages_async(urls))

        # 当前已有事件循环（多在 asyncio 异步交互框架下运行）
        if loop.is_running():
            # 通过在独立线程中执行以避免 RuntimeError: This event loop is already running
            import threading
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.download_pages_async(urls))
                return future.result()
        else:
            return loop.run_until_complete(self.download_pages_async(urls))
