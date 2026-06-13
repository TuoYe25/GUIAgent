"""
Puppeteer-based GUI Agent for benchmark comparison.

Uses Puppeteer (via pyppeteer) to control Chromium.
Represents the traditional browser automation approach.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class PuppeteerAgent:
    """Browser automation using Puppeteer (pyppeteer)."""

    def __init__(
        self,
        headless: bool = True,
        viewport: Optional[Dict[str, int]] = None,
        timeout_ms: int = 30000,
        slow_mo_ms: int = 0,
    ) -> None:
        self.headless = headless
        self.viewport = viewport or {"width": 1280, "height": 720}
        self.timeout_ms = timeout_ms
        self.slow_mo_ms = slow_mo_ms

        self.browser: Any = None
        self.page: Any = None

    async def start(self) -> None:
        """Launch Puppeteer browser."""
        try:
            from pyppeteer import launch

            self.browser = await launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    f"--window-size={self.viewport['width']},{self.viewport['height']}",
                ],
                slowMo=self.slow_mo_ms,
            )
            self.page = await self.browser.newPage()
            await self.page.setViewport(self.viewport)
            logger.info("Puppeteer browser launched")
        except ImportError:
            logger.error("pyppeteer not installed. Run: pip install pyppeteer")
            raise
        except Exception as e:
            logger.error(f"Failed to launch Puppeteer: {e}")
            raise

    async def stop(self) -> None:
        """Close browser."""
        if self.browser:
            await self.browser.close()
            logger.info("Puppeteer browser closed")

    # ------------------------------------------------------------------
    # Core Actions
    # ------------------------------------------------------------------

    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to URL."""
        start = time.perf_counter()
        try:
            response = await self.page.goto(url, waitUntil="networkidle2", timeout=self.timeout_ms)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "success": response is not None and response.ok,
                "url": url,
                "status": response.status if response else None,
                "elapsed_ms": round(elapsed_ms, 1),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def click(self, selector: str) -> Dict[str, Any]:
        """Click element by selector."""
        start = time.perf_counter()
        try:
            await self.page.waitForSelector(selector, timeout=self.timeout_ms)
            await self.page.click(selector)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"success": True, "selector": selector, "elapsed_ms": round(elapsed_ms, 1)}
        except Exception as e:
            return {"success": False, "selector": selector, "error": str(e)}

    async def click_xy(self, x: int, y: int) -> Dict[str, Any]:
        """Click at screen coordinates."""
        start = time.perf_counter()
        try:
            await self.page.mouse.click(x, y)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"success": True, "x": x, "y": y, "elapsed_ms": round(elapsed_ms, 1)}
        except Exception as e:
            return {"success": False, "x": x, "y": y, "error": str(e)}

    async def type_text(self, selector: str, text: str) -> Dict[str, Any]:
        """Type text into element."""
        start = time.perf_counter()
        try:
            await self.page.waitForSelector(selector, timeout=self.timeout_ms)
            await self.page.click(selector)
            await self.page.type(selector, text, delay=50)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"success": True, "selector": selector, "text": text, "elapsed_ms": round(elapsed_ms, 1)}
        except Exception as e:
            return {"success": False, "selector": selector, "error": str(e)}

    async def scroll(self, direction: str, amount: int = 300) -> Dict[str, Any]:
        """Scroll page."""
        start = time.perf_counter()
        try:
            y_offset = -amount if direction == "up" else amount
            await self.page.evaluate(f"window.scrollBy(0, {y_offset})")
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"success": True, "direction": direction, "elapsed_ms": round(elapsed_ms, 1)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def screenshot(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Take a screenshot."""
        start = time.perf_counter()
        try:
            kwargs: Dict[str, Any] = {"fullPage": False}
            if path:
                kwargs["path"] = str(path)
            screenshot_data = await self.page.screenshot(kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"success": True, "elapsed_ms": round(elapsed_ms, 1)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def evaluate(self, script: str) -> Dict[str, Any]:
        """Execute JavaScript in page context."""
        try:
            result = await self.page.evaluate(script)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_page_state(self) -> Dict[str, Any]:
        """Extract interactive elements from the page (similar to action_executor.js)."""
        script = """
        () => {
            const elements = [];
            document.querySelectorAll('a, button, input, select, textarea, [role="button"]')
                .forEach(el => {
                    const rect = el.getBoundingClientRect();
                    elements.push({
                        tag: el.tagName.toLowerCase(),
                        id: el.id || null,
                        text: (el.textContent || '').trim().slice(0, 100),
                        position: { x: Math.round(rect.x+rect.width/2), y: Math.round(rect.y+rect.height/2) },
                    });
                });
            return { url: window.location.href, title: document.title, elements };
        }
        """
        return await self.evaluate(script)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    agent = PuppeteerAgent(headless=True)
    await agent.start()

    result = await agent.navigate("https://example.com")
    logger.info(f"Navigate: {result}")

    state = await agent.get_page_state()
    logger.info(f"Page state: {state.get('result', {})}")

    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
