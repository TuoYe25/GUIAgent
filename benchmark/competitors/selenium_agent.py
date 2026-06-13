"""
Selenium-based GUI Agent for benchmark comparison.

Uses Selenium WebDriver to control browser.
Represents the most widely-used traditional browser automation approach.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class SeleniumAgent:
    """Browser automation using Selenium WebDriver."""

    def __init__(
        self,
        browser: str = "chrome",
        headless: bool = True,
        viewport: Optional[Dict[str, int]] = None,
        timeout_sec: int = 30,
        implicit_wait_sec: float = 5.0,
    ) -> None:
        self.browser = browser
        self.headless = headless
        self.viewport = viewport or {"width": 1280, "height": 720}
        self.timeout_sec = timeout_sec
        self.implicit_wait_sec = implicit_wait_sec

        self.driver: Any = None

    def start(self) -> None:
        """Launch Selenium WebDriver."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.chrome.service import Service as ChromeService

            options = ChromeOptions()
            if self.headless:
                options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument(f"--window-size={self.viewport['width']},{self.viewport['height']}")
            options.add_argument("--disable-gpu")

            self.driver = webdriver.Chrome(options=options)
            self.driver.implicitly_wait(self.implicit_wait_sec)
            self.driver.set_window_size(self.viewport["width"], self.viewport["height"])

            logger.info("Selenium Chrome WebDriver started")
        except ImportError:
            logger.error("selenium not installed. Run: pip install selenium")
            raise
        except Exception as e:
            logger.error(f"Failed to start Selenium: {e}")
            raise

    def stop(self) -> None:
        """Close WebDriver."""
        if self.driver:
            self.driver.quit()
            logger.info("Selenium WebDriver closed")

    # ------------------------------------------------------------------
    # Core Actions
    # ------------------------------------------------------------------

    def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to URL."""
        start = time.perf_counter()
        try:
            self.driver.get(url)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "success": True,
                "url": url,
                "title": self.driver.title,
                "elapsed_ms": round(elapsed_ms, 1),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def click(self, selector: str) -> Dict[str, Any]:
        """Click element by CSS selector."""
        start = time.perf_counter()
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            element = WebDriverWait(self.driver, self.timeout_sec).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            element.click()
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"success": True, "selector": selector, "elapsed_ms": round(elapsed_ms, 1)}
        except Exception as e:
            return {"success": False, "selector": selector, "error": str(e)}

    def click_xy(self, x: int, y: int) -> Dict[str, Any]:
        """Click at coordinates using JavaScript."""
        start = time.perf_counter()
        try:
            self.driver.execute_script(
                f"document.elementFromPoint({x}, {y}).click()"
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"success": True, "x": x, "y": y, "elapsed_ms": round(elapsed_ms, 1)}
        except Exception as e:
            return {"success": False, "x": x, "y": y, "error": str(e)}

    def type_text(self, selector: str, text: str) -> Dict[str, Any]:
        """Type text into element."""
        start = time.perf_counter()
        try:
            from selenium.webdriver.common.by import By
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            element.clear()
            element.send_keys(text)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"success": True, "selector": selector, "text": text, "elapsed_ms": round(elapsed_ms, 1)}
        except Exception as e:
            return {"success": False, "selector": selector, "error": str(e)}

    def scroll(self, direction: str, amount: int = 300) -> Dict[str, Any]:
        """Scroll page."""
        start = time.perf_counter()
        try:
            y_offset = -amount if direction == "up" else amount
            self.driver.execute_script(f"window.scrollBy(0, {y_offset})")
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"success": True, "direction": direction, "elapsed_ms": round(elapsed_ms, 1)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def screenshot(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Take a screenshot."""
        start = time.perf_counter()
        try:
            if path:
                self.driver.save_screenshot(str(path))
            else:
                self.driver.get_screenshot_as_png()
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"success": True, "elapsed_ms": round(elapsed_ms, 1)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def evaluate(self, script: str) -> Dict[str, Any]:
        """Execute JavaScript."""
        try:
            result = self.driver.execute_script(f"return {script}")
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_page_state(self) -> Dict[str, Any]:
        """Extract interactive elements (similar to action_executor.js)."""
        script = """
        var elements = [];
        document.querySelectorAll('a, button, input, select, textarea, [role="button"]')
            .forEach(function(el) {
                var rect = el.getBoundingClientRect();
                elements.push({
                    tag: el.tagName.toLowerCase(),
                    id: el.id || null,
                    text: (el.textContent || '').trim().substring(0, 100),
                    position: { x: Math.round(rect.x+rect.width/2), y: Math.round(rect.y+rect.height/2) },
                });
            });
        return { url: window.location.href, title: document.title, elements: elements };
        """
        return self.evaluate(script)

    def select(self, selector: str, value: str) -> Dict[str, Any]:
        """Select an option from a dropdown."""
        start = time.perf_counter()
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import Select
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            Select(element).select_by_value(value)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"success": True, "selector": selector, "elapsed_ms": round(elapsed_ms, 1)}
        except Exception as e:
            return {"success": False, "selector": selector, "error": str(e)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    agent = SeleniumAgent(headless=True)
    agent.start()

    result = agent.navigate("https://example.com")
    logger.info(f"Navigate: {result}")

    state = agent.get_page_state()
    logger.info(f"Page state: {state.get('result', {})}")

    agent.stop()


if __name__ == "__main__":
    main()
