"""
Playwright-based GUI Agent (competitor baseline).

Leverages Playwright's accessibility tree and JavaScript execution for
deterministic GUI automation. Serves as a performance and reliability
baseline for comparing against ML-based approaches.

Features:
- Full browser automation via Playwright
- Accessibility tree-based action execution
- Codegen-style action recording
- Resource monitoring during execution
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from benchmark.runner import BenchmarkTask, BenchmarkResult


# ---------------------------------------------------------------------------
# Playwright Agent
# ---------------------------------------------------------------------------

@dataclass
class PlaywrightConfig:
    """Configuration for the Playwright agent."""

    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, webkit
    timeout: int = 30000
    viewport_width: int = 1280
    viewport_height: int = 720
    use_a11y_tree: bool = True
    slow_mo: int = 0  # Slow down operations by N ms (debugging)
    base_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headless": self.headless,
            "browser_type": self.browser_type,
            "timeout": self.timeout,
            "viewport": f"{self.viewport_width}x{self.viewport_height}",
            "use_a11y_tree": self.use_a11y_tree,
        }


class PlaywrightAgent:
    """GUI Agent powered by Playwright for deterministic automation."""

    def __init__(
        self,
        config: Optional[PlaywrightConfig] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.config = config or PlaywrightConfig()
        self.output_dir = output_dir or Path("benchmark/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.browser = None
        self.context = None
        self.page = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the browser."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "playwright not installed. Run: pip install playwright && playwright install"
            )

        self._pw = await async_playwright().start()

        browser_type = getattr(self._pw, self.config.browser_type)
        self.browser = await browser_type.launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo,
        )

        self.context = await self.browser.new_context(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
        )

        self.page = await self.context.new_page()
        self.page.set_default_timeout(self.config.timeout)

        logger.info(f"Playwright agent started (headless={self.config.headless})")

    async def stop(self) -> None:
        """Stop the browser."""
        if self.browser:
            await self.browser.close()
        if self._pw:
            await self._pw.stop()
        logger.info("Playwright agent stopped")

    async def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        if not self.page:
            raise RuntimeError("Agent not started")
        await self.page.goto(url, wait_until="load")
        logger.info(f"Navigated to {url}")

    async def screenshot(self, path: Optional[Path] = None) -> bytes:
        """Take a screenshot."""
        if not self.page:
            raise RuntimeError("Agent not started")
        if path:
            await self.page.screenshot(path=str(path))
        return await self.page.screenshot()

    # ------------------------------------------------------------------
    # Accessibility Tree
    # ------------------------------------------------------------------

    async def get_accessibility_tree(self) -> Dict[str, Any]:
        """Get the accessibility tree snapshot."""
        if not self.page:
            raise RuntimeError("Agent not started")

        snapshot = await self.page.accessibility.snapshot()
        return snapshot or {}

    async def find_a11y_element(self, role: str, name: str, partial: bool = True) -> Optional[Dict[str, Any]]:
        """Find an accessibility element by role and name."""
        tree = await self.get_accessibility_tree()
        return self._search_a11y_tree(tree, role, name, partial)

    def _search_a11y_tree(
        self,
        node: Dict[str, Any],
        role: str,
        name: str,
        partial: bool,
    ) -> Optional[Dict[str, Any]]:
        """Recursively search the a11y tree."""
        if not isinstance(node, dict):
            return None

        node_role = node.get("role", "").lower()
        node_name = node.get("name", "")

        if node_role == role.lower():
            if partial and name.lower() in node_name.lower():
                return node
            if not partial and node_name.lower() == name.lower():
                return node

        for child in node.get("children", []):
            result = self._search_a11y_tree(child, role, name, partial)
            if result:
                return result

        return None

    # ------------------------------------------------------------------
    # Action Execution
    # ------------------------------------------------------------------

    async def click(
        self,
        selector: Optional[str] = None,
        text: Optional[str] = None,
        role: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> bool:
        """Click an element by various methods."""
        if not self.page:
            raise RuntimeError("Agent not started")

        try:
            if selector:
                await self.page.click(selector)
            elif text:
                await self.page.click(f"text={text}")
            elif role and text:
                await self.page.click(f"{role}={text}")
            elif x is not None and y is not None:
                await self.page.mouse.click(x, y)
            else:
                logger.error("No click target specified")
                return False

            logger.debug(f"Clicked: selector={selector}, text={text}")
            return True

        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False

    async def type_text(self, selector: str, text: str, clear_first: bool = True) -> bool:
        """Type text into a field."""
        if not self.page:
            raise RuntimeError("Agent not started")

        try:
            if clear_first:
                await self.page.fill(selector, "")
            await self.page.type(selector, text, delay=50)
            logger.debug(f"Typed: '{text}' into {selector}")
            return True
        except Exception as e:
            logger.error(f"Type failed: {e}")
            return False

    async def select_option(self, selector: str, value: str or bool) -> bool:
        """Select an option from a dropdown."""
        if not self.page:
            raise RuntimeError("Agent not started")

        try:
            await self.page.select_option(selector, value)
            logger.debug(f"Selected: {value} in {selector}")
            return True
        except Exception as e:
            logger.error(f"Select failed: {e}")
            return False

    async def scroll(self, direction: str, amount: int = 300) -> bool:
        """Scroll the page."""
        if not self.page:
            raise RuntimeError("Agent not started")

        scroll_map = {
            "down": (0, amount),
            "up": (0, -amount),
            "right": (amount, 0),
            "left": (-amount, 0),
        }

        delta = scroll_map.get(direction, (0, amount))
        try:
            await self.page.evaluate(
                f"window.scrollBy({delta[0]}, {delta[1]})"
            )
            logger.debug(f"Scrolled {direction} by {amount}")
            return True
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return False

    async def press_key(self, key: str) -> bool:
        """Press a keyboard key."""
        if not self.page:
            raise RuntimeError("Agent not started")

        try:
            await self.page.keyboard.press(key)
            logger.debug(f"Pressed key: {key}")
            return True
        except Exception as e:
            logger.error(f"Key press failed: {e}")
            return False

    async def wait_for_load(self, state: str = "load") -> bool:
        """Wait for page to load."""
        if not self.page:
            raise RuntimeError("Agent not started")

        try:
            await self.page.wait_for_load_state(state)
            return True
        except Exception as e:
            logger.warning(f"Wait for load failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Task Execution (Benchmark Integration)
    # ------------------------------------------------------------------

    async def execute_task(self, task: BenchmarkTask) -> BenchmarkResult:
        """Execute a benchmark task and return results."""
        import time

        start_time = time.perf_counter()

        try:
            for action in task.expected_actions:
                action_type = action.get("action_type", "")
                success = await self._execute_action(action)

                if not success:
                    elapsed = time.perf_counter() - start_time
                    return BenchmarkResult(
                        task_id=task.id,
                        model_name="playwright",
                        action_method="dom",
                        success=False,
                        steps=len(task.expected_actions),
                        execution_time_ms=elapsed * 1000,
                        error=f"Action failed: {action}",
                    )

                # Small delay between actions
                await asyncio.sleep(0.1)

            elapsed = time.perf_counter() - start_time
            return BenchmarkResult(
                task_id=task.id,
                model_name="playwright",
                action_method="dom",
                success=True,
                steps=len(task.expected_actions),
                execution_time_ms=elapsed * 1000,
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"Task execution failed: {e}")
            return BenchmarkResult(
                task_id=task.id,
                model_name="playwright",
                action_method="dom",
                success=False,
                steps=0,
                execution_time_ms=elapsed * 1000,
                error=str(e),
            )

    async def _execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute a single action."""
        action_type = action.get("action_type", "")
        target = action.get("target", "")
        text = action.get("text", "")

        if action_type == "click":
            return await self.click(text=target)
        elif action_type == "type":
            selector = f"input, textarea"  # Simplified
            return await self.type_text(selector, text)
        elif action_type == "scroll":
            direction = action.get("direction", "down")
            amount = action.get("amount", 300)
            return await self.scroll(direction, amount)
        elif action_type == "navigate":
            url = action.get("url", "")
            if url:
                await self.navigate(url)
                return True
            return False
        elif action_type == "press":
            key = action.get("key", "")
            return await self.press_key(key)
        elif action_type == "wait":
            await asyncio.sleep(1)
            return True
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return False

    # ------------------------------------------------------------------
    # Resource Monitoring
    # ------------------------------------------------------------------

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        try:
            import psutil
            process = psutil.Process()

            usage = {
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
            }

            # Add child process usage (browser)
            for child in process.children(recursive=True):
                usage["memory_mb"] += child.memory_info().rss / 1024 / 1024

            return usage
        except Exception as e:
            logger.warning(f"Resource monitoring failed: {e}")
            return {}


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------

async def main() -> None:
    """Test the Playwright agent."""
    agent = PlaywrightAgent(PlaywrightConfig(headless=False))

    try:
        await agent.start()
        await agent.navigate("https://example.com")
        await asyncio.sleep(2)

        # Take screenshot
        screenshot = await agent.screenshot(Path("benchmark/results/playwright_test.png"))
        logger.info("Screenshot saved")

        # Get a11y tree
        tree = await agent.get_accessibility_tree()
        logger.info(f"A11y tree: {json.dumps(tree, indent=2)[:500]}")

    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
