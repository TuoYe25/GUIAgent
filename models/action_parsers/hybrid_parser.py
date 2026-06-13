"""
Hybrid Action Parser.

Combines multiple approaches (coordinate + DOM + a11y) into a single,
robust parsing pipeline with fallback strategies.

Architecture:
1. Primary: Coordinate-based (UI-TARS style) — fastest, simplest
2. Secondary: DOM-based — more robust for web apps
3. Tertiary: Accessibility tree — cross-platform, works with native apps

The parser chooses the best strategy based on context and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from models.action_parsers.coordinate_parser import (
    ActionType,
    CoordinateAction,
    CoordinateParser,
)
from models.action_parsers.dom_parser import DOMAction, DOMActionType, DOMParser
from models.action_parsers.a11y_parser import A11yAction, A11yActionType, A11yParser


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class ParseStrategy(str, Enum):
    """Which parsing strategy to use."""
    COORDINATE = "coordinate"
    DOM = "dom"
    A11Y = "a11y"
    AUTO = "auto"       # Automatically select best strategy


@dataclass
class ParseResult:
    """Result from any parsing strategy, normalized to a common format."""

    success: bool
    strategy: ParseStrategy
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    confidence: float = 0.0
    execution_time_ms: float = 0.0

    @property
    def is_click(self) -> bool:
        return self.data.get("action_type", "") == "click"

    @property
    def is_type(self) -> bool:
        return self.data.get("action_type", "") == "type"

    @property
    def is_finished(self) -> bool:
        return self.data.get("action_type", "") == "finished"


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class ExecutionContext:
    """Context information for choosing parsing strategy."""

    app_type: str = "web"           # web, desktop, mobile
    platform: str = "windows"       # windows, macos, linux
    has_dom: bool = False
    has_a11y_tree: bool = False
    screen_width: int = 1920
    screen_height: int = 1080
    page_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Hybrid Parser
# ---------------------------------------------------------------------------

class HybridParser:
    """
    Multi-strategy action parser with automatic fallback.

    Usage:
        parser = HybridParser(context=ExecutionContext(app_type="web"))
        result = parser.parse(raw_text, screenshot=screenshot)
    """

    def __init__(
        self,
        context: ExecutionContext,
        primary: ParseStrategy = ParseStrategy.AUTO,
        enable_fallback: bool = True,
    ) -> None:
        self.context = context
        self.primary = primary
        self.enable_fallback = enable_fallback

        # Initialize sub-parsers
        self.coord_parser = CoordinateParser(
            screen_width=context.screen_width,
            screen_height=context.screen_height,
        )
        self.dom_parser = DOMParser(page_url=context.page_url)
        self.a11y_parser = A11yParser(
            platform=context.platform,
            screen_width=context.screen_width,
            screen_height=context.screen_height,
        )

        # Strategy selection order
        self._strategy_order = self._determine_strategy_order()

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse(self, raw_text: str) -> ParseResult:
        """
        Parse raw model output using the best available strategy.
        """
        import time

        results: List[ParseResult] = []

        for strategy in self._strategy_order:
            t0 = time.perf_counter()
            try:
                result = self._parse_with(raw_text, strategy)
                result.execution_time_ms = (time.perf_counter() - t0) * 1000
                results.append(result)

                if result.success and result.confidence >= 0.7:
                    logger.debug(f"Hybrid parser: success with {strategy} (conf={result.confidence:.2f})")
                    return result

            except Exception as e:
                logger.warning(f"Parser {strategy} failed: {e}")
                results.append(ParseResult(
                    success=False,
                    strategy=strategy,
                    error=str(e),
                ))

        # If no strategy succeeded, return the best attempt
        if results:
            best = max(results, key=lambda r: r.confidence)
            if best.confidence > 0:
                return best

        # Total failure
        return ParseResult(
            success=False,
            strategy=ParseStrategy.AUTO,
            error="All parsing strategies failed",
        )

    def parse_with_fallback(self, raw_text: str) -> ParseResult:
        """Parse with explicit fallback chain."""
        return self.parse(raw_text)

    # ------------------------------------------------------------------
    # Strategy execution
    # ------------------------------------------------------------------

    def _parse_with(self, raw_text: str, strategy: ParseStrategy) -> ParseResult:
        """Execute a single parsing strategy."""
        if strategy == ParseStrategy.COORDINATE:
            action = self.coord_parser.parse(raw_text)
            return ParseResult(
                success=action.action_type != ActionType.UNKNOWN,
                strategy=strategy,
                data=action.to_dict(),
                confidence=self._estimate_confidence(action),
            )

        elif strategy == ParseStrategy.DOM:
            action = self.dom_parser.parse(raw_text)
            success = action.action_type != DOMActionType.FINISHED or action.selector is not None
            return ParseResult(
                success=success,
                strategy=strategy,
                data=action.to_dict(),
                confidence=action.confidence,
            )

        elif strategy == ParseStrategy.A11Y:
            action = self.a11y_parser.parse(raw_text)
            success = action.action_type != A11yActionType.FINISHED
            return ParseResult(
                success=success,
                strategy=strategy,
                data=action.to_dict(),
                confidence=action.confidence,
            )

        else:
            return ParseResult(
                success=False,
                strategy=strategy,
                error=f"Unknown strategy: {strategy}",
            )

    # ------------------------------------------------------------------
    # Strategy ordering
    # ------------------------------------------------------------------

    def _determine_strategy_order(self) -> List[ParseStrategy]:
        """Determine the best strategy order based on context."""
        if self.primary != ParseStrategy.AUTO:
            order = [self.primary]
            # Add others as fallback
            for s in ParseStrategy:
                if s not in (ParseStrategy.AUTO, self.primary):
                    order.append(s)
            return order

        # Auto selection based on context
        order: List[ParseStrategy] = []

        if self.context.app_type == "web" and self.context.has_dom:
            order = [ParseStrategy.DOM, ParseStrategy.COORDINATE, ParseStrategy.A11Y]
        elif self.context.app_type == "desktop" and self.context.has_a11y_tree:
            order = [ParseStrategy.A11Y, ParseStrategy.COORDINATE, ParseStrategy.DOM]
        else:
            order = [ParseStrategy.COORDINATE, ParseStrategy.DOM, ParseStrategy.A11Y]

        return order

    # ------------------------------------------------------------------
    # Confidence estimation
    # ------------------------------------------------------------------

    def _estimate_confidence(self, action: CoordinateAction) -> float:
        """Estimate confidence for a coordinate action."""
        if action.action_type == ActionType.UNKNOWN:
            return 0.0

        base = 0.8

        # Penalize if coords are out of bounds
        if action.x is not None and action.y is not None:
            if not (0 <= action.x <= self.context.screen_width):
                base -= 0.2
            if not (0 <= action.y <= self.context.screen_height):
                base -= 0.2

        # Bonus for explicit text/key
        if action.text or action.key:
            base += 0.1

        return max(0.0, min(1.0, base))

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def parse_batch(self, raw_outputs: List[str]) -> List[ParseResult]:
        """Parse multiple outputs."""
        return [self.parse(text) for text in raw_outputs]

    def statistics(self, results: List[ParseResult]) -> Dict[str, Any]:
        """Compute statistics over a batch of parse results."""
        total = len(results)
        if total == 0:
            return {}

        success_count = sum(1 for r in results if r.success)
        strategy_counts: Dict[str, int] = {}
        for r in results:
            strategy_counts[r.strategy.value] = strategy_counts.get(r.strategy.value, 0) + 1

        avg_confidence = sum(r.confidence for r in results) / total
        avg_latency = sum(r.execution_time_ms for r in results) / total

        return {
            "total": total,
            "success_rate": success_count / total,
            "strategy_distribution": strategy_counts,
            "avg_confidence": round(avg_confidence, 3),
            "avg_latency_ms": round(avg_latency, 1),
        }


# ---------------------------------------------------------------------------
# Action Normalizer
# ---------------------------------------------------------------------------

class ActionNormalizer:
    """Convert actions from any parser format to a unified format for execution."""

    @staticmethod
    def normalize(result: ParseResult) -> Dict[str, Any]:
        """
        Convert any ParseResult to a unified action dict.
        This is what gets sent to the action executor.
        """
        data = result.data

        normalized: Dict[str, Any] = {
            "action_type": data.get("action_type", "unknown"),
            "strategy": result.strategy.value,
            "confidence": round(result.confidence, 3),
        }

        # Coordinates (click/hover/drag)
        if "x" in data and "y" in data:
            normalized["x"] = data["x"]
            normalized["y"] = data["y"]
        if "x2" in data and "y2" in data:
            normalized["x1"] = data["x"]
            normalized["y1"] = data["y"]
            normalized["x2"] = data["x2"]
            normalized["y2"] = data["y2"]

        # Selector (DOM-based)
        if "selector" in data and data["selector"]:
            normalized["selector"] = data["selector"]
            normalized["locator_strategy"] = data.get("locator_strategy", "css")

        # Accessibility ref
        if "node_id" in data:
            normalized["node_id"] = data["node_id"]
        if "role" in data:
            normalized["role"] = data["role"]
        if "name" in data:
            normalized["a11y_name"] = data["name"]

        # Value / text / key
        if "text" in data or "value" in data:
            normalized["text"] = data.get("text") or data.get("value")
        if "key" in data:
            normalized["key"] = data["key"]
        if "direction" in data:
            normalized["direction"] = data["direction"]
        if "amount" in data:
            normalized["amount"] = data["amount"]

        # Message (for finished)
        if "message" in data:
            normalized["message"] = data["message"]

        return normalized
