"""
Coordinate-based action parser (UI-TARS style).

Parses model outputs into absolute screen coordinates for direct
mouse/keyboard automation via pyautogui or similar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    PRESS = "press"
    SCROLL = "scroll"
    DRAG = "drag"
    HOVER = "hover"
    WAIT = "wait"
    FINISHED = "finished"
    UNKNOWN = "unknown"


@dataclass
class CoordinateAction:
    """A parsed coordinate-based action."""

    action_type: ActionType
    x: Optional[int] = None
    y: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None
    text: Optional[str] = None
    key: Optional[str] = None
    direction: Optional[str] = None
    amount: Optional[int] = None
    duration_ms: Optional[int] = None
    message: Optional[str] = None

    # Metadata
    raw_output: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (excluding None values)."""
        d: Dict[str, Any] = {"action_type": self.action_type.value}
        for attr in (
            "x", "y", "x2", "y2", "text", "key",
            "direction", "amount", "duration_ms", "message",
            "confidence",
        ):
            val = getattr(self, attr, None)
            if val is not None:
                d[attr] = val
        return d


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class CoordinateParser:
    """
    Parse model outputs into coordinate-based actions.

    Supports multiple output formats:
    - UI-TARS:   click(x=123, y=456)
    - CogAgent:  CLICK(box=[[x1,y1,x2,y2]], element_info='...')
    - Generic:   {"action": "click", "x": 123, "y": 456}
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        normalize: bool = False,
    ) -> None:
        """
        Args:
            screen_width: Actual screen width for denormalization
            screen_height: Actual screen height for denormalization
            normalize: If True, model outputs 0-1000 normalized coords
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Main parse entry
    # ------------------------------------------------------------------

    def parse(self, raw_text: str, image_size: Optional[Tuple[int, int]] = None) -> CoordinateAction:
        """
        Parse raw model output text into a CoordinateAction.

        Tries multiple parsing strategies in order.
        """
        if not raw_text or not raw_text.strip():
            return CoordinateAction(
                action_type=ActionType.UNKNOWN,
                raw_output=raw_text,
            )

        text = raw_text.strip()

        # Strategy 1: UI-TARS format — click(x=123, y=456)
        result = self._parse_ui_tars(text)
        if result and result.action_type != ActionType.UNKNOWN:
            return result

        # Strategy 2: CogAgent format — CLICK(box=[[...]])
        result = self._parse_cogagent(text)
        if result and result.action_type != ActionType.UNKNOWN:
            return result

        # Strategy 3: Generic key=value format
        result = self._parse_generic(text)
        if result and result.action_type != ActionType.UNKNOWN:
            return result

        # Strategy 4: JSON
        result = self._parse_json(text)
        if result and result.action_type != ActionType.UNKNOWN:
            return result

        logger.warning(f"Could not parse coordinate action from: {text[:200]}")
        return CoordinateAction(action_type=ActionType.UNKNOWN, raw_output=text)

    # ------------------------------------------------------------------
    # UI-TARS format
    # ------------------------------------------------------------------

    def _parse_ui_tars(self, text: str) -> Optional[CoordinateAction]:
        """Parse UI-TARS style: click(x=123, y=456)"""
        text_lower = text.lower().strip()

        if text_lower.startswith("click"):
            m = re.search(r"x\s*=\s*(-?\d+).*?y\s*=\s*(-?\d+)", text)
            if m:
                x, y = self._denorm(int(m.group(1)), int(m.group(2)))
                return CoordinateAction(
                    action_type=ActionType.CLICK,
                    x=x, y=y, raw_output=text,
                )

        elif text_lower.startswith("double_click"):
            m = re.search(r"x\s*=\s*(-?\d+).*?y\s*=\s*(-?\d+)", text)
            if m:
                x, y = self._denorm(int(m.group(1)), int(m.group(2)))
                return CoordinateAction(
                    action_type=ActionType.DOUBLE_CLICK,
                    x=x, y=y, raw_output=text,
                )

        elif text_lower.startswith("right_click"):
            m = re.search(r"x\s*=\s*(-?\d+).*?y\s*=\s*(-?\d+)", text)
            if m:
                x, y = self._denorm(int(m.group(1)), int(m.group(2)))
                return CoordinateAction(
                    action_type=ActionType.RIGHT_CLICK,
                    x=x, y=y, raw_output=text,
                )

        elif text_lower.startswith("type"):
            m = re.search(r"text\s*=\s*['\"](.+?)['\"]", text)
            if m:
                return CoordinateAction(
                    action_type=ActionType.TYPE,
                    text=m.group(1), raw_output=text,
                )

        elif text_lower.startswith("press"):
            m = re.search(r"key\s*=\s*['\"](\w+)['\"]", text)
            if m:
                return CoordinateAction(
                    action_type=ActionType.PRESS,
                    key=m.group(1), raw_output=text,
                )

        elif text_lower.startswith("scroll"):
            m_dir = re.search(r"direction\s*=\s*['\"](\w+)['\"]", text)
            m_amt = re.search(r"amount\s*=\s*(\d+)", text)
            direction = m_dir.group(1) if m_dir else "down"
            amount = int(m_amt.group(1)) if m_amt else 100
            return CoordinateAction(
                action_type=ActionType.SCROLL,
                direction=direction, amount=amount, raw_output=text,
            )

        elif text_lower.startswith("drag"):
            m = re.search(
                r"x1\s*=\s*(\d+).*?y1\s*=\s*(\d+).*?x2\s*=\s*(\d+).*?y2\s*=\s*(\d+)",
                text,
            )
            if m:
                x1, y1 = self._denorm(int(m.group(1)), int(m.group(2)))
                x2, y2 = self._denorm(int(m.group(3)), int(m.group(4)))
                return CoordinateAction(
                    action_type=ActionType.DRAG,
                    x=x1, y=y1, x2=x2, y2=y2, raw_output=text,
                )

        elif text_lower.startswith("hover"):
            m = re.search(r"x\s*=\s*(\d+).*?y\s*=\s*(\d+)", text)
            if m:
                x, y = self._denorm(int(m.group(1)), int(m.group(2)))
                return CoordinateAction(
                    action_type=ActionType.HOVER,
                    x=x, y=y, raw_output=text,
                )

        elif text_lower.startswith("wait"):
            m = re.search(r"(\d+)", text)
            if m:
                return CoordinateAction(
                    action_type=ActionType.WAIT,
                    duration_ms=int(m.group(1)), raw_output=text,
                )

        elif text_lower.startswith("finished"):
            m = re.search(r"message\s*=\s*['\"](.+?)['\"]", text)
            msg = m.group(1) if m else ""
            return CoordinateAction(
                action_type=ActionType.FINISHED,
                message=msg, raw_output=text,
            )

        return None

    # ------------------------------------------------------------------
    # CogAgent format
    # ------------------------------------------------------------------

    def _parse_cogagent(self, text: str) -> Optional[CoordinateAction]:
        """Parse CogAgent style: CLICK(box=[[100,200,300,400]], element_info='button')"""
        text_upper = text.strip().upper()

        # CLICK(box=[[x1,y1,x2,y2]], ...)
        if text_upper.startswith("CLICK"):
            m = re.search(r"box\s*=\s*\[\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\]", text)
            if m:
                x1, y1, x2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cx, cy = self._denorm(center_x, center_y)
                return CoordinateAction(
                    action_type=ActionType.CLICK,
                    x=cx, y=cy,
                    metadata={"bbox": [x1, y1, x2, y2]},
                    raw_output=text,
                )

        elif text_upper.startswith("TYPE"):
            m = re.search(r"text\s*=\s*['\"](.+?)['\"]", text, re.IGNORECASE)
            if m:
                return CoordinateAction(
                    action_type=ActionType.TYPE,
                    text=m.group(1), raw_output=text,
                )

        return None

    # ------------------------------------------------------------------
    # Generic format
    # ------------------------------------------------------------------

    def _parse_generic(self, text: str) -> Optional[CoordinateAction]:
        """Parse generic 'action: click x: 123 y: 456' format."""
        text_lower = text.lower().strip()

        # action: <type>
        m_action = re.search(r"action\s*[:=]\s*(\w+)", text_lower)
        if not m_action:
            return None

        action_str = m_action.group(1)
        action_type = self._resolve_action_type(action_str)
        if action_type == ActionType.UNKNOWN:
            return None

        action = CoordinateAction(action_type=action_type, raw_output=text)

        # x and y
        m_x = re.search(r"x\s*[:=]\s*(-?\d+)", text_lower)
        m_y = re.search(r"y\s*[:=]\s*(-?\d+)", text_lower)
        if m_x and m_y:
            x, y = self._denorm(int(m_x.group(1)), int(m_y.group(1)))
            action.x = x
            action.y = y

        # text
        m_text = re.search(r"text\s*[:=]\s*['\"](.+?)['\"]", text)
        if m_text:
            action.text = m_text.group(1)

        # key
        m_key = re.search(r"key\s*[:=]\s*['\"](\w+)['\"]", text_lower)
        if m_key:
            action.key = m_key.group(1)

        return action

    # ------------------------------------------------------------------
    # JSON format
    # ------------------------------------------------------------------

    def _parse_json(self, text: str) -> Optional[CoordinateAction]:
        """Try to parse as JSON."""
        import json

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from within text
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    return None
            else:
                return None

        if not isinstance(data, dict):
            return None

        action_str = data.get("action") or data.get("action_type", "")
        action_type = self._resolve_action_type(str(action_str))
        if action_type == ActionType.UNKNOWN:
            return None

        action = CoordinateAction(action_type=action_type, raw_output=text)

        if "x" in data and "y" in data:
            x, y = self._denorm(int(data["x"]), int(data["y"]))
            action.x = x
            action.y = y

        for key in ("text", "key", "direction", "message"):
            if key in data:
                setattr(action, key, data[key])

        if "amount" in data:
            action.amount = int(data["amount"])
        if "duration_ms" in data:
            action.duration_ms = int(data["duration_ms"])
        if "confidence" in data:
            action.confidence = float(data["confidence"])

        return action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _denorm(self, x: int, y: int) -> Tuple[int, int]:
        """Denormalize coordinates if model outputs 0-1000 range."""
        if self.normalize:
            x = int(x * self.screen_width / 1000)
            y = int(y * self.screen_height / 1000)
        return x, y

    @staticmethod
    def _resolve_action_type(name: str) -> ActionType:
        """Map string to ActionType enum."""
        mapping: Dict[str, ActionType] = {
            "click": ActionType.CLICK,
            "double_click": ActionType.DOUBLE_CLICK,
            "doubleclick": ActionType.DOUBLE_CLICK,
            "right_click": ActionType.RIGHT_CLICK,
            "rightclick": ActionType.RIGHT_CLICK,
            "type": ActionType.TYPE,
            "input": ActionType.TYPE,
            "press": ActionType.PRESS,
            "key": ActionType.PRESS,
            "keypress": ActionType.PRESS,
            "scroll": ActionType.SCROLL,
            "drag": ActionType.DRAG,
            "hover": ActionType.HOVER,
            "wait": ActionType.WAIT,
            "sleep": ActionType.WAIT,
            "finished": ActionType.FINISHED,
            "done": ActionType.FINISHED,
            "complete": ActionType.FINISHED,
        }
        return mapping.get(name.lower(), ActionType.UNKNOWN)


# ---------------------------------------------------------------------------
# Batch Parser
# ---------------------------------------------------------------------------

class BatchCoordinateParser:
    """Parse multiple coordinate outputs efficiently."""

    def __init__(self, parser: CoordinateParser) -> None:
        self.parser = parser

    def parse_batch(self, raw_outputs: List[str]) -> List[CoordinateAction]:
        """Parse a batch of raw outputs."""
        return [self.parser.parse(text) for text in raw_outputs]

    def filter_by_type(
        self,
        actions: List[CoordinateAction],
        action_type: ActionType,
    ) -> List[CoordinateAction]:
        """Filter actions by type."""
        return [a for a in actions if a.action_type == action_type]
