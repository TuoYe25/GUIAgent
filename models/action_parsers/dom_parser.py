"""
DOM-based action parser.

Uses Playwright's accessibility tree / DOM snapshot to convert natural language
actions into precise DOM selectors and browser automation commands.

Key differences from coordinate-based:
- Uses semantic selectors (CSS, XPath, ARIA roles) instead of (x, y)
- More robust to layout changes and resolution differences
- Requires browser context (with Playwright)
- Native support for form filling, navigation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class DOMActionType(str, Enum):
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    CHECK = "check"
    HOVER = "hover"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    PRESS_KEY = "press_key"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    EVALUATE = "evaluate"
    FINISHED = "finished"


@dataclass
class DOMAction:
    """A parsed DOM-based action."""

    action_type: DOMActionType
    selector: Optional[str] = None        # CSS / XPath / ARIA selector
    value: Optional[str] = None           # Fill text, URL, JS code
    options: Optional[List[str]] = None   # Select options
    key: Optional[str] = None
    pixel_scroll: Optional[int] = None

    # Metadata
    raw_output: str = ""
    confidence: float = 1.0
    locator_strategy: str = "css"         # css, xpath, text, role, testid
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"action_type": self.action_type.value}
        if self.selector:
            d["selector"] = self.selector
        if self.value:
            d["value"] = self.value
        if self.options:
            d["options"] = self.options
        if self.key:
            d["key"] = self.key
        if self.pixel_scroll is not None:
            d["pixel_scroll"] = self.pixel_scroll
        d["locator_strategy"] = self.locator_strategy
        return d


# ---------------------------------------------------------------------------
# Accessibility Tree Extractor
# ---------------------------------------------------------------------------

class AccessibilityTreeExtractor:
    """
    Extract interactive elements from a DOM / accessibility tree snapshot.

    This runs in the browser context via Playwright's accessibility snapshot.
    """

    @staticmethod
    def build_element_map(
        ax_tree: Dict[str, Any],
        prefix: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Walk an accessibility tree and build a flat element map.

        Returns dict mapping element IDs/names to their properties.
        """
        elements: Dict[str, Dict[str, Any]] = {}

        def _walk(node: Dict[str, Any], depth: int) -> None:
            node_id = node.get("nodeId") or node.get("id") or node.get("name")
            if node_id:
                elements[str(node_id)] = {
                    "role": node.get("role", ""),
                    "name": node.get("name", ""),
                    "value": node.get("value", ""),
                    "description": node.get("description", ""),
                    "checked": node.get("checked"),
                    "disabled": node.get("disabled", False),
                    "level": depth,
                    "children": [c.get("nodeId", c.get("name", "")) for c in node.get("children", [])],
                }

            for child in node.get("children", []):
                _walk(child, depth + 1)

        _walk(ax_tree, 0)
        return elements

    @staticmethod
    def find_clickable(elements: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter elements to only interactive (clickable/focusable) ones."""
        interactive_roles = {
            "button", "link", "textbox", "searchbox", "combobox",
            "checkbox", "radio", "switch", "slider", "menuitem",
            "option", "tab", "listitem", "treeitem",
        }
        return [
            {"id": eid, **props}
            for eid, props in elements.items()
            if props["role"].lower() in interactive_roles and not props.get("disabled", False)
        ]


# ---------------------------------------------------------------------------
# DOM Parser
# ---------------------------------------------------------------------------

class DOMParser:
    """
    Parse model outputs into DOM-based actions.

    Works with:
    - Playwright locators (page.locator(selector))
    - Accessibility tree references
    - Text-based selectors
    """

    # Playwright locator strategies
    LOCATOR_STRATEGIES = {
        "css": "locator",
        "xpath": "locator",
        "text": "getByText",
        "role": "getByRole",
        "label": "getByLabel",
        "placeholder": "getByPlaceholder",
        "testid": "getByTestId",
        "alt": "getByAltText",
        "title": "getByTitle",
    }

    def __init__(self, page_url: Optional[str] = None) -> None:
        self.page_url = page_url

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse(self, raw_text: str) -> DOMAction:
        """Parse raw model output into DOMAction."""
        if not raw_text or not raw_text.strip():
            return DOMAction(action_type=DOMActionType.FINISHED, raw_output=raw_text)

        text = raw_text.strip()

        # Try JSON first (structured model output)
        result = self._parse_json(text)
        if result:
            return result

        # Try WebAgent / Mind2Web format
        result = self._parse_webagent(text)
        if result:
            return result

        # Generic key=value
        result = self._parse_generic(text)
        if result:
            return result

        logger.warning(f"Could not parse DOM action: {text[:200]}")
        return DOMAction(action_type=DOMActionType.FINISHED, raw_output=text)

    # ------------------------------------------------------------------
    # Parsing strategies
    # ------------------------------------------------------------------

    def _parse_json(self, text: str) -> Optional[DOMAction]:
        """Parse JSON-format action."""
        import re

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
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
        action_type = self._resolve_type(str(action_str))

        return DOMAction(
            action_type=action_type,
            selector=data.get("selector") or data.get("locator"),
            value=data.get("value") or data.get("text"),
            options=data.get("options"),
            key=data.get("key"),
            locator_strategy=data.get("strategy", "css"),
            raw_output=text,
            confidence=float(data.get("confidence", 1.0)),
        )

    def _parse_webagent(self, text: str) -> Optional[DOMAction]:
        """
        Parse WebAgent / Mind2Web format:
            [action] CLICK [element] button "Submit" [ref] e123
        """
        import re

        m_action = re.search(r"\[action\]\s*(\w+)", text, re.IGNORECASE)
        if not m_action:
            return None

        action_type = self._resolve_type(m_action.group(1))

        selector: Optional[str] = None
        value: Optional[str] = None

        # Extract selector from [element] or [ref]
        m_elem = re.search(r"\[element\]\s*(.+?)(?:\s*\[ref\]|\s*$)", text, re.IGNORECASE)
        if m_elem:
            elem_text = m_elem.group(1).strip()
            # If it looks like an element reference ID
            m_ref = re.search(r"\[ref\]\s*(\w+)", text, re.IGNORECASE)
            if m_ref:
                selector = f'[data-ref="{m_ref.group(1)}"]'
            elif elem_text:
                selector = elem_text

        # Extract value
        m_val = re.search(r"\[value\]\s*['\"](.+?)['\"]", text, re.IGNORECASE)
        if m_val:
            value = m_val.group(1)

        return DOMAction(
            action_type=action_type,
            selector=selector,
            value=value,
            raw_output=text,
        )

    def _parse_generic(self, text: str) -> Optional[DOMAction]:
        """Parse generic key=value format."""
        import re

        text_lower = text.lower()

        m_action = re.search(r"action\s*[:=]\s*(\w+)", text_lower)
        if not m_action:
            return None

        action_type = self._resolve_type(m_action.group(1))
        if action_type == DOMActionType.FINISHED:
            return DOMAction(action_type=action_type, raw_output=text)

        action = DOMAction(action_type=action_type, raw_output=text)

        # selector
        m_sel = re.search(r"(?:selector|locator|element)\s*[:=]\s*['\"](.+?)['\"]", text)
        if m_sel:
            action.selector = m_sel.group(1)

        # value
        m_val = re.search(r"(?:value|text|input)\s*[:=]\s*['\"](.+?)['\"]", text)
        if m_val:
            action.value = m_val.group(1)

        # strategy
        m_strat = re.search(r"strategy\s*[:=]\s*['\"](\w+)['\"]", text_lower)
        if m_strat:
            action.locator_strategy = m_strat.group(1)

        return action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_type(name: str) -> DOMActionType:
        mapping: Dict[str, DOMActionType] = {
            "click": DOMActionType.CLICK,
            "fill": DOMActionType.FILL,
            "input": DOMActionType.FILL,
            "type": DOMActionType.FILL,
            "select": DOMActionType.SELECT,
            "check": DOMActionType.CHECK,
            "toggle": DOMActionType.CHECK,
            "hover": DOMActionType.HOVER,
            "scroll": DOMActionType.SCROLL,
            "navigate": DOMActionType.NAVIGATE,
            "goto": DOMActionType.NAVIGATE,
            "press_key": DOMActionType.PRESS_KEY,
            "keypress": DOMActionType.PRESS_KEY,
            "wait": DOMActionType.WAIT,
            "screenshot": DOMActionType.SCREENSHOT,
            "evaluate": DOMActionType.EVALUATE,
            "finished": DOMActionType.FINISHED,
            "done": DOMActionType.FINISHED,
        }
        return mapping.get(name.lower(), DOMActionType.CLICK)

    def to_playwright_command(self, action: DOMAction) -> Dict[str, Any]:
        """
        Convert a DOMAction to a Playwright-compatible command dict.

        Returns a dict that can be serialized and sent to a Playwright executor.
        """
        cmd: Dict[str, Any] = {"action": action.action_type.value}

        if action.selector:
            cmd["selector"] = action.selector
            cmd["strategy"] = action.locator_strategy

        if action.value:
            cmd["value"] = action.value
        if action.options:
            cmd["options"] = action.options
        if action.key:
            cmd["key"] = action.key
        if action.pixel_scroll is not None:
            cmd["pixel_scroll"] = action.pixel_scroll

        return cmd


# ---------------------------------------------------------------------------
# Element Descriptor
# ---------------------------------------------------------------------------

@dataclass
class ElementDescriptor:
    """Rich description of a DOM element for VLM prompting."""

    tag: str
    role: str
    name: str
    text: str
    id: Optional[str] = None
    css_classes: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    bbox: Optional[Tuple[int, int, int, int]] = None
    children_count: int = 0

    def to_prompt_str(self) -> str:
        """Format for inclusion in a model prompt."""
        parts = [f"<{self.tag} role='{self.role}'"]
        if self.id:
            parts.append(f" id='{self.id}'")
        if self.name:
            parts.append(f" name='{self.name}'")
        if self.text:
            parts.append(f" text='{self.text[:80]}'")
        parts.append("/>")
        return "".join(parts)
