"""
Accessibility Tree Parser.

Parses actions based on accessibility tree (a11y) nodes, which are
platform-native and more robust than DOM for desktop and mobile apps.

Key features:
- Uses platform-specific accessibility APIs (Windows UIA, macOS AX, Linux AT‑SPI2)
- Works for both web and native applications
- Semantic roles (button, link, textbox) and properties (name, value, state)
- No dependency on DOM structure or CSS selectors
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class A11yActionType(str, Enum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    CLEAR = "clear"
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"
    EXPAND = "expand"
    COLLAPSE = "collapse"
    SCROLL = "scroll"
    NAVIGATE = "navigate"
    PRESS_KEY = "press_key"
    WAIT = "wait"
    FINISHED = "finished"


@dataclass
class A11yNode:
    """Represents an accessibility tree node."""

    role: str
    name: str
    value: str = ""
    description: str = ""
    state: Dict[str, Any] = field(default_factory=dict)
    bounds: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    children: List[A11yNode] = field(default_factory=list)
    node_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "role": self.role,
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "state": self.state,
        }
        if self.bounds:
            d["bounds"] = self.bounds
        if self.node_id:
            d["node_id"] = self.node_id
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    def is_interactive(self) -> bool:
        """Check if this node is interactive (clickable/focusable)."""
        interactive_roles = {
            "button", "link", "textbox", "searchbox", "combobox",
            "checkbox", "radio", "switch", "slider", "menuitem",
            "option", "tab", "listitem", "treeitem", "cell",
            "menu", "menubar", "toolbar", "scrollbar",
        }
        return self.role.lower() in interactive_roles

    def find_by_role(self, role: str) -> List[A11yNode]:
        """Find descendant nodes with a given role."""
        results: List[A11yNode] = []
        if self.role.lower() == role.lower():
            results.append(self)
        for child in self.children:
            results.extend(child.find_by_role(role))
        return results

    def find_by_name(self, name: str, partial: bool = True) -> List[A11yNode]:
        """Find nodes whose name contains the given string."""
        results: List[A11yNode] = []
        if name.lower() in self.name.lower() if partial else self.name.lower() == name.lower():
            results.append(self)
        for child in self.children:
            results.extend(child.find_by_name(name, partial))
        return results


@dataclass
class A11yAction:
    """A parsed accessibility-based action."""

    action_type: A11yActionType
    node_id: Optional[str] = None
    role: Optional[str] = None
    name: Optional[str] = None
    value: Optional[str] = None
    text: Optional[str] = None
    key: Optional[str] = None
    direction: Optional[str] = None
    url: Optional[str] = None

    # Metadata
    raw_output: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"action_type": self.action_type.value}
        for attr in ("node_id", "role", "name", "value", "text", "key", "direction", "url", "confidence"):
            val = getattr(self, attr, None)
            if val is not None:
                d[attr] = val
        return d


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class A11yParser:
    """
    Parse model outputs into accessibility tree actions.

    Works with:
    - Windows UIA (UI Automation)
    - macOS AX (Accessibility)
    - Linux AT‑SPI2
    - Web accessibility tree (ARIA)
    """

    def __init__(
        self,
        platform: str = "windows",
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> None:
        self.platform = platform.lower()
        self.screen_width = screen_width
        self.screen_height = screen_height

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse(self, raw_text: str, a11y_tree: Optional[A11yNode] = None) -> A11yAction:
        """Parse raw model output into A11yAction."""
        if not raw_text or not raw_text.strip():
            return A11yAction(action_type=A11yActionType.FINISHED, raw_output=raw_text)

        text = raw_text.strip()

        # Try JSON first
        result = self._parse_json(text)
        if result:
            return result

        # Try platform-specific formats
        if self.platform == "windows":
            result = self._parse_windows_uia(text)
        elif self.platform == "macos":
            result = self._parse_macos_ax(text)
        elif self.platform == "linux":
            result = self._parse_linux_atspi(text)
        else:
            result = self._parse_generic(text)

        if result and result.action_type != A11yActionType.FINISHED:
            return result

        # Fallback to generic
        result = self._parse_generic(text)
        if result:
            return result

        logger.warning(f"Could not parse a11y action: {text[:200]}")
        return A11yAction(action_type=A11yActionType.FINISHED, raw_output=text)

    # ------------------------------------------------------------------
    # Platform-specific parsers
    # ------------------------------------------------------------------

    def _parse_windows_uia(self, text: str) -> Optional[A11yAction]:
        """Parse Windows UIA AutomationElement format."""
        import re

        # UIA pattern: AutomationElement: Name="Submit", Role="Button"
        m = re.search(
            r"AutomationElement\s*:\s*Name\s*=\s*['\"](.+?)['\"].*?Role\s*=\s*['\"](\w+)['\"]",
            text,
            re.IGNORECASE,
        )
        if m:
            name, role = m.group(1), m.group(2)
            action_type = self._role_to_action(role)
            return A11yAction(
                action_type=action_type,
                name=name,
                role=role,
                raw_output=text,
            )

        # UIA click: Click(Name="Submit", Role="Button")
        m = re.search(
            r"Click\s*\(\s*Name\s*=\s*['\"](.+?)['\"].*?Role\s*=\s*['\"](\w+)['\"]\s*\)",
            text,
            re.IGNORECASE,
        )
        if m:
            name, role = m.group(1), m.group(2)
            return A11yAction(
                action_type=A11yActionType.CLICK,
                name=name,
                role=role,
                raw_output=text,
            )

        return None

    def _parse_macos_ax(self, text: str) -> Optional[A11yAction]:
        """Parse macOS AX (Accessibility) format."""
        import re

        # AX pattern: AXButton: "Submit"
        m = re.search(r"(AX\w+)\s*:\s*['\"](.+?)['\"]", text)
        if m:
            role, name = m.group(1), m.group(2)
            action_type = self._role_to_action(role)
            return A11yAction(
                action_type=action_type,
                name=name,
                role=role,
                raw_output=text,
            )

        # AX click: click(AXButton: "Submit")
        m = re.search(r"click\s*\(\s*(AX\w+)\s*:\s*['\"](.+?)['\"]\s*\)", text, re.IGNORECASE)
        if m:
            role, name = m.group(1), m.group(2)
            return A11yAction(
                action_type=A11yActionType.CLICK,
                name=name,
                role=role,
                raw_output=text,
            )

        return None

    def _parse_linux_atspi(self, text: str) -> Optional[A11yAction]:
        """Parse Linux AT‑SPI2 format."""
        import re

        # AT-SPI pattern: role='button', name='Submit'
        m = re.search(r"role\s*=\s*['\"](\w+)['\"].*?name\s*=\s*['\"](.+?)['\"]", text, re.IGNORECASE)
        if m:
            role, name = m.group(1), m.group(2)
            action_type = self._role_to_action(role)
            return A11yAction(
                action_type=action_type,
                name=name,
                role=role,
                raw_output=text,
            )

        return None

    def _parse_generic(self, text: str) -> Optional[A11yAction]:
        """Parse generic key=value format."""
        import re

        text_lower = text.lower()

        m_action = re.search(r"action\s*[:=]\s*(\w+)", text_lower)
        if not m_action:
            return None

        action_type = self._resolve_type(m_action.group(1))
        if action_type == A11yActionType.FINISHED:
            return A11yAction(action_type=action_type, raw_output=text)

        action = A11yAction(action_type=action_type, raw_output=text)

        # role
        m_role = re.search(r"role\s*[:=]\s*['\"](\w+)['\"]", text, re.IGNORECASE)
        if m_role:
            action.role = m_role.group(1)

        # name
        m_name = re.search(r"name\s*[:=]\s*['\"](.+?)['\"]", text)
        if m_name:
            action.name = m_name.group(1)

        # node_id
        m_node = re.search(r"(?:node_id|id)\s*[:=]\s*['\"](.+?)['\"]", text)
        if m_node:
            action.node_id = m_node.group(1)

        # value / text
        m_val = re.search(r"(?:value|text)\s*[:=]\s*['\"](.+?)['\"]", text)
        if m_val:
            action.value = m_val.group(1)

        # key
        m_key = re.search(r"key\s*[:=]\s*['\"](\w+)['\"]", text)
        if m_key:
            action.key = m_key.group(1)

        return action

    def _parse_json(self, text: str) -> Optional[A11yAction]:
        """Parse JSON format."""
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

        return A11yAction(
            action_type=action_type,
            node_id=data.get("node_id"),
            role=data.get("role"),
            name=data.get("name"),
            value=data.get("value") or data.get("text"),
            key=data.get("key"),
            raw_output=text,
            confidence=float(data.get("confidence", 1.0)),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_type(name: str) -> A11yActionType:
        mapping: Dict[str, A11yActionType] = {
            "click": A11yActionType.CLICK,
            "double_click": A11yActionType.DOUBLE_CLICK,
            "right_click": A11yActionType.RIGHT_CLICK,
            "type": A11yActionType.TYPE,
            "input": A11yActionType.TYPE,
            "clear": A11yActionType.CLEAR,
            "select": A11yActionType.SELECT,
            "check": A11yActionType.CHECK,
            "uncheck": A11yActionType.UNCHECK,
            "expand": A11yActionType.EXPAND,
            "collapse": A11yActionType.COLLAPSE,
            "scroll": A11yActionType.SCROLL,
            "navigate": A11yActionType.NAVIGATE,
            "press_key": A11yActionType.PRESS_KEY,
            "wait": A11yActionType.WAIT,
            "finished": A11yActionType.FINISHED,
        }
        return mapping.get(name.lower(), A11yActionType.CLICK)

    @staticmethod
    def _role_to_action(role: str) -> A11yActionType:
        """Map accessibility role to default action type."""
        role_lower = role.lower()
        if "button" in role_lower or "link" in role_lower:
            return A11yActionType.CLICK
        elif "text" in role_lower or "edit" in role_lower or "field" in role_lower:
            return A11yActionType.TYPE
        elif "check" in role_lower:
            return A11yActionType.CHECK
        elif "radio" in role_lower:
            return A11yActionType.SELECT
        elif "combo" in role_lower or "dropdown" in role_lower:
            return A11yActionType.SELECT
        elif "scroll" in role_lower:
            return A11yActionType.SCROLL
        elif "tab" in role_lower:
            return A11yActionType.NAVIGATE
        else:
            return A11yActionType.CLICK

    # ------------------------------------------------------------------
    # Tree Matching
    # ------------------------------------------------------------------

    def match_node(
        self,
        tree: A11yNode,
        action: A11yAction,
    ) -> Optional[A11yNode]:
        """Find the best matching node in the accessibility tree for the given action."""
        candidates: List[A11yNode] = []

        def _collect(node: A11yNode) -> None:
            if not node.is_interactive():
                return

            # Match by node_id (exact)
            if action.node_id and node.node_id == action.node_id:
                candidates.append(node)
                return

            # Match by role
            if action.role and node.role.lower() == action.role.lower():
                candidates.append(node)
                return

            # Match by name (partial)
            if action.name and action.name.lower() in node.name.lower():
                candidates.append(node)
                return

        self._walk_tree(tree, _collect)

        if not candidates:
            return None

        # Prefer nodes with matching role + name
        for node in candidates:
            if action.role and action.name:
                if node.role.lower() == action.role.lower() and action.name.lower() in node.name.lower():
                    return node

        # Fallback to first candidate
        return candidates[0]

    @staticmethod
    def _walk_tree(node: A11yNode, callback) -> None:
        callback(node)
        for child in node.children:
            A11yParser._walk_tree(child, callback)
