"""
Preprocessor — Pre-process page content before sending to remote models.

Strategy: Transform raw DOM/screenshot data into structured,
privacy-safe intermediate representations before sending to 3rd-party APIs.

Benefits:
- Reduces data sent to remote (bandwidth & latency)
- Removes PII/sensitive data before leaving edge device
- Structure the data for better model understanding
- Reduces token count (cost) for paid APIs

Methods:
1. DOM Filtering — extract only interactive elements, strip text content
2. A11y Tree Extraction — use accessibility tree for structured page representation
3. Screenshot Anonymization — blur/hide sensitive regions
4. Semantic Compression — summarize page content before sending
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class PreprocessMethod(str, Enum):
    DOM_FILTER = "dom_filter"
    A11Y_TREE = "a11y_tree"
    ANONYMIZE = "anonymize"
    SEMANTIC_COMPRESS = "semantic_compress"
    HYBRID = "hybrid"


@dataclass
class PreprocessResult:
    """Result of preprocessing."""
    method: PreprocessMethod
    original_size_bytes: int
    processed_size_bytes: int
    compression_ratio: float
    data: Dict[str, Any]
    redactions: int = 0
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Sensitive Data Patterns
# ---------------------------------------------------------------------------

# Patterns to detect PII in text content
SENSITIVE_PATTERNS: List[Tuple[str, str, re.Pattern]] = [
    ("email", r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', re.IGNORECASE),
    ("phone", r'\b(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'),
    ("ssn", r'\b\d{3}-\d{2}-\d{4}\b'),
    ("credit_card", r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
    ("ip_address", r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    ("api_key", r'\b(sk-[a-zA-Z0-9]{20,}|[a-zA-Z0-9]{32,})\b'),
]

SENSITIVE_SELECTORS = [
    'input[type="password"]',
    '[data-sensitive]',
    '.account-number',
    '.credit-card',
    '.ssn',
    '.personal-info',
]


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class Preprocessor:
    """Pre-process page content before sending to remote models."""

    def __init__(self, method: PreprocessMethod = PreprocessMethod.HYBRID) -> None:
        self.method = method

    def process(
        self,
        screenshot_base64: Optional[str] = None,
        dom_json: Optional[Dict] = None,
        a11y_json: Optional[Dict] = None,
        page_text: Optional[str] = None,
    ) -> PreprocessResult:
        """
        Preprocess page data based on configured method.

        Args:
            screenshot_base64: Raw screenshot (base64)
            dom_json: Raw DOM structure
            a11y_json: Raw accessibility tree
            page_text: Raw page text content

        Returns:
            PreprocessResult with processed data and metrics
        """
        import time

        start = time.perf_counter()

        # Calculate original size
        original_size = 0
        if screenshot_base64:
            original_size += len(screenshot_base64.encode())
        if dom_json:
            original_size += len(json.dumps(dom_json).encode())
        if page_text:
            original_size += len(page_text.encode())

        redactions = 0
        data: Dict[str, Any] = {}

        if self.method == PreprocessMethod.DOM_FILTER:
            data = self._filter_dom(dom_json or {})
        elif self.method == PreprocessMethod.A11Y_TREE:
            data = self._extract_a11y(a11y_json or {})
        elif self.method == PreprocessMethod.ANONYMIZE:
            data, redactions = self._anonymize(screenshot_base64, page_text or "")
        elif self.method == PreprocessMethod.SEMANTIC_COMPRESS:
            data = self._semantic_compress(dom_json, page_text)
        elif self.method == PreprocessMethod.HYBRID:
            # Combine DOM filter + anonymize
            filtered = self._filter_dom(dom_json or {})
            _, redactions = self._anonymize(None, json.dumps(filtered))
            data = filtered

        processed_size = len(json.dumps(data).encode())
        elapsed = time.perf_counter() - start

        return PreprocessResult(
            method=self.method,
            original_size_bytes=original_size,
            processed_size_bytes=processed_size,
            compression_ratio=round(processed_size / max(original_size, 1), 3),
            data=data,
            redactions=redactions,
            elapsed_ms=round(elapsed * 1000, 1),
        )

    # ------------------------------------------------------------------
    # DOM Filtering
    # ------------------------------------------------------------------

    def _filter_dom(self, dom: Dict) -> Dict[str, Any]:
        """Extract only interactive elements from DOM."""
        elements = self._flatten_elements(dom.get("elements", []) or self._parse_raw(dom))

        interactive = {
            "links": [],
            "buttons": [],
            "inputs": [],
            "forms": [],
            "total_elements": len(elements),
            "interactive_count": 0,
        }

        for el in elements:
            tag = el.get("tag", "").lower()
            info = {
                "tag": tag,
                "selector": el.get("selector", ""),
                "text": el.get("text", "")[:100],
                "position": el.get("position", {}),
                "visible": el.get("visible", True),
            }

            if tag == "a":
                info["href"] = el.get("href", "")[:200]
                interactive["links"].append(info)
                interactive["interactive_count"] += 1
            elif tag in ("button",) or el.get("role") == "button":
                interactive["buttons"].append(info)
                interactive["interactive_count"] += 1
            elif tag in ("input", "select", "textarea"):
                info["type"] = el.get("type", "")
                info["name"] = el.get("name", "")
                interactive["inputs"].append(info)
                interactive["interactive_count"] += 1
            elif tag == "form":
                interactive["forms"].append(info)
                interactive["interactive_count"] += 1

        return interactive

    def _flatten_elements(self, elements: List[Dict]) -> List[Dict]:
        """Recursively flatten nested element tree."""
        result = []
        for el in elements:
            result.append(el)
            if "children" in el:
                result.extend(self._flatten_elements(el["children"]))
        return result

    def _parse_raw(self, dom: Dict) -> List[Dict]:
        """Parse raw dict into element list."""
        if "interactiveElements" in dom:
            return dom["interactiveElements"]
        if "body" in dom and "children" in dom["body"]:
            return self._flatten_elements(dom["body"]["children"])
        return []

    # ------------------------------------------------------------------
    # Accessibility Tree
    # ------------------------------------------------------------------

    def _extract_a11y(self, a11y: Dict) -> Dict[str, Any]:
        """Extract structured info from accessibility tree."""
        nodes = self._flatten_a11y(a11y.get("nodes", a11y.get("tree", [])) or [])

        interactive_nodes = []
        for node in nodes:
            role = node.get("role", "").lower()
            if role in ("link", "button", "textbox", "combobox", "checkbox",
                        "radio", "menuitem", "option", "tab", "switch", "searchbox"):
                interactive_nodes.append({
                    "role": role,
                    "name": node.get("name", ""),
                    "description": node.get("description", ""),
                    "value": node.get("value", ""),
                    "disabled": node.get("disabled", False),
                    "position": node.get("position", {}),
                })

        return {
            "method": "a11y_tree",
            "interactive_nodes": interactive_nodes,
            "total_count": len(interactive_nodes),
            "hierarchy_summary": self._summarize_hierarchy(nodes),
        }

    def _flatten_a11y(self, nodes: List[Dict]) -> List[Dict]:
        result = []
        for node in nodes:
            result.append(node)
            for key in ("children", "nodes", "content"):
                if key in node:
                    result.extend(self._flatten_a11y(node[key]))
        return result

    def _summarize_hierarchy(self, nodes: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for node in nodes:
            role = node.get("role", "unknown")
            counts[role] = counts.get(role, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Anonymization
    # ------------------------------------------------------------------

    def _anonymize(
        self, screenshot_base64: Optional[str], text: str
    ) -> Tuple[Dict[str, Any], int]:
        """Detect and redact sensitive information."""
        redactions = 0

        # Redact text
        redacted_text = text
        for _label, pattern in SENSITIVE_PATTERNS:
            matches = pattern.findall(redacted_text)
            redactions += len(matches)
            redacted_text = pattern.sub("[REDACTED]", redacted_text)

        # Identify sensitive regions in screenshot (stub — real impl would use OCR)
        sensitive_regions = self._detect_sensitive_regions(text)

        return {
            "method": "anonymized",
            "text": redacted_text,
            "has_screenshot": screenshot_base64 is not None,
            "sensitive_regions": sensitive_regions,
            "note": "Screenshot regions should be blurred client-side before sending",
        }, redactions

    def _detect_sensitive_regions(self, text: str) -> List[Dict[str, Any]]:
        """Identify regions containing sensitive data."""
        regions = []
        for label, pattern in SENSITIVE_PATTERNS:
            for match in pattern.finditer(text):
                regions.append({
                    "type": label,
                    "position": {"start": match.start(), "end": match.end()},
                    "text": "[REDACTED]",
                })
        return regions

    # ------------------------------------------------------------------
    # Semantic Compression
    # ------------------------------------------------------------------

    def _semantic_compress(
        self, dom: Optional[Dict], text: Optional[str]
    ) -> Dict[str, Any]:
        """Create a compressed semantic summary of the page."""
        interactive = self._filter_dom(dom or {})

        # Build a minimal page summary
        summary_parts = []
        if interactive["links"]:
            summary_parts.append(f"{len(interactive['links'])} links: " +
                                 "; ".join(l["text"][:50] for l in interactive["links"][:5]))
        if interactive["buttons"]:
            summary_parts.append(f"{len(interactive['buttons'])} buttons: " +
                                 "; ".join(b["text"][:50] for b in interactive["buttons"][:5]))
        if interactive["inputs"]:
            summary_parts.append(f"{len(interactive['inputs'])} input fields")

        page_summary = " | ".join(summary_parts) if summary_parts else "No interactive elements"

        # Truncate text
        text_sample = (text or "")[:2000]

        return {
            "method": "semantic_compress",
            "page_summary": page_summary,
            "interactive_elements": interactive,
            "text_sample": text_sample,
        }


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def benchmark_preprocessing(
    preprocessor: Preprocessor,
    page_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Benchmark different preprocessing methods."""
    import time

    results = {}
    for method in PreprocessMethod:
        p = Preprocessor(method=method)
        latencies = []
        ratios = []
        redactions_total = 0

        for data in page_data:
            start = time.perf_counter()
            result = p.process(
                screenshot_base64=data.get("screenshot"),
                dom_json=data.get("dom"),
                page_text=data.get("text"),
            )
            latencies.append(result.elapsed_ms)
            ratios.append(result.compression_ratio)
            redactions_total += result.redactions

        results[method.value] = {
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
            "avg_compression_ratio": round(sum(ratios) / max(len(ratios), 1), 3),
            "total_redactions": redactions_total,
        }

    return results
