"""
Privacy Impact Assessment for Edge GUI Agent Deployments.

Analyzes privacy implications across strategies:
- Fully local: all processing on-device, zero data leaves
- Local GUI + remote planner: screenshots/actions stay local, only queries sent
- Hybrid: screenshots may contain sensitive page content sent to remote
- Preprocess-then-send: structured data after redaction

Provides:
- Data flow analysis per strategy
- PII detection and redaction metrics
- Compliance guidance table
- Risk scoring
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

class ComplianceStandard(str, Enum):
    GDPR = "GDPR"
    CCPA = "CCPA"
    HIPAA = "HIPAA"
    SOC2 = "SOC2"
    ISO27001 = "ISO27001"


class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DataFlow:
    """Description of data flowing in a strategy."""
    stage: str
    data_type: str
    contains_pii: bool
    destination: str
    encrypted: bool
    retention: str


@dataclass
class PrivacyProfile:
    """Privacy profile for a single strategy."""
    strategy: str
    risk_level: RiskLevel
    data_flows: List[DataFlow] = field(default_factory=list)
    pii_exposure_score: float = 0.0  # 0=no exposure, 100=full exposure
    recommendations: List[str] = field(default_factory=list)
    compliance_status: Dict[str, bool] = field(default_factory=dict)


@dataclass
class PrivacyReport:
    """Full privacy assessment report."""
    profiles: Dict[str, PrivacyProfile] = field(default_factory=dict)
    overall_recommendation: str = ""
    best_strategy: str = ""


# ---------------------------------------------------------------------------
# PII Detector (in-memory, no network)
# ---------------------------------------------------------------------------

class PIIDetector:
    """Detects PII in text content. All processing is local and offline."""

    def __init__(self) -> None:
        self.patterns = {
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "phone": r'(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}',
            "ssn": r'\d{3}-\d{2}-\d{4}',
            "credit_card": r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "address": r'\d{1,5}\s+\w+\s+(?:street|st|avenue|ave|road|rd|blvd|lane|ln|drive|dr)',
            "dob": r'\b(?:dob|date of birth)[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            "api_key": r'(?:api[_-]?key|token|secret)[:\s]*["\']?[a-zA-Z0-9]{20,}["\']?',
            "password_field": r'<input[^>]*type=["\']password["\']',
        }

    def scan(self, text: str) -> Dict[str, Any]:
        """Scan text for PII occurrences."""
        import re

        findings: Dict[str, List[str]] = {}
        total_count = 0

        for category, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                findings[category] = matches[:10]  # cap at 10 per category
                total_count += len(matches)

        return {
            "has_pii": total_count > 0,
            "total_findings": total_count,
            "categories": list(findings.keys()),
            "details": findings,
        }

    def redact(self, text: str) -> Tuple[str, int]:
        """Redact PII from text. Returns (redacted_text, redaction_count)."""
        import re

        count = 0
        redacted = text

        for pattern in self.patterns.values():
            matches = re.findall(pattern, redacted, re.IGNORECASE)
            count += len(matches)
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

        return redacted, count

    def assess_risk(self, data: str, data_type: str) -> RiskLevel:
        """Assess privacy risk of data."""
        findings = self.scan(data)

        if not findings["has_pii"]:
            return RiskLevel.NONE

        # Count and categorize
        pii_count = findings["total_findings"]
        categories = findings["categories"]

        high_risk_categories = {"ssn", "credit_card", "password_field", "api_key"}
        medium_risk_categories = {"email", "phone", "address", "dob"}
        low_risk_categories = {"ip_address"}

        has_high = bool(set(categories) & high_risk_categories)
        has_medium = bool(set(categories) & medium_risk_categories)

        if has_high:
            return RiskLevel.CRITICAL
        elif has_medium or pii_count >= 5:
            return RiskLevel.HIGH
        elif pii_count >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


# ---------------------------------------------------------------------------
# Privacy Assessor
# ---------------------------------------------------------------------------

class PrivacyAssessor:
    """
    Assesses privacy implications across strategies.
    """

    def __init__(self) -> None:
        self.detector = PIIDetector()

    def assess_all(self) -> PrivacyReport:
        """Create full privacy assessment across all strategies."""
        report = PrivacyReport()

        profiles = {
            "fully_local": self._assess_fully_local(),
            "hybrid": self._assess_hybrid(),
            "preprocess": self._assess_preprocess(),
        }

        for name, profile in profiles.items():
            # Assess compliance
            profile.compliance_status = self._assess_compliance(profile)

            # Sort by PII exposure score (lower is better)
            report.profiles[name] = profile

        # Determine best strategy for privacy
        best = min(report.profiles.items(), key=lambda x: x[1].pii_exposure_score)
        report.best_strategy = best[0]

        # Overall recommendation
        report.overall_recommendation = (
            f"For maximum privacy, use the '{best[0]}' strategy "
            f"(PII exposure score: {best[1].pii_exposure_score:.0f}/100). "
            f"Preprocessing local data before sending only structured, "
            f"redacted information to remote APIs provides the best balance "
            f"of performance and privacy."
        )

        return report

    # ------------------------------------------------------------------
    # Per-Strategy Assessments
    # ------------------------------------------------------------------

    def _assess_fully_local(self) -> PrivacyProfile:
        profile = PrivacyProfile(
            strategy="Fully Local",
            risk_level=RiskLevel.NONE,
            pii_exposure_score=0.0,
            data_flows=[
                DataFlow("Screenshot capture", "image/screenshot", True, "Local GPU", True, "ephemeral"),
                DataFlow("GUI model inference", "pixel data", True, "Local GPU/CPU", True, "ephemeral"),
                DataFlow("VLM planning", "text + image", True, "Local GPU/CPU", True, "ephemeral"),
                DataFlow("Action execution", "coordinates/selectors", False, "Local browser", True, "ephemeral"),
            ],
            recommendations=[
                "No external data transfer — ideal for sensitive use cases",
                "Ensure model weights are from trusted sources",
                "Log files may contain page content — sanitize or disable",
            ],
        )
        return profile

    def _assess_hybrid(self) -> PrivacyProfile:
        profile = PrivacyProfile(
            strategy="Hybrid (Local GUI + Remote Planner)",
            risk_level=RiskLevel.MEDIUM,
            pii_exposure_score=40.0,
            data_flows=[
                DataFlow("Screenshot capture", "image/screenshot", True, "Local GPU", True, "ephemeral"),
                DataFlow("GUI model inference", "pixel data", True, "Local GPU/CPU", True, "ephemeral"),
                DataFlow("Planning query", "task description text", False, "Remote API", True, "varies by provider"),
                DataFlow("Evaluation", "step result text", False, "Remote API", True, "varies by provider"),
            ],
            recommendations=[
                "Task descriptions should NOT include page content — only abstract instructions",
                "Verify remote API provider's data retention policy",
                "Use zero-data-retention providers (e.g., OpenAI API with data opt-out)",
                "Consider using self-hosted remote models if regulatory compliance required",
                "Add PII scanning before sending any data to remote",
            ],
        )
        return profile

    def _assess_preprocess(self) -> PrivacyProfile:
        profile = PrivacyProfile(
            strategy="Preprocess + Remote",
            risk_level=RiskLevel.LOW,
            pii_exposure_score=15.0,
            data_flows=[
                DataFlow("Screenshot capture", "image/screenshot", True, "Local GPU", True, "ephemeral"),
                DataFlow("GUI model inference", "pixel data", True, "Local GPU/CPU", True, "ephemeral"),
                DataFlow("PII detection & redaction", "text + patterns", True, "Local CPU", True, "ephemeral"),
                DataFlow("Structured data", "JSON (element types, positions)", False, "Remote API", True, "varies"),
            ],
            recommendations=[
                "Best balance of performance and privacy",
                "Ensure redaction pipeline is comprehensive",
                "Consider differential privacy for aggregate data sent to remote",
                "Regularly audit redaction patterns for new PII types",
                "Document data minimization practices for compliance audits",
            ],
        )
        return profile

    # ------------------------------------------------------------------
    # Compliance Assessment
    # ------------------------------------------------------------------

    def _assess_compliance(self, profile: PrivacyProfile) -> Dict[str, bool]:
        """Check compliance with major standards."""
        status = {}

        # GDPR
        status[ComplianceStandard.GDPR.value] = (
            profile.risk_level in (RiskLevel.NONE, RiskLevel.LOW)
            and profile.pii_exposure_score < 30
        )

        # CCPA
        status[ComplianceStandard.CCPA.value] = (
            profile.risk_level in (RiskLevel.NONE, RiskLevel.LOW)
            and profile.pii_exposure_score < 40
        )

        # HIPAA
        status[ComplianceStandard.HIPAA.value] = (
            profile.risk_level == RiskLevel.NONE
            and profile.pii_exposure_score == 0
        )

        # SOC2
        status[ComplianceStandard.SOC2.value] = (
            profile.pii_exposure_score < 50
        )

        # ISO27001
        status[ComplianceStandard.ISO27001.value] = (
            profile.pii_exposure_score < 50
        )

        return status

    # ------------------------------------------------------------------
    # Data Sanitization (utility)
    # ------------------------------------------------------------------

    def sanitize_for_remote(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Sanitize text before sending to remote API.

        Returns (sanitized_text, audit_log).
        """
        pii_result = self.detector.scan(text)
        redacted_text, count = self.detector.redact(text)
        risk = self.detector.assess_risk(text, "text")

        audit = {
            "original_length": len(text),
            "sanitized_length": len(redacted_text),
            "pii_found": pii_result["has_pii"],
            "pii_categories": pii_result["categories"],
            "redactions": count,
            "risk_level": risk.value,
            "action": "send" if risk in (RiskLevel.NONE, RiskLevel.LOW) else "review_before_send",
        }

        if risk in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
            logger.warning(
                f"Data contains {risk.value} risk PII ({count} findings). "
                f"Review before sending to remote."
            )

        return redacted_text, audit

    # ------------------------------------------------------------------
    # Report Export
    # ------------------------------------------------------------------

    def export_report(self, path: str) -> None:
        """Export privacy report as JSON."""
        report = self.assess_all()

        data = {
            "best_strategy": report.best_strategy,
            "overall_recommendation": report.overall_recommendation,
            "profiles": {},
        }

        for name, profile in report.profiles.items():
            data["profiles"][name] = {
                "strategy": profile.strategy,
                "risk_level": profile.risk_level.value,
                "pii_exposure_score": profile.pii_exposure_score,
                "recommendations": profile.recommendations,
                "compliance_status": profile.compliance_status,
                "data_flows": [
                    {
                        "stage": df.stage,
                        "data_type": df.data_type,
                        "contains_pii": df.contains_pii,
                        "destination": df.destination,
                        "encrypted": df.encrypted,
                    }
                    for df in profile.data_flows
                ],
            }

        import json
        from pathlib import Path

        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"Privacy report exported to {path}")


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

def demo() -> None:
    assessor = PrivacyAssessor()

    # Test PII detection
    detector = PIIDetector()
    sample_text = "Contact john@example.com or call 555-123-4567. SSN: 123-45-6789."
    result = detector.scan(sample_text)
    print(f"PII scan: {json.dumps(result, indent=2)}")

    redacted, count = detector.redact(sample_text)
    print(f"Redacted ({count}): {redacted}")

    # Full assessment
    report = assessor.assess_all()
    print(f"\nBest privacy strategy: {report.best_strategy}")
    print(f"Recommendation: {report.overall_recommendation}")

    # Compliance table
    print("\nCompliance Status:")
    for name, profile in report.profiles.items():
        print(f"  {name}:")
        for standard, compliant in profile.compliance_status.items():
            icon = "✓" if compliant else "✗"
            print(f"    {icon} {standard}")


if __name__ == "__main__":
    demo()
