"""
Unified model registry and configuration management for GUI interaction models.

Supports:
- UI-TARS (1.5 / 2) — ByteDance, coordinate-based
- ShowUI — visual grounding + action
- OS-ATLAS — OS-level GUI agent
- Other models as they become available
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class ActionMethod(str, Enum):
    """GUI interaction method categories."""
    COORDINATE = "coordinate"       # absolute coordinates (UI-TARS style)
    DOM = "dom"                     # DOM / accessibility tree
    ACCESSIBILITY = "accessibility" # accessibility tree only
    HYBRID = "hybrid"               # combined approach
    VLM_ONLY = "vlm_only"           # pure VLM reasoning


class ModelFamily(str, Enum):
    """Known model families."""
    UI_TARS = "ui-tars"
    UI_TARS_2 = "ui-tars-2"
    SHOWUI = "showui"
    OS_ATLAS = "os-atlas"
    FARA = "fara"
    ULTRA_CUA = "ultra-cua"
    AGENT_CPM = "agent-cpm"
    MOBILE_AGENT_V3 = "mobile-agent-v3"
    UI_VENUS = "ui-venus"
    UITRON = "uitron"
    CUSTOM = "custom"


class DeviceTarget(str, Enum):
    """Target deployment device."""
    DESKTOP = "desktop"
    MOBILE = "mobile"
    WEB = "web"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a single GUI model instance."""

    name: str
    family: ModelFamily
    hf_repo_id: str
    action_method: ActionMethod
    device_target: DeviceTarget = DeviceTarget.DESKTOP

    # Model loading
    dtype: str = "auto"                     # torch dtype
    device_map: str = "auto"
    trust_remote_code: bool = True
    max_memory: Optional[Dict[int, str]] = None

    # Inference
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.95
    repetition_penalty: float = 1.0

    # Image processing
    image_size: int = 1024
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28

    # vLLM-specific
    use_vllm: bool = False
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.90
    vllm_max_model_len: Optional[int] = None

    # Extra
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Enum):
                d[k] = v.value
            elif isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Deserialize from dictionary."""
        data = dict(data)
        if "family" in data and isinstance(data["family"], str):
            data["family"] = ModelFamily(data["family"])
        if "action_method" in data and isinstance(data["action_method"], str):
            data["action_method"] = ActionMethod(data["action_method"])
        if "device_target" in data and isinstance(data["device_target"], str):
            data["device_target"] = DeviceTarget(data["device_target"])
        return cls(**data)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Pre-defined model registry entries
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "ui-tars-1.5-7b": ModelConfig(
        name="UI-TARS-1.5-7B",
        family=ModelFamily.UI_TARS,
        hf_repo_id="bytedance/UI-TARS-1.5-7B",
        action_method=ActionMethod.COORDINATE,
        device_target=DeviceTarget.DESKTOP,
        image_size=1024,
        max_new_tokens=1024,
        metadata={
            "description": "ByteDance UI-TARS 1.5 7B — coordinate-based GUI agent",
            "paper": "https://arxiv.org/abs/2501.12326",
            "supports": ["desktop", "mobile", "web"],
            "license": "Apache-2.0",
        },
    ),

    "ui-tars-2-7b": ModelConfig(
        name="UI-TARS-2-7B",
        family=ModelFamily.UI_TARS_2,
        hf_repo_id="bytedance/UI-TARS-2-7B",
        action_method=ActionMethod.COORDINATE,
        device_target=DeviceTarget.DESKTOP,
        image_size=1024,
        max_new_tokens=1024,
        metadata={
            "description": "ByteDance UI-TARS 2 7B — next-gen coordinate-based GUI agent",
            "supports": ["desktop", "mobile", "web"],
        },
    ),

    "showui-2b": ModelConfig(
        name="ShowUI-2B",
        family=ModelFamily.SHOWUI,
        hf_repo_id="showlab/ShowUI-2B",
        action_method=ActionMethod.COORDINATE,
        device_target=DeviceTarget.WEB,
        image_size=1344,
        min_pixels=256 * 28 * 28,
        max_pixels=1344 * 28 * 28,
        metadata={
            "description": "ShowUI — UI-guided visual grounding model",
            "paper": "https://arxiv.org/abs/2411.03152",
            "supports": ["web", "mobile"],
            "license": "Apache-2.0",
        },
    ),

    "os-atlas-base": ModelConfig(
        name="OS-ATLAS-Base",
        family=ModelFamily.OS_ATLAS,
        hf_repo_id="OS-Copilot/OS-ATLAS",
        action_method=ActionMethod.HYBRID,
        device_target=DeviceTarget.DESKTOP,
        image_size=1024,
        metadata={
            "description": "OS-ATLAS — OS-level GUI grounding foundation model",
            "paper": "https://arxiv.org/abs/2410.03568",
            "supports": ["desktop", "linux", "macos", "windows"],
            "license": "Apache-2.0",
        },
    ),

    "fara-7b": ModelConfig(
        name="Fara-7B",
        family=ModelFamily.FARA,
        hf_repo_id="agiedit/Fara-7B",
        action_method=ActionMethod.COORDINATE,
        device_target=DeviceTarget.HYBRID,
        metadata={
            "description": "Fara-7B — lightweight GUI agent",
            "supports": ["desktop", "web"],
        },
    ),
}


class ModelRegistry:
    """Central registry for GUI model configurations."""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        self._registry: Dict[str, ModelConfig] = dict(MODEL_REGISTRY)
        self.config_dir = config_dir or Path(__file__).parent.parent / "configs"
        self._load_custom_configs()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, key: str, config: ModelConfig) -> None:
        """Register a new or override an existing model config."""
        self._registry[key] = config
        logger.info(f"Registered model: {key} ({config.name})")

    def unregister(self, key: str) -> None:
        """Remove a model config from registry."""
        if key in self._registry:
            del self._registry[key]
            logger.info(f"Unregistered model: {key}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[ModelConfig]:
        """Retrieve a model config by key."""
        return self._registry.get(key)

    def list_all(self) -> Dict[str, ModelConfig]:
        """Return full registry."""
        return dict(self._registry)

    def list_by_method(self, method: ActionMethod) -> Dict[str, ModelConfig]:
        """Filter models by action method."""
        return {k: v for k, v in self._registry.items() if v.action_method == method}

    def list_by_device(self, target: DeviceTarget) -> Dict[str, ModelConfig]:
        """Filter models by target device."""
        return {k: v for k, v in self._registry.items() if v.device_target == target}

    def list_by_family(self, family: ModelFamily) -> Dict[str, ModelConfig]:
        """Filter models by model family."""
        return {k: v for k, v in self._registry.items() if v.family == family}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, key: str, path: Optional[Path] = None) -> Path:
        """Save a model config to JSON file."""
        config = self._registry.get(key)
        if config is None:
            raise KeyError(f"Model config '{key}' not found in registry")

        save_path = path or (self.config_dir / f"{key}.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved config for '{key}' → {save_path}")
        return save_path

    def load(self, path: Path) -> str:
        """Load a model config from JSON file and register it."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = ModelConfig.from_dict(data)
        key = path.stem
        self.register(key, config)
        return key

    def _load_custom_configs(self) -> None:
        """Load any JSON configs from the configs directory."""
        if not self.config_dir.exists():
            return
        for cfg_file in self.config_dir.glob("*.json"):
            try:
                self.load(cfg_file)
            except Exception as exc:
                logger.warning(f"Failed to load config {cfg_file}: {exc}")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the registry."""
        lines = [f"{'='*60}", f"Model Registry — {len(self._registry)} entries", f"{'='*60}"]
        for key, cfg in self._registry.items():
            lines.append(
                f"  [{key}] {cfg.name} | {cfg.family.value} | "
                f"{cfg.action_method.value} | {cfg.device_target.value}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

# Global singleton
_registry_instance: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or create the global model registry singleton."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    reg = get_registry()
    print(reg.summary())
