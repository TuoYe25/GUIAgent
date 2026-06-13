"""
OS-ATLAS Model Deployment Script.

OS-ATLAS: A Foundation Model for GUI Action Agents
Paper: https://arxiv.org/abs/2410.03568
Repo: https://github.com/OS-Copilot/OS-ATLAS

Key features:
- OS-level GUI grounding
- Hybrid approach: combines visual grounding + coordinate prediction
- Cross-platform: Linux, macOS, Windows support
- Supports complex multi-step interactions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from PIL import Image

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HF_REPO = "OS-Copilot/OS-ATLAS"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class OSAtlasConfig:
    """OS-ATLAS configuration."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_HF_REPO,
        device: str = DEFAULT_DEVICE,
        dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        image_size: int = 1024,
        **kwargs: Any,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = self._resolve_dtype(dtype)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.image_size = image_size

    @staticmethod
    def _resolve_dtype(dtype_str: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        }
        return mapping.get(dtype_str, torch.float16)


# ---------------------------------------------------------------------------
# Action Primitives
# ---------------------------------------------------------------------------

ACTION_SPACE: Dict[str, Dict[str, Any]] = {
    "click": {
        "description": "Click at specific coordinates",
        "params": {"x": "int", "y": "int"},
    },
    "double_click": {
        "description": "Double click at coordinates",
        "params": {"x": "int", "y": "int"},
    },
    "right_click": {
        "description": "Right click at coordinates",
        "params": {"x": "int", "y": "int"},
    },
    "type": {
        "description": "Type text at current focus",
        "params": {"text": "str"},
    },
    "key_press": {
        "description": "Press a keyboard key or combination",
        "params": {"keys": "str"},
    },
    "scroll": {
        "description": "Scroll in direction",
        "params": {"direction": "str", "amount": "int"},
    },
    "drag": {
        "description": "Drag from one point to another",
        "params": {"x1": "int", "y1": "int", "x2": "int", "y2": "int"},
    },
    "wait": {
        "description": "Wait for specified duration",
        "params": {"duration_ms": "int"},
    },
    "screenshot": {
        "description": "Take a screenshot (for verification)",
        "params": {},
    },
    "finished": {
        "description": "Task completed",
        "params": {"message": "str"},
    },
}


# ---------------------------------------------------------------------------
# Deployer
# ---------------------------------------------------------------------------

class OSAtlasDeployer:
    """Handle OS-ATLAS model loading and inference."""

    def __init__(self, config: OSAtlasConfig) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load OS-ATLAS model."""
        if self._loaded:
            logger.warning("OS-ATLAS already loaded.")
            return

        logger.info(f"Loading OS-ATLAS from {self.config.model_name_or_path} ...")

        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
            )

            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.model_name_or_path,
                torch_dtype=self.config.dtype,
                device_map=self.config.device if self.config.device != "cpu" else None,
                trust_remote_code=True,
            )
            self.model.eval()
            self._loaded = True
            logger.info("OS-ATLAS loaded successfully.")

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load OS-ATLAS: {e}")
            raise

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Image.Image,
        instruction: str,
        screen_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Predict next action for a GUI task.

        Args:
            image: Current screenshot
            instruction: User task description
            screen_info: Optional metadata (screen size, OS, app context)

        Returns:
            Parsed action dict
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Build context-aware prompt
        prompt = self._build_prompt(instruction, screen_info)

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]

        formatted = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=formatted,
            images=[image],
            return_tensors="pt",
        )
        if self.config.device != "cpu":
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
            )

        response = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]

        if formatted in response:
            response = response[len(formatted):].strip()

        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Multi-step Execution
    # ------------------------------------------------------------------

    def execute_task(
        self,
        screenshots: List[Image.Image],
        instruction: str,
        max_steps: int = 20,
    ) -> List[Dict[str, Any]]:
        """Execute a multi-step task across multiple screenshots.

        Used with an external environment loop that actually performs the actions.
        """
        actions: List[Dict[str, Any]] = []
        for step_idx, screenshot in enumerate(screenshots):
            if step_idx >= max_steps:
                break

            action = self.predict(screenshot, instruction)
            actions.append(action)

            if action.get("action_type") == "finished":
                logger.info(f"Task completed at step {step_idx + 1}")
                break

            logger.info(f"Step {step_idx + 1}: {action.get('action_type')}")

        return actions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        """Build system prompt for OS-ATLAS."""
        return (
            "You are OS-ATLAS, an operating system GUI agent. "
            "You control the computer by viewing screenshots and outputting precise actions. "
            "Actions: click(x,y), double_click(x,y), right_click(x,y), type('text'), "
            "key_press('keys'), scroll(direction, amount), "
            "drag(x1,y1,x2,y2), wait(ms), finished('message')."
        )

    def _build_prompt(
        self,
        instruction: str,
        screen_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a context-rich instruction prompt."""
        parts = [f"Task: {instruction}"]

        if screen_info:
            if "screen_width" in screen_info and "screen_height" in screen_info:
                parts.append(
                    f"Screen: {screen_info['screen_width']}x{screen_info['screen_height']}"
                )
            if "app" in screen_info:
                parts.append(f"Application: {screen_info['app']}")
            if "os" in screen_info:
                parts.append(f"OS: {screen_info['os']}")

        parts.append("\nOutput the next action:")
        return "\n".join(parts)

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse OS-ATLAS action output."""
        import re

        result: Dict[str, Any] = {
            "raw": text,
            "action_type": "unknown",
        }

        text_clean = text.strip()

        # click(123, 456)
        m = re.match(r"click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text_clean, re.IGNORECASE)
        if m:
            result["action_type"] = "click"
            result["x"] = int(m.group(1))
            result["y"] = int(m.group(2))
            return result

        # double_click(123, 456)
        m = re.match(r"double_click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text_clean, re.IGNORECASE)
        if m:
            result["action_type"] = "double_click"
            result["x"] = int(m.group(1))
            result["y"] = int(m.group(2))
            return result

        # right_click(123, 456)
        m = re.match(r"right_click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text_clean, re.IGNORECASE)
        if m:
            result["action_type"] = "right_click"
            result["x"] = int(m.group(1))
            result["y"] = int(m.group(2))
            return result

        # type('text')
        m = re.match(r"type\s*\(\s*['\"](.+?)['\"]\s*\)", text_clean, re.IGNORECASE)
        if m:
            result["action_type"] = "type"
            result["text"] = m.group(1)
            return result

        # key_press('ctrl+c')
        m = re.match(r"key_press\s*\(\s*['\"](.+?)['\"]\s*\)", text_clean, re.IGNORECASE)
        if m:
            result["action_type"] = "key_press"
            result["keys"] = m.group(1)
            return result

        # scroll(down, 100)
        m = re.match(
            r"scroll\s*\(\s*['\"]?(\w+)['\"]?\s*,\s*(\d+)\s*\)",
            text_clean,
            re.IGNORECASE,
        )
        if m:
            result["action_type"] = "scroll"
            result["direction"] = m.group(1)
            result["amount"] = int(m.group(2))
            return result

        # drag(100,200,300,400)
        m = re.match(
            r"drag\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)",
            text_clean,
            re.IGNORECASE,
        )
        if m:
            result["action_type"] = "drag"
            result["x1"] = int(m.group(1))
            result["y1"] = int(m.group(2))
            result["x2"] = int(m.group(3))
            result["y2"] = int(m.group(4))
            return result

        # finished('message')
        m = re.match(r"finished\s*\(\s*['\"](.*?)['\"]\s*\)", text_clean, re.IGNORECASE)
        if m:
            result["action_type"] = "finished"
            result["message"] = m.group(1)
            return result

        # wait(500)
        m = re.match(r"wait\s*\(\s*(\d+)\s*\)", text_clean, re.IGNORECASE)
        if m:
            result["action_type"] = "wait"
            result["duration_ms"] = int(m.group(1))
            return result

        logger.warning(f"Unrecognized action format: {text_clean[:200]}")
        return result

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Verify model is functional."""
        if not self._loaded:
            return False
        try:
            dummy = Image.new("RGB", (256, 256), color="white")
            result = self.predict(dummy, "Click the OK button")
            return "action_type" in result
        except Exception as e:
            logger.warning(f"OS-ATLAS health check failed: {e}")
            return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="OS-ATLAS Model Deployer")
    parser.add_argument("--model", default=DEFAULT_HF_REPO)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--image", type=str, help="Test screenshot")
    parser.add_argument("--instruction", type=str, default="Open the Settings app")
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    args = parser.parse_args()

    config = OSAtlasConfig(model_name_or_path=args.model, device=args.device)
    deployer = OSAtlasDeployer(config)
    deployer.load()

    screen_info = {"screen_width": args.screen_width, "screen_height": args.screen_height, "os": "Windows"}
    if args.image and Path(args.image).exists():
        image = Image.open(args.image).convert("RGB")
        result = deployer.predict(image, args.instruction, screen_info)
        logger.info(f"Result: {result}")
    else:
        logger.info("Model loaded. Provide --image to test.")


if __name__ == "__main__":
    main()
