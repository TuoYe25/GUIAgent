"""
ShowUI Model Deployment Script.

ShowUI: UI-guided visual grounding model for GUI interaction.
Paper: https://arxiv.org/abs/2411.03152
Repo: https://github.com/showlab/ShowUI

Key features:
- Unified modeling of UI grounding and instruction following
- Visual token selection for efficient UI representation
- Supports web and mobile UI tasks
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from PIL import Image

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HF_REPO = "showlab/ShowUI-2B"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ShowUIConfig:
    """ShowUI model configuration."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_HF_REPO,
        device: str = DEFAULT_DEVICE,
        dtype: str = "bfloat16",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        image_size: int = 1344,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1344 * 28 * 28,
        **kwargs: Any,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = self._resolve_dtype(dtype)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.image_size = image_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

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
# Deployer
# ---------------------------------------------------------------------------

class ShowUIDeployer:
    """Handle ShowUI model loading and inference."""

    # Action space definition (ref: ShowUI paper)
    ACTION_TYPES: List[str] = [
        "CLICK", "TYPE", "SCROLL_UP", "SCROLL_DOWN",
        "PRESS_ENTER", "PRESS_BACK", "PRESS_HOME",
        "LONG_PRESS", "SWIPE_UP", "SWIPE_DOWN",
        "SWIPE_LEFT", "SWIPE_RIGHT", "WAIT", "FINISHED",
    ]

    def __init__(self, config: ShowUIConfig) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load ShowUI model."""
        if self._loaded:
            logger.warning("ShowUI already loaded.")
            return

        logger.info(f"Loading ShowUI from {self.config.model_name_or_path} ...")

        try:
            from transformers import AutoModel, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
                min_pixels=self.config.min_pixels,
                max_pixels=self.config.max_pixels,
            )

            self.model = AutoModel.from_pretrained(
                self.config.model_name_or_path,
                torch_dtype=self.config.dtype,
                device_map=self.config.device if self.config.device != "cpu" else None,
                trust_remote_code=True,
            )
            self.model.eval()
            self._loaded = True
            logger.info("ShowUI loaded successfully.")

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load ShowUI: {e}")
            raise

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Dict[str, Any]:
        """Predict action from screenshot + instruction.

        Returns dict with:
            action_type: one of CLICK, TYPE, SCROLL, etc.
            point: (x, y) in range [0, 999] for click actions
            text: text for TYPE actions
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Prepare messages in ShowUI format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=prompt,
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

        # Extract the model's response (after the prompt)
        if prompt in response:
            response = response[len(prompt):].strip()

        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Response Parsing
    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse ShowUI action output.

        ShowUI outputs in format:
            <action> CLICK </action> <point> x y </point>
        or  <action> TYPE </action> <text> hello world </text>
        """
        import re

        result: Dict[str, Any] = {
            "raw": text,
            "action_type": "UNKNOWN",
        }

        # Extract action
        action_match = re.search(r"<action>\s*(.*?)\s*</action>", text, re.DOTALL)
        if action_match:
            result["action_type"] = action_match.group(1).strip().upper()

        # Extract point for click/long_press
        point_match = re.search(r"<point>\s*(\d+)\s+(\d+)\s*</point>", text)
        if point_match:
            result["point"] = (int(point_match.group(1)), int(point_match.group(2)))

        # Extract text for type actions
        text_match = re.search(r"<text>\s*(.*?)\s*</text>", text, re.DOTALL)
        if text_match:
            result["text"] = text_match.group(1).strip()

        logger.info(f"ShowUI parsed: {result['action_type']}")
        return result

    # ------------------------------------------------------------------
    # Grounding
    # ------------------------------------------------------------------

    def ground_element(
        self,
        image: Image.Image,
        element_description: str,
    ) -> Optional[Tuple[int, int]]:
        """
        Ground a natural language description to coordinates.

        This is ShowUI's core capability — finding UI elements by description.
        """
        result = self.predict(image, f"Find and click: {element_description}")
        if result.get("action_type") == "CLICK" and "point" in result:
            return result["point"]
        return None

    def list_elements(
        self,
        image: Image.Image,
    ) -> List[Dict[str, Any]]:
        """Enumerate visible UI elements with their coordinates."""
        result = self.predict(image, "List all clickable elements and their positions")
        return [result]  # Simplified; real impl parses structured output

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Verify model is functional."""
        if not self._loaded:
            return False
        try:
            dummy = Image.new("RGB", (256, 256), color="white")
            result = self.predict(dummy, "click the button")
            return "action_type" in result
        except Exception as e:
            logger.warning(f"ShowUI health check failed: {e}")
            return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ShowUI Model Deployer")
    parser.add_argument("--model", default=DEFAULT_HF_REPO)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--image", type=str, help="Test screenshot")
    parser.add_argument("--instruction", type=str, default="Click the submit button")
    parser.add_argument("--ground", type=str, help="Element description to ground")
    args = parser.parse_args()

    config = ShowUIConfig(model_name_or_path=args.model, device=args.device)
    deployer = ShowUIDeployer(config)
    deployer.load()

    if args.image and Path(args.image).exists():
        image = Image.open(args.image).convert("RGB")
        if args.ground:
            pt = deployer.ground_element(image, args.ground)
            logger.info(f"Grounded '{args.ground}' → {pt}")
        else:
            result = deployer.predict(image, args.instruction)
            logger.info(f"Result: {result}")
    else:
        logger.info("Model loaded. Provide --image to test.")


if __name__ == "__main__":
    main()
