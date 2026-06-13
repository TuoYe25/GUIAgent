"""
UI-TARS Model Deployment Script.

Supports:
- HuggingFace transformers (default)
- vLLM backend for high-throughput serving
- Text Generation Inference (TGI) compatible API

Model: ByteDance UI-TARS 1.5 / 2
Action Method: Absolute coordinate (x, y) on screen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_HF_REPO = "bytedance/UI-TARS-1.5-7B"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class UITARSConfig:
    """UI-TARS model configuration."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_HF_REPO,
        device: str = DEFAULT_DEVICE,
        dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.95,
        image_size: int = 1024,
        use_fast_tokenizer: bool = True,
        **kwargs: Any,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = self._resolve_dtype(dtype)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.image_size = image_size
        self.use_fast_tokenizer = use_fast_tokenizer

    @staticmethod
    def _resolve_dtype(dtype_str: str) -> torch.dtype:
        mapping: Dict[str, torch.dtype] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        }
        return mapping.get(dtype_str, torch.float16)


# ---------------------------------------------------------------------------
# Deployer
# ---------------------------------------------------------------------------

class UITARSDeployer:
    """Handle UI-TARS model loading and inference."""

    def __init__(self, config: UITARSConfig) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model and processor from HuggingFace."""
        if self._loaded:
            logger.warning("Model already loaded, skipping.")
            return

        logger.info(f"Loading UI-TARS model from {self.config.model_name_or_path} ...")

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

            if self.config.device == "cpu":
                self.model = self.model.to("cpu")
            else:
                self.model = self.model.to(self.config.device)

            self.model.eval()
            self._loaded = True
            logger.info(f"UI-TARS model loaded successfully on {self.config.device}")

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error(
                "Install with: pip install transformers accelerate pillow torch"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Image.Image,
        instruction: str,
        previous_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on a screenshot + instruction.

        Args:
            image: PIL Image screenshot
            instruction: Natural language instruction
            previous_actions: Optional history of previous actions

        Returns:
            Dict with keys: action_type, x, y, text, description
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Build conversation messages
        messages = self._build_messages(image, instruction, previous_actions)

        # Process inputs
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

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                top_p=self.config.top_p if self.config.temperature > 0 else None,
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Remove the prompt portion to get only the response
        if prompt in generated_text:
            response_text = generated_text[len(prompt):].strip()
        else:
            response_text = generated_text.strip()

        # Parse the coordinate action
        parsed = self._parse_response(response_text)
        logger.info(f"UI-TARS prediction: {parsed}")
        return parsed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        image: Image.Image,
        instruction: str,
        previous_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Build the conversation messages for UI-TARS."""
        system_prompt = (
            "You are a GUI agent. Given a screenshot and an instruction, "
            "output the next action as a click at specific coordinates. "
            "Output format: click(x=<int>, y=<int>) or type(text='<str>') "
            "or scroll(direction='up'|'down') or press(key='<str>') or "
            "finished(message='<str>')."
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]

        if previous_actions:
            for action in previous_actions:
                messages.append({"role": "assistant", "content": str(action)})

        return messages

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse UI-TARS action output."""

        result: Dict[str, Any] = {
            "raw": text,
            "action_type": "unknown",
        }

        text_lower = text.strip().lower()

        # click(x=123, y=456)
        if text_lower.startswith("click"):
            import re
            match = re.search(r"x\s*=\s*(\d+).*?y\s*=\s*(\d+)", text)
            if match:
                result["action_type"] = "click"
                result["x"] = int(match.group(1))
                result["y"] = int(match.group(2))

        # type(text='hello')
        elif text_lower.startswith("type"):
            import re
            match = re.search(r"text\s*=\s*['\"](.+?)['\"]", text)
            if match:
                result["action_type"] = "type"
                result["text"] = match.group(1)

        # scroll(direction='up'|'down')
        elif text_lower.startswith("scroll"):
            import re
            match = re.search(r"direction\s*=\s*['\"](\w+)['\"]", text)
            if match:
                result["action_type"] = "scroll"
                result["direction"] = match.group(1)

        # press(key='enter')
        elif text_lower.startswith("press"):
            import re
            match = re.search(r"key\s*=\s*['\"](\w+)['\"]", text)
            if match:
                result["action_type"] = "press"
                result["key"] = match.group(1)

        # finished(message='...')
        elif text_lower.startswith("finished"):
            import re
            match = re.search(r"message\s*=\s*['\"](.+?)['\"]", text)
            if match:
                result["action_type"] = "finished"
                result["message"] = match.group(1)

        return result

    # ------------------------------------------------------------------
    # vLLM Deployment
    # ------------------------------------------------------------------

    def deploy_vllm(
        self,
        port: int = 8000,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
    ) -> None:
        """
        Deploy UI-TARS via vLLM OpenAI-compatible API server.

        This starts a subprocess; use for production inference.
        """
        import subprocess
        import sys

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_name_or_path,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--trust-remote-code",
            "--max-model-len", "8192",
        ]

        logger.info(f"Starting vLLM server: {' '.join(cmd)}")
        try:
            process = subprocess.Popen(cmd)
            logger.info(f"vLLM server PID {process.pid} on port {port}")
        except FileNotFoundError:
            logger.error("vLLM not installed. Install with: pip install vllm")
            raise

    def health_check(self) -> bool:
        """Check if the model is loaded and functional."""
        if not self._loaded:
            return False
        try:
            # Simple smoke test with a tiny dummy image
            dummy = Image.new("RGB", (128, 128), color="white")
            result = self.predict(dummy, "test")
            return "action_type" in result
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UI-TARS Model Deployer")
    parser.add_argument("--model", default=DEFAULT_HF_REPO, help="HF repo or local path")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device: cuda, cpu, mps")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype")
    parser.add_argument("--image", type=str, help="Path to screenshot for test inference")
    parser.add_argument("--instruction", type=str, default="Click the submit button", help="Test instruction")
    parser.add_argument("--serve", action="store_true", help="Start vLLM server")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    args = parser.parse_args()

    config = UITARSConfig(
        model_name_or_path=args.model,
        device=args.device,
        dtype=args.dtype,
    )
    deployer = UITARSDeployer(config)

    if args.serve:
        deployer.deploy_vllm(port=args.port)
    else:
        deployer.load()
        if args.image and Path(args.image).exists():
            image = Image.open(args.image).convert("RGB")
            result = deployer.predict(image, args.instruction)
            logger.info(f"Result: {result}")
        else:
            logger.info("Model loaded. Use --image and --instruction to test.")


if __name__ == "__main__":
    main()
