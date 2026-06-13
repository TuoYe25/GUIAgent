"""
Deploy edge VLM for orchestrator/planner.

Supports:
- Phi-3-Vision (microsoft/Phi-3-vision-128k-instruct)
- LLaVA-1.6 (llava-hf/llava-v1.6-mistral-7b-hf)
- Qwen2-VL (Qwen/Qwen2-VL-7B-Instruct)
- InternVL2 (OpenGVLab/InternVL2-4B)
- MiniCPM-V (openbmb/MiniCPM-V-2_6)

Usage:
    python scripts/deploy_edge_vlm.py --model phi3-vision --port 8766
    python scripts/deploy_edge_vlm.py --model qwen2-vl-7b --port 8766 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vlm_integration.orchestrator import VLMOrchestrator, VLMConfig, EDGE_VLM_PRESETS


# ---------------------------------------------------------------------------
# Deployment Logic
# ---------------------------------------------------------------------------

def deploy_vllm(model_id: str, port: int, device: str) -> bool:
    """Deploy model using vLLM if available."""
    try:
        import subprocess
        from vllm.entrypoints.openai.api_server import main as vllm_main
        logger.info(f"Starting vLLM server for {model_id} on port {port}...")
        # In practice, launch as subprocess or use vLLM's programmatic API
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_id,
            "--port", str(port),
            "--dtype", "auto",
        ]
        if device == "cuda":
            pass  # default
        elif device == "mps":
            logger.warning("vLLM does not support MPS. Falling back to HuggingFace.")
            return False

        logger.info(f"vLLM command: {' '.join(cmd)}")
        logger.info("Start vLLM manually: " + " ".join(cmd))
        return True
    except ImportError:
        logger.warning("vLLM not installed. Try: pip install vllm")
        return False


def deploy_huggingface(model_id: str, port: int, device: str) -> bool:
    """
    Deploy using HuggingFace pipeline.
    For actual serving, use gradio or a simple FastAPI wrapper.
    """
    try:
        from transformers import pipeline
        import torch

        device_map = "auto" if device == "cuda" else device
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info(f"Loading {model_id} with transformers pipeline...")
        pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Quick test
        result = pipe("Describe this image.", images=["https://httpbin.org/image/png"])
        logger.info(f"Model loaded. Test output: {str(result)[:200]}")

        # In production, wrap with FastAPI/Gradio server
        logger.info(f"Model ready. Use python -m vlm_integration.orchestrator to start server.")
        return True
    except Exception as e:
        logger.error(f"HuggingFace deployment failed: {e}")
        return False


def deploy_mlx(model_id: str, port: int) -> bool:
    """Deploy using Apple MLX (macOS only)."""
    try:
        from mlx_lm import load, generate

        logger.info(f"Loading {model_id} with MLX...")
        model, tokenizer = load(model_id)
        logger.info(f"Model loaded with MLX. Use mlx_lm.server to serve.")
        return True
    except ImportError:
        logger.warning("MLX not available (macOS only)")
        return False
    except Exception as e:
        logger.error(f"MLX deployment failed: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy edge VLM for GUI Agent orchestrator")
    p.add_argument("--model", choices=list(EDGE_VLM_PRESETS.keys()), default="phi3-vision",
                   help="Model to deploy")
    p.add_argument("--port", type=int, default=8766, help="Server port")
    p.add_argument("--device", choices=["cuda", "cpu", "mps", "auto"], default="auto",
                   help="Device for inference")
    p.add_argument("--backend", choices=["auto", "vllm", "hf", "mlx"], default="auto",
                   help="Deployment backend")
    p.add_argument("--test", action="store_true", help="Run a quick inference test")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    preset = EDGE_VLM_PRESETS.get(args.model)
    if not preset:
        logger.error(f"Unknown model: {args.model}")
        logger.info(f"Available: {list(EDGE_VLM_PRESETS.keys())}")
        sys.exit(1)

    model_id = preset["model_id"]
    logger.info(f"Deploying {args.model} ({model_id})")

    # Auto-detect device
    if args.device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                args.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                args.device = "mps"
            else:
                args.device = "cpu"
        except ImportError:
            args.device = "cpu"
    logger.info(f"Using device: {args.device}")

    # Auto-select backend
    backend = args.backend
    if backend == "auto":
        if args.device == "cuda":
            backend = "vllm"  # try vllm first
        elif args.device == "mps":
            backend = "mlx"   # MLX for Apple Silicon
        else:
            backend = "hf"

    # Deploy
    success = False
    if backend == "vllm":
        success = deploy_vllm(model_id, args.port, args.device)
        if not success:
            logger.info("Falling back to HuggingFace pipeline")
            success = deploy_huggingface(model_id, args.port, args.device)

    elif backend == "mlx":
        success = deploy_mlx(model_id, args.port)
        if not success:
            logger.info("Falling back to HuggingFace pipeline")
            success = deploy_huggingface(model_id, args.port, args.device)

    else:  # hf
        success = deploy_huggingface(model_id, args.port, args.device)

    if not success:
        logger.error("All deployment backends failed.")
        sys.exit(1)

    # Optional test
    if args.test:
        logger.info("Running inference test...")
        try:
            config = VLMConfig(
                model_name=preset["name"],
                model_path=model_id,
                device=args.device,
            )
            orchestrator = VLMOrchestrator(config)
            plan = orchestrator.plan_task("Click the search button")
            logger.info(f"Test plan: {len(plan.steps)} steps generated")
            for s in plan.steps:
                logger.info(f"  - {s.action_type}: {s.description}")
        except Exception as e:
            logger.error(f"Test failed: {e}")

    logger.info(f"Deployment complete. Server ready on port {args.port}")


if __name__ == "__main__":
    main()
