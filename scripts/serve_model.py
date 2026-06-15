"""
Local model server — exposes HuggingFace GUI models as OpenAI-compatible API.

Usage:
    python scripts/serve_model.py --model ui-tars-1.5-7b --port 8000

Supports: ui-tars-1.5-7b, showui-2b, os-atlas-7b, fara-7b, agentcpm-gui
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic models (OpenAI-compatible subset)
# ---------------------------------------------------------------------------

class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class Message(BaseModel):
    role: str
    content: Any  # str or List[ContentPart]


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0


class ChatResponse(BaseModel):
    id: str = "local-0"
    object: str = "chat.completion"
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int] = Field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
    })


class ModelInfo(BaseModel):
    id: str
    object: str = "model"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="GUI Agent Local Server")
deployer_instance: Any = None
model_name: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_image_and_text(messages: List[Message]) -> tuple[Image.Image, str]:
    """Extract PIL Image and instruction text from an OpenAI-format message list."""
    last_user = None
    for m in reversed(messages):
        if m.role == "user":
            last_user = m
            break

    if last_user is None:
        raise HTTPException(400, "No user message found")

    content = last_user.content
    image = None
    text_parts = []

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if hasattr(part, "type"):
                part_type = part.type
                if part_type == "text" and part.text:
                    text_parts.append(part.text)
                elif part_type == "image_url" and part.image_url:
                    url = part.image_url.get("url", "")
                    if url.startswith("data:"):
                        # data:image/png;base64,...
                        b64_data = url.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_data)
                        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        b64_data = url.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_data)
                        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    instruction = " ".join(t.strip() for t in text_parts if t.strip())
    if not instruction:
        instruction = "Analyze the screenshot and determine the next action."

    if image is None:
        # Create a dummy black image to avoid crashes
        image = Image.new("RGB", (1920, 1080), (0, 0, 0))

    return image, instruction


def create_deployer(model_id: str):
    """Factory: instantiate the right deployer for a model id."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    if model_id.startswith("ui-tars"):
        from models.deploy.ui_tars_deploy import UITARSConfig, UITARSDeployer
        repo_map = {
            "ui-tars-1.5-7b": "bytedance/UI-TARS-1.5-7B",
            "ui-tars-1.5-2b": "bytedance/UI-TARS-1.5-2B",
        }
        repo_id = repo_map.get(model_id, model_id)
        cfg = UITARSConfig(model_name_or_path=repo_id)
        return UITARSDeployer(cfg), cfg

    elif model_id.startswith("showui"):
        from models.deploy.showui_deploy import ShowUIConfig, ShowUIDeployer
        repo_map = {
            "showui-2b": "showlab/ShowUI-2B",
        }
        repo_id = repo_map.get(model_id, model_id)
        cfg = ShowUIConfig(model_name_or_path=repo_id)
        return ShowUIDeployer(cfg), cfg

    elif model_id.startswith("os-atlas"):
        from models.deploy.os_atlas_deploy import OSAtlasConfig, OSAtlasDeployer
        repo_map = {
            "os-atlas-7b": "OS-Copilot/OS-ATLAS-7B",
            "os-atlas-4b": "OS-Copilot/OS-ATLAS-4B",
        }
        repo_id = repo_map.get(model_id, model_id)
        cfg = OSAtlasConfig(model_name_or_path=repo_id)
        return OSAtlasDeployer(cfg), cfg

    elif model_id.startswith("fara"):
        # Fara-7B uses a standard transformers setup; fall back to UI-TARS style
        from models.deploy.ui_tars_deploy import UITARSConfig, UITARSDeployer
        repo_id = "fara-ai/Fara-7B"
        cfg = UITARSConfig(model_name_or_path=repo_id)
        return UITARSDeployer(cfg), cfg

    elif model_id.startswith("agentcpm"):
        raise HTTPException(400, "AgentCPM-GUI model is not yet available. "
                                "Model checkpoint not released publicly.")

    else:
        raise HTTPException(400, f"Unknown model: {model_id}. "
                                 f"Supported: ui-tars-1.5-7b, showui-2b, "
                                 f"os-atlas-7b, fara-7b")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    return ModelList(data=[ModelInfo(id=model_name)])


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    global deployer_instance

    if deployer_instance is None:
        raise HTTPException(503, "Model not loaded yet. Server is starting up.")

    image, instruction = extract_image_and_text(req.messages)

    try:
        result = deployer_instance.predict(image, instruction)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(500, f"Inference error: {str(e)}")

    # Convert result dict to natural language response
    if hasattr(result, "get"):
        action = result.get("action_type", result.get("action", "unknown"))
        x = result.get("x", result.get("click_x", 0))
        y = result.get("y", result.get("click_y", 0))
        text = result.get("text", "")
        desc = result.get("description", f"{action}({x},{y})")
    else:
        desc = str(result)

    response_text = desc

    return ChatResponse(
        model=req.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok" if deployer_instance else "loading",
        "model": model_name,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global deployer_instance, model_name

    parser = argparse.ArgumentParser(description="GUI Agent Local Model Server")
    parser.add_argument("--model", required=True,
                        help="Model id: ui-tars-1.5-7b, showui-2b, os-atlas-7b, fara-7b")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    model_name = args.model
    deployer, config = create_deployer(model_name)

    logger.info(f"Loading {model_name} from HuggingFace (first run will download)...")
    deployer.load()
    deployer_instance = deployer

    logger.info(f"Server ready at http://{args.host}:{args.port}")
    logger.info(f"Test: curl http://{args.host}:{args.port}/health")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
