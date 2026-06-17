# Edge GUI Agent — Deployment Guide

## Supported Environments

| Platform | GPU | Memory | Storage | Status |
|----------|-----|--------|---------|--------|
| Windows 10/11 + NVIDIA GPU | RTX 3060+, 8GB+ VRAM | 32GB+ | 50GB | Supported |
| macOS (Apple Silicon) | M1/M2/M3, 16GB+ unified | 16GB+ | 50GB | Experimental |
| Linux (Ubuntu 22.04+) + NVIDIA GPU | RTX 3060+, 8GB+ VRAM | 32GB+ | 50GB | Supported |

## Step 1: Environment Setup

### Windows
```powershell
# CUDA Toolkit (if using NVIDIA GPU)
# Download from: https://developer.nvidia.com/cuda-downloads

# Python 3.10+
python -m venv venv
.\venv\Scripts\activate

# PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
pip install -r requirements.txt
pip install -e .
```

### macOS (Apple Silicon)
```bash
python3 -m venv venv
source venv/bin/activate

# PyTorch with MPS
pip install torch torchvision torchaudio

# MLX for better performance
pip install mlx mlx-lm

# Project dependencies
pip install -r requirements.txt
pip install -e .
```

### Linux
```bash
python3 -m venv venv
source venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

## Step 2: Start Local Model Server

```bash
# Start the FastAPI server (first run downloads model weights from HuggingFace)
python scripts/serve_model.py --model ui-tars-1.5-7b --port 8000

# Verify
curl http://localhost:8000/health
# → {"status":"ok","model":"ui-tars-1.5-7b"}

# List models
curl http://localhost:8000/v1/models
# → {"object":"list","data":[{"id":"ui-tars-1.5-7b",...}]}
```

### Available Models

| Model ID | Repo | Size |
|----------|------|------|
| `ui-tars-1.5-7b` | `bytedance/UI-TARS-1.5-7B` | ~14GB |
| `showui-2b` | `showlab/ShowUI-2B` | ~4GB |
| `os-atlas-7b` | `OS-Copilot/OS-ATLAS-7B` | ~14GB |
| `fara-7b` | `fara-ai/Fara-7B` | ~14GB |

### Cross-Machine Setup (e.g., Mac Mini inference, Windows UI)

On the model server machine:
```bash
python scripts/serve_model.py --model ui-tars-1.5-7b --host 0.0.0.0 --port 8000
```

On the UI machine, edit `electron_demo/renderer/app.js`:
```js
const LOCAL_MODEL_BASE = 'http://<tailscale-ip>:8000';  // e.g. http://100.85.1.23:8000
```

## Step 3: Launch Electron Demo

```bash
cd electron_demo

# Install Node.js dependencies
npm install

# Start the demo
npm start

# Or in dev mode with hot reload
npm run dev
```

The Electron app will:
1. Open a control panel on the left
2. Navigate to the target URL in a sandboxed BrowserView
3. Accept prompts and execute GUI actions

## Step 4: Run Benchmarks

```bash
# Basic benchmark
python -m scripts.run_benchmarks --strategy fully_local

# Comparison across strategies
python -m scripts.run_benchmarks --compare

# With plots
python -m scripts.run_benchmarks --compare --plot
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GUI_AGENT_MODEL_PATH` | auto | Path to GUI agent model weights |
| `GUI_AGENT_SERVER_PORT` | 8000 | FastAPI model server port |
| `LOCAL_MODEL_BASE` | localhost:8000 | Model server URL (change for cross-machine) |
| `OPENAI_API_KEY` | — | For hybrid mode with OpenAI |
| `ANTHROPIC_API_KEY` | — | For hybrid mode with Anthropic |
| `GROQ_API_KEY` | — | For hybrid mode with Groq |
| `LOG_LEVEL` | INFO | Logging level |

### Model Configurations

Edit `models/configs/` YAML files:
```yaml
# models/configs/ui-tars-1.5.yaml
model:
  name: "UI-TARS-1.5-7B"
  repo_id: "bytedance/UI-TARS-1.5-7B"
  framework: transformers
  approach: coordinate_based
  gpu_memory_gb: 8
  parameters: 7B
```

## Troubleshooting

### CUDA Out of Memory
- Use CPU-only fallback: `python scripts/serve_model.py --model ui-tars-1.5-7b --port 8000` (auto-detects device)
- Try a smaller model (ShowUI-2B or UI-TARS-2B)
- Reduce `max_new_tokens` in the deploy config

### Model Server Connection Refused
- Ensure the server is running: `curl http://localhost:8000/health`
- Check port conflicts: `netstat -ano | findstr 8000` (Windows) or `lsof -i :8000` (macOS)
- For cross-machine: verify Tailscale is connected and `LOCAL_MODEL_BASE` IP is correct

### First Run Slow
- First run downloads model weights from HuggingFace (~4-14GB), subsequent runs use cached weights
- Set `HF_HOME` environment variable to control cache location

### Electron App Blank Screen
- Check console for CSP errors
- Verify BrowserView sandbox permissions
- Try with `--disable-web-security` flag (dev only)
