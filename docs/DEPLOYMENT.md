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

## Step 2: Deploy GUI Interaction Model

### UI-TARS (ByteDance)
```bash
# Clone the official repo
git clone https://github.com/bytedance/ui-tars.git
cd ui-tars
pip install ui-tars

# Deploy with vLLM (recommended for speed)
python -m models.ui_tars_deploy --backend vllm --port 8765

# Or deploy with HuggingFace pipeline
python -m models.ui_tars_deploy --backend hf --port 8765

# Verify deployment
curl http://localhost:8765/health
```

### Alternative Models
See `models/configs/` for configuration files for ShowUI, OS-ATLAS, Fara-7B, etc.

## Step 3: Edge VLM Orchestrator (Optional)

```bash
# Deploy Phi-3-Vision as edge planner
python -m scripts.deploy_edge_vlm --model phi3-vision --port 8766

# Or Qwen2-VL
python -m scripts.deploy_edge_vlm --model qwen2-vl-7b --port 8766
```

## Step 4: Launch Electron Demo

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

## Step 5: Run Benchmarks

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
| `GUI_AGENT_BACKEND` | vllm | Model backend: vllm, hf, gradio |
| `GUI_AGENT_PORT` | 8765 | WebSocket server port |
| `EDGE_VLM_MODEL` | phi3-vision | Edge VLM for orchestrator |
| `EDGE_VLM_PORT` | 8766 | Edge VLM server port |
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
- Reduce batch size in model config
- Use `dtype: float16` instead of bfloat16
- Disable vLLM and use HuggingFace pipeline
- Try a smaller model (e.g., UI-TARS-1.5-2B)

### WebSocket Connection Refused
- Ensure Python backend is running: `curl http://localhost:8765/health`
- Check port conflicts: `netstat -ano | findstr 8765`
- Firewall may block localhost WebSocket — add exception

### Electron App Blank Screen
- Check console for CSP errors
- Verify BrowserView sandbox permissions
- Try with `--disable-web-security` flag (dev only)
