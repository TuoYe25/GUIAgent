# Edge Device GUI Agent

> Deploy, benchmark, and experiment with open-source GUI interaction models on local edge devices — integrated with an Electron-based sandboxed demo.

## Architecture Overview

```
edge-gui-agent/
├── models/                    # Model deployment & action parsing
│   ├── deploy/               # UI-TARS, ShowUI, OS-ATLAS deployment scripts
│   ├── action_parsers/       # coordinate / DOM / a11y / hybrid parsers
│   └── configs/              # Per-model YAML configuration (gui_models.yaml)
├── benchmark/                 # Benchmarking framework
│   ├── runner.py             # Unified benchmark runner
│   ├── tasks/                # Standard benchmark task definitions
│   ├── metrics.py            # Action-level metrics
│   ├── system_monitor.py     # CPU/GPU/memory resource monitoring
│   ├── comparison.py         # Statistical analysis & visualization
│   └── competitors/          # Playwright / Puppeteer / Selenium agents
├── vlm_integration/           # VLM orchestration experiments
│   ├── orchestrator.py       # Edge VLM as planner/evaluator
│   ├── hybrid.py             # Local GUI model + remote VLM/LLM
│   ├── preprocessor.py       # Preprocess → send intermediate data
│   ├── comparison.py         # Three-way comparison framework
│   └── privacy.py            # Privacy impact assessment
├── electron_demo/             # Electron sandboxed demo
│   ├── main.js               # BrowserWindow / BrowserView + IPC proxy
│   ├── preload.js            # Secure sandbox preload
│   ├── renderer/             # Control panel UI (HTML/CSS/JS)
│   ├── gui_agent/            # Agent bridge & action execution
│   └── workflows/            # Sample workflow definitions
├── docs/                      # API.md, ARCHITECTURE.md, BENCHMARK.md, DEPLOYMENT.md
├── scripts/                   # setup.py, run_benchmarks.py, deploy_edge_vlm.py, clean.py
├── pyproject.toml             # Python project config (pip install -e ".[dev]")
└── requirements.txt           # Python dependencies
```

## Key Features

| Module | Description |
|--------|-------------|
| **Model Deployment** | Deploy UI-TARS, ShowUI, OS-ATLAS via HuggingFace transformers or vLLM |
| **Action Parsers** | Coordinate-based, DOM-based, accessibility-tree, and hybrid approaches |
| **Benchmark Suite** | Compare model performance (success rate, latency, CPU/GPU/memory) |
| **Competitor Comparison** | Head-to-head vs Playwright, Puppeteer, Selenium |
| **VLM Integration** | Experiment with edge VLM orchestrator, hybrid, and preprocessor strategies |
| **Electron Demo** | Sandboxed BrowserView with unified Model Registry — switch between text LLMs (DeepSeek v4), vision models (Qwen-VL-Max), local GUI models, or regex parser at runtime |

## Quick Start

### Prerequisites

- **Python**: ≥3.10
- **Node.js**: ≥20 (for Electron demo)
- **GPU**: NVIDIA GPU with ≥8GB VRAM (CUDA 12.1+) recommended; CPU-only fallback available
- **OS**: Windows 10/11 or macOS 14+

### Installation

```bash
# Python environment
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # macOS / Linux

pip install -e ".[dev]"

# Electron demo
cd electron_demo
npm install
```

### Run Benchmark

```bash
python scripts/run_benchmarks.py --models ui-tars-1.5-7b --methods coordinate dom a11y --runs 3
```

### Start Electron Demo

```bash
cd electron_demo
npm start
```

## Supported Models

The Electron demo uses a **Model Registry** (`renderer/app.js`) — a single config table where each model entry specifies endpoint, API key, model name, and type (`text` or `vision`). Switching models at runtime requires no code changes.

| Model | Type | API | Status |
|-------|------|-----|:------:|
| DeepSeek v4 | Text | DMXAPI (cloud) | Enabled |
| Qwen-VL-Max | Vision | DashScope (cloud) | Enabled |
| UI-TARS-7B | Vision | vLLM / Ollama (local) | Self-host |
| ShowUI-2B | Vision | vLLM / Ollama (local) | Self-host |
| OS-ATLAS | Vision | vLLM / Ollama (local) | Self-host |
| Fara-7B | Vision | vLLM / Ollama (local) | Self-host |
| AgentCPM-GUI | Vision | vLLM / Ollama (local) | Self-host |

## Action Methods Compared

| Method | How it works | Pros | Cons |
|--------|-------------|------|------|
| **Coordinate** | Model outputs (x, y) directly | Simple, model-agnostic | Resolution-dependent |
| **DOM** | Parses DOM tree for selectors | Precise element targeting | Page-dependent |
| **Accessibility** | Uses ARIA/a11y tree | Semantic, robust | Limited adoption |
| **Hybrid** | Combines coordinate + semantic | Best of both worlds | Higher complexity |

## Research Goals

1. Deploy recent open-source GUI interaction models on local edge devices
2. Benchmark performance vs. resource usage in realistic edge environments
3. Experiment with edge VLMs as orchestrator/planner/evaluator
4. Compare local vs. hybrid (local + remote) strategies
5. Prototype Electron-based sandboxed GUI agent demo

## License

Apache-2.0
