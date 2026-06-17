# Architecture

## Edge GUI Agent — High-Level Design

```
┌──────────────────────────────────────────────────────────────────┐
│                     Electron Sandbox Demo                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Control Panel │  │  BrowserView  │  │   Action Executor    │   │
│  │  (renderer)   │  │  (sandboxed)  │  │ (inject/execute API) │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         │                 │                      │               │
│         └────────┬────────┴──────────┬───────────┘               │
│                  │       IPC         │                           │
│         ┌────────▼───────────────────▼──────────┐                │
│         │           main.js (Electron)           │                │
│         │   - Window/BrowserView management       │                │
│         │   - IPC handlers                       │                │
│         │   - HTTP API proxy to model server     │                │
│         └──────────────────┬─────────────────────┘                │
└─────────────────────────────┼──────────────────────────────────────┘
                              │ HTTP (localhost:8000 or Tailscale IP)
┌─────────────────────────────┼──────────────────────────────────────┐
│                   Model Server (FastAPI)                            │
│         ┌───────────────────▼───────────────┐                      │
│         │  scripts/serve_model.py            │                      │
│         │  - OpenAI-compatible /v1 API       │                      │
│         │  - Loads models via transformers   │                      │
│         │  - UI-TARS / ShowUI / OS-ATLAS     │                      │
│         └───────────────────────────────────┘                      │
│                                                                     │
│         ┌───────────────────────────────────┐                      │
│         │         models/                    │                      │
│         │  ┌───────────────────────────┐    │                      │
│         │  │  ui_tars_deploy.py         │    │                      │
│         │  │  showui_deploy.py          │    │                      │
│         │  │  os_atlas_deploy.py        │    │                      │
│         │  ├───────────────────────────┤    │                      │
│         │  │  model_registry.py         │    │                      │
│         │  │  (ModelConfig, registry)   │    │                      │
│         │  └───────────────────────────┘    │                      │
│         └───────────────────────────────────┘                      │
└────────────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### 1. models/
GUI interaction model deployment and registry. Currently focused on UI-TARS.
- **Registry**: Central model config management
- **Deployer**: vLLM, HuggingFace pipeline, Gradio launch
- **Configs**: JSON/YAML model configurations

### 2. vlm_integration/
Edge VLM as orchestrator/planner/evaluator.
- **Orchestrator**: Plan-execute-evaluate loop with local VLM
- **Hybrid**: Local GUI + remote VLM/LLM strategies
- **Preprocessor**: DOM filtering, A11y tree, anonymization before remote send
- **Comparison**: Side-by-side strategy benchmarking
- **Privacy**: PII detection, redaction, compliance assessment

### 3. benchmark/
Standardized GUI agent benchmarking framework.
- **Runner**: Execution engine with metrics collection
- **SystemMonitor**: Real-time CPU/GPU/memory tracking
- **Competitors**: Playwright, Puppeteer, Selenium traditional approaches

### 4. electron_demo/
Rapid prototype sandbox demo.
- **main.js**: Electron main process with BrowserWindow + BrowserView
- **preload.js**: Secure IPC bridge
- **renderer/**: Control panel UI (HTML/CSS/JS)
- **gui_agent/**: Action executor + Agent bridge
- **workflows/**: Sample automation workflows

## Data Flow

```
User Prompt → Control Panel (renderer)
    → IPC → main.js
        → HTTP POST → FastAPI Model Server (localhost:8000 or Tailscale remote)
            → GUI Model Predict (local transformers)
        ← HTTP Response ← Server
    ← IPC ← Result
← Render (log, status)
```

## Approach Taxonomy

| Approach | Element Location | Planning | Privacy | Latency |
|----------|-----------------|----------|---------|---------|
| **Coordinate-based** (UI-TARS) | Absolute (x,y) | VLM | None | Medium |
| **DOM-based** (Playwright) | CSS/XPath selectors | Scripted | None | Low |
| **A11y-tree** | Accessibility roles | Scripted | None | Low |
| **Hybrid (local GUI + remote LLM)** | From local model | Remote API | Medium | High |
| **Hybrid (preprocess + remote)** | Structured/filtered | Remote API | High | High |
| **Fully Local** | All on-device | Local VLM | High | Medium-High |

## Key Design Decisions

1. **Electron BrowserView**: Isolates sandbox from control panel — same stack as real product
2. **FastAPI model server**: OpenAI-compatible `/v1/chat/completions` — any client can call it; cross-machine via Tailscale
3. **Model registry pattern**: Easy to add new models (ShowUI, OS-ATLAS, Fara-7B, etc.)
4. **Privacy-first preprocessing**: Redact PII before any remote API call
5. **Strategy abstraction**: Same benchmark can run local/hybrid/preprocess comparisons
6. **macOS MPS support**: Automatic device fallback (cuda → mps → cpu) for Apple Silicon
