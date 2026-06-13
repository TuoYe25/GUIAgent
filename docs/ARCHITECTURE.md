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
│         │   - Agent bridge (WS to Python BE)     │                │
│         └──────────────────┬─────────────────────┘                │
└─────────────────────────────┼──────────────────────────────────────┘
                              │ WebSocket (ws://localhost:8765)
┌─────────────────────────────┼──────────────────────────────────────┐
│                   Python Backend                                    │
│         ┌───────────────────▼───────────────┐                      │
│         │         models/                    │                      │
│         │  ┌───────────────────────────┐    │                      │
│         │  │  ui_tars_deploy.py         │    │                      │
│         │  │  (UI-TARS vLLM deploy,      │    │                      │
│         │  │   HF pipeline, gradio)      │    │                      │
│         │  ├───────────────────────────┤    │                      │
│         │  │  model_registry.py         │    │                      │
│         │  │  (ModelConfig, registry)   │    │                      │
│         │  └───────────────────────────┘    │                      │
│         └─────────────────┬─────────────────┘                      │
│                           │                                        │
│         ┌─────────────────▼─────────────────┐                      │
│         │       vlm_integration/             │                      │
│         │  ┌─────────────┐ ┌──────────────┐ │                      │
│         │  │ orchestrator │ │   hybrid     │ │                      │
│         │  │(edge VLM     │ │(local+remote)│ │                      │
│         │  │ planner/eval)│ │              │ │                      │
│         │  ├─────────────┤ ├──────────────┤ │                      │
│         │  │ preprocessor │ │  comparison   │ │                      │
│         │  │(PII filter,  │ │(strategy bench│ │                      │
│         │  │ DOM→structured│ │ & visualization│ │                      │
│         │  ├──────────────┤ ├──────────────┤ │                      │
│         │  │  privacy      │ │              │ │                      │
│         │  │ (GDPR/HIPAA   │ │              │ │                      │
│         │  │  compliance)  │ │              │ │                      │
│         │  └──────────────┘ └──────────────┘ │                      │
│         └─────────────────┬─────────────────┘                      │
│                           │                                        │
│         ┌─────────────────▼─────────────────┐                      │
│         │        benchmark/                  │                      │
│         │  ┌────────────┐ ┌───────────────┐ │                      │
│         │  │  runner    │ │  competitors   │ │                      │
│         │  │(Benchmark  │ │ puppeteer.py   │ │                      │
│         │  │ Runner,    │ │ selenium.py    │ │                      │
│         │  │ SystemMon) │ │ playwright.py  │ │                      │
│         │  └────────────┘ └───────────────┘ │                      │
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
        → WebSocket → Python Agent Backend
            → VLM Planner (local or remote)
            → GUI Model Predict (local)
            → Action Executor (BrowserView sandbox)
        ← WebSocket ← Result
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
2. **WebSocket protocol**: Enables local-only communication, zero network exposure
3. **Model registry pattern**: Easy to add new models (ShowUI, OS-ATLAS, Fara-7B, etc.)
4. **Privacy-first preprocessing**: Redact PII before any remote API call
5. **Strategy abstraction**: Same benchmark can run local/hybrid/preprocess comparisons
