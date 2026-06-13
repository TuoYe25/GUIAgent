# Edge GUI Agent — Benchmark Guide

## Overview

The benchmark framework evaluates GUI agent performance across multiple dimensions:

1. **Task Success Rate** — Did the agent complete the task?
2. **Latency** — How fast? (step-level and end-to-end)
3. **Resource Usage** — GPU memory, CPU, RAM
4. **Privacy Exposure** — How much data leaves the device?
5. **Cost** — API cost for hybrid strategies

## Quick Start

### Install dependencies
```bash
pip install -r benchmark/requirements.txt
playwright install chromium
```

### Run a single benchmark
```bash
cd benchmark
python runner.py --url https://example.com --task "Click the first link"
```

### Run comparison (multiple strategies)
```bash
python runner.py --compare --tasks tasks/sample_tasks.json
```

## Task Format

Tasks are defined in JSON:

```json
{
  "tasks": [
    {
      "id": "search_github",
      "name": "Search GitHub",
      "url": "https://github.com/search?q=ui-tars",
      "prompt": "Search for UI-TARS and click the first result",
      "expected_action": "click",
      "expected_target": "first search result link"
    }
  ]
}
```

## Competitors

Three traditional approaches are included for comparison:

### Playwright (PlaywrightAgent)
```bash
python competitors/playwright_agent.py
```

### Puppeteer (PuppeteerAgent)
```bash
python competitors/puppeteer_agent.py
```

### Selenium (SeleniumAgent)
```bash
python competitors/selenium_agent.py
```

## Metrics Reference

| Metric | Description | Unit |
|--------|-------------|------|
| task_success | Boolean success/failure | boolean |
| total_latency_ms | End-to-end task latency | ms |
| step_latency_ms | Per-step (click, type, etc.) latency | ms |
| gpu_memory_mb | GPU VRAM usage | MB |
| cpu_percent | CPU utilization | % |
| ram_mb | System RAM usage | MB |
| prediction_latency_ms | ML model inference time | ms |
| element_count | Interactive elements on page | count |
| data_sent_mb | Data sent to remote APIs | MB |
| api_cost_usd | Estimated API cost | USD |
| privacy_score | 0-100 (higher = better) | score |

## Strategy Comparison

```
Strategy           Success Rate   Avg Latency   GPU Mem   Privacy   Cost
─────────────────────────────────────────────────────────────────────────
Playwright         95%+           ~50ms         ~200MB    100       $0
Puppeteer          95%+           ~60ms         ~250MB    100       $0
Selenium           90%+           ~150ms        ~300MB    100       $0
UI-TARS (local)    75-85%         ~500ms        ~8GB      100       $0
Local + Remote LLM 80-90%         ~2000ms       ~8GB      50        ~$0.01/task
Preprocess+Remote  85-90%         ~1500ms       ~8GB      85        ~$0.005/task
```

> Numbers are projected based on model papers and architecture analysis.
> Actual benchmarks should be run on target hardware.

## Running Full Suite

```bash
# Full benchmark across all strategies
python scripts/run_benchmarks.py --all

# Specific strategy
python scripts/run_benchmarks.py --strategy fully_local

# With visualization
python scripts/run_benchmarks.py --all --plot

# Export results
python scripts/run_benchmarks.py --all --export json csv
```

## Adding a New Model

1. Register in `models/model_registry.py`:
```python
ModelRegistry.register(
    ModelConfig(
        name="my-model",
        repo_id="org/my-model",
        framework=ModelFramework.TRANSFORMERS,
        task="gui-action-prediction",
        approach=Approach.COORDINATE_BASED,
        gpu_memory_gb=8,
    )
)
```

2. Add deployment code in `models/` if needed
3. Add strategy mapping in `vlm_integration/orchestrator.py`
