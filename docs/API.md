# Edge GUI Agent — API Reference

## Python Backend API

### WebSocket Endpoint: `ws://localhost:8765`

Standard JSON protocol with request/response pairs.

### Message Types

#### 1. Predict Action
```json
{
  "type": "predict",
  "requestId": 1,
  "payload": {
    "image": "<base64_encoded_screenshot>",
    "instruction": "Click the search button"
  }
}
```

**Response:**
```json
{
  "requestId": 1,
  "success": true,
  "data": {
    "action": "click",
    "x": 150,
    "y": 320,
    "confidence": 0.92,
    "explanation": "Clicking search button at coordinates (150, 320)"
  }
}
```

#### 2. Execute Command
```json
{
  "type": "execute",
  "requestId": 2,
  "payload": {
    "action": "click",
    "params": { "x": 150, "y": 320 }
  }
}
```

#### 3. Status Check
```json
{
  "type": "status",
  "requestId": 3,
  "payload": {}
}
```

**Response:**
```json
{
  "requestId": 3,
  "success": true,
  "data": {
    "model_loaded": true,
    "model_name": "bytedance/UI-TARS-1.5-7B",
    "gpu_memory_used_mb": 6500,
    "uptime_seconds": 3600
  }
}
```

## Electron IPC API

### Renderer → Main Process (via `window.electronAPI`)

#### Navigate
```javascript
const result = await window.electronAPI.navigate("https://example.com");
// { success: true, url: "https://example.com" }
```

#### Execute Action
```javascript
const result = await window.electronAPI.execute({
  type: "click",
  x: 150,
  y: 320
});
// { success: true, result: { x: 150, y: 320 } }
```

#### Screenshot
```javascript
const result = await window.electronAPI.screenshot();
// { success: true, data: "base64..." }
```

#### Reload Sandbox
```javascript
await window.electronAPI.reloadSandbox();
```

### Main Process → Sandbox (via `executeJavaScript`)

Actions are injected into the BrowserView sandbox context:

```javascript
// Action types and their JS implementations in action_executor.js:
// - click(x, y)
// - type(text, selector?)
// - scroll(direction, amount)
// - navigate(url)
// - wait(ms)
// - select(selector, value)
// - hover(x, y)
// - press(key)
// - getState()
```

## VLM Orchestrator API

### Plan Task
```python
from vlm_integration.orchestrator import VLMOrchestrator

orchestrator = VLMOrchestrator(config)
plan = orchestrator.plan_task("Search for flights to Paris")
# plan.steps: List[Step] with action_type, description, params
```

### Evaluate Step
```python
screenshot_base64 = capture_screenshot()
evaluation = orchestrator.evaluate_step(step, screenshot_base64, task_goal)
# { status: "success"|"failed", reason: "...", next_action: "..." }
```

### Execute Plan
```python
completed_plan = orchestrator.execute_plan(
    plan=plan,
    execute_fn=lambda step: execute_step(step),
    screenshot_fn=lambda: capture_screenshot(),
)
```

## Hybrid Executor API

```python
from vlm_integration.hybrid import HybridExecutor, HybridConfig, HybridMode

config = HybridConfig(
    mode=HybridMode.LOCAL_GUI,
    remote_provider="openai",
    remote_model="gpt-4o-mini"
)

executor = HybridExecutor(config)
executor.set_local_gui_fn(my_local_predict_function)

metrics = executor.execute(
    task="Search for AI papers on arxiv",
    screenshot_fn=lambda: capture_screenshot()
)
# metrics: List[HybridMetrics] with latency, resource, cost per step
```

## Benchmark Runner API

```python
from benchmark.runner import BenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(
    tasks_file="benchmark/tasks/sample_tasks.json",
    strategy="fully_local",
    headless=True,
    repeat=5,
)

runner = BenchmarkRunner(config)
results = runner.run()
# results: List[BenchmarkResult] with all metrics

report = runner.generate_report(results)
# JSON-serializable report
```

## Preprocessor API

```python
from vlm_integration.preprocessor import Preprocessor, PreprocessMethod

preprocessor = Preprocessor(method=PreprocessMethod.HYBRID)

result = preprocessor.process(
    dom_json=page_dom,
    page_text=page_text,
)

# result.compression_ratio: 0.15 (85% data reduction)
# result.data: structured, redacted JSON
# result.redactions: 3 (PII instances removed)
```

## Model Registry API

```python
from models.model_registry import ModelRegistry

# List all registered models
models = ModelRegistry.list()

# Get specific model config
config = ModelRegistry.get("ui-tars-1.5-7b")

# Register custom model
ModelRegistry.register(MyCustomConfig(...))

# Get compatible models for a task
compatible = ModelRegistry.get_compatible(task="gui-action-prediction")
```
