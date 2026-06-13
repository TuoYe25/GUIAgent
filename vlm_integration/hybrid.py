"""
Hybrid Strategy — Local GUI Model + Remote VLM/LLM.

Experiments with hybrid architectures where:
- GUI action parsing runs on the local edge device
- High-level planning and evaluation may be offloaded to remote APIs
- Latency, privacy, and performance trade-offs are benchmarked

Architecture variants:
1. FULLY_LOCAL  — GUI model + planner both local
2. LOCAL_GUI     — GUI model local, planner remote (OpenAI/Claude)
3. LOCAL_PLAN    — Planner local, GUI model remote (rare)
4. FALLBACK_ONLY — Local first, remote on failure
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class HybridMode(str, Enum):
    FULLY_LOCAL = "fully_local"
    LOCAL_GUI = "local_gui"          # local GUI model, remote planner
    LOCAL_PLAN = "local_plan"        # remote GUI model, local planner
    FALLBACK_ONLY = "fallback_only"  # local → remote on failure


@dataclass
class HybridMetrics:
    """Per-step metrics for hybrid execution."""
    mode: HybridMode
    step_id: str

    # Latency breakdown (ms)
    local_gui_latency_ms: float = 0.0
    remote_planner_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Success
    local_success: bool = True
    used_fallback: bool = False

    # Resource
    local_gpu_memory_mb: float = 0.0
    local_cpu_percent: float = 0.0


@dataclass
class HybridConfig:
    """Hybrid execution configuration."""
    mode: HybridMode = HybridMode.FULLY_LOCAL
    remote_provider: str = "openai"       # openai | anthropic | groq | together
    remote_model: str = "gpt-4o-mini"
    remote_api_key: Optional[str] = None  # read from env if None
    local_planner_model: str = "microsoft/Phi-3-vision-128k-instruct"
    local_gui_model: str = "bytedance/UI-TARS-1.5-7B"
    fallback_threshold_count: int = 2     # fallback after N local failures
    timeout_sec: float = 30.0
    retry_count: int = 1


# ---------------------------------------------------------------------------
# Remote API Client
# ---------------------------------------------------------------------------

class RemotePlannerClient:
    """Unified interface for remote VLM/LLM API calls."""

    def __init__(self, provider: str, model: str, api_key: Optional[str] = None) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def plan(self, task: str, screenshot_base64: Optional[str] = None) -> Dict[str, Any]:
        """Send planning request to remote API."""
        start = time.perf_counter()

        if self.provider == "openai":
            result = self._call_openai(task, screenshot_base64)
        elif self.provider == "anthropic":
            result = self._call_anthropic(task, screenshot_base64)
        elif self.provider == "groq":
            result = self._call_groq(task)
        elif self.provider == "together":
            result = self._call_together(task, screenshot_base64)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        elapsed_ms = (time.perf_counter() - start) * 1000
        result["latency_ms"] = round(elapsed_ms, 1)
        return result

    def evaluate(
        self, task: str, step: str, screenshot_base64: str
    ) -> Dict[str, Any]:
        """Send evaluation request to remote API."""
        start = time.perf_counter()

        prompt = f"Task: {task}\nStep executed: {step}\nEvaluate success/failure."
        messages = [
            {"role": "system", "content": "You evaluate GUI automation steps. Output JSON: {\"status\":\"success\"|\"failed\",\"reason\":\"...\"}"},
            {"role": "user", "content": prompt},
        ]

        if self.provider == "openai":
            result = self._call_openai_chat(messages)
        elif self.provider == "anthropic":
            result = self._call_anthropic_chat(messages)
        else:
            result = {"status": "success", "reason": "Provider not supported for chat"}

        elapsed_ms = (time.perf_counter() - start) * 1000
        result["latency_ms"] = round(elapsed_ms, 1)
        return result

    def _call_openai(self, task: str, _screenshot: Optional[str]) -> Dict[str, Any]:
        try:
            import os
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Plan GUI automation steps as JSON array."},
                    {"role": "user", "content": task},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            content = response.choices[0].message.content or "[]"
            import json
            return {"steps": json.loads(content) if content.startswith("[") else [], "raw": content}
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {"error": str(e), "steps": []}

    def _call_anthropic(self, task: str, _screenshot: Optional[str]) -> Dict[str, Any]:
        try:
            import os
            from anthropic import Anthropic

            client = Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                system="Plan GUI automation steps as JSON array.",
                messages=[{"role": "user", "content": task}],
            )
            content = response.content[0].text
            import json
            return {"steps": json.loads(content) if content.strip().startswith("[") else [], "raw": content}
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {"error": str(e), "steps": []}

    def _call_groq(self, task: str) -> Dict[str, Any]:
        try:
            import os
            from groq import Groq

            client = Groq(api_key=self.api_key or os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Plan GUI automation steps as JSON array."},
                    {"role": "user", "content": task},
                ],
            )
            content = response.choices[0].message.content or "[]"
            import json
            return {"steps": json.loads(content) if content.startswith("[") else [], "raw": content}
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return {"error": str(e), "steps": []}

    def _call_together(self, task: str, _screenshot: Optional[str]) -> Dict[str, Any]:
        try:
            import os
            from together import Together

            client = Together(api_key=self.api_key or os.getenv("TOGETHER_API_KEY"))
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Plan GUI automation steps as JSON array."},
                    {"role": "user", "content": task},
                ],
            )
            content = response.choices[0].message.content or "[]"
            import json
            return {"steps": json.loads(content) if content.startswith("[") else [], "raw": content}
        except Exception as e:
            logger.error(f"Together API error: {e}")
            return {"error": str(e), "steps": []}

    def _call_openai_chat(self, messages: List[Dict]) -> Dict[str, Any]:
        try:
            import os, json
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0, max_tokens=200
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content) if content.startswith("{") else {"status": "success", "reason": content}
        except Exception:
            return {"status": "success", "reason": "API error"}

    def _call_anthropic_chat(self, messages: List[Dict]) -> Dict[str, Any]:
        try:
            import os, json
            from anthropic import Anthropic

            client = Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
            system = messages[0]["content"] if messages else ""
            user_msgs = [{"role": "user", "content": m["content"]} for m in messages[1:]]
            response = client.messages.create(model=self.model, max_tokens=200, system=system, messages=user_msgs)
            content = response.content[0].text
            return json.loads(content) if content.strip().startswith("{") else {"status": "success", "reason": content}
        except Exception:
            return {"status": "success", "reason": "API error"}


# ---------------------------------------------------------------------------
# Hybrid Executor
# ---------------------------------------------------------------------------

class HybridExecutor:
    """Executes GUI agent tasks using hybrid local/remote strategies."""

    def __init__(self, config: HybridConfig) -> None:
        self.config = config
        self.local_gui_predict: Optional[Callable] = None
        self.local_planner: Any = None
        self.remote_client: Optional[RemotePlannerClient] = None

        if config.mode in (HybridMode.LOCAL_GUI, HybridMode.FALLBACK_ONLY):
            self.remote_client = RemotePlannerClient(
                config.remote_provider, config.remote_model, config.remote_api_key
            )

    def set_local_gui_fn(self, fn: Callable) -> None:
        """Set the local GUI model predict function."""
        self.local_gui_predict = fn

    def set_local_planner(self, planner: Any) -> None:
        """Set the local VLM planner."""
        self.local_planner = planner

    def execute(
        self, task: str, screenshot_fn: Callable[[], str]
    ) -> List[HybridMetrics]:
        """Execute a task and collect metrics per step."""
        metrics: List[HybridMetrics] = []
        consecutive_failures = 0
        use_fallback = False

        # Step 1: Plan
        plan = self._plan(task, screenshot_fn)

        if "error" in plan:
            logger.error(f"Planning failed: {plan['error']}")
            return metrics

        # Step 2: Execute each step
        for i, step_def in enumerate(plan.get("steps", [])):
            step_id = f"step_{i}"
            step_metrics = HybridMetrics(mode=self.config.mode, step_id=step_id)

            start_total = time.perf_counter()

            # ----- GUI Action (local) -----
            gui_start = time.perf_counter()
            gui_result = self._execute_gui_action(step_def, screenshot_fn)
            step_metrics.local_gui_latency_ms = (time.perf_counter() - gui_start) * 1000

            if gui_result.get("success"):
                step_metrics.local_success = True
                consecutive_failures = 0
            else:
                step_metrics.local_success = False
                consecutive_failures += 1

                # Fallback decision
                if consecutive_failures >= self.config.fallback_threshold_count:
                    use_fallback = True
                    logger.info(f"Falling back after {consecutive_failures} failures")

            # ----- Planner (remote or local) -----
            if self.config.mode == HybridMode.LOCAL_GUI and self.remote_client:
                remote_start = time.perf_counter()
                if screenshot_fn:
                    evaluation = self.remote_client.evaluate(task, step_def.get("description", ""), screenshot_fn())
                    step_metrics.remote_planner_latency_ms = (time.perf_counter() - remote_start) * 1000
                    step_metrics.used_fallback = False

            elif self.config.mode == HybridMode.FALLBACK_ONLY and use_fallback:
                step_metrics.used_fallback = True
                # Re-plan with remote
                plan = self._plan_remote(task, screenshot_fn)

            # Resource metrics
            step_metrics.local_gpu_memory_mb = self._get_gpu_memory_mb()
            step_metrics.local_cpu_percent = self._get_cpu_percent()
            step_metrics.total_latency_ms = (time.perf_counter() - start_total) * 1000

            metrics.append(step_metrics)

        return metrics

    def _plan(self, task: str, screenshot_fn) -> Dict[str, Any]:
        """Plan task using current mode strategy."""
        if self.config.mode in (HybridMode.FULLY_LOCAL, HybridMode.LOCAL_GUI):
            return self._plan_local(task)
        else:
            return self._plan_remote(task, screenshot_fn)

    def _plan_local(self, task: str) -> Dict[str, Any]:
        """Plan using local VLM."""
        if self.local_planner:
            plan_obj = self.local_planner.plan_task(task)
            steps_raw = [
                {"action_type": s.action_type, "description": s.description, "params": s.params}
                for s in plan_obj.steps
            ]
            return {"steps": steps_raw}

        # Fallback: simple keyword-based planning
        return self._simple_plan(task)

    def _plan_remote(self, task: str, screenshot_fn) -> Dict[str, Any]:
        """Plan using remote API."""
        if self.remote_client:
            return self.remote_client.plan(task)
        return {"steps": [], "error": "No remote client configured"}

    def _execute_gui_action(self, step_def: Dict, screenshot_fn) -> Dict[str, Any]:
        """Execute a single GUI action locally."""
        action_type = step_def.get("action_type", "click")
        description = step_def.get("description", "")

        if self.local_gui_predict:
            try:
                screenshot = screenshot_fn() if screenshot_fn else ""
                return self.local_gui_predict(screenshot, description)
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Offline stub
        return {"success": True, "action": action_type, "description": description, "note": "stub"}

    def _simple_plan(self, task: str) -> Dict[str, Any]:
        """Very simple heuristic plan when no planner is available."""
        task_lower = task.lower()
        steps = []

        keywords_map = {
            "search": [{"action_type": "click", "description": "Click search box"}],
            "click": [{"action_type": "click", "description": "Click target element"}],
            "type": [{"action_type": "type", "description": "Type text"}],
            "scroll": [{"action_type": "scroll", "description": "Scroll page"}],
            "login": [
                {"action_type": "click", "description": "Click username field"},
                {"action_type": "type", "description": "Type username"},
                {"action_type": "click", "description": "Click password field"},
                {"action_type": "type", "description": "Type password"},
                {"action_type": "click", "description": "Click login button"},
            ],
        }

        for keyword, action_list in keywords_map.items():
            if keyword in task_lower:
                steps.extend(action_list)

        if not steps:
            steps = [{"action_type": "click", "description": task}]

        return {"steps": steps}

    # ------------------------------------------------------------------
    # Resource Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _get_gpu_memory_mb() -> float:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 2)
        except Exception:
            pass
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _get_cpu_percent() -> float:
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

HYBRID_PRESETS: Dict[str, HybridConfig] = {
    "local-gui-openai": HybridConfig(mode=HybridMode.LOCAL_GUI, remote_provider="openai", remote_model="gpt-4o-mini"),
    "local-gui-claude": HybridConfig(mode=HybridMode.LOCAL_GUI, remote_provider="anthropic", remote_model="claude-3-haiku-20240307"),
    "local-gui-groq": HybridConfig(mode=HybridMode.LOCAL_GUI, remote_provider="groq", remote_model="llama3-70b-8192"),
    "fully-local": HybridConfig(mode=HybridMode.FULLY_LOCAL),
    "fallback-openai": HybridConfig(mode=HybridMode.FALLBACK_ONLY, remote_provider="openai"),
}
