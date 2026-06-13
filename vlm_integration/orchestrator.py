"""
VLM Orchestrator — Edge VLM as Planner/Evaluator for GUI Agent tasks.

Implements the "edge VLM as orchestrator" strategy:
1. A lightweight local VLM plans the task into sub-steps
2. The GUI interaction model executes each step
3. The same (or another) VLM evaluates completed steps and adjusts the plan

Supports: phi-3-vision, llava, qwen2-vl, internvl, minicpm-v
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Enums & Types
# ---------------------------------------------------------------------------

class OrchestratorRole(str, Enum):
    PLANNER = "planner"
    EVALUATOR = "evaluator"
    REACT = "react"  # reason + act loop


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Step:
    """A single planned step."""
    id: str
    action_type: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class Plan:
    """A task plan produced by the orchestrator."""
    task: str
    steps: List[Step]
    context: Dict[str, Any] = field(default_factory=dict)
    current_step_idx: int = 0
    status: TaskStatus = TaskStatus.PENDING

    def current_step(self) -> Optional[Step]:
        if 0 <= self.current_step_idx < len(self.steps):
            return self.steps[self.current_step_idx]
        return None

    def advance(self) -> bool:
        self.current_step_idx += 1
        return self.current_step_idx < len(self.steps)


# ---------------------------------------------------------------------------
# Orchestrator Config
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorConfig:
    """Configuration for the VLM orchestrator."""

    model_name: str = "microsoft/Phi-3-vision-128k-instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_plan_steps: int = 20
    evaluation_enabled: bool = True
    replan_on_failure: bool = True
    max_replans: int = 3

    # Template prompts
    planner_system_prompt: str = (
        "You are a GUI task planner. Given a user instruction, break it down into "
        "a sequence of atomic GUI actions: click, type, scroll, wait, navigate, press. "
        "Output a JSON array of steps. Each step must have: id, action_type, description."
    )
    evaluator_system_prompt: str = (
        "You are a GUI task evaluator. Given a screenshot and the task goal, evaluate "
        "whether the current step was executed correctly. Output JSON: "
        "{'status': 'success'|'failed'|'blocked', 'reason': '...', 'next_action': '...'}"
    )


VLMConfig = OrchestratorConfig


# ---------------------------------------------------------------------------
# VLM Orchestrator
# ---------------------------------------------------------------------------

class VLMOrchestrator:
    """Edge VLM as planner/evaluator for GUI Agent tasks."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self._loaded: bool = False

        # Callbacks
        self.on_step_start: Optional[Callable[[Step], None]] = None
        self.on_step_complete: Optional[Callable[[Step], None]] = None
        self.on_plan_update: Optional[Callable[[Plan], None]] = None

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the edge VLM model."""
        if self._loaded:
            return

        logger.info(f"Loading VLM orchestrator model: {self.config.model_name}")

        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name, trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.model_name,
                torch_dtype=getattr(torch, self.config.dtype),
                device_map=self.config.device,
                trust_remote_code=True,
            )
            self.model.eval()
            self._loaded = True
            logger.info("VLM orchestrator loaded")
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            raise

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------

    def plan_task(self, task: str, screenshot_base64: Optional[str] = None) -> Plan:
        """
        Generate a task plan from a user instruction.

        Args:
            task: User instruction (e.g., "Search for flights to Paris")
            screenshot_base64: Optional screenshot for context

        Returns:
            Plan with decomposed steps
        """
        logger.info(f"Planning task: {task}")

        messages = [
            {"role": "system", "content": self.config.planner_system_prompt},
            {"role": "user", "content": f"Task: {task}\n\nOutput a JSON array of steps."},
        ]

        # Generate plan via VLM
        plan_json = self._generate(messages, screenshot_base64)
        steps = self._parse_plan_json(plan_json)

        plan = Plan(task=task, steps=steps)
        logger.info(f"Plan generated: {len(steps)} steps")
        return plan

    def replan(self, plan: Plan, failure_reason: str) -> Plan:
        """
        Replan from current state after a failure.

        Args:
            plan: Current (failed) plan
            failure_reason: What went wrong

        Returns:
            New/updated plan
        """
        logger.info(f"Replanning after failure: {failure_reason}")

        # Keep completed steps, re-plan remaining
        completed = [s for s in plan.steps if s.status == TaskStatus.SUCCESS]
        remaining_task = f"Task: {plan.task}\nCompleted: {len(completed)} steps\nFailed at: {failure_reason}\nPlan remaining steps."

        messages = [
            {"role": "system", "content": self.config.planner_system_prompt},
            {"role": "user", "content": remaining_task},
        ]

        plan_json = self._generate(messages)
        new_steps = self._parse_plan_json(plan_json)

        # Merge with completed steps
        merged_steps = completed + new_steps
        return Plan(task=plan.task, steps=merged_steps, status=TaskStatus.IN_PROGRESS)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate_step(
        self, step: Step, screenshot_base64: str, task_goal: str
    ) -> Dict[str, Any]:
        """
        Evaluate whether a step was executed successfully.

        Args:
            step: The step that was executed
            screenshot_base64: Screenshot after step execution
            task_goal: Overall task goal

        Returns:
            Evaluation dict with status and reason
        """
        messages = [
            {"role": "system", "content": self.config.evaluator_system_prompt},
            {
                "role": "user",
                "content": (
                    f"Task goal: {task_goal}\n"
                    f"Step executed: {step.description}\n"
                    f"Evaluate if the step was successful."
                ),
            },
        ]

        result_json = self._generate(messages, screenshot_base64)
        return self._parse_evaluation(result_json)

    # ------------------------------------------------------------------
    # Execute Plan (Orchestration Loop)
    # ------------------------------------------------------------------

    def execute_plan(
        self,
        plan: Plan,
        execute_fn: Callable[[Step], Dict[str, Any]],
        screenshot_fn: Callable[[], str],
        max_steps: Optional[int] = None,
    ) -> Plan:
        """
        Execute a plan step-by-step with evaluation.

        Args:
            plan: Task plan to execute
            execute_fn: Function to execute each step (returns result dict)
            screenshot_fn: Function to capture current screenshot (returns base64)
            max_steps: Maximum steps to execute (None = all)

        Returns:
            Completed plan with results
        """
        remaining_replans = self.config.max_replans
        step_count = 0
        max_allowed = max_steps or len(plan.steps)

        while step_count < max_allowed and plan.current_step_idx < len(plan.steps):
            step = plan.current_step()
            if step is None:
                break

            # Callback
            if self.on_step_start:
                self.on_step_start(step)

            step.status = TaskStatus.IN_PROGRESS
            logger.info(f"Executing step {plan.current_step_idx + 1}/{len(plan.steps)}: {step.description}")

            try:
                # Execute the step
                step.result = execute_fn(step)

                # Evaluate if enabled
                if self.config.evaluation_enabled:
                    screenshot = screenshot_fn()
                    evaluation = self.evaluate_step(step, screenshot, plan.task)

                    if evaluation.get("status") == "failed":
                        step.status = TaskStatus.FAILED
                        step.error = evaluation.get("reason", "Unknown failure")

                        if self.config.replan_on_failure and remaining_replans > 0:
                            remaining_replans -= 1
                            plan = self.replan(plan, step.error)
                            if self.on_plan_update:
                                self.on_plan_update(plan)
                            continue
                    else:
                        step.status = TaskStatus.SUCCESS
                else:
                    step.status = TaskStatus.SUCCESS

            except Exception as e:
                step.status = TaskStatus.FAILED
                step.error = str(e)
                logger.error(f"Step failed: {e}")

                if self.config.replan_on_failure and remaining_replans > 0:
                    remaining_replans -= 1
                    plan = self.replan(plan, str(e))
                    if self.on_plan_update:
                        self.on_plan_update(plan)
                    continue

            # Callback
            if self.on_step_complete:
                self.on_step_complete(step)

            plan.advance()
            step_count += 1

        # Final status
        all_success = all(s.status == TaskStatus.SUCCESS for s in plan.steps)
        plan.status = TaskStatus.SUCCESS if all_success else TaskStatus.FAILED

        return plan

    # ------------------------------------------------------------------
    # VLM Generation
    # ------------------------------------------------------------------

    def _generate(self, messages: List[Dict[str, Any]], image_base64: Optional[str] = None) -> str:
        """Generate text from the VLM model."""
        if not self._loaded:
            return "[]"  # offline fallback

        try:
            import torch
            from PIL import Image
            import io
            import base64

            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if image_base64:
                # Decode base64 image
                img_data = base64.b64decode(image_base64.split(",")[-1])
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            else:
                inputs = self.processor(text=prompt, return_tensors="pt")

            if self.config.device != "cpu":
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=512, temperature=0.0)

            result = self.processor.batch_decode(generated, skip_special_tokens=True)[0]

            # Extract response after prompt
            if prompt in result:
                result = result[len(prompt):].strip()

            return result

        except Exception as e:
            logger.warning(f"VLM generation failed: {e}")
            return "[]"

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_plan_json(self, raw: str) -> List[Step]:
        """Parse VLM output into a list of Steps."""
        try:
            # Try direct JSON parse
            data = json.loads(raw)
            if isinstance(data, list):
                return [
                    Step(
                        id=s.get("id", f"step_{i}"),
                        action_type=s.get("action_type", "unknown"),
                        description=s.get("description", ""),
                        params=s.get("params", {}),
                    )
                    for i, s in enumerate(data)
                ]
            elif isinstance(data, dict) and "steps" in data:
                return self._parse_plan_json(json.dumps(data["steps"]))
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            match = re.search(r"\[[\s\S]*\]", raw)
            if match:
                return self._parse_plan_json(match.group(0))

        # Fallback: create a single-step plan
        logger.warning(f"Could not parse plan JSON, using fallback. Raw: {raw[:200]}")
        return [Step(id="step_0", action_type="unknown", description="Parse failed, manual intervention needed")]

    def _parse_evaluation(self, raw: str) -> Dict[str, Any]:
        """Parse evaluation result."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return {"status": "success", "reason": "Could not parse evaluation"}


# ---------------------------------------------------------------------------
# Pre-set edge VLM configurations
# ---------------------------------------------------------------------------

EDGE_VLM_CONFIGS: Dict[str, OrchestratorConfig] = {
    "phi3-vision": OrchestratorConfig(
        model_name="microsoft/Phi-3-vision-128k-instruct",
        device="cuda",
    ),
    "llava-1.6": OrchestratorConfig(
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        device="cuda",
    ),
    "qwen2-vl-7b": OrchestratorConfig(
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        device="cuda",
    ),
    "internvl2-4b": OrchestratorConfig(
        model_name="OpenGVLab/InternVL2-4B",
        device="cuda",
    ),
    "minicpm-v-2.6": OrchestratorConfig(
        model_name="openbmb/MiniCPM-V-2_6",
        device="cuda",
    ),
}


def create_orchestrator(name: str) -> VLMOrchestrator:
    """Factory for creating orchestrators from preset configs."""
    if name not in EDGE_VLM_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(EDGE_VLM_CONFIGS.keys())}")
    return VLMOrchestrator(EDGE_VLM_CONFIGS[name])
