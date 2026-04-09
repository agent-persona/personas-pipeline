from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


StageFn = Callable[[Any], Any] | Callable[[Any], Awaitable[Any]]


@dataclass
class Stage:
    """One stage of the pipeline.

    A stage wraps a function (sync or async) that takes the previous stage's
    output and returns a new value. The orchestrator handles state, errors,
    and timing.
    """

    name: str
    fn: StageFn
    description: str = ""


@dataclass
class StageResult:
    name: str
    success: bool
    duration_ms: int
    output_summary: str = ""
    error: str | None = None


@dataclass
class RunState:
    """The result of running a Pipeline. Mirrors what real orchestration
    (Temporal/LangGraph) would persist in Postgres."""

    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    tenant_id: str = ""
    stages: list[StageResult] = field(default_factory=list)
    final_output: Any = None
    success: bool = True
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def total_duration_ms(self) -> int:
        end = self.finished_at or time.time()
        return int((end - self.started_at) * 1000)

    def stage_by_name(self, name: str) -> StageResult | None:
        for s in self.stages:
            if s.name == name:
                return s
        return None


class PipelineError(Exception):
    def __init__(self, stage_name: str, original: Exception, state: RunState):
        super().__init__(f"Pipeline failed at stage '{stage_name}': {original}")
        self.stage_name = stage_name
        self.original = original
        self.state = state


class Pipeline:
    """Sequential DAG of stages.

    Each stage receives the previous stage's output. Errors short-circuit
    the run and surface in `RunState`. Resume support is intentionally
    simple — a future production version would persist `RunState` to
    Postgres after every stage and skip already-completed stages on retry.
    """

    def __init__(self, stages: list[Stage]) -> None:
        self.stages = stages

    async def run(
        self,
        initial_input: Any,
        tenant_id: str = "",
    ) -> RunState:
        state = RunState(tenant_id=tenant_id)
        current = initial_input

        for stage in self.stages:
            stage_start = time.monotonic()
            logger.info("[%s] Starting stage: %s", state.run_id, stage.name)
            try:
                if inspect.iscoroutinefunction(stage.fn):
                    current = await stage.fn(current)
                else:
                    current = stage.fn(current)
            except Exception as e:
                duration = int((time.monotonic() - stage_start) * 1000)
                state.stages.append(
                    StageResult(
                        name=stage.name,
                        success=False,
                        duration_ms=duration,
                        error=f"{type(e).__name__}: {e}",
                    )
                )
                state.success = False
                state.finished_at = time.time()
                logger.error("[%s] Stage %s failed: %s", state.run_id, stage.name, e)
                raise PipelineError(stage.name, e, state) from e

            duration = int((time.monotonic() - stage_start) * 1000)
            state.stages.append(
                StageResult(
                    name=stage.name,
                    success=True,
                    duration_ms=duration,
                    output_summary=_summarize(current),
                )
            )
            logger.info(
                "[%s] Stage %s ok (%dms)",
                state.run_id,
                stage.name,
                duration,
            )

        state.final_output = current
        state.finished_at = time.time()
        return state


def _summarize(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)} keys]"
    return type(value).__name__
