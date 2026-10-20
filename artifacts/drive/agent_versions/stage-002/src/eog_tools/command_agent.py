"""SDK CommandAgent with independent adaptation and scored-task limits."""

from __future__ import annotations

from typing import Any

from simple_agentic_evals import CommandAgent


class TimedCommandAgent(CommandAgent):
    def __init__(self, *args: Any, adapt_timeout: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.adapt_timeout = float(adapt_timeout)

    def adapt(self, stage: int, tasks: Any) -> dict[str, Any]:
        # The SDK runs adaptation and scored tasks sequentially but currently
        # exposes one timeout for both subprocess paths.
        task_timeout = self.timeout
        self.timeout = self.adapt_timeout
        try:
            return super().adapt(stage, tasks)
        finally:
            self.timeout = task_timeout
