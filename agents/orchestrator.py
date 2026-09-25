from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .state import AgentState


class AgentWorkflow:
    """Simple stateful orchestrator for the AI reasoning pipeline.

    The core principle is that the CNN remains the source of truth for the model
    prediction. The agent stages only transform and package the same result into a
    research-oriented, explainable report.
    """

    def __init__(
        self,
        image_analysis_agent: Callable[[AgentState], AgentState] | None = None,
        explanation_agent: Callable[[AgentState], AgentState] | None = None,
        research_agent: Callable[[AgentState], AgentState] | None = None,
        report_agent: Callable[[AgentState], AgentState] | None = None,
    ) -> None:
        self.image_analysis_agent = image_analysis_agent or self._noop_agent
        self.explanation_agent = explanation_agent or self._noop_agent
        self.research_agent = research_agent or self._noop_agent
        self.report_agent = report_agent or self._noop_agent

    @staticmethod
    def _noop_agent(state: AgentState) -> AgentState:
        return state

    def run(self, payload: dict[str, Any] | AgentState | None = None) -> AgentState:
        state = payload if isinstance(payload, AgentState) else AgentState.from_dict(payload)
        state.status = "started"

        if not state.imagePath:
            state.validationError = "No image path provided for analysis."
            state.status = "invalid_image"
            state.error = "Image validation failed."
            return state

        state = self.image_analysis_agent(state)
        if state.validationError:
            state.status = "invalid_image"
            state.error = state.validationError
            return state

        state.status = "image_validated"
        state = self.explanation_agent(state)
        state.status = "explanation_ready"

        state = self.research_agent(state)
        if state.research.get("status") == "unavailable":
            state.status = "research_unavailable"
        else:
            state.status = "research_ready"

        state = self.report_agent(state)
        if state.report:
            state.status = "report_ready"
        else:
            state.status = "report_pending"

        return state


DEFAULT_AGENT_WORKFLOW = AgentWorkflow()
