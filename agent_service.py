from __future__ import annotations

from dataclasses import asdict

from agents.explanation import explain_model_output
from agents.image_analysis import analyze_image
from agents.report import generate_report
from agents.research import ResearchAgent
from agents.state import AgentState


def run_agentic_analysis(image_path: str, model_version: str = "v1") -> AgentState:
    state = AgentState(imagePath=image_path, modelVersion=model_version)
    state = analyze_image(state)

    if state.validationError:
        state.status = "invalid_image"
        state.error = state.validationError
        return state

    state = explain_model_output(state)
    state.status = "explanation_ready"

    state = ResearchAgent().run(state)
    if state.research.get("status") == "unavailable":
        state.status = "research_unavailable"
    else:
        state.status = "research_ready"

    state.report = generate_report(state)
    state.status = "report_ready"
    return state


def run_agentic_analysis_dict(image_path: str, model_version: str = "v1") -> dict:
    state = run_agentic_analysis(image_path=image_path, model_version=model_version)
    return asdict(state)
