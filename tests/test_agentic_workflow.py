from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

import app as app_module
from agents.explanation import explain_model_output
from agents.report import generate_report
from agents.research import ResearchAgent
from agents.state import AgentState
from agents.orchestrator import AgentWorkflow


@pytest.fixture
def valid_ct_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "scan.png"
    image = Image.new("L", (224, 224), 25)
    image.save(image_path)
    return image_path


def test_image_validation_valid_and_invalid(valid_ct_image: Path):
    is_valid, message = app_module._looks_like_ct_scan(valid_ct_image)
    assert is_valid is True
    assert message == ""

    bad_path = valid_ct_image.with_name("bad.txt")
    bad_path.write_text("not an image", encoding="utf-8")
    is_valid, message = app_module._looks_like_ct_scan(bad_path)
    assert is_valid is False
    assert "readable image" in message.lower()


def test_explanation_agent_generates_safe_summary():
    state = AgentState(prediction="Adenocarcinoma", confidence=0.87, modelVersion="v1")
    explanation = explain_model_output(state)

    assert explanation.explanation["summary"]
    assert "not a definitive medical diagnosis" in explanation.explanation["summary"].lower()
    assert explanation.explanation["limitations"]


def test_research_agent_handles_unknown_class_without_fabrication():
    state = AgentState(prediction="Rare Unknown Class", confidence=0.5, modelVersion="v1")
    result = ResearchAgent().run(state)

    assert result.research["status"] == "unavailable"
    assert result.research["summary"]
    assert result.sources == []


def test_workflow_reports_successful_end_to_end():
    def image_agent(state: AgentState) -> AgentState:
        state.prediction = "Adenocarcinoma"
        state.confidence = 0.87
        state.modelVersion = "v1"
        return state

    def explanation_agent(state: AgentState) -> AgentState:
        return explain_model_output(state)

    def research_agent(state: AgentState) -> AgentState:
        return ResearchAgent().run(state)

    def report_agent(state: AgentState) -> AgentState:
        state.report = generate_report(state)
        return state

    workflow = AgentWorkflow(image_agent, explanation_agent, research_agent, report_agent)
    result = workflow.run({"imagePath": "sample.png"})

    assert result.status in {"report_ready", "report_pending"}
    assert "Adenocarcinoma" in result.report
    assert "not a medical diagnosis" in result.report.lower()


def test_api_analysis_route_returns_structured_response():
    client = app_module.app.test_client()
    image = Image.new("L", (224, 224), 25)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    payload = {
        "image": (buffer, "scan.png"),
    }

    response = client.post("/api/analyze", data=payload, content_type="multipart/form-data")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] in {"report_ready", "research_unavailable", "invalid_image"}
    assert "prediction" in data
