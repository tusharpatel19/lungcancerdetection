from __future__ import annotations

from .state import AgentState


def generate_report(state: AgentState) -> str:
    prediction = state.prediction or "Unknown"
    confidence = float(state.confidence or 0.0)
    explanation = state.explanation or {}
    research = state.research or {}
    sources = state.sources or []

    summary_lines = [
        "Analysis Summary",
        "",
        f"Predicted Class: {prediction}",
        f"Model Confidence: {confidence * 100:.1f}%",
        "",
        "AI model output for educational/research purposes — not a medical diagnosis.",
        "",
        "Model Interpretation:",
        explanation.get("summary", "No summary available."),
        "",
        "Educational Information:",
        research.get("summary", "Research information could not be retrieved."),
        "",
        "Limitations:",
    ]

    limitations = explanation.get("limitations") or [
        "The model is not a definitive medical diagnosis.",
        "Research information may be unavailable in some cases.",
    ]
    summary_lines.extend(f"- {item}" for item in limitations)

    if sources:
        summary_lines.extend(["", "Sources:"])
        summary_lines.extend(f"- {source}" for source in sources)

    return "\n".join(summary_lines)
