from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Structured state passed between the agent stages."""

    imagePath: str | None = None
    prediction: str | None = None
    confidence: float | None = None
    modelVersion: str = "v1"
    explanation: dict[str, Any] = field(default_factory=dict)
    research: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    report: str = ""
    status: str = "initialized"
    validationError: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "imagePath": self.imagePath,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "modelVersion": self.modelVersion,
            "explanation": self.explanation,
            "research": self.research,
            "sources": self.sources,
            "report": self.report,
            "status": self.status,
            "validationError": self.validationError,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AgentState":
        if not payload:
            return cls()

        return cls(
            imagePath=payload.get("imagePath"),
            prediction=payload.get("prediction"),
            confidence=payload.get("confidence"),
            modelVersion=str(payload.get("modelVersion", "v1")),
            explanation=dict(payload.get("explanation") or {}),
            research=dict(payload.get("research") or {}),
            sources=list(payload.get("sources") or []),
            report=str(payload.get("report") or ""),
            status=str(payload.get("status", "initialized")),
            validationError=payload.get("validationError"),
            error=payload.get("error"),
        )

    def update_status(self, value: str) -> "AgentState":
        self.status = value
        return self
