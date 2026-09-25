from __future__ import annotations

from .state import AgentState


class ResearchAgent:
    """Controlled research layer that avoids inventing medical facts."""

    SAFE_SOURCES = [
        "National Cancer Institute (NCI)",
        "National Institutes of Health (NIH)",
        "U.S. Department of Health and Human Services",
        "Major academic medical centers / peer-reviewed literature",
    ]

    def run(self, state: AgentState) -> AgentState:
        prediction = (state.prediction or "Unknown").strip()
        if not prediction:
            state.research = {
                "summary": "Research information could not be retrieved because no model prediction was available.",
                "status": "unavailable",
                "notes": ["No prediction available for controlled lookup."],
            }
            state.sources = []
            return state

        if "adenocarcinoma" in prediction.lower():
            state.research = {
                "summary": "Adenocarcinoma is a type of lung cancer that often arises from glandular cells. In educational settings, it is discussed as one of the major histologic patterns of lung cancer, typically distinguished from other subtypes using pathology and imaging context.",
                "status": "ok",
                "notes": [
                    "This is general educational information about a cancer class and is not a personal medical diagnosis."
                ],
            }
            state.sources = self.SAFE_SOURCES[:2]
            return state

        if "squamous" in prediction.lower() or "large cell" in prediction.lower():
            state.research = {
                "summary": "This predicted class falls within broader lung cancer subtype educational categories that are often discussed through pathology, imaging, and clinical context rather than as a direct diagnosis from a single model output.",
                "status": "ok",
                "notes": [
                    "The model result is educational and research-oriented only; it does not establish a diagnosis."
                ],
            }
            state.sources = self.SAFE_SOURCES[:3]
            return state

        if "normal" in prediction.lower():
            state.research = {
                "summary": "The model did not identify the image as a cancer pattern in this classification setting. This result is still educational and should be interpreted carefully without assuming clinical certainty.",
                "status": "ok",
                "notes": [
                    "Normal classification is still a model output, not a clinical exclusion of disease."
                ],
            }
            state.sources = self.SAFE_SOURCES[:2]
            return state

        state.research = {
            "summary": "Research retrieval was unavailable for this class because the prediction did not match a recognized educational reference in the controlled lookup set.",
            "status": "unavailable",
            "notes": [
                "The model output remains available, but additional research details were not retrieved."
            ],
        }
        state.sources = []
        return state


def research_for_prediction(state: AgentState) -> AgentState:
    return ResearchAgent().run(state)
