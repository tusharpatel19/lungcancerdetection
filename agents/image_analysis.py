from __future__ import annotations

from pathlib import Path

from .state import AgentState


class ImageAnalysisAgent:
    """Agent that validates the upload and runs the existing CNN model."""

    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def run(self, state: AgentState) -> AgentState:
        if not state.imagePath:
            state.validationError = "No image path provided for analysis."
            return state

        import app as app_module

        image_path = Path(state.imagePath)
        if not image_path.exists():
            state.validationError = "Uploaded image could not be found."
            return state

        if image_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            state.validationError = "Unsupported file type. Please upload a PNG or JPG image."
            return state

        is_valid, message = app_module._looks_like_ct_scan(image_path)
        if not is_valid:
            state.validationError = message
            return state

        result, error = app_module._predict(image_path)
        if error:
            state.validationError = str(error)
            state.error = str(error)
            return state

        if not result:
            state.validationError = "CNN model did not return a valid prediction."
            state.error = state.validationError
            return state

        state.prediction = result.get("display_label") or result.get("label")
        state.confidence = float(result.get("confidence", 0.0))
        state.modelVersion = "v1"
        state.status = "cnn_predicted"
        return state


def analyze_image(state: AgentState) -> AgentState:
    """Convenience wrapper for the workflow."""
    return ImageAnalysisAgent().run(state)
