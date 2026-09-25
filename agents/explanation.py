from __future__ import annotations

from .state import AgentState


def explain_model_output(state: AgentState) -> AgentState:
    prediction = state.prediction or "Unknown"
    confidence = float(state.confidence or 0.0)
    confidence_pct = max(0.0, min(100.0, confidence * 100.0))

    summary = (
        f"The current model output predicts {prediction} with an estimated confidence of {confidence_pct:.1f}%. "
        "This result is generated for educational and research purposes only and is not a definitive medical diagnosis."
    )

    educational = (
        f"In a general educational context, a prediction of {prediction} indicates the model identified visual patterns "
        "consistent with that class based on the trained CNN. This should be interpreted as a model-based classification signal, "
        "not a clinical determination."
    )

    limitations = [
        "The model output is not a definitive medical diagnosis and should not be used to make treatment or care decisions.",
        "Model confidence is a statistical estimate from the trained classifier, not a clinical certainty measure.",
        "Performance can vary with image quality, preprocessing, class balance, and dataset limitations.",
        "A final clinical assessment requires review by a qualified healthcare professional using appropriate diagnostic tools."
    ]

    state.explanation = {
        "summary": summary,
        "educationalExplanation": educational,
        "limitations": limitations,
    }
    return state
