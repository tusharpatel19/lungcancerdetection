# Agentic Lung Cancer Research Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?logo=keras&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)

This project started as a CNN-based lung image classifier and was upgraded into an educational, research-oriented Agentic AI platform. The existing TensorFlow/Keras model is preserved as the source of truth for image classification, while a lightweight multi-agent workflow explains the result, summarizes educational context, retrieves controlled research information, and assembles a final report.

## Project overview

The platform accepts a lung CT or related image, validates it, preprocesses it using the existing pipeline, passes it to the CNN model, and then routes the structured result through:

- Image Analysis Agent
- Explanation Agent
- Research Agent
- Report Agent

This is an educational/research decision-support workflow, not an autonomous medical diagnosis system.

## Existing CNN architecture

The current model and preprocessing pipeline are preserved from the original project:

- Input size: 224 x 224 RGB
- TensorFlow/Keras CNN
- Image normalization and augmentation in training
- Model saved as:
  - model/model.h5
  - model/model.keras
- Class labels loaded from metadata or directory structure

The CNN remains the definitive source of the classification result.

## Agentic AI architecture

The upgrade adds a stateful orchestration layer around the existing model.

Workflow:

```text
User uploads CT/Image
        ↓
Image Analysis Agent
        ↓
Existing CNN Model
        ↓
Prediction + Confidence
        ↓
Explanation Agent
        ↓
Research Agent
        ↓
Report Agent
        ↓
Final Research Report
```

### Agent responsibilities

1. Image Analysis Agent
   - validates file type and CT-likeness
   - preprocesses the image through the existing pipeline
   - invokes the CNN model
   - returns structured prediction output

2. Explanation Agent
   - translates the raw model result into simple educational language
   - states that the result is not a definitive medical diagnosis
   - includes the key limitations

3. Research Agent
   - retrieves controlled educational background information
   - uses only safe, general information sources
   - fails gracefully if retrieval is unavailable

4. Report Agent
   - merges explanation, research summaries, sources, and limitations
   - produces a final structured report

This qualifies as Agentic AI because the workflow is stateful and orchestrated across specialized roles with explicit structured handoff between stages.

## Tech stack

- Python
- Flask
- TensorFlow/Keras
- PIL / NumPy
- SQLite
- HTML / JavaScript frontend
- Optional future LLM integration through a controlled API interface

## Setup

1. Create a virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the model exists:

```bash
model/model.keras
# or model/model.h5
```

4. Run the app:

```bash
python app.py
```

5. Open http://localhost:5000

## Environment variables

Create a .env file or export variables before running the app:

```bash
FLASK_SECRET_KEY=your-secret-key
MODEL_PATH=./model/model.keras
CORS_ORIGINS=http://localhost:5000
MAX_UPLOAD_BYTES=5242880
REQUIRE_API_LOGIN=false
```

## API endpoints

- POST /api/analyze
- GET /api/analysis/<id>
- GET /api/analysis/history
- POST /predict
- GET /healthz

## Example output

```json
{
  "status": "report_ready",
  "prediction": "Adenocarcinoma",
  "confidence": 0.87,
  "modelVersion": "v1",
  "explanation": {
    "summary": "The current model output predicts Adenocarcinoma with an estimated confidence of 87.0%. This result is generated for educational and research purposes only and is not a definitive medical diagnosis.",
    "educationalExplanation": "In a general educational context, a prediction of Adenocarcinoma indicates the model identified patterns consistent with that class.",
    "limitations": [
      "The model output is not a definitive medical diagnosis."
    ]
  },
  "research": {
    "summary": "Adenocarcinoma is a type of lung cancer discussed in educational settings as a major subtype.",
    "status": "ok"
  },
  "sources": [
    "National Cancer Institute (NCI)",
    "National Institutes of Health (NIH)"
  ]
}
```

## Medical safety and limitations

This application is designed for research and educational understanding only.

Important safeguards:

- model results are not a definitive diagnosis
- no personalized treatment recommendations are made
- no clinician replacement or patient-specific advice is provided
- no fabricated medical findings or citations are allowed
- a final clinical assessment still requires a qualified medical professional

## Medical disclaimer

This application provides an AI-generated research/educational analysis based on an image classification model. It is not a medical diagnosis and should not be used to make medical decisions.

## Future improvements

- add a real LLM-backed agent layer with structured prompts and source-grounding
- add a richer retrieval store with vetted medical references
- add clinician review workflow and audit logs
- improve PDF export and downloadable summary files
- add stricter security, authentication, and deployment configuration for production use

## Why this is not a full clinical diagnosis system

The app is intentionally limited to research and educational support. It does not claim certainty and does not replace a physician. The CNN is used as a classification tool, while the agent layers simply interpret and package the result in a safe, explainable, and bounded way.
