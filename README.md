# LungGuard CNN

A Flask-based demo for lung cancer detection using a CNN model. Upload a CT slice and receive a prediction from `model/model.h5`.

## Features
- Dashboard-style frontend
- Flask backend with `/predict` endpoint
- CNN model loader and image preprocessing
- Supports sigmoid or softmax outputs

## Quickstart
1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your trained model at `model/model.h5`.
4. Run the app:

```bash
python app.py
```

5. Open `http://localhost:5000`.

## Model Expectations
- Input image resized to **224x224** RGB.
- Output can be:
  - Sigmoid (binary) -> `Normal` vs `Lung Cancer`
  - Softmax (multi-class) -> uses `CLASS_LABELS` env var if provided

Example:

```bash
set CLASS_LABELS=Normal,Lung Cancer
```

## Disclaimer
This is a research demo and not a medical device. Do not use for clinical decisions.
