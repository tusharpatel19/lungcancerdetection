# LungGuard CNN

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?logo=keras&logoColor=white)

A Flask-based dashboard for lung cancer detection using a CNN model. Upload a CT slice and receive a prediction from `model/model.h5` or `model/model.keras`.

## Highlights
- Dashboard-style frontend
- Flask backend with `/predict` endpoint
- CNN model loader and image preprocessing
- Supports sigmoid or softmax outputs
- Static GitHub Pages demo in `docs/`

## Quickstart
1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your trained model at `model/model.h5` or `model/model.keras`.
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
set CLASS_LABELS=Normal,Lung Cancer,Benign,Malignant
```

## GitHub Pages Demo
The `docs/` folder contains a static demo UI. It does not run inference without a backend.

To enable GitHub Pages:
1. Go to the repo settings in GitHub.
2. Pages -> Deploy from a branch -> `main` -> `/docs`.

## Screenshots
Add images to `assets/` and update this section.

## Disclaimer
This is a research demo and not a medical device. Do not use for clinical decisions.
