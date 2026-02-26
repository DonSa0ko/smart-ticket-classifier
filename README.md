# Smart Ticket Classifier

An NLP-powered system that automatically classifies IT support tickets by priority and responsible area using Machine Learning.

Built to solve a real helpdesk problem: instead of a technician manually reading and routing every ticket, the model predicts in real time which team should handle it and how urgent it is.

---

## Demo

Send a POST request to the API with a ticket description and receive an instant classification:

```json
POST /classify
{
  "text": "The WiFi in the warehouse is completely down"
}
```

```json
{
  "ticket": "The WiFi in the warehouse is completely down",
  "priority": "High",
  "priority_confidence": "91.3%",
  "area": "Networks",
  "area_confidence": "96.7%"
}
```

---

## Model Performance

| Classifier | Test Accuracy | CV Accuracy (5-fold) |
|---|---|---|
| Priority (High / Medium / Low) | 85% | 83% ± 3% |
| Area (Networks / Hardware / Access) | 91% | 89% ± 2% |

The area classifier correctly identifies the responsible team 9 out of 10 times. Priority errors only occur between adjacent levels (High ↔ Medium) — the model never confuses High with Low, which mirrors human performance on the same task.

---

## Features

- Classifies tickets by **priority** (High / Medium / Low) and **responsible area** (Networks / Hardware / Access)
- Returns a **confidence score** for each prediction
- REST API built with FastAPI with auto-generated interactive documentation
- Custom text preprocessing pipeline (stopword removal, lowercasing, n-gram TF-IDF)
- LinearSVC classifier — chosen over Random Forest for superior performance on short text
- Evaluated with 5-fold cross-validation for reliable accuracy estimates
- Generates visual training reports (confusion matrices, precision/recall charts, CV plots)

---

## Project Structure

```
smart-ticket-classifier/
├── api/
│   └── main.py             # FastAPI application and endpoints
├── data/
│   └── tickets.csv         # 270 labeled training tickets
├── model/
│   ├── train.py            # Training script — generates .pkl models and reports
│   ├── priority_model.pkl  # Trained priority classifier
│   └── area_model.pkl      # Trained area classifier
├── reports/                # Auto-generated training evaluation charts
│   ├── confusion_matrix_priority.png
│   ├── confusion_matrix_area.png
│   ├── metrics_priority.png
│   ├── metrics_area.png
│   ├── cross_validation.png
│   └── summary.csv
├── classifier.py           # Loads models and exposes predict()
├── preprocessor.py         # Shared text cleaning transformer
└── requirements.txt
```

---

## Requirements

- Python 3.10+

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train the model

```bash
python model/train.py
```

This will train both classifiers, evaluate them, and save:
- Trained models to `model/`
- Visual reports to `reports/`

### 2. Start the API

```bash
uvicorn api.main:app --reload
```

### 3. Open the interactive docs

```
http://127.0.0.1:8000/docs
```

From here you can test the API directly in the browser — write any ticket description and see the classification in real time.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/classify` | Classify a support ticket |

### Example request

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I forgot my password and cannot log in"}'
```

### Example response

```json
{
  "ticket": "I forgot my password and cannot log in",
  "priority": "High",
  "priority_confidence": "88.4%",
  "area": "Access",
  "area_confidence": "97.1%"
}
```

---

## How It Works

```
User writes ticket
       ↓
TextPreprocessor  →  lowercase, remove punctuation, filter stopwords
       ↓
TF-IDF Vectorizer →  convert text to weighted term-frequency matrix
       ↓
LinearSVC         →  classify against learned decision boundaries
       ↓
CalibratedCV      →  convert decision scores to confidence probabilities
       ↓
API response      →  priority + area + confidence scores
```

---

## Training Data

The dataset contains 270 unique labeled tickets evenly distributed across all classes:

| Area | Priority | Count |
|---|---|---|
| Networks | High / Medium / Low | 30 each |
| Hardware | High / Medium / Low | 30 each |
| Access | High / Medium / Low | 30 each |

Example tickets per category:

- **Networks / High**: `"The entire plant network is down"`
- **Networks / Medium**: `"WiFi disconnects every few minutes on the floor"`
- **Hardware / High**: `"The label printer on the production line stopped working"`
- **Access / High**: `"My account has been locked out of the system"`
- **Access / Low**: `"Need to update my display name in the directory"`

---

## Stack

- `Python` — core language
- `scikit-learn` — ML pipeline, LinearSVC, TF-IDF, cross-validation
- `pandas` — dataset loading and manipulation
- `FastAPI` — REST API framework
- `uvicorn` — ASGI server
- `matplotlib` / `seaborn` — training evaluation charts
- `pydantic` — request and response schema validation
