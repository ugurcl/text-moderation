# Text Moderation

Text moderation system for e-commerce and messaging platforms. Analyzes incoming texts and classifies them as product listings, adult content, or toxic content.

## What does it do?

The system classifies each text into 3 categories:

- **product** - Regular product listings (Samsung phone, Nike shoes, etc.) → allowed
- **adult** - Adult content products → blocked
- **toxic** - Profanity, hate speech, threats, spam, scams → blocked

Supports both Turkish and English texts. Works with character n-grams so it's language-agnostic and handles mixed-language inputs without issues.

## Installation

Requires Python 3.10+

```bash
pip install -r requirements.txt
```

## Quick Start

The project doesn't include training data or a pre-trained model. To get started:

1. Create your training data as `data/train.txt` (fastText format: `__label__category text`)
2. Train the model with `python scripts/train.py`
3. Start using it with `python run.py`

Training data format example:
```
__label__product Samsung Galaxy S24 Ultra 256GB
__label__adult Vibrator Wand Massager 10 Speed
__label__toxic go fuck yourself
```

## Usage

### Training the model

```bash
python scripts/train.py
```

Reads the training data, trains the model, and saves it as `models/classifier.pkl`. Also creates a timestamped backup on each run.

### Running the demo

```bash
python scripts/demo.py
```

Runs classification on sample texts and outputs a speed benchmark. Automatically trains the model first if it doesn't exist.

### Interactive mode

```bash
python run.py
```

Type a text and press Enter to see the result. Type `q` to quit.

```
Enter text: Samsung Galaxy S24 Ultra 256GB
  Result  : PASSED (ALLOW)
  Category: product
  Confidence: 96.3%

Enter text: Vibrator Wand Massager 10 Speed
  Result  : BLOCKED (BLOCK)
  Category: adult
  Confidence: 100.0%

Enter text: go fuck yourself moron
  Result  : BLOCKED (BLOCK)
  Category: toxic
  Confidence: 100.0%

Enter text: q
Exiting...
```

### REST API

```bash
python api.py
```

Starts the server at `http://localhost:8000`. Swagger docs are auto-generated at `http://localhost:8000/docs`.

**Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health check (version, uptime, model & DB status) |
| `/predict` | POST | Single text classification |
| `/predict/batch` | POST | Batch classification (max 100) |
| `/predict/explain` | POST | Prediction + feature contribution analysis |
| `/feedback` | POST | Submit label correction for a misprediction |
| `/feedback/list` | GET | Recent feedback entries |
| `/stats` | GET | Prediction statistics |
| `/history` | GET | Recent prediction history |
| `/metrics` | GET | Prometheus metrics (prediction count, latency, feedback) |

To enable API key protection, set `API_KEY=your-secret-key` in your `.env` file. If left empty, auth is disabled. When active, include the `X-API-Key` header in requests:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"text": "Samsung Galaxy S24"}'
```

**Examples:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Samsung Galaxy S24 Ultra 256GB"}'
```
```json
{"text": "Samsung Galaxy S24 Ultra 256GB", "label": "product", "confidence": 0.9634, "allowed": true, "needs_review": false}
```

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Nike Air Max 270", "Vibrator Wand Massager", "go fuck yourself"]}'
```
```json
[
  {"text": "Nike Air Max 270", "label": "product", "confidence": 1.0, "allowed": true, "needs_review": false},
  {"text": "Vibrator Wand Massager", "label": "adult", "confidence": 1.0, "allowed": false, "needs_review": false},
  {"text": "go fuck yourself", "label": "toxic", "confidence": 1.0, "allowed": false, "needs_review": false}
]
```

**Feedback (label correction):**

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"text": "some misclassified text", "correct_label": "product"}'
```
```json
{"text": "some misclassified text", "predicted_label": "toxic", "correct_label": "product"}
```

**Health check:**

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "version": "1.1.0", "uptime_seconds": 123.4, "model_loaded": true, "database_connected": true}
```

**Prometheus metrics:**

```bash
curl http://localhost:8000/metrics
```

Returns metrics in Prometheus exposition format, including `prediction_total`, `prediction_latency_seconds`, and `feedback_total`.

### Streamlit UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in the browser. Supports single and batch text analysis.

### Docker

```bash
docker-compose up --build
```

API runs at `http://localhost:8000`, Streamlit at `http://localhost:8501`. The `data/` and `models/` directories are mounted as volumes.

### Python usage

```python
from src.classifier import TextClassifier

clf = TextClassifier()

label, confidence = clf.predict("Samsung Galaxy S24 Ultra 256GB")
print(label, confidence)  # product 1.0

print(clf.is_allowed("Nike Air Max 270"))  # True
print(clf.is_allowed("go fuck yourself"))  # False

detail = clf.get_detail("some text")
print(detail)  # {"text": "...", "label": "...", "confidence": 0.99, "allowed": True, "needs_review": False}

results = clf.predict_batch(["text1", "text2", "text3"])
```

### Tests

```bash
pytest tests/ -v
```

## Configuration

Copy `.env.example` to `.env` and adjust the settings:

| Variable | Default | Description |
|---|---|---|
| `API_HOST` | 0.0.0.0 | API server address |
| `API_PORT` | 8000 | API port |
| `LOG_LEVEL` | INFO | Log level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | json | Log format (`json` for structured, `text` for plain) |
| `RATE_LIMIT` | 60/minute | API rate limit |
| `CONFIDENCE_THRESHOLD` | 0.5 | Minimum confidence threshold |
| `REVIEW_THRESHOLD` | 0.85 | Below this confidence, predictions are flagged as `needs_review` |
| `API_KEY` | (empty) | API key (auth disabled when empty) |

## Project Structure

```
text-moderation/
├── run.py               # Interactive text moderation CLI
├── api.py               # FastAPI REST API
├── app.py               # Streamlit web UI
├── Makefile             # Shortcut commands
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml       # black/isort config
├── .pre-commit-config.yaml
├── requirements.txt
├── .env.example
├── src/
│   ├── classifier.py    # TextClassifier class
│   ├── config.py        # Configuration management
│   ├── database.py      # SQLite prediction history & feedback
│   ├── logger.py        # Logging setup (JSON/text)
│   └── metrics.py       # Prometheus metrics definitions
├── scripts/
│   ├── train.py         # Model training script
│   └── demo.py          # Test and benchmark script
├── tests/
│   ├── conftest.py      # Test fixtures
│   ├── test_classifier.py
│   └── test_api.py
├── data/
│   ├── train.txt        # Training data (30,000 samples)
│   └── test.txt         # Test data (450 samples)
└── models/
    └── classifier.pkl   # Trained model
```

> Note: `data/`, `models/`, `.env` and `scripts/generate_data.py` are excluded from the repo due to sensitive content.

## Technical Details

- **Algorithm:** TF-IDF (character n-grams) + SGD Classifier (Linear SVM)
- **N-gram range:** 2-5 characters
- **Vocabulary:** 80,000 features
- **Dataset:** 30,000 training, 450 test (balanced across classes)
- **API:** FastAPI + API key auth + rate limiting + CORS + SQLite history
- **Explainability:** Per-prediction feature contribution analysis
- **Human-in-the-Loop:** Feedback endpoint for label corrections + confidence-based `needs_review` flag
- **Observability:** Structured JSON logging + Prometheus metrics (latency, counters)
- **CI/CD:** GitHub Actions automated testing
- **Code Quality:** black + flake8 + isort + pre-commit hooks
- **Container:** Docker + docker-compose

## Requirements

- scikit-learn >= 1.3
- numpy
- joblib
- fastapi
- uvicorn
- python-dotenv
- slowapi
- prometheus-client
- streamlit
- pytest
- httpx
- black
- flake8
- isort
- pre-commit
