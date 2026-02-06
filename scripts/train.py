import os
import sys
import shutil
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.config import DATA_DIR, MODEL_DIR
from src.logger import get_logger

log = get_logger("train")

MODEL_PATH = os.path.join(MODEL_DIR, "classifier.pkl")


def load_data(filepath: str) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            label = parts[0].replace("__label__", "")
            text = parts[1]
            labels.append(label)
            texts.append(text)
    return texts, labels


def train() -> None:
    train_path = os.path.join(DATA_DIR, "train.txt")
    test_path = os.path.join(DATA_DIR, "test.txt")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    log.info("Loading training data...")
    X_train, y_train = load_data(train_path)
    log.info(f"{len(X_train)} samples loaded")

    dist = Counter(y_train)
    for label, count in sorted(dist.items()):
        pct = count / len(y_train) * 100
        log.info(f"  {label}: {count} ({pct:.1f}%)")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=80000,
            sublinear_tf=True,
            strip_accents=None,
        )),
        ("clf", SGDClassifier(
            loss="modified_huber",
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            random_state=42,
        )),
    ])

    log.info("Training...")
    pipeline.fit(X_train, y_train)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(MODEL_DIR, f"classifier_{timestamp}.pkl")
    joblib.dump(pipeline, versioned_path)
    versioned_size = os.path.getsize(versioned_path) / 1024
    log.info(f"Saved versioned: {versioned_path} ({versioned_size:.0f} KB)")

    shutil.copy2(versioned_path, MODEL_PATH)
    log.info(f"Saved latest: {MODEL_PATH}")

    if os.path.isfile(test_path):
        X_test, y_test = load_data(test_path)
        y_pred = pipeline.predict(X_test)
        log.info(f"Test results ({len(X_test)} samples):")
        print(classification_report(y_test, y_pred))
    else:
        log.warning(f"No test file at {test_path}, skipping evaluation.")


if __name__ == "__main__":
    train()
