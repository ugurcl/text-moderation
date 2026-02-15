import os
import re

import joblib
import numpy as np

from src.config import MODEL_DIR, REVIEW_THRESHOLD
from src.logger import get_logger

log = get_logger("classifier")


class TextClassifier:
    def __init__(self):
        model_path = os.path.join(MODEL_DIR, "classifier.pkl")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run 'python scripts/train.py' first."
            )

        self._model = joblib.load(model_path)
        log.info(f"Model loaded from {model_path}")

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\n", " ")
        return text

    def predict(self, text: str) -> tuple[str, float]:
        text = self._clean(text)
        if not text:
            return ("product", 1.0)

        label = self._model.predict([text])[0]
        proba = self._model.predict_proba([text])[0]
        confidence = float(np.max(proba))
        return (label, confidence)

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        cleaned = [self._clean(t) for t in texts]
        labels = self._model.predict(cleaned)
        probas = self._model.predict_proba(cleaned)
        return [
            (label, float(np.max(proba)))
            for label, proba in zip(labels, probas)
        ]

    def is_allowed(self, text: str, threshold: float = 0.5) -> bool:
        label, confidence = self.predict(text)
        if label == "product" and confidence >= threshold:
            return True
        return False

    def get_detail(self, text: str) -> dict:
        label, confidence = self.predict(text)
        return {
            "text": text[:100],
            "label": label,
            "confidence": round(confidence, 4),
            "allowed": self.is_allowed(text),
            "needs_review": confidence < REVIEW_THRESHOLD,
        }

    def explain(self, text: str, top_n: int = 10) -> dict:
        text = self._clean(text)
        if not text:
            return {"text": "", "label": "product", "confidence": 1.0, "probabilities": {}, "top_features": []}

        vectorizer = self._model.named_steps["tfidf"]
        classifier = self._model.named_steps["clf"]

        tfidf_vec = vectorizer.transform([text])
        proba = self._model.predict_proba([text])[0]
        pred_idx = proba.argmax()
        pred_label = classifier.classes_[pred_idx]

        weights = classifier.coef_[pred_idx]
        feature_names = vectorizer.get_feature_names_out()

        nonzero = tfidf_vec.nonzero()[1]
        scores = []
        for i in nonzero:
            s = float(tfidf_vec[0, i] * weights[i])
            scores.append((feature_names[i], s))

        scores.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            "text": text[:100],
            "label": pred_label,
            "confidence": round(float(proba[pred_idx]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(classifier.classes_, proba)},
            "top_features": [{"feature": f, "weight": round(w, 4)} for f, w in scores[:top_n]],
        }
