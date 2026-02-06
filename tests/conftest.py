import os
import pytest
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    model_path = os.path.join(model_dir, "classifier.pkl")

    if os.path.isfile(model_path):
        yield
        return

    texts = [
        "Samsung Galaxy S24", "Nike Air Max", "Apple MacBook",
        "iPhone 15 Pro", "Sony Headphones", "Laptop Dell",
        "vibrator massager", "adult toy item", "lingerie set",
        "dildo silicone", "bondage kit", "sexy costume",
        "fuck you idiot", "go to hell moron", "piece of shit",
        "kill yourself", "scam fraud money", "hate speech text",
    ]
    labels = ["product"] * 6 + ["adult"] * 6 + ["toxic"] * 6

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=5000)),
        ("clf", SGDClassifier(loss="modified_huber", max_iter=200, random_state=42)),
    ])
    pipeline.fit(texts, labels)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, model_path)

    yield

    os.remove(model_path)
