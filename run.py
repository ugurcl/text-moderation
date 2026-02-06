import os
from src.classifier import TextClassifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "classifier.pkl")

if not os.path.isfile(MODEL_PATH):
    print("Model not found. Training...\n")
    from scripts.train import train
    train()
    print()

clf = TextClassifier()

print("=" * 50)
print("  TEXT MODERATION SYSTEM")
print("  Type 'q' to quit")
print("=" * 50)

while True:
    print()
    text = input("Enter text: ").strip()

    if not text:
        continue

    if text.lower() == "q":
        print("Exiting...")
        break

    detail = clf.get_detail(text)
    status = "PASSED (ALLOW)" if detail["allowed"] else "BLOCKED (BLOCK)"

    print(f"  Result  : {status}")
    print(f"  Category: {detail['label']}")
    print(f"  Confidence: {detail['confidence'] * 100:.1f}%")
