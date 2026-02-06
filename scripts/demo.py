import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import TextClassifier

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "classifier.pkl")

if not os.path.isfile(MODEL_PATH):
    print("Model not found. Training...\n")
    from scripts.train import train
    train()
    print()

clf = TextClassifier()

test_messages = [
    "Samsung Galaxy S24 Ultra 256GB",
    "Nike Air Max 270 Running Shoes",
    "Apple MacBook Pro 16 inch M3 Max",
    "Arçelik No Frost Buzdolabı 540 Litre",
    "Sony WH-1000XM5 Headphones",
    "Dyson V15 Detect Cordless Vacuum Cleaner",
    "Philips Airfryer XXL Sıcak Hava Fritözü",
    "CeraVe Moisturizing Cream for Dry Skin 19 oz",
    "Optimum Nutrition Gold Standard Whey Protein 2.27kg",
    "IKEA Billy Kitaplık Beyaz",
    "Decathlon Koşu Ayakkabısı Erkek",
    "Lenovo IdeaPad Slim 5 Laptop 16 inç",

    "Dildo Realistic Silicone 8 Inch with Suction Cup",
    "Vibrator Wand Massager 10 Speed Rechargeable",
    "Satisfyer Pro 2 Next Generation",
    "Anal Plug Silicone Set for Beginners",
    "Bondage Kit Set 10 Piece Restraint BDSM",
    "Vibratör Klitoral Stimülatör Şarj Edilebilir",
    "Kayganlaştırıcı Jel Su Bazlı 200ml",
    "Fantezi Kıyafet Hemşire Kostüm Seti",
    "Prezervatif Ultra İnce 12li Paket",
    "Lovense Lush 3 Bluetooth Egg",

    "siktir git buradan aptal herif",
    "orospu çocuğu seni",
    "you are a piece of shit",
    "go fuck yourself moron",
    "hırsız piç kurusu seni",
    "kill yourself nobody will miss you",
    "sen gerçekten mal mısın ya",
    "suriyelilerin hepsini denize dökelim",
    "eşcinseller hastalıklı tedavi edilmeli",
    "selling cocaine and meth fast delivery DM",
    "I will find you and kill your family",
    "bu kızın ifşa fotoğrafları var bakın",
    "tebrikler 100000 TL kazandınız tıklayın",
    "hesabınız tehlikede hemen şifrenizi girin",
]

print("=" * 72)
print(f"{'TEXT':<45} {'LABEL':<10} {'CONF':<8} {'PASS'}")
print("=" * 72)

for msg in test_messages:
    detail = clf.get_detail(msg)
    status = "ALLOW" if detail["allowed"] else "BLOCK"
    print(f"{detail['text']:<45} {detail['label']:<10} {detail['confidence']:<8} {status}")

print("=" * 72)

print("\nSpeed benchmark (1000 predictions)...")
start = time.perf_counter()
for _ in range(1000):
    clf.predict("Vibrator Wand Massager 10 Speed Rechargeable")
elapsed = time.perf_counter() - start
print(f"  1000 predictions in {elapsed*1000:.1f}ms")
print(f"  {elapsed*1000/1000:.3f}ms per prediction")
print(f"  {1000/elapsed:.0f} predictions/sec")

print("\nBatch benchmark (1000 predictions at once)...")
batch = ["Samsung Galaxy S24 Ultra 256GB"] * 1000
start = time.perf_counter()
clf.predict_batch(batch)
elapsed = time.perf_counter() - start
print(f"  1000 batch predictions in {elapsed*1000:.1f}ms")
print(f"  {elapsed*1000/1000:.3f}ms per prediction")
print(f"  {1000/elapsed:.0f} predictions/sec")
print("=" * 72)
