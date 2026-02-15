import time

from fastapi import FastAPI, Request, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.classifier import TextClassifier
from src.config import API_HOST, API_PORT, RATE_LIMIT, CONFIDENCE_THRESHOLD, API_KEY
from src.database import PredictionDB
from src.logger import get_logger

APP_VERSION = "1.1.0"
_start_time = time.time()

log = get_logger("api")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Text Moderation API")
app.state.limiter = limiter

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)):
    if not API_KEY:
        return
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "Too many requests"})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

clf = TextClassifier()
db = PredictionDB()

log.info(f"API ready on {API_HOST}:{API_PORT}")


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


@app.get("/health")
def health():
    uptime = time.time() - _start_time
    db_ok = True
    try:
        db.conn.execute("SELECT 1")
    except Exception:
        db_ok = False

    return {
        "status": "ok",
        "version": APP_VERSION,
        "uptime_seconds": round(uptime, 1),
        "model_loaded": clf._model is not None,
        "database_connected": db_ok,
    }


@app.post("/predict")
@limiter.limit(RATE_LIMIT)
def predict(request: Request, req: TextRequest, _=Depends(verify_api_key)):
    detail = clf.get_detail(req.text)
    db.save(detail["text"], detail["label"], detail["confidence"], detail["allowed"])
    log.info(f"{detail['label']} ({detail['confidence']}) {'ALLOW' if detail['allowed'] else 'BLOCK'}")
    return detail


@app.post("/predict/batch")
@limiter.limit(RATE_LIMIT)
def predict_batch(request: Request, req: BatchRequest, _=Depends(verify_api_key)):
    results = clf.predict_batch(req.texts)
    response = []
    for text, (label, conf) in zip(req.texts, results):
        allowed = label == "product" and conf >= CONFIDENCE_THRESHOLD
        entry = {
            "text": text[:100],
            "label": label,
            "confidence": round(conf, 4),
            "allowed": allowed,
        }
        db.save(entry["text"], label, round(conf, 4), allowed)
        response.append(entry)
    log.info(f"Batch: {len(req.texts)} texts")
    return response


@app.post("/predict/explain")
@limiter.limit(RATE_LIMIT)
def explain(request: Request, req: TextRequest, _=Depends(verify_api_key)):
    result = clf.explain(req.text)
    log.info(f"Explain: {result['label']} ({result['confidence']})")
    return result


@app.get("/stats")
def stats(_=Depends(verify_api_key)):
    return db.get_stats()


@app.get("/history")
def history(limit: int = 20, _=Depends(verify_api_key)):
    return db.get_recent(min(limit, 100))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=int(API_PORT))
