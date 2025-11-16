from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Literal
from backend.collector import fetch_twitter_posts, fetch_reddit_posts
from backend.sentiment import SentimentAnalyzer
from backend.preprocess import clean_texts

app = FastAPI(title="Sentiment Dashboard Backend")

class CollectResponse(BaseModel):
    source: Literal["twitter","reddit"]
    query: str
    limit: int
    texts: List[str]

class AnalyzeRequest(BaseModel):
    texts: List[str]
    mode: Literal["lexicon","ml","lstm"] = "lexicon"

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/collect", response_model=CollectResponse)
def collect(query: str = Query(..., min_length=1), source: Literal["twitter","reddit"]="twitter", limit: int = 50):
    if source == "twitter":
        texts = fetch_twitter_posts(query, limit=limit)
    else:
        texts = fetch_reddit_posts(query, limit=limit)
    return {"source": source, "query": query, "limit": limit, "texts": texts}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    mode = req.mode
    texts = clean_texts(req.texts)

    if mode == "ml":
        analyzer = SentimentAnalyzer(
            mode="ml",
            ml_model_path="backend/tfidf_linear_svc.joblib",
            vectorizer_path="backend/tfidf_vectorizer.joblib",
        )
    elif mode == "lstm":
        analyzer = SentimentAnalyzer(
            mode="lstm",
            lstm_model_path="backend/lstm_model.h5",
            lstm_tokenizer_path="backend/tokenizer.pkl",
        )
    else:
        analyzer = SentimentAnalyzer(mode="lexicon")

    preds = analyzer.predict(texts)
    return {"mode": mode, "results": preds}
