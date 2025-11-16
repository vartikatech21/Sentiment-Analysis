# Sentiment Analysis: Fixed Project

## How to run

### 1) Install
```
pip install -r requirements.txt
```

### 2) Train models (uses provided Twitter/Reddit CSVs from your original ZIP)
Run from the repository root:
```
python backend/model_train.py
```
This will train:
- TF-IDF + LinearSVC (baseline ML) and save to `backend/tfidf_linear_svc.joblib` and `backend/tfidf_vectorizer.joblib`.
- LSTM (preferred RNN variant) and save to `backend/lstm_model.h5` and `backend/tokenizer.pkl`.

### 3) Run Streamlit dashboard (no separate API needed)
```
streamlit run app.py
```

### 4) Optional: Run FastAPI backend
```
uvicorn backend.main:app --reload --port 8000
```
