import os
import glob
import pickle
import joblib
import pandas as pd
import numpy as np
import nltk
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from backend.preprocess import clean_text

nltk.download("stopwords", quiet=True)

DATA_DIR = Path("data")

# --- helpers -------------------------------------------------
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")

TEXT_CANDIDATES = [
    "text","clean_text","tweet","Tweet","content","SentimentText","review","tweet_text"
]
LABEL_CANDIDATES = [
    "label","target","class","sentiment","Sentiment","category"
]

def _map_label(v):
    # numeric-style
    try:
        iv = int(v)
        if iv in (-1,0,1): return iv
        if iv == 0: return -1     # common mapping: 0=neg
        if iv == 2: return 0      # neutral
        if iv == 4: return 1      # positive (Sentiment140)
        return -1 if iv < 0 else (1 if iv > 0 else 0)
    except Exception:
        pass
    # string-style
    s = str(v).strip().lower()
    if "pos" in s or s == "positive": return 1
    if "neu" in s or s == "neutral":  return 0
    if "neg" in s or s == "negative": return -1
    return 0

def _read_any(path):
    # Try common encodings + separators; allow header or no header
    for enc in ("utf-8","utf-8-sig","latin-1","cp1252"):
        for sep in (",","\t",";","|"):
            # Try with header row
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
                if df.shape[1] >= 2:
                    return df
            except Exception:
                pass
            # Try no header
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python", header=None, on_bad_lines="skip")
                if df.shape[1] >= 2:
                    # fabricate column names
                    df.columns = [f"col_{i}" for i in range(df.shape[1])]
                    return df
            except Exception:
                pass
    raise RuntimeError(f"Failed to read {path}")

def _pick_col(df, candidates):
    cols = list(df.columns)
    # exact or case-insensitive match
    for c in candidates:
        if c in cols: return c
        for col in cols:
            if col.lower() == c.lower():
                return col
    return None

def _infer_text_and_label(df):
    # 1) Sentiment140 special-case
    if set(df.columns) >= {"target", "text"}:
        return "text", "target"

    # 2) Try explicit candidates
    tcol = _pick_col(df, TEXT_CANDIDATES)
    lcol = _pick_col(df, LABEL_CANDIDATES)
    if tcol and lcol:
        return tcol, lcol

    # 3) Heuristics for headerless / unknown schemas
    # Find a "label-like" column: small number of unique values, often ints or small set of strings
    label_like = None
    for col in df.columns:
        try:
            uniq = df[col].dropna().astype(str).str.lower().unique()
            if 1 < len(uniq) <= 6:
                tokens = set(uniq)
                if tokens & {"-1","0","1","2","4","negative","neutral","positive","neg","neu","pos"}:
                    label_like = col
                    break
        except Exception:
            continue

    # Find a "text-like" column: stringy and with large average length
    text_like = None
    best_len = -1
    for col in df.columns:
        try:
            sample = df[col].dropna().astype(str).head(200)
            avg_len = sample.map(len).mean() if not sample.empty else 0
            if avg_len > best_len:
                best_len = avg_len
                text_like = col
        except Exception:
            continue

    # Avoid picking the same column twice
    if text_like == label_like:
        text_like = None

    if text_like and label_like:
        return text_like, label_like
    return None, None

def _load_one(path, max_rows=None):
    df = _read_any(path)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)

    tcol, lcol = _infer_text_and_label(df)
    if not (tcol and lcol):
        raise RuntimeError(f"Could not infer text/label columns in {path}. Columns: {list(df.columns)}")

    out = pd.DataFrame({
        "text": df[tcol].astype(str),
        "label": df[lcol].map(_map_label)
    })
    return out

def load_and_merge():
    frames = []
    for p in DATA_DIR.glob("*.csv"):
        try:
            # Use a higher cap for small files; sample very large files (e.g., Sentiment140)
            max_rows = 400_000 if p.name.lower().startswith(("training.1600000", "sentiment140", "twitter_data")) else None
            part = _load_one(p, max_rows=max_rows)
            frames.append(part)
            print(f"[OK] Loaded {p.name}: {len(part)} rows")
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")

    if not frames:
        raise RuntimeError("No usable CSVs found in data/. Add at least one labeled dataset.")

    data = pd.concat(frames, ignore_index=True)
    data.dropna(subset=["text","label"], inplace=True)

    # Clean text
    data["text"] = data["text"].map(clean_text)
    data = data[data["text"].str.len() > 0]

    # Clip labels to {-1,0,1}
    data["label"] = data["label"].astype(int).clip(-1,1)

    print(f"[OK] Merged dataset size: {len(data)}")
    return data

# --- models --------------------------------------------------

def train_tfidf_linear_svc(data: pd.DataFrame):
    X = data["text"].values
    y = data["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    clf = LinearSVC()
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)
    print("TF-IDF LinearSVC Accuracy:", accuracy_score(y_test, preds))

    joblib.dump(clf, "backend/tfidf_linear_svc.joblib")
    joblib.dump(vectorizer, "backend/tfidf_vectorizer.joblib")
    print("Saved backend/tfidf_linear_svc.joblib and backend/tfidf_vectorizer.joblib")

def train_lstm(data: pd.DataFrame, max_words: int = 30000, max_len: int = 200, epochs: int = 10, batch_size: int = 64):
    X = data["text"].tolist()
    y = data["label"].values

    # encode labels as 0..n-1
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, "backend/label_encoder.joblib")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    X_test_pad  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=max_len)

    num_classes = len(np.unique(y_enc))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes=num_classes)

    model = Sequential([
        Embedding(input_dim=min(max_words, len(tokenizer.word_index)+1), output_dim=128, input_length=max_len),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(
    X_train_pad, y_train_cat,
    validation_data=(X_test_pad, y_test_cat),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# Save training history
    # Save training history
    import pickle
    with open("backend/lstm_history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    print("✅ Saved LSTM training history -> backend/lstm_history.pkl")

    # Evaluate model
    loss, acc = model.evaluate(X_test_pad, y_test_cat, verbose=0)
    print(f"LSTM Accuracy: {acc:.3f}")

    # Save model and tokenizer
    model.save("backend/lstm_model.h5")
    with open("backend/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Saved backend/lstm_model.h5 and backend/tokenizer.pkl")

# --- main ----------------------------------------------------

if __name__ == "__main__":
    data = load_and_merge()
    print("Dataset size:", len(data))
    train_tfidf_linear_svc(data)
    train_lstm(data)
