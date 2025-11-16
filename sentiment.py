from typing import List, Union
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Download resources
nltk.download("vader_lexicon", quiet=True)

class SentimentAnalyzer:
    def __init__(self, mode: str = "lexicon",
                 ml_model_path: str | None = None,
                 vectorizer_path: str | None = None,
                 lstm_model_path: str | None = None,
                 lstm_tokenizer_path: str | None = None):
        self.mode = mode.lower()
        self.analyzer = None
        self.model = None
        self.vectorizer = None
        self.lstm_model = None
        self.lstm_tokenizer = None

        if self.mode == "lexicon":
            self.analyzer = SentimentIntensityAnalyzer()

        elif self.mode == "ml":
            if not (ml_model_path and vectorizer_path):
                raise ValueError("For ML mode, provide ml_model_path and vectorizer_path")
            self.model = joblib.load(ml_model_path)
            self.vectorizer = joblib.load(vectorizer_path)

        elif self.mode == "lstm":
            from tensorflow.keras.models import load_model
            import pickle
            if not (lstm_model_path and lstm_tokenizer_path):
                raise ValueError("For LSTM mode, provide lstm_model_path and lstm_tokenizer_path")
            self.lstm_model = load_model(lstm_model_path)
            with open(lstm_tokenizer_path, "rb") as f:
                self.lstm_tokenizer = pickle.load(f)
        else:
            raise ValueError("mode must be one of: 'lexicon', 'ml', 'lstm'")

    def predict(self, texts: Union[str, List[str]]) -> List[str]:
        if isinstance(texts, str):
            texts = [texts]

        if self.mode == "lexicon":
            out = []
            for t in texts:
                scores = self.analyzer.polarity_scores(t)
                c = scores["compound"]
                if c >= 0.05:
                    out.append("positive")
                elif c <= -0.05:
                    out.append("negative")
                else:
                    out.append("neutral")
            return out

        elif self.mode == "ml":
            X = self.vectorizer.transform(texts)
            preds = self.model.predict(X)
            # Ensure strings
            return [str(p) for p in preds]

        elif self.mode == "lstm":
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seqs = self.lstm_tokenizer.texts_to_sequences(texts)
            X = pad_sequences(seqs, maxlen=200)
            probs = self.lstm_model.predict(X, verbose=0)
            idxs = probs.argmax(axis=1)
            # Map indices back to labels saved alongside the ML model for consistency
            # Expect joblib label encoder at 'backend/label_encoder.joblib' if trained via provided script
            try:
                le = joblib.load("backend/label_encoder.joblib")
                labels = le.inverse_transform(idxs)
                return [str(l) for l in labels]
            except Exception:
                # Fallback to common mapping {-1,0,1}
                mapping = {0: "negative", 1: "neutral", 2: "positive"}
                return [mapping.get(int(i), "neutral") for i in idxs]
