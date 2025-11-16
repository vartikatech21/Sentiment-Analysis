import re
import nltk
from nltk.corpus import stopwords

# Ensure necessary NLTK data
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", " ", text)        # remove links
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)  # remove mentions
    text = re.sub(r"#", " ", text)               # remove hashtag symbol
    text = re.sub(r"[^a-zA-Z\s]", " ", text)    # keep only letters/space
    text = text.lower()
    text = " ".join([w for w in text.split() if w not in STOPWORDS])
    return text.strip()

def clean_texts(texts):
    return [clean_text(t) for t in texts]
