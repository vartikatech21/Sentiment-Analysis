import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# ---- STEP 3: load cookies as soon as the app starts ----
import backend.twitter_cookies as _tc
_tc.ensure_cookies()
# --------------------------------------------------------
st.caption(f"Cookie file: {os.getenv('SNSCRAPE_TWITTER_COOKIES_FILE') or 'None'}")

from backend.collector import fetch_twitter_posts, fetch_reddit_posts
from backend.preprocess import clean_texts
from backend.sentiment import SentimentAnalyzer

st.set_page_config(page_title="Dynamic Sentiment Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dynamic Sentiment Analysis Dashboard")

with st.sidebar:
    st.header("Settings")
    query = st.text_input("Brand / Query", value="Apple")
    source = st.selectbox("Source", ["twitter", "reddit"])
    # 👇 Fixed limit = 5 posts only
    limit = 5
    st.caption("Fetching limited to 5 posts for faster demo/testing.")
    mode = st.selectbox("Model", ["lexicon", "ml", "lstm"])
    run = st.button("Run Analysis")

placeholder_info = st.empty()
col1, col2 = st.columns([2, 1])

def analyze(texts, mode):
    texts_clean = clean_texts(texts)
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
    preds = analyzer.predict(texts_clean)
    return texts_clean, preds

def draw_wordcloud(texts, title):
    text = " ".join(texts)
    if not text.strip():
        return
    wc = WordCloud(width=800, height=300).generate(text)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

if run:
    q = f"{query} lang:en"

    with st.spinner("Collecting posts (max 5)..."):
        try:
            if source == "twitter":
                raw = fetch_twitter_posts(q, limit=limit)
            else:
                raw = fetch_reddit_posts(query, limit=limit)
        except Exception as e:
            st.warning(
                "Unable to fetch posts right now.\n\n"
                "Try again later or switch the source.\n\n"
                f"Details: {e}"
            )
            st.stop()

    if not raw:
        st.warning("No posts found. Try another query.")
    else:
        texts_clean, preds = analyze(raw, mode)

        counts = Counter(preds)
        total = sum(counts.values())
        pos = counts.get("positive", 0) + counts.get("1", 0)
        neu = counts.get("neutral", 0) + counts.get("0", 0)
        neg = counts.get("negative", 0) + counts.get("-1", 0)

        placeholder_info.info(
            f"Found {total} posts for **{query}** from **{source}** using **{mode}** model."
        )

        with col1:
            df = pd.DataFrame({"text": raw, "clean": texts_clean, "sentiment": preds})
            st.dataframe(df.head(5))  # show only 5 posts

            st.subheader("Sentiment Distribution")
            chart_df = pd.DataFrame(
                {"Sentiment": ["negative", "neutral", "positive"], "Count": [neg, neu, pos]}
            )
            chart = alt.Chart(chart_df).mark_bar().encode(
                x="Sentiment:N", y="Count:Q", tooltip=["Sentiment", "Count"]
            )
            st.altair_chart(chart, use_container_width=True)

            st.subheader("Examples")
            st.write("**Most Positive**")
            pos_examples = [t for t, s in zip(raw, preds) if s in ("positive", "1")]
            neg_examples = [t for t, s in zip(raw, preds) if s in ("negative", "-1")]
            st.write("- " + "\n- ".join(pos_examples[:2]) if pos_examples else "_None_")
            st.write("**Most Negative**")
            st.write("- " + "\n- ".join(neg_examples[:2]) if neg_examples else "_None_")

        with col2:
            st.subheader("Wordcloud (All)")
            draw_wordcloud(texts_clean, "All")

            st.subheader("Wordcloud (Positive)")
            draw_wordcloud([t for t, s in zip(texts_clean, preds) if s in ("positive", "1")], "Positive")

            st.subheader("Wordcloud (Negative)")
            draw_wordcloud([t for t, s in zip(texts_clean, preds) if s in ("negative", "-1")], "Negative")

else:
    st.info("Enter a brand/query, pick a source and model, then click **Run Analysis**.")
