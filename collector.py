# backend/collector.py
from __future__ import annotations
from typing import List
from pathlib import Path
import os, json, time, hashlib, asyncio

# Load .env (for TW_BEARER_TOKEN and Reddit creds)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import snscrape.modules.twitter as sntwitter
import snscrape.modules.reddit as snreddit

from .twitter_cookies import ensure_cookies
ensure_cookies()  # sets SNSCRAPE_TWITTER_COOKIES_FILE if a cookies txt exists (used only by snscrape fallback)

# ---------------- cache ----------------
CACHE_DIR = Path("data/.cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _ckey(source: str, query: str, limit: int) -> Path:
    h = hashlib.sha1(f"{source}|{query}|{limit}".encode()).hexdigest()
    return CACHE_DIR / f"{source}.{h}.json"

def _load_cache(p: Path):
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except:
            return None

def _save_cache(p: Path, data: list[str]):
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def _dedupe_trim(texts: list[str], n: int) -> list[str]:
    seen, out = set(), []
    for t in texts:
        t = (t or "").strip()
        if t and t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= n: break
    return out

# ---------------- helpers ----------------
def _ensure_lang_and_filters(query: str) -> str:
    q = query.strip()
    if "lang:" not in q:
        q += " lang:en"
    # prefer to remove pure retweets from sentiment
    if "-is:retweet" not in q and "is:retweet" not in q:
        q += " -is:retweet"
    return q

# ---------------- Twitter via Tweepy (primary / official API) ----------------
def _fetch_twitter_tweepy(query: str, limit: int) -> list[str]:
    """
    Uses Twitter API v2 via Tweepy to fetch recent tweets.
    Requires TW_BEARER_TOKEN in environment/.env.
    """
    try:
        import tweepy
    except Exception as e:
        raise RuntimeError("Tweepy not installed. Run: pip install tweepy") from e

    bearer = os.getenv("TW_BEARER_TOKEN")
    if not bearer:
        raise RuntimeError("Missing TW_BEARER_TOKEN in environment/.env")

    client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)

    q = _ensure_lang_and_filters(query)
    max_results = 100 if limit > 100 else max(10, limit)
    texts: list[str] = []

    try:
        paginator = tweepy.Paginator(
            client.search_recent_tweets,
            query=q,
            tweet_fields=["created_at", "lang", "public_metrics"],
            max_results=max_results,
        )

        for page in paginator:
            if not page or not page.data:
                break
            for tw in page.data:
                # Keep English if you left lang:en in query; otherwise allow all
                txt = getattr(tw, "text", None)
                if txt:
                    texts.append(txt)
                    if len(texts) >= limit:
                        return texts
            time.sleep(0.2)  # small courtesy pause; wait_on_rate_limit handles 429s
    except tweepy.TooManyRequests as e:
        # API quota hit
        raise RuntimeError(f"Tweepy rate limited: {e}")
    except tweepy.TweepyException as e:
        raise RuntimeError(f"Tweepy error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected Tweepy error: {e}")

    return texts

# ---------------- Twitter via snscrape (fallback) ----------------
def _fetch_twitter_snscrape(query: str, limit: int, max_wait: int = 20) -> list[str]:
    start = time.time()
    out: list[str] = []
    it = sntwitter.TwitterSearchScraper(query).get_items()
    try:
        for tw in it:
            if tw.content and len(tw.content.strip()) > 3:
                out.append(tw.content)
                if len(out) >= limit:
                    break
            if not out and (time.time() - start) > max_wait:
                raise RuntimeError("snscrape timed out (no results) — likely blocked/429.")
    except Exception as e:
        raise RuntimeError(f"snscrape failed: {e}")
    return out

# ---------------- Public: Twitter ----------------
def fetch_twitter_posts(query: str, limit: int = 50) -> List[str]:
    """
    Primary: Tweepy (official API).
    Fallback: snscrape (no API).
    Caches results on success.
    """
    # normalize query and clamp limit
    query = _ensure_lang_and_filters(query)
    limit = max(5, min(limit, 50))

    # cache
    cpath = _ckey("twitter", query, limit)
    cached = _load_cache(cpath)
    if cached:
        return cached[:limit]

    # 1) Tweepy (official)
    tw_err = None
    try:
        texts = _fetch_twitter_tweepy(query, limit)
        texts = _dedupe_trim(texts, limit)
        if texts:
            _save_cache(cpath, texts)
            return texts
    except Exception as e1:
        tw_err = e1

    # 2) snscrape (fallback)
    sn_err = None
    try:
        texts = _fetch_twitter_snscrape(query, limit)
        texts = _dedupe_trim(texts, limit)
        if texts:
            _save_cache(cpath, texts)
            return texts
    except Exception as e2:
        sn_err = e2

    raise RuntimeError(f"Twitter fetch failed. tweepy: {tw_err} | snscrape: {sn_err}")

# ---------------- Reddit via PRAW (official API) ----------------
def _fetch_reddit_praw(query: str, limit: int) -> list[str]:
    import praw
    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    uag = os.getenv("REDDIT_USER_AGENT", "sentiment-dashboard/0.1")

    if not (cid and csec):
        raise RuntimeError("Missing Reddit API credentials (REDDIT_CLIENT_ID/SECRET).")

    reddit = praw.Reddit(client_id=cid, client_secret=csec, user_agent=uag, check_for_async=False)

    out: list[str] = []
    for s in reddit.subreddit("all").search(query, sort="new", limit=limit):
        title = getattr(s, "title", "") or ""
        out: list[str] = []
    for s in reddit.subreddit("all").search(query, sort="new", limit=limit):
        title = getattr(s, "title", "") or ""
        body  = getattr(s, "selftext", "") or ""
        t = f"{title} {body}".strip()
        if len(t) > 3:
            out.append(t)
        if len(out) >= limit:
            break
    return out

# ---------------- Reddit via snscrape (fallback) ----------------
def _fetch_reddit_snscrape(query: str, limit: int) -> list[str]:
    posts: list[str] = []
    for i, sub in enumerate(snreddit.RedditSearchScraper(query).get_items()):
        if i >= limit:
            break
        title = getattr(sub, "title", "") or ""
        body  = getattr(sub, "selftext", "") or ""
        t = f"{title} {body}".strip()
        if len(t) > 3:
            posts.append(t)
    return posts

# ---------------- Public: Reddit ----------------
def fetch_reddit_posts(query: str, limit: int = 50) -> List[str]:
    cpath = _ckey("reddit", query, limit)
    cached = _load_cache(cpath)
    if cached:
        return cached[:limit]

    api_err = None
    try:
        posts = _fetch_reddit_praw(query, limit)
        posts = _dedupe_trim(posts, limit)
        if posts:
            _save_cache(cpath, posts)
            return posts
    except Exception as e_api:
        api_err = e_api

    scr_err = None
    try:
        posts = _fetch_reddit_snscrape(query, limit)
        posts = _dedupe_trim(posts, limit)
        if posts:
            _save_cache(cpath, posts)
            return posts
    except Exception as e_scr:
        scr_err = e_scr

    raise RuntimeError(f"Reddit fetch failed. praw: {api_err} | snscrape: {scr_err}")

