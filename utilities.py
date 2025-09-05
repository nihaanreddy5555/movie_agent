import os, json, time, math, random, pathlib, threading
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from dataclasses import dataclass, asdict
import requests
import numpy as np
from rich import print as rprint
from rich.table import Table
from config import *
from light_persistence import *
from tmdb_client import *
import embeddings as emb

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(1, len(a | b))

def mmr(cands: List[Tuple[str, float, np.ndarray]], lambda_=0.7, k=10):
    # cands: [(movie_id, relevance_score, vec)]
    selected = []
    remaining = cands.copy()
    while remaining and len(selected) < k:
        best = None; best_val = -1
        for mid, rel, vec in remaining:
            if not selected:
                val = rel
            else:
                div = max([float(np.dot(vec, sv)) for _, __, sv in selected] + [0.0])
                val = lambda_ * rel - (1 - lambda_) * div
            if val > best_val:
                best_val = val; best = (mid, rel, vec)
        selected.append(best); remaining.remove(best)
    return [mid for mid, _, _ in selected]

def now_ts(): return int(time.time())

def favorites_and_nopes():
    likes = [e["movie_id"] for e in EVENTS if e["action"] == "like"]
    dislikes = [e["movie_id"] for e in EVENTS if e["action"] == "dislike"]
    return likes, dislikes

def build_user_vector(alpha=1.0, beta=0.6) -> np.ndarray:
    emb.ensure_embed_arrays()
    likes, dislikes = favorites_and_nopes()
    like_vecs = [emb.EMBEDS[emb.INDEX[mid]] for mid in likes if mid in emb.INDEX]
    dislike_vecs = [emb.EMBEDS[emb.INDEX[mid]] for mid in dislikes if mid in emb.INDEX]

    if not like_vecs and not dislike_vecs:
        dims = emb.EMBEDS.shape[1] if isinstance(emb.EMBEDS, np.ndarray) and emb.EMBEDS.size else 384
        return np.zeros((dims,), dtype=np.float32)

    pos = np.mean(like_vecs, axis=0) if like_vecs else 0
    neg = np.mean(dislike_vecs, axis=0) if dislike_vecs else 0
    v = alpha * pos - beta * neg
    n = np.linalg.norm(v)
    return (v / n) if n else v

def get_filters() -> Dict[str, Any]:
    return FILTERS or {}

def set_filters(**kwargs) -> None:
    # Accept keys: year_min, year_max, runtime_min, runtime_max, language, genres (list[str])
    for k, v in kwargs.items():
        if v is None: continue
        FILTERS[k] = v
    save_json(CACHE_FILTERS, FILTERS)

def clear_filters() -> None:
    FILTERS.clear()
    save_json(CACHE_FILTERS, FILTERS)

def _match_filters(m: Dict[str, Any], f: Dict[str, Any]) -> bool:
    # Year range
    y = None
    try:
        y = int(m.get("year") or 0)
    except Exception:
        y = None
    ymin = f.get("year_min"); ymax = f.get("year_max")
    if ymin is not None and y is not None and y < int(ymin):
        return False
    if ymax is not None and y is not None and y > int(ymax):
        return False


    # Runtime range
    r = m.get("runtime")
    rmin = f.get("runtime_min"); rmax = f.get("runtime_max")
    if rmin is not None and isinstance(r, (int, float)) and r < int(rmin):
        return False
    if rmax is not None and isinstance(r, (int, float)) and r > int(rmax):
        return False


    # Language exact code match (e.g., 'en', 'hi', 'te')
    lang = normalize_lang_code(f.get("language"))
    if lang:
        movie_lang = (m.get("original_language") or "").lower()
        if movie_lang != lang:
            return False



    # Genre intersection (user-provided genres ⊆ movie genres?)
    want = set([g.lower() for g in (f.get("genres") or [])])
    if want:
        have = set([g.lower() for g in (m.get("genres") or [])])
        if not want.intersection(have):
            return False


    return True

def filter_pool(pool: List[str]) -> List[str]:
    f = get_filters()
    if not f: return pool
    out = []
    for mid in pool:
        m = MOVIES.get(mid)
        if not m: continue
        if _match_filters(m, f):
            out.append(mid)
    return out

def keyword_search(query: str, limit: int = 30) -> List[str]:
    """Case-insensitive substring search over title, overview, and cast."""
    q = (query or "").strip().lower()
    if not q: return []
    hits = []
    for mid, m in MOVIES.items():
        title = (m.get("title") or "").lower()
        ov = (m.get("overview") or "").lower()
        cast = " ".join(m.get("cast") or []).lower()
        hay = " ".join([title, ov, cast])
        if q in hay:
            hits.append(mid)
    # Respect active filters if any
    hits = filter_pool(hits)
    # Order: simple popularity desc (then title)
    hits.sort(key=lambda x: (MOVIES[x].get("popularity", 0.0), MOVIES[x].get("title", "")), reverse=True)
    return hits[:limit]

# --- Language normalization ---
LANGUAGE_ALIASES = {
    "en": "en", "english": "en",
    "hi": "hi", "hindi": "hi",
    "te": "te", "telugu": "te", "తెలుగు": "te",
}

def normalize_lang_code(s: str | None) -> str | None:
    code = (s or "").strip().lower()
    if not code:
        return None
    return LANGUAGE_ALIASES.get(code, code if len(code) == 2 else None)
