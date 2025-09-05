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
