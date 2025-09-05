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
from data_ingestion import *
from utilities import *

def base_score(mid: str, user_vec: np.ndarray) -> float:
    if mid not in emb.INDEX: return -1e9
    mv = emb.EMBEDS[emb.INDEX[mid]]
    rel = float(np.dot(user_vec, mv))
    pop = MOVIES[mid].get("popularity", 0.0) / 100.0
    return 0.90*rel + 0.10*pop

def recommend(k=10) -> List[str]:
    emb.ensure_embed_arrays()
    pool = list(MOVIES.keys())
    seen = {e["movie_id"] for e in EVENTS}
    pool = [mid for mid in pool if mid not in seen]
    pool = filter_pool(pool)
    if not pool:
        pool = fetch_seed_pool(pages=2, size_cap=80)
    user_vec = build_user_vector()
    cands = []
    for mid in pool:
        if mid not in emb.INDEX: continue
        mv = emb.EMBEDS[emb.INDEX[mid]]
        if np.linalg.norm(user_vec) < 1e-6:
            rel = MOVIES[mid].get("popularity", 0.0) / 100.0
        else:
            rel = base_score(mid, user_vec)
        cands.append((mid, rel, mv))
    cands.sort(key=lambda x: x[1], reverse=True)
    top = cands[: min(50, len(cands))]
    order = mmr(top, lambda_=0.7, k=min(k, len(top)))
    return order