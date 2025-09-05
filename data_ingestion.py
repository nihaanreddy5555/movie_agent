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

def cache_movie(mid: int):
    key = str(mid)
    if key in MOVIES: return
    d = tmdb_movie(mid)
    item = {
        "id": str(d["id"]),
        "title": d.get("title") or d.get("original_title"),
        "year": (d.get("release_date") or "????")[:4],
        "overview": d.get("overview") or "",
        "genres": [g["name"] for g in d.get("genres", [])],
        "popularity": d.get("popularity", 0.0),
        "cast": [c["name"] for c in (d.get("credits", {}).get("cast", [])[:10])],
        "crew": [c["name"] for c in (d.get("credits", {}).get("crew", [])[:10])],
        "poster_path": d.get("poster_path"),
    }
    MOVIES[key] = item
    save_json(CACHE_MOVIES, MOVIES)
    emb.add_embedding(key, item["overview"] or item["title"] or "")
    # embed overview
    emb.add_embedding(key, item["overview"] or item["title"] or "")

def fetch_seed_pool(pages=2, size_cap=60) -> List[str]:
    ids = []
    for p in range(1, pages+1):
        for r in tmdb_trending(page=p):
            ids.append(str(r["id"]))
    ids = list(dict.fromkeys(ids))  # dedupe
    random.shuffle(ids)
    ids = ids[:size_cap]
    for mid in ids: cache_movie(int(mid))
    return ids
