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
        "runtime": d.get("runtime"),
        "original_language": d.get("original_language"),
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
    # Embeddings will be rebuilt once after seeding.


def fetch_seed_pool(pages=2, size_cap=60) -> List[str]:
    ids = []
    for p in range(1, pages + 1):
        for r in tmdb_trending(page=p):
            ids.append(str(r["id"]))
    ids = list(dict.fromkeys(ids))  # dedupe
    random.shuffle(ids)
    ids = ids[:size_cap]

    for mid in ids:
        cache_movie(int(mid))

    rprint(f"[dim]Cached {len(ids)} movies. Building embeddings…[/dim]")
    emb.rebuild_embeddings(quiet=True)
    rprint("[dim]Done.[/dim]")

    # Build embeddings once, quietly, after caching all movies
    emb.rebuild_embeddings(quiet=True)
    return ids

def fetch_seed_pool_by_language(lang_code: str, pages: int = 2, size_cap: int = 60, **discover_params) -> List[str]:
    """
    Fetch a seed pool using TMDB discover filtered by original language,
    with extra discover params forwarded to the TMDB API.
    
    Args:
        lang_code: TMDB two-letter language code (e.g., "hi" for Hindi).
        pages: number of discover pages to fetch.
        size_cap: maximum number of unique movie IDs to return.
        **discover_params: extra TMDB discover parameters (sort_by, primary_release_date.lte/gte, etc.).

    Returns:
        A list of movie IDs (as strings), shuffled and capped by size_cap.
    """
    ids: List[str] = []
    for p in range(1, pages + 1):
        results = tmdb_discover_by_language(lang_code, page=p, **discover_params)
        for r in results:
            ids.append(str(r["id"]))

    # Deduplicate while preserving order
    ids = list(dict.fromkeys(ids))

    # Shuffle for some randomness
    random.shuffle(ids)

    # Limit the pool size
    ids = ids[:size_cap]

    # Cache metadata locally and rebuild embeddings once
    for mid in ids:
        cache_movie(int(mid))
    emb.rebuild_embeddings(quiet=True)

    return ids
