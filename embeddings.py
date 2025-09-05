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

EMBEDS: np.ndarray | None = None           # shape: [n_items, n_features]
INDEX: Dict[str, int] = load_json(CACHE_INDEX, {})  # movie_id -> row
TFIDF_PATH = DATA_DIR / "tfidf.pkl"

def _movie_text(mid: str) -> str:
    m = MOVIES.get(mid, {})
    title = (m.get("title") or "").strip()
    overview = (m.get("overview") or "").strip()
    return f"{title}. {overview}".strip()

def _build_corpus(order: List[str]) -> list[str]:
    return [_movie_text(mid) for mid in order]

def rebuild_embeddings(quiet: bool = False) -> None:
    """Rebuild TF-IDF embeddings from current MOVIES and write caches."""
    global EMBEDS, INDEX

    if not MOVIES:
        # Nothing to embed yet
        EMBEDS = np.zeros((0, 0), dtype=np.float32)
        INDEX = {}
        save_json(CACHE_INDEX, INDEX)
        try:
            # Make sure we don't leave a stale vectorizer
            if TFIDF_PATH.exists():
                TFIDF_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        np.save(CACHE_EMBEDS, EMBEDS)
        if not quiet:
            rprint("[green]Rebuilt embeddings:[/green] items=0 dims=0")
        return

    # Stable order for reproducibility
    order = sorted(MOVIES.keys())
    corpus = _build_corpus(order)

    # Build TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        stop_words="english"
    )
    X = vectorizer.fit_transform(corpus)           # scipy sparse
    M = X.astype(np.float32).toarray()            # dense float32

    EMBEDS = M
    INDEX = {mid: i for i, mid in enumerate(order)}

    # Save caches
    np.save(CACHE_EMBEDS, EMBEDS)
    save_json(CACHE_INDEX, INDEX)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    if not quiet:
        rprint(f"[green]Rebuilt embeddings:[/green] items={len(order)} dims={EMBEDS.shape[1]}")


def ensure_embed_arrays() -> None:
    """Load or rebuild the global EMBEDS/INDEX if needed; always leave EMBEDS as an ndarray."""
    global EMBEDS, INDEX

    # If already loaded and sane, nothing to do
    if isinstance(EMBEDS, np.ndarray):
        return

    # Try loading from cache
    if CACHE_EMBEDS.exists() and CACHE_INDEX.exists() and TFIDF_PATH.exists():
        try:
            EMBEDS = np.load(CACHE_EMBEDS)
            INDEX = load_json(CACHE_INDEX, {})
            if isinstance(EMBEDS, np.ndarray) and isinstance(INDEX, dict):
                return
        except Exception:
            pass  # fall through to rebuild

    # No valid cache → rebuild from MOVIES (may be empty)
    rebuild_embeddings()

def add_embedding(movie_id: str, overview: str) -> None:
    """
    For TF-IDF we rebuild the whole matrix so all rows share the same vocabulary.
    Called after you cache a new/updated movie.
    """
    rebuild_embeddings()