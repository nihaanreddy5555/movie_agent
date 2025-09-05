import os, json, time, math, random, pathlib, threading
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from dataclasses import dataclass, asdict
import requests
import numpy as np
from rich import print as rprint
from rich.table import Table

DATA_DIR = pathlib.Path("./data"); DATA_DIR.mkdir(exist_ok=True)
CACHE_MOVIES = DATA_DIR / "movies.json"     # TMDB details cache
CACHE_EVENTS = DATA_DIR / "events.json"     # likes/dislikes/views
CACHE_EMBEDS = DATA_DIR / "embeddings.npy"  # movie vectors
CACHE_INDEX  = DATA_DIR / "embed_index.json" # movie_id -> row index
TMDB_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"

if not TMDB_KEY:
    rprint("[red]ERROR:[/red] Please set TMDB_API_KEY environment variable.")
    raise SystemExit(1)
