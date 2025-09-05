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


def load_json(p: pathlib.Path, default):
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except: pass
    return default

def save_json(p: pathlib.Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

MOVIES: Dict[str, Any] = load_json(CACHE_MOVIES, {})       # movie_id -> dict
EVENTS: List[Dict[str, Any]] = load_json(CACHE_EVENTS, [])  # list of {movie_id, ts, action}
EMBEDS = None   # numpy array
INDEX: Dict[str, int] = load_json(CACHE_INDEX, {})  # movie_id -> row