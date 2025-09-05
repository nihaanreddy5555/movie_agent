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

def tmdb(path, **params):
    params["api_key"] = TMDB_KEY
    url = f"{TMDB_BASE}/{path}"
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()

def tmdb_trending(page=1):
    return tmdb("trending/movie/week", page=page)["results"]

def tmdb_movie(mid: int):
    return tmdb(f"movie/{mid}", append_to_response="credits,keywords,release_dates")
