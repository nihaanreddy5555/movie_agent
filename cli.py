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
from embeddings import *
from tmdb_client import *
from data_ingestion import *
from pretty_print import *
from utilities import *
from scoring import *

HELP = """
Commands:
/seed                → fetch trending and let you quickly like/dislike a few
/recommend [k]       → show k (default 10) recommendations
/like <id>           → mark a movie as liked (updates your taste vector)
/dislike <id>        → mark a movie as disliked
/skip <id>           → mark as seen/skip (won't be recommended again)
/why <id>            → show details and the score signal
/add-filters         → (stub) later you can add runtime/year/language filters
/help                → show commands
/quit                → exit
"""

def log_event(mid: str, action: str):
    EVENTS.append({"movie_id": mid, "ts": now_ts(), "action": action})
    save_json(CACHE_EVENTS, EVENTS)

def cmd_seed():
    rprint("[cyan]Fetching a seed pool…[/cyan]")
    ids = fetch_seed_pool(pages=2, size_cap=30)
    rprint("Rate a handful. For each movie, type: [green]l[/green]=like, [red]d[/red]=dislike, [yellow]s[/yellow]=skip, [blue]q[/blue]=stop.")
    for mid in ids:
        m = MOVIES[mid]
        rprint(f"\n[bold]{m['title']}[/bold] ({m['year']})  –  {', '.join(m['genres'])}")
        rprint((m["overview"] or "No overview.")[:300])
        ans = input("[l/d/s/q] > ").strip().lower()
        if ans == "l": log_event(mid, "like")
        elif ans == "d": log_event(mid, "dislike")
        elif ans == "s": log_event(mid, "skip")
        elif ans == "q": break

def cmd_recommend(args: List[str]):
    k = 10
    if args and args[0].isdigit():
        k = int(args[0])
    ids = recommend(k=k)
    if not ids:
        rprint("[yellow]No candidates. Try /seed first.[/yellow]")
        return
    show_list(ids, title="Your recommendations")
    rprint("Tip: /like <id>, /dislike <id>, /skip <id>, /why <id>")

def cmd_like(mid: str): 
    if mid not in MOVIES: rprint("[yellow]Unknown id.[/yellow]"); return
    log_event(mid, "like"); rprint(f"👍 Liked: {MOVIES[mid]['title']}")

def cmd_dislike(mid: str): 
    if mid not in MOVIES: rprint("[yellow]Unknown id.[/yellow]"); return
    log_event(mid, "dislike"); rprint(f"👎 Disliked: {MOVIES[mid]['title']}")

def cmd_skip(mid: str):
    if mid not in MOVIES: rprint("[yellow]Unknown id.[/yellow]"); return
    log_event(mid, "skip"); rprint(f"⏭️ Skipped: {MOVIES[mid]['title']}")

def cmd_why(mid: str):
    if mid not in MOVIES: rprint("[yellow]Unknown id.[/yellow]"); return
    user_vec = build_user_vector()
    show_movie(mid)
    if np.linalg.norm(user_vec) < 1e-6:
        rprint("[dim]Reason: cold-start (popularity & diversification).[/dim]")
        return
    sc = base_score(mid, user_vec)
    rprint(f"[dim]Reason: content similarity vector; score={sc:.3f}[/dim]")
