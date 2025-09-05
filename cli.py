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
from collections import Counter

HELP = """
Commands:
/seed                → fetch trending and rate
/recommend [k]       → show k (default 10) recommendations
/like <id>           → mark a movie as liked (updates your taste vector)
/dislike <id>        → mark a movie as disliked
/skip <id>           → mark as seen/skip (won't be recommended again)
/why <id>            → show details and the score signal
/add-filters         → set runtime/year/language/genre filters
/clear-filters       → remove all active filters
/search <query>      → keyword search (title/overview/cast), obeying filters
/languages           → list language codes currently in cache
/list-lang <code>   → show movies whose original_language matches <code>
/seed-lang <code|name> → fetch trending-by-language and rate (e.g., te / telugu)
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

def _parse_range(s: str) -> Tuple[int | None, int | None]:
    s = (s or "").strip()
    if not s: return None, None
    if "-" in s:
        a, b = s.split("-", 1)
        return (int(a) if a else None, int(b) if b else None)
    v = int(s); return v, v

def cmd_add_filters():
    rprint("[cyan]Add filters — leave blank to skip a field.[/cyan]")
    yr = input("Year range (e.g., 1990-2005 or 2010): ").strip()
    rt = input("Runtime range minutes (e.g., 80-120): ").strip()
    lg = input("Language code (e.g., en, es, hi, te) or name (e.g., telugu): ").strip()
    gs = input("Genres (comma-separated, e.g., Action, Comedy): ").strip()

    y_min, y_max = _parse_range(yr)
    r_min, r_max = _parse_range(rt)
    lg_code = normalize_lang_code(lg)
    genres = [g.strip() for g in gs.split(',') if g.strip()] if gs else []

    set_filters(
        year_min=y_min, year_max=y_max,
        runtime_min=r_min, runtime_max=r_max,
        language=lg_code,              # normalized ('te' if 'telugu')
        genres=genres or None,
    )
    rprint("[green]Filters updated.[/green]")
    rprint(f"Active: {get_filters()}")

def cmd_clear_filters():
    clear_filters(); rprint("[green]Filters cleared.[/green]")

def cmd_search(args: List[str]):
    if not args:
        rprint("[yellow]Usage: /search <query>[/yellow]"); return
    query = " ".join(args)
    ids = keyword_search(query, limit=30)
    if not ids:
        rprint("[yellow]No matches.[/yellow]"); return
    show_list(ids, title=f"Search results for: {query}")

def cmd_languages():
    counts = Counter()
    for m in MOVIES.values():
        code = (m.get("original_language") or "").lower().strip()
        if code:
            counts[code] += 1
    if not counts:
        rprint("[yellow]No languages found. Seed or migrate first.[/yellow]")
        return
    rprint("[bold]Languages in cache (original_language):[/bold]")
    for code, n in counts.most_common():
        rprint(f"{code}: {n}")

def cmd_list_lang(args: List[str]):
    if not args:
        rprint("[yellow]Usage: /list-lang <code>[/yellow]")
        return
    code = args[0].lower().strip()
    ids = [mid for mid, m in MOVIES.items()
           if (m.get("original_language") or "").lower().strip() == code]
    if not ids:
        rprint("[yellow]No matches for that language code.[/yellow]")
        return
    # Sort newest + popular first
    def _year(mid: str) -> int:
        y = (MOVIES[mid].get("year") or "0")[:4]
        return int(y) if y.isdigit() else 0
    ids.sort(key=lambda mid: (_year(mid), MOVIES[mid].get("popularity", 0.0)), reverse=True)
    show_list(ids, title=f"Language: {code}")

def cmd_seed_lang(args: List[str]):
    if not args:
        rprint("[yellow]Usage: /seed-lang <code|name>[/yellow] (e.g., te or telugu)")
        return
    code = normalize_lang_code(args[0])
    if not code:
        rprint("[yellow]Unknown language. Try a two-letter code like 'te' or the name 'telugu'.[/yellow]")
        return
    rprint("[cyan]Fetching a seed pool…[/cyan]")
    ids = fetch_seed_pool_by_language(code, pages=2, size_cap=30)
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
