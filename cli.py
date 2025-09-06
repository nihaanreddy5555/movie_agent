import os, json, time, math, random, pathlib, threading
from typing import List, Dict, Any, Tuple
from datetime import datetime
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
/seed [--no-filters]     → fetch trending and rate (respects active filters unless --no-filters)
/recommend [k]           → show k (default 10) recommendations
/like <id>               → mark a movie as liked (updates your taste vector)
/dislike <id>            → mark a movie as disliked
/skip <id>               → mark as seen/skip (won't be recommended again)
/why <id>                → show details and the score signal
/add-filters             → set runtime/year/language/genre filters
/clear-filters           → remove all active filters
/search <query>          → keyword search (title/overview/cast), obeying filters
/languages               → list language codes currently in cache
/list-lang <code>        → show movies whose original_language matches <code>
/seed-lang <code|name> [asc|desc] [--no-filters]
/help                    → show commands
/quit                    → exit
"""



def log_event(mid: str, action: str):
    EVENTS.append({"movie_id": mid, "ts": now_ts(), "action": action})
    save_json(CACHE_EVENTS, EVENTS)


def cmd_seed(args: List[str]):
    no_filters = any(a.strip().lower() in ("--no-filters", "--nofilters") for a in (args or []))
    rprint("[cyan]Fetching a seed pool…[/cyan]")
    ids = fetch_seed_pool(pages=2, size_cap=30)  # builds/refreshes cache

    # Optionally filter the seed pool
    if not no_filters:
        filtered = filter_pool(ids)  # uses active FILTERS
        if filtered:
            ids = filtered
            rprint("[dim]Applied active filters to seed pool.[/dim]")
        else:
            rprint("[yellow]Your filters excluded all seeded movies. Showing unfiltered seed.[/yellow]")

    rprint("Rate a handful. For each movie, type: [green]l[/green]=like, [red]d[/red]=dislike, [yellow]s[/yellow]=skip, [blue]q[/blue]=stop.")
    for mid in ids:
        m = MOVIES[mid]
        rprint(f"\n[bold]{m['title']}[/bold] ({m['year']})  –  {', '.join(m['genres'])}")
        rprint((m['overview'] or 'No overview.')[:300])
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


# -----------------------------
# Helpers (no duplicates below)
# -----------------------------

def _parse_range(s: str) -> Tuple[int | None, int | None]:
    """Parse 'A-B' or single 'A' as minutes/years. Returns (min, max) possibly None."""
    s = (s or "").strip()
    if not s: return None, None
    if "-" in s:
        a, b = s.split("-", 1)
        return (int(a) if a else None, int(b) if b else None)
    v = int(s); return v, v


def _parse_year_interval_to_bounds(tok: str) -> Tuple[int | None, int | None]:
    """
    Convert compact interval token into integer year bounds:
      'YYYY'       -> (YYYY, YYYY)
      'YYYY-YYYY'  -> (YYYY, YYYY)
      '<=YYYY'     -> (None, YYYY)
      '>=YYYY'     -> (YYYY, None)
    Returns (None, None) if invalid/empty.
    """
    s = (tok or "").strip()
    if not s:
        return None, None
    if s.startswith("<="):
        y = s[2:].strip()
        return (None, int(y)) if y.isdigit() and len(y) == 4 else (None, None)
    if s.startswith(">="):
        y = s[2:].strip()
        return (int(y), None) if y.isdigit() and len(y) == 4 else (None, None)
    if "-" in s:
        a, b = s.split("-", 1)
        a = a.strip(); b = b.strip()
        if a.isdigit() and len(a) == 4 and b.isdigit() and len(b) == 4 and int(a) <= int(b):
            return int(a), int(b)
        return None, None
    if s.isdigit() and len(s) == 4:
        y = int(s)
        return y, y
    return None, None

# -----------------------------
# Filters
# -----------------------------

def cmd_add_filters():
    rprint("[cyan]Add filters — leave blank to remove that constraint.[/cyan]")
    yr = input("Year (YYYY, YYYY-YYYY, <=YYYY, >=YYYY): ").strip()
    rt = input("Runtime range minutes (e.g., 80-120): ").strip()
    lg = input("Language code (e.g., en, es, hi, te) or name (e.g., telugu): ").strip()
    gs = input("Genres (comma-separated, e.g., Action, Comedy): ").strip()

    # --- Year handling via compact interval ---
    if yr:
        y_min, y_max = _parse_year_interval_to_bounds(yr)
        if y_min is not None: FILTERS["year_min"] = int(y_min)
        else: FILTERS.pop("year_min", None)
        if y_max is not None: FILTERS["year_max"] = int(y_max)
        else: FILTERS.pop("year_max", None)
    else:
        FILTERS.pop("year_min", None)
        FILTERS.pop("year_max", None)

    # --- Runtime handling ---
    if rt:
        r_min, r_max = _parse_range(rt)
        if r_min is not None: FILTERS["runtime_min"] = int(r_min)
        else: FILTERS.pop("runtime_min", None)
        if r_max is not None: FILTERS["runtime_max"] = int(r_max)
        else: FILTERS.pop("runtime_max", None)
    else:
        FILTERS.pop("runtime_min", None)
        FILTERS.pop("runtime_max", None)

    # --- Language handling ---
    if lg:
        lg_code = normalize_lang_code(lg)
        if lg_code: FILTERS["language"] = lg_code
        else: FILTERS.pop("language", None)
    else:
        FILTERS.pop("language", None)

    # --- Genres handling ---
    if gs:
        genres = [g.strip() for g in gs.split(',') if g.strip()]
        FILTERS["genres"] = genres
    else:
        FILTERS.pop("genres", None)

    # persist
    save_json(CACHE_FILTERS, FILTERS)
    rprint("[green]Filters updated.[/green]")
    rprint(f"Active: {get_filters()}")


def cmd_clear_filters():
    clear_filters(); rprint("[green]Filters cleared.[/green]")


# -----------------------------
# Search & Language Lists
# -----------------------------

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


# -----------------------------
# Seeding by language (interval-aware)
# -----------------------------

def cmd_seed_lang(args: List[str]):
    """
    Usage:
      /seed-lang <code|name> [asc|desc] [--no-filters]

    Notes:
      • Year interval is NOT accepted here. It is taken from /add-filters (year_min/year_max).
      • If year_min/year_max exist, they are mapped to TMDB discover bounds automatically.
      • Sort: 'asc' → release_date.asc, 'desc' → release_date.desc;
              if year bounds exist and no sort given, default to release_date.asc.
              otherwise TMDB default (popularity.desc) is used.
    """
    if not args:
        rprint("[yellow]Usage: /seed-lang <code|name> [asc|desc] [--no-filters][/yellow]")
        return

    code_raw = args[0]
    rest = [t for t in args[1:] if t]

    # Flags & options
    no_filters = any(t.lower() in ("--no-filters", "--nofilters") for t in rest)
    rest = [t for t in rest if t.lower() not in ("--no-filters", "--nofilters")]

    sort_tok = None
    if rest and rest[0].lower() in ("asc", "desc"):
        sort_tok = rest[0].lower()

    code = normalize_lang_code(code_raw)
    if not code:
        rprint("[yellow]Unknown language. Try a two-letter code like 'hi' or a name like 'hindi'.[/yellow]")
        return

    # Build TMDB discover params from saved /add-filters years
    discover_params = {}
    f = get_filters()
    y_min = f.get("year_min")
    y_max = f.get("year_max")

    if isinstance(y_min, int):
        discover_params["primary_release_date.gte"] = f"{y_min}-01-01"
    if isinstance(y_max, int):
        discover_params["primary_release_date.lte"] = f"{y_max}-12-31"

    # Sorting behavior
    if sort_tok == "asc":
        discover_params["sort_by"] = "release_date.asc"
    elif sort_tok == "desc":
        discover_params["sort_by"] = "release_date.desc"
    else:
        # If year bounds exist but no explicit sort, pick chronological asc
        if ("primary_release_date.gte" in discover_params) or ("primary_release_date.lte" in discover_params):
            discover_params["sort_by"] = "release_date.asc"
        # else: leave TMDB default (popularity.desc)

    rprint("[cyan]Fetching a seed pool…[/cyan]")
    # Use your widened sampling if desired (e.g., pages=10, size_cap=200)
    ids = fetch_seed_pool_by_language(code, pages=10, size_cap=200, **discover_params)

    # Optionally apply local filters (year/runtime/genres/language) on the seed
    if not no_filters:
        filtered = filter_pool(ids)
        if filtered:
            ids = filtered
            rprint("[dim]Applied active filters to language seed pool.[/dim]")
        else:
            rprint("[yellow]Your filters excluded all seeded movies for this language. Showing unfiltered seed.[/yellow]")

    rprint("Rate a handful. For each movie, type: [green]l[/green]=like, [red]d[/red]=dislike, [yellow]s[/yellow]=skip, [blue]q[/blue]=stop.")
    for mid in ids:
        m = MOVIES[mid]
        rprint(f"\n[bold]{m['title']}[/bold] ({m['year']})  –  {', '.join(m.get('genres') or [])}")
        rprint((m.get('overview') or 'No overview.')[:300])
        ans = input("[l/d/s/q] > ").strip().lower()
        if ans == "l": log_event(mid, "like")
        elif ans == "d": log_event(mid, "dislike")
        elif ans == "s": log_event(mid, "skip")
        elif ans == "q": break
