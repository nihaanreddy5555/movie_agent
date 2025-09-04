# movie_agent.py
# Local, terminal-based movie recommender with /recommend slash command.
# No filters at start (they’re stubbed and can be added later).

import os, json, time, math, random, pathlib, threading
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from dataclasses import dataclass, asdict
import requests
import numpy as np
from rich import print as rprint
from rich.table import Table

# ---------- Config ----------
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

# ---------- Light persistence ----------
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

# ---------- Embeddings ----------
EMBEDS = None           # numpy array [n_items, n_features]
INDEX: Dict[str, int] = load_json(CACHE_INDEX, {})  # movie_id -> row
TFIDF_PATH = DATA_DIR / "tfidf.pkl"

def _get_overview_text(m):
    # fallback to title if overview empty
    txt = (m.get("overview") or "").strip()
    return txt if txt else (m.get("title") or "")

def rebuild_embeddings():
    """Fit/Refit a TF-IDF vectorizer on all movie overviews and rebuild EMBEDS/INDEX."""
    global EMBEDS, INDEX
    corpus_ids = list(MOVIES.keys())
    corpus_txt = [_get_overview_text(MOVIES[mid]) for mid in corpus_ids]

    if not corpus_ids:
        EMBEDS = np.zeros((0, 1), dtype=np.float32)
        INDEX = {}
        np.save(CACHE_EMBEDS, EMBEDS)
        save_json(CACHE_INDEX, INDEX)
        return

    # Fit a new vectorizer every time; for small N this is fine.
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
    X = vectorizer.fit_transform(corpus_txt)  # sparse matrix
    EMBEDS = X.toarray().astype(np.float32)

    # Persist vectorizer for future sessions
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    # Rebuild index map: movie_id -> row index
    INDEX = {mid: i for i, mid in enumerate(corpus_ids)}

    # Save embeddings and index on disk
    np.save(CACHE_EMBEDS, EMBEDS)
    save_json(CACHE_INDEX, INDEX)

def ensure_embed_arrays():
    """Ensure in-memory EMBEDS/INDEX exist. If files missing (first run), build them."""
    global EMBEDS, INDEX
    if EMBEDS is not None and INDEX:
        return
    if CACHE_EMBEDS.exists() and CACHE_INDEX.exists() and TFIDF_PATH.exists():
        try:
            EMBEDS = np.load(CACHE_EMBEDS)
            INDEX = load_json(CACHE_INDEX, {})
            return
        except Exception:
            pass
    # Fallback: rebuild from current MOVIES
    rebuild_embeddings()

def add_embedding(movie_id: str, overview: str):
    """For TF-IDF we just rebuild for the whole catalog so vectors share the same vocab."""
    rebuild_embeddings()

# ---------- TMDB client ----------
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

# ---------- Utilities ----------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(1, len(a | b))

def mmr(cands: List[Tuple[str, float, np.ndarray]], lambda_=0.7, k=10):
    # cands: [(movie_id, relevance_score, vec)]
    selected = []
    remaining = cands.copy()
    while remaining and len(selected) < k:
        best = None; best_val = -1
        for mid, rel, vec in remaining:
            if not selected:
                val = rel
            else:
                div = max([float(np.dot(vec, sv)) for _, __, sv in selected] + [0.0])
                val = lambda_ * rel - (1 - lambda_) * div
            if val > best_val:
                best_val = val; best = (mid, rel, vec)
        selected.append(best); remaining.remove(best)
    return [mid for mid, _, _ in selected]

def now_ts(): return int(time.time())

def favorites_and_nopes():
    likes = [e["movie_id"] for e in EVENTS if e["action"] == "like"]
    dislikes = [e["movie_id"] for e in EVENTS if e["action"] == "dislike"]
    return likes, dislikes

def build_user_vector(alpha=1.0, beta=0.6) -> np.ndarray:
    ensure_embed_arrays()
    likes, dislikes = favorites_and_nopes()
    like_vecs = [EMBEDS[INDEX[mid]] for mid in likes if mid in INDEX]
    dislike_vecs = [EMBEDS[INDEX[mid]] for mid in dislikes if mid in INDEX]
    if not like_vecs and not dislike_vecs:
        return np.zeros((EMBEDS.shape[1] if EMBEDS.size else 384,), dtype=np.float32)
    pos = np.mean(like_vecs, axis=0) if like_vecs else 0
    neg = np.mean(dislike_vecs, axis=0) if dislike_vecs else 0
    v = alpha*pos - beta*neg
    n = np.linalg.norm(v)
    return (v / n) if n else v

# ---------- Data ingestion ----------
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
    add_embedding(key, item["overview"] or item["title"] or "")
    # embed overview
    add_embedding(key, item["overview"] or item["title"] or "")

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

# ---------- Scoring ----------
def base_score(mid: str, user_vec: np.ndarray) -> float:
    # Start simple: pure content similarity (overview embedding) + a tiny popularity prior.
    if mid not in INDEX: return -1e9
    mv = EMBEDS[INDEX[mid]]
    rel = float(np.dot(user_vec, mv))
    pop = MOVIES[mid].get("popularity", 0.0) / 100.0
    return 0.90*rel + 0.10*pop

def recommend(k=10) -> List[str]:
    ensure_embed_arrays()
    pool = list(MOVIES.keys())
    seen = {e["movie_id"] for e in EVENTS}
    pool = [mid for mid in pool if mid not in seen]  # avoid already acted-on
    if not pool:
        pool = fetch_seed_pool(pages=2, size_cap=80)
    user_vec = build_user_vector()
    # If user_vec is zero (no feedback yet), just rank by popularity; then MMR diversify on vectors
    cands = []
    for mid in pool:
        if mid not in INDEX: continue
        mv = EMBEDS[INDEX[mid]]
        if np.linalg.norm(user_vec) < 1e-6:
            rel = MOVIES[mid].get("popularity", 0.0) / 100.0
        else:
            rel = base_score(mid, user_vec)
        cands.append((mid, rel, mv))
    cands.sort(key=lambda x: x[1], reverse=True)
    top = cands[: min(50, len(cands))]
    order = mmr(top, lambda_=0.7, k=min(k, len(top)))
    return order

# ---------- Pretty print ----------
def show_list(mids: List[str], title="Recommendations"):
    tbl = Table(title=title)
    tbl.add_column("ID"); tbl.add_column("Title"); tbl.add_column("Year")
    tbl.add_column("Genres"); tbl.add_column("Why / Signal")
    user_vec = build_user_vector()
    for mid in mids:
        m = MOVIES[mid]
        reason = f"score={base_score(mid, user_vec):.3f}" if np.linalg.norm(user_vec)>1e-6 else "seed/popularity"
        tbl.add_row(mid, m["title"] or "?", str(m["year"]), ", ".join(m["genres"][:3]), reason)
    rprint(tbl)

def show_movie(mid: str):
    if mid not in MOVIES:
        rprint(f"[yellow]Unknown movie id {mid}[/yellow]"); return
    m = MOVIES[mid]
    rprint(f"[bold]{m['title']}[/bold] ({m['year']})")
    rprint(f"[dim]{', '.join(m['genres'])}[/dim]")
    rprint(m["overview"][:600] + ("..." if len(m["overview"])>600 else ""))

# ---------- CLI loop ----------
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

def main():
    rprint("[bold green]Personal Movie Agent[/bold green] — local, simple, no filters yet.")
    rprint("Type /help to see commands.")
    # Prime a bit of data so /recommend can work even before /seed
    if not MOVIES:
        fetch_seed_pool(pages=2, size_cap=60)
    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not cmd: continue
        if not cmd.startswith("/"):
            rprint("[dim]Commands start with '/'. Try /help.[/dim]")
            continue
        parts = cmd.split()
        name, args = parts[0].lower(), parts[1:]

        if name == "/help": print(HELP)
        elif name == "/seed": cmd_seed()
        elif name == "/recommend": cmd_recommend(args)
        elif name == "/like" and args: cmd_like(args[0])
        elif name == "/dislike" and args: cmd_dislike(args[0])
        elif name == "/skip" and args: cmd_skip(args[0])
        elif name == "/why" and args: cmd_why(args[0])
        elif name == "/add-filters":
            rprint("[cyan]Filters are off for now. This is where we’ll add runtime/year/language later.[/cyan]")
        elif name == "/quit": break
        else:
            rprint("[yellow]Unknown command.[/yellow] Try /help.")

if __name__ == "__main__":
    main()
