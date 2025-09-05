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
import embeddings as emb
from data_ingestion import *
from utilities import *
from scoring import *

def show_list(mids: List[str], title="Recommendations"):
    tbl = Table(title=title)
    tbl.add_column("ID")
    tbl.add_column("Title")
    tbl.add_column("Year")
    tbl.add_column("Genres")
    tbl.add_column("Why / Signal")

    f = get_filters()
    if f and title == "Recommendations":
        title = f"Recommendations (filtered)"
    tbl = Table(title=title)

    user_vec = build_user_vector()
    has_profile = np.linalg.norm(user_vec) > 1e-6

    for mid in mids:
        if mid not in MOVIES:
            continue  # or handle gracefully
        m = MOVIES[mid]
        reason = f"score={base_score(mid, user_vec):.3f}" if has_profile else "seed/popularity"
        tbl.add_row(
            mid,
            m.get("title") or "?",
            str(m.get("year") or ""),
            ", ".join((m.get("genres") or [])[:3]),
            reason,
        )

    rprint(tbl)

def show_movie(mid: str):
    if mid not in MOVIES:
        rprint(f"[yellow]Unknown movie id {mid}[/yellow]"); return
    m = MOVIES[mid]
    rprint(f"[bold]{m['title']}[/bold] ({m['year']})")
    rprint(f"[dim]{', '.join(m['genres'])}[/dim]")
    rprint(m["overview"][:600] + ("..." if len(m["overview"])>600 else ""))
