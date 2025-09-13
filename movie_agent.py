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
from config import *
from light_persistence import *
from tmdb_client import *
from embeddings import *
from data_ingestion import *
from pretty_print import *
from cli import *


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
        elif name == "/seed": cmd_seed(args)
        elif name == "/recommend": cmd_recommend(args)
        elif name == "/like" and args: cmd_like(args[0])
        elif name == "/dislike" and args: cmd_dislike(args[0])
        elif name == "/skip" and args: cmd_skip(args[0])
        elif name == "/why" and args: cmd_why(args[0])
        elif name == "/add-filters": cmd_add_filters()
        elif name == "/clear-filters": cmd_clear_filters()
        elif name == "/search": cmd_search(args)
        elif name == "/languages": cmd_languages()
        elif name == "/list-lang": cmd_list_lang(args)
        elif name == "/seed-lang" and args: cmd_seed_lang(args)
        elif name == "/stats": cmd_stats()
        elif name == "/quit": break
        else:
            rprint("[yellow]Unknown command.[/yellow] Try /help.")

if __name__ == "__main__":
    main()
