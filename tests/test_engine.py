"""Offline test of the query engine with hand-written plans for the 9 assignment questions.
Run:  python tests/test_engine.py     (no API key needed)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from imdb_agent.data import load_movies
from imdb_agent.vectorstore import VectorStore
from imdb_agent.engine import run_plan, recommend

pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 40)
df = load_movies()
store = VectorStore(df)
print("vector backend:", store.backend, "\n")

PLANS = {
 "1 Matrix release": {"filters": {"title": "The Matrix"}},
 "2 top5 2019 meta": {"filters": {"year_min": 2019, "year_max": 2019}, "sort_by": "meta_score", "limit": 5},
 "3 top7 comedy 2010-20": {"filters": {"genres_any": ["Comedy"], "year_min": 2010, "year_max": 2020}, "sort_by": "rating", "limit": 7},
 "4 horror meta>85 imdb>8": {"filters": {"genres_any": ["Horror"], "min_meta": 85, "min_rating": 8}, "sort_by": "rating"},
 "5 directors 500M x2": {"aggregate": {"type": "director_threshold", "gross_threshold": 5e8, "min_count": 2}},
 "6 1M votes, low gross": {"filters": {"min_votes": 1_000_000}, "sort_by": "gross", "ascending": True, "limit": 10},
 "7 comedy + death": {"filters": {"genres_any": ["Comedy"]}, "semantic_query": "someone dies, death, dead people, funeral, corpse, murder"},
 "8 Spielberg sci-fi": {"filters": {"director": "Steven Spielberg", "genres_any": ["Sci-Fi"]}, "sort_by": "rating", "need_overview": True},
 "9 pre-1990 police": {"filters": {"year_max": 1989}, "semantic_query": "police officers, cops, detective, law enforcement, crime investigation"},
 "Pacino lead >50M >=8": {"filters": {"actor": "Al Pacino", "actor_role": "lead", "min_gross": 5e7, "min_rating": 8}},
 "Pacino any >50M >=8": {"filters": {"actor": "Al Pacino", "actor_role": "any", "min_gross": 5e7, "min_rating": 8}},
}
for name, plan in PLANS.items():
    r = run_plan(df, store, plan)
    t = r["table"].drop(columns=["Overview", "Genre"], errors="ignore").head(8)
    print(f"=== {name}  (matches: {r['total_matches']}) ===")
    print(t.to_string(index=False))
    if name.startswith(("7", "9", "Pacino lead")):
        print("  recs ->", recommend(df, store, r["indices"]).to_string(index=False).replace("\n", "\n         "))
    print()
