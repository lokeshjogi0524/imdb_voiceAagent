"""Deterministic query engine: executes a structured 'plan' (produced by the LLM) over the
dataframe + vector store. No LLM calls here, so it is fully testable offline."""
from __future__ import annotations
import difflib
import pandas as pd

SORT_COLS = {
    "rating": "IMDB_Rating", "imdb_rating": "IMDB_Rating", "meta_score": "Meta_score",
    "gross": "GrossNum", "votes": "No_of_Votes", "year": "Year", "runtime": "RuntimeMin",
    "title": "Series_Title",
}
SHOW_COLS = ["Series_Title", "Year", "Genre", "Director", "Star1", "IMDB_Rating",
             "Meta_score", "No_of_Votes", "GrossNum"]
SEMANTIC_POOL = 30


def _contains(series: pd.Series, needle: str) -> pd.Series:
    return series.str.contains(needle, case=False, regex=False, na=False)


def _apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if f.get("title"):
        t = f["title"]
        mm = _contains(df["Series_Title"], t)
        if not mm.any():  # fuzzy fallback for typos
            close = difflib.get_close_matches(t, df["Series_Title"].tolist(), n=3, cutoff=0.6)
            mm = df["Series_Title"].isin(close)
        m &= mm
    if f.get("director"):
        m &= _contains(df["Director"], f["director"])
    if f.get("actor"):
        a = f["actor"].lower()
        if f.get("actor_role") == "lead":
            m &= df["Star1"].str.lower().str.contains(a, regex=False)
        else:
            m &= df["Stars"].apply(lambda s: any(a in x.lower() for x in s))
    low = lambda names: [n.lower() for n in names]
    if f.get("genres_any"):
        want = low(f["genres_any"])
        m &= df["Genres"].apply(lambda g: any(x in want for x in low(g)))
    if f.get("genres_all"):
        want = low(f["genres_all"])
        m &= df["Genres"].apply(lambda g: all(w in low(g) for w in want))
    rng = [("year_min", "Year", ">="), ("year_max", "Year", "<="),
           ("min_rating", "IMDB_Rating", ">="), ("max_rating", "IMDB_Rating", "<="),
           ("min_meta", "Meta_score", ">="), ("max_meta", "Meta_score", "<="),
           ("min_gross", "GrossNum", ">="), ("max_gross", "GrossNum", "<="),
           ("min_votes", "No_of_Votes", ">="), ("max_votes", "No_of_Votes", "<="),
           ("min_runtime", "RuntimeMin", ">="), ("max_runtime", "RuntimeMin", "<=")]
    for key, col, op in rng:
        v = f.get(key)
        if v is not None:
            m &= ((df[col] >= v) if op == ">=" else (df[col] <= v)).fillna(False)
    if f.get("certificate"):
        m &= df["Certificate"].fillna("").str.upper() == str(f["certificate"]).upper()
    return df[m]


def _director_aggregate(sub: pd.DataFrame, agg: dict) -> pd.DataFrame:
    thr = agg.get("gross_threshold", 5e8)
    min_count = agg.get("min_count", 2)
    big = sub[sub["GrossNum"] >= thr]
    rows = []
    for d, g in big.groupby("Director"):
        if len(g) >= min_count:
            g = g.sort_values("GrossNum", ascending=False)
            top = g.iloc[0]
            rows.append({"Director": d, "Movies_over_threshold": len(g),
                         "Highest_grossing_movie": top["Series_Title"],
                         "Gross": top["GrossNum"], "_idx": top.name,
                         "All_qualifying": "; ".join(g["Series_Title"])})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Movies_over_threshold", "Gross"], ascending=False)
    return out


def run_plan(df: pd.DataFrame, store, plan: dict) -> dict:
    """Returns {kind, table, total_matches, indices}; `indices` = dataframe row ids of results."""
    filters = plan.get("filters") or {}
    limit = int(plan.get("limit") or 10)
    sub = _apply_filters(df, filters)

    agg = plan.get("aggregate")
    if agg and agg.get("type") == "director_threshold":
        table = _director_aggregate(sub, agg)
        idx = table["_idx"].tolist()[:limit] if not table.empty else []
        return {"kind": "aggregate", "table": table.drop(columns=["_idx"], errors="ignore").head(limit),
                "total_matches": len(table), "indices": idx}

    sem = plan.get("semantic_query")
    sort_by = SORT_COLS.get(str(plan.get("sort_by") or "").lower())
    asc = bool(plan.get("ascending", False))
    kind = "structured"
    if sem and len(sub):
        sims = pd.Series(store.similarity(sem), index=df.index).loc[sub.index]
        sub = sub.assign(Similarity=sims).sort_values("Similarity", ascending=False)
        kind = "semantic"
        if sort_by:  # hybrid: semantic pool, then re-sort by the requested column
            sub = sub.head(SEMANTIC_POOL).sort_values(sort_by, ascending=asc)
        limit = max(limit, SEMANTIC_POOL) if not sort_by else limit
    elif sort_by:
        sub = sub.dropna(subset=[sort_by])
        by, order = [sort_by], [asc]
        if sort_by == "Meta_score":
            by, order = [sort_by, "IMDB_Rating"], [asc, False]
        elif sort_by == "IMDB_Rating":
            by, order = [sort_by, "No_of_Votes"], [asc, False]
        sub = sub.sort_values(by, ascending=order)
    else:
        sub = sub.sort_values(["IMDB_Rating", "No_of_Votes"], ascending=False)

    total = len(sub)
    top = sub.head(limit)
    cols = SHOW_COLS + (["Similarity"] if "Similarity" in top else [])
    if plan.get("need_overview") or kind == "semantic":
        cols = cols + ["Overview"]
    return {"kind": kind, "table": top[cols], "total_matches": total, "indices": top.index.tolist()}


def recommend(df: pd.DataFrame, store, idx: list[int], n: int = 5) -> pd.DataFrame:
    """Movies whose plots are closest to the centroid of the result movies; shows rating & meta."""
    if not idx:
        return pd.DataFrame()
    sims = pd.Series(store.similar_to(idx[:5]), index=df.index).drop(index=idx, errors="ignore")
    cand = df.loc[sims.sort_values(ascending=False).head(n * 3).index].assign(Similarity=sims)
    cand = cand.assign(_score=cand["Similarity"] + 0.03 * (cand["IMDB_Rating"] - 8))  # nudge to better-rated
    cand = cand.sort_values("_score", ascending=False).head(n)
    return cand[["Series_Title", "Year", "Genre", "IMDB_Rating", "Meta_score", "Similarity"]]
