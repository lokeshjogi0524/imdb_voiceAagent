"""Load and clean the IMDB Top-1000 dataset."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "imdb_top_1000.csv"


def load_movies(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # One known bad row: Apollo 13 has Released_Year == "PG"
    df.loc[df["Series_Title"] == "Apollo 13", "Released_Year"] = "1995"
    df["Year"] = pd.to_numeric(df["Released_Year"], errors="coerce").astype("Int64")
    df["GrossNum"] = pd.to_numeric(
        df["Gross"].astype("string").str.replace(",", "", regex=False), errors="coerce"
    )
    df["RuntimeMin"] = pd.to_numeric(df["Runtime"].str.extract(r"(\d+)")[0], errors="coerce")
    df["Genres"] = df["Genre"].apply(lambda s: [g.strip() for g in s.split(",")])
    df["Stars"] = df[["Star1", "Star2", "Star3", "Star4"]].values.tolist()
    # Text used for semantic search
    df["SearchText"] = df["Series_Title"] + ". " + df["Genre"] + ". " + df["Overview"]
    return df.reset_index(drop=True)
