"""Vector database layer.

Primary store  : ChromaDB (open source, embedded, persisted on disk in data/chroma_db/).
                 Each movie = one record: embedding + document (title/genre/overview) + metadata.
Embeddings     : sentence-transformers `all-MiniLM-L6-v2` (free, local).
Fallbacks      : TF-IDF+LSA embeddings if the transformer model can't be loaded;
                 an in-memory numpy index if chromadb isn't installed (so the app never hard-fails).
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma_db"
NPY_CACHE = ROOT / "data" / "embeddings_minilm.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _metadata(df: pd.DataFrame) -> list[dict]:
    """Chroma metadata: scalar values only, no NaN/None."""
    out = []
    for r in df.itertuples():
        m = {"title": r.Series_Title, "genre": r.Genre, "director": r.Director, "star1": r.Star1,
             "year": int(r.Year), "imdb_rating": float(r.IMDB_Rating), "votes": int(r.No_of_Votes)}
        if pd.notna(r.Meta_score):
            m["meta_score"] = float(r.Meta_score)
        if pd.notna(r.GrossNum):
            m["gross"] = float(r.GrossNum)
        out.append(m)
    return out


class VectorStore:
    def __init__(self, df: pd.DataFrame, persist_dir: Path = CHROMA_DIR):
        self.n = len(df)
        texts = df["SearchText"].tolist()
        self._init_encoder(texts)
        self.collection = None
        self.emb = None
        self.db = "numpy in-memory (chromadb not installed)"
        try:
            import chromadb

            persist_dir = Path(persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_dir))
            name = "imdb_movies_" + re.sub(r"[^a-z0-9]+", "_", self.backend.lower()).strip("_")[:40]
            col = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
            if col.count() != self.n:  # first run (or dataset changed): (re)index
                col.upsert(ids=[str(i) for i in range(self.n)],
                           embeddings=self._doc_embeddings(texts).tolist(),
                           documents=texts, metadatas=_metadata(df))
            self.collection = col
            self.db = f"ChromaDB · {col.count()} vectors · {persist_dir.name}/"
        except ImportError:
            self.emb = self._doc_embeddings(texts)
        except Exception as e:  # corrupted db etc. -> keep working in memory
            self.emb = self._doc_embeddings(texts)
            self.db = f"numpy in-memory (Chroma error: {type(e).__name__})"

    # ---------------------------------------------------------------- encoders
    def _init_encoder(self, texts: list[str]):
        self._doc_cache = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_NAME)
            self.backend = f"sentence-transformers/{MODEL_NAME}"
        except Exception:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
            X = self._tfidf.fit_transform(texts)
            self._svd = TruncatedSVD(n_components=150, random_state=0)
            self._doc_cache = _norm(self._svd.fit_transform(X))
            self.backend = "tfidf-lsa (fallback)"

    def _doc_embeddings(self, texts: list[str]) -> np.ndarray:
        if self._doc_cache is not None:
            return self._doc_cache
        if NPY_CACHE.exists() and np.load(NPY_CACHE).shape[0] == len(texts):
            emb = np.load(NPY_CACHE)
        else:
            emb = self._model.encode(texts, batch_size=64, show_progress_bar=False)
            np.save(NPY_CACHE, emb)
        self._doc_cache = _norm(emb)
        return self._doc_cache

    def encode(self, text: str) -> np.ndarray:
        if self.backend.startswith("sentence"):
            v = self._model.encode([text])
        else:
            v = self._svd.transform(self._tfidf.transform([text]))
        return _norm(v)[0]

    # ------------------------------------------------------------------ search
    def _scores(self, vec: np.ndarray) -> np.ndarray:
        """Cosine similarity of `vec` to every movie, aligned with dataframe row order."""
        if self.collection is None:
            return self.emb @ vec
        res = self.collection.query(query_embeddings=[vec.tolist()], n_results=self.n, include=["distances"])
        sims = np.zeros(self.n, dtype=np.float32)
        for id_, dist in zip(res["ids"][0], res["distances"][0]):
            sims[int(id_)] = 1.0 - dist  # cosine distance -> similarity
        return sims

    def similarity(self, query: str) -> np.ndarray:
        return self._scores(self.encode(query))

    def similar_to(self, idx: list[int]) -> np.ndarray:
        """Similarity of every movie to the centroid of the given movies' vectors."""
        if self.collection is None:
            vecs = self.emb[idx]
        else:
            got = self.collection.get(ids=[str(i) for i in idx], include=["embeddings"])
            vecs = np.asarray(got["embeddings"], dtype=np.float32)
        return self._scores(_norm(vecs.mean(axis=0))[0])
