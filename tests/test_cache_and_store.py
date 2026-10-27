import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, tempfile
import fakes
fakes.install_fake_chromadb(); fakes.install_fake_redis()
from imdb_agent.data import load_movies
from imdb_agent.vectorstore import VectorStore
from imdb_agent.cache import QACache, make_kv, MemoryKV, signature

df = load_movies()
# ---- Chroma path vs numpy path must agree
vs = VectorStore(df, persist_dir=tempfile.mkdtemp())
assert vs.collection is not None and "ChromaDB" in vs.db, vs.db
s_chroma = vs.similarity("police officers, detective, law enforcement")
s_np = vs.encode("police officers, detective, law enforcement") @ vs._doc_embeddings(df.SearchText.tolist()).T
assert np.allclose(s_chroma, s_np, atol=1e-4)
assert np.allclose(vs.similar_to([0, 1]).shape, (len(df),))
print("chroma store ok:", vs.db, "| top:", df.Series_Title[int(np.argmax(s_chroma))])

# ---- cache over (fake) Redis and in-memory
resp = {"answer": "Parasite is #1", "reasoning": "r", "suggestions": "s", "plan": {"a": 1},
        "table": pd.DataFrame({"Title": ["Parasite"], "Meta": [96.0]}), "recs": None}
for kv, label, is_r in [(*make_kv()[:1], "redis", True), (MemoryKV(), "mem", False)]:
    c = QACache(kv, label, is_r)
    q = "What are the top 5 movies of 2019 by meta score?"
    assert c.get(q, vs.encode) is None
    c.put(q, resp, vs.encode)
    p, sc, _ = c.get(q, vs.encode); assert sc == 1.0 and p["table"].Title[0] == "Parasite"
    p, sc, m = c.get("what are the top 5 movies of 2019 by meta score", vs.encode); assert sc == 1.0   # normalised exact
    assert c.get("What are the top 5 movies of 2018 by meta score?", vs.encode) is None      # number differs
    assert c.get("What are the top 7 movies of 2019 by meta score?", vs.encode) is None
    assert c.get("Top 5 horror movies of 2019 by meta score?", vs.encode) is None            # genre/qualifier differs
    print(label, "cache ok | popular:", c.popular(3))
assert signature("Top 7 comedy movies between 2010-2020") != signature("Top 7 horror movies between 2010-2020")
print("ALL CACHE/STORE TESTS PASSED")
