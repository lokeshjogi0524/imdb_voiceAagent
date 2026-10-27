"""Minimal fakes of the `chromadb` and `redis` client APIs so integration code paths can be tested offline."""
import sys, types
import numpy as np


def install_fake_chromadb():
    class Col:
        def __init__(self): self.ids, self.emb, self.docs, self.meta = [], None, [], []
        def count(self): return len(self.ids)
        def upsert(self, ids, embeddings, documents, metadatas):
            assert all(isinstance(v, (str, int, float)) for m in metadatas for v in m.values()), "bad metadata"
            self.ids, self.emb, self.docs, self.meta = list(ids), np.array(embeddings, dtype=np.float32), documents, metadatas
        def query(self, query_embeddings, n_results, include):
            q = np.array(query_embeddings[0], dtype=np.float32)
            d = 1 - (self.emb / np.linalg.norm(self.emb, axis=1, keepdims=True)) @ (q / np.linalg.norm(q))
            o = np.argsort(d)[:n_results]
            return {"ids": [[self.ids[i] for i in o]], "distances": [[float(d[i]) for i in o]]}
        def get(self, ids, include):
            return {"ids": ids, "embeddings": np.array([self.emb[self.ids.index(i)] for i in ids])}
    class Client:
        _cols = {}
        def __init__(self, path): self.path = path
        def get_or_create_collection(self, name, metadata=None):
            assert metadata == {"hnsw:space": "cosine"}
            return Client._cols.setdefault(name, Col())
    mod = types.ModuleType("chromadb"); mod.PersistentClient = Client
    sys.modules["chromadb"] = mod


def install_fake_redis():
    class R:
        store = {"kv": {}, "h": {}, "z": {}}
        def ping(self): return True
        def set(self, k, v, ex=None): assert ex is None or isinstance(ex, int); self.store["kv"][k] = v
        def get(self, k): return self.store["kv"].get(k)
        def delete(self, k): self.store["kv"].pop(k, None)
        def hset(self, n, k, v): self.store["h"].setdefault(n, {})[k] = v
        def hgetall(self, n): return dict(self.store["h"].get(n, {}))
        def hdel(self, n, k): self.store["h"].get(n, {}).pop(k, None)
        def zincrby(self, n, amount, value):
            z = self.store["z"].setdefault(n, {}); z[value] = z.get(value, 0) + amount
        def zrevrange(self, n, a, b, withscores=False):
            return sorted(self.store["z"].get(n, {}).items(), key=lambda x: -x[1])[a:b + 1]
    mod = types.ModuleType("redis")
    mod.Redis = types.SimpleNamespace(from_url=lambda url, **kw: R())
    sys.modules["redis"] = mod
