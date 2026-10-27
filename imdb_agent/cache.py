"""Redis-backed question/answer cache with *semantic* matching.

* exact match   : normalised question -> key
* semantic match: cosine similarity between question embeddings (same encoder as the vector DB),
                  accepted only if the question 'signatures' agree (numbers, genres, proper nouns, direction words)
                  so "top 5 comedies of 2015" never reuses the answer for "top 7 comedies of 2016".
* popularity    : a Redis sorted set counts hits, so the UI can show the most-asked questions.
If Redis isn't reachable the app transparently falls back to an in-process dict with the same behaviour.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
import re
import time

import numpy as np
import pandas as pd

from .guardrails import GENRES

PREFIX = os.getenv("CACHE_PREFIX", "imdb:v1:")
TTL = int(os.getenv("CACHE_TTL_SECONDS", 24 * 3600))
THRESHOLD = float(os.getenv("CACHE_SIM_THRESHOLD", 0.92))
_QUALIFIERS = {"before", "after", "above", "below", "over", "under", "less", "more", "lower", "higher", "not",
               "without", "worst", "lowest", "highest", "best", "top", "bottom", "lead", "least", "most",
               "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twice"}
_GENRE_WORDS = {g.lower() for g in GENRES} | {"scifi", "sci-fi"}


# ------------------------------------------------------------------ KV back-ends
class MemoryKV:
    """Tiny in-process stand-in for Redis (same method names as RedisKV)."""
    def __init__(self):
        self._d, self._exp, self._h, self._z = {}, {}, {}, {}

    def ping(self): return True

    def set(self, k, v, ttl=None):
        self._d[k] = v
        self._exp[k] = time.time() + ttl if ttl else None

    def get(self, k):
        if k in self._d and self._exp.get(k) and self._exp[k] < time.time():
            self._d.pop(k, None)
        return self._d.get(k)

    def delete(self, k): self._d.pop(k, None)
    def hset(self, name, k, v): self._h.setdefault(name, {})[k] = v
    def hgetall(self, name): return dict(self._h.get(name, {}))
    def hdel(self, name, k): self._h.get(name, {}).pop(k, None)
    def zincr(self, name, member): self._z.setdefault(name, {}); self._z[name][member] = self._z[name].get(member, 0) + 1
    def ztop(self, name, n): return sorted(self._z.get(name, {}).items(), key=lambda x: -x[1])[:n]


class RedisKV:
    def __init__(self, url: str):
        import redis
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
        self.r.ping()

    def ping(self): return self.r.ping()
    def set(self, k, v, ttl=None): self.r.set(k, v, ex=ttl)
    def get(self, k): return self.r.get(k)
    def delete(self, k): self.r.delete(k)
    def hset(self, name, k, v): self.r.hset(name, k, v)
    def hgetall(self, name): return self.r.hgetall(name)
    def hdel(self, name, k): self.r.hdel(name, k)
    def zincr(self, name, member): self.r.zincrby(name, 1, member)
    def ztop(self, name, n): return self.r.zrevrange(name, 0, n - 1, withscores=True)


def make_kv(url: str | None = None):
    url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        return RedisKV(url), f"Redis ({url})", True
    except Exception as e:
        return MemoryKV(), f"in-memory fallback — Redis unreachable ({type(e).__name__})", False


# --------------------------------------------------------------- (de)serialisation
def pack(resp: dict) -> str:
    j = lambda t: t.to_json(orient="split") if isinstance(t, pd.DataFrame) else None
    return json.dumps({"answer": resp["answer"], "reasoning": resp.get("reasoning"),
                       "suggestions": resp.get("suggestions"), "plan": resp.get("plan"),
                       "table": j(resp.get("table")), "recs": j(resp.get("recs"))})


def unpack(s: str) -> dict:
    d = json.loads(s)
    for k in ("table", "recs"):
        d[k] = pd.read_json(io.StringIO(d[k]), orient="split") if d.get(k) else None
    return d


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower()).strip(" ?!.")


def signature(q: str) -> list[str]:
    nums = [n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", q)]
    words = re.findall(r"[a-z\-]+", q.lower())
    first = re.match(r"\W*(\S+)", q)
    caps = [w.lower() for w in re.findall(r"\b[A-Z][a-z]+", q)
            if not (first and w == first.group(1).strip("?,.'\""))]
    sig = nums + [w for w in words if w in _QUALIFIERS or w in _GENRE_WORDS] + caps
    return sorted(set(sig))


class QACache:
    def __init__(self, kv, label: str, is_redis: bool, threshold: float = THRESHOLD, ttl: int = TTL):
        self.kv, self.label, self.is_redis, self.threshold, self.ttl = kv, label, is_redis, threshold, ttl
        self.ans, self.index, self.pop = PREFIX + "ans:", PREFIX + "index", PREFIX + "popular"

    @staticmethod
    def _key(q: str) -> str:
        return hashlib.sha1(_norm(q).encode()).hexdigest()[:16]

    def get(self, question: str, encode) -> tuple[dict, float, str] | None:
        key = self._key(question)
        if (raw := self.kv.get(self.ans + key)):
            self.kv.zincr(self.pop, key)
            return unpack(raw), 1.0, question
        entries = self.kv.hgetall(self.index)
        if not entries:
            return None
        qv, sig = encode(question), signature(question)
        best = None
        for k, v in entries.items():
            e = json.loads(v)
            if e["sig"] != sig:
                continue
            s = float(np.dot(qv, np.asarray(e["emb"], dtype=np.float32)))
            if s >= self.threshold and (best is None or s > best[0]):
                best = (s, k, e["q"])
        if best:
            raw = self.kv.get(self.ans + best[1])
            if raw is None:  # answer expired -> drop stale index entry
                self.kv.hdel(self.index, best[1])
                return None
            self.kv.zincr(self.pop, best[1])
            return unpack(raw), best[0], best[2]
        return None

    def put(self, question: str, resp: dict, encode) -> None:
        key = self._key(question)
        self.kv.set(self.ans + key, pack(resp), self.ttl)
        self.kv.hset(self.index, key, json.dumps({"q": question, "sig": signature(question),
                                                  "emb": [round(float(x), 5) for x in encode(question)]}))
        self.kv.zincr(self.pop, key)

    def popular(self, n: int = 6) -> list[tuple[str, int]]:
        idx = self.kv.hgetall(self.index)
        out = []
        for key, score in self.kv.ztop(self.pop, n):
            if key in idx and self.kv.get(self.ans + key):
                out.append((json.loads(idx[key])["q"], int(score)))
        return out
