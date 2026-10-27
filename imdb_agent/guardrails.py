"""Guardrails for the movie agent (all deterministic, no extra LLM calls).

INPUT   : empty/too long, self-harm (supportive reply), prompt-injection, harmful-how-to, abuse, PII redaction
PLAN    : tool arguments from the LLM are whitelisted, type-coerced and clamped before touching the data
OUTPUT  : grounding check (answer may only name movies that were actually retrieved), secret redaction,
          system-prompt-leak check, deterministic fallback answer
RATE    : per-session sliding-window limiter
"""
from __future__ import annotations
import re
import time
from collections import deque
from dataclasses import dataclass, field

GENRES = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Drama", "Family", "Fantasy",
          "Film-Noir", "History", "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi", "Sport",
          "Thriller", "War", "Western"]
_GENRE_MAP = {g.lower(): g for g in GENRES} | {"scifi": "Sci-Fi", "sci fi": "Sci-Fi", "science fiction": "Sci-Fi",
                                               "film noir": "Film-Noir", "romcom": "Romance"}
SORTS = {"rating", "imdb_rating", "meta_score", "gross", "votes", "year", "runtime"}

MAX_CHARS = 500
SCOPE_MESSAGE = ("I can only help with questions about the IMDB Top-1000 movies dataset — release years, rankings, "
                 "genres, directors, actors, plots and recommendations. Try e.g. “Top 5 movies of 2019 by meta score”.")

INJECTION = [
    r"ignore\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules|messages)",
    r"disregard\s+.{0,40}(instructions|rules|guidelines|guardrails)",
    r"(reveal|show|print|repeat|display|leak|output)\s+.{0,30}(system\s+prompt|your\s+(prompt|instructions|rules))",
    r"what\s+(is|are)\s+your\s+(system\s+prompt|instructions)",
    r"you\s+are\s+now\s+(?!a\s+movie)", r"\bjailbreak\b", r"\bdeveloper\s+mode\b", r"\bDAN\s+mode\b",
    r"pretend\s+(to\s+be|you\s+are|that\s+you)", r"</?\s*(system|assistant)\s*>", r"\[/?INST\]",
    r"override\s+.{0,30}(rules|guardrails|safety|restrictions)", r"\bbypass\b.{0,30}(filter|guardrail|safety)",
    r"(api[_\s-]?key|secret|password)s?\b.{0,20}(show|give|tell|reveal|print)",
]
SELF_HARM = [r"\b(i\s+want\s+to\s+(die|kill\s+myself)|kill\s+myself|end\s+my\s+life|end\s+it\s+all|suicidal)\b"]
HARMFUL = [r"\bhow\s+(do|can|to|would)\s+(i\s+|you\s+|we\s+)?(make|build|create|synthesi[sz]e|cook)\s+(a\s+|an\s+)?"
           r"(bomb|explosives?|nerve\s+agent|meth\b|methamphetamine|ransomware|malware)",
           r"\bhack\s+into\b", r"\b(steal|crack)\s+(a\s+)?passwords?\b"]
ABUSE = [r"\b(fuck|fucking|shit|bitch|asshole|cunt|bastard)\b"]
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in
             dict(injection=INJECTION, self_harm=SELF_HARM, harmful=HARMFUL, abuse=ABUSE).items()}

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+?\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SECRET = re.compile(r"(AIza[0-9A-Za-z_-]{20,}|gsk_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})")
_LEAK = re.compile(r"(query planner|agentic rag controller|json schema|system prompt|AGENT_SYSTEM)", re.I)


def _luhn(num: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", num)][::-1]
    return 13 <= len(d) <= 19 and sum(x if i % 2 == 0 else (x * 2 - 9 if x * 2 > 9 else x * 2)
                                      for i, x in enumerate(d)) % 10 == 0


@dataclass
class InputCheck:
    allowed: bool
    text: str
    reason: str | None = None
    message: str | None = None
    flags: list[str] = field(default_factory=list)


class Guardrails:
    # ---------------------------------------------------------------- input
    def check_input(self, text: str) -> InputCheck:
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text or "").strip()
        if not text:
            return InputCheck(False, "", "empty", "I didn't catch that — could you ask again?")
        if len(text) > MAX_CHARS:
            return InputCheck(False, text[:80], "too_long",
                              f"That's a bit long ({len(text)} characters). Please keep questions under {MAX_CHARS}.")
        for p in _COMPILED["self_harm"]:
            if p.search(text):
                return InputCheck(False, text, "self_harm",
                                  "I'm really sorry you're feeling this way. I'm just a movie assistant, but you "
                                  "don't have to go through this alone — please reach out to someone you trust or a "
                                  "local crisis line or emergency service right now.")
        for p in _COMPILED["injection"]:
            if p.search(text):
                return InputCheck(False, text, "prompt_injection",
                                  "I can't change or reveal my instructions. " + SCOPE_MESSAGE)
        for p in _COMPILED["harmful"]:
            if p.search(text):
                return InputCheck(False, text, "harmful_request", "I can't help with that. " + SCOPE_MESSAGE)
        for p in _COMPILED["abuse"]:
            if p.search(text):
                return InputCheck(False, text, "abusive_language",
                                  "Let's keep it friendly 🙂 Ask me anything about movies and I'll gladly help.")
        flags = []
        clean = text
        for name, rx in (("email", _EMAIL), ("phone", _PHONE), ("ssn", _SSN)):
            if rx.search(clean):
                clean = rx.sub(f"[{name.upper()} REDACTED]", clean)
                flags.append(f"pii_redacted:{name}")
        clean2 = _CARD.sub(lambda m: "[CARD REDACTED]" if _luhn(m.group()) else m.group(), clean)
        if clean2 != clean:
            flags.append("pii_redacted:card")
        return InputCheck(True, clean2, flags=flags)

    # ---------------------------------------------------------- tool arguments
    @staticmethod
    def _num(v, lo, hi, cast=float):
        try:
            x = cast(float(str(v).replace(",", "")))
        except (TypeError, ValueError):
            return None
        return min(max(x, lo), hi)

    @staticmethod
    def _str(v, n=100):
        return re.sub(r"\s+", " ", v).strip()[:n] if isinstance(v, str) and v.strip() else None

    def sanitize_search_args(self, args) -> tuple[dict, list[str]]:
        """Whitelist + clamp what the LLM wants to run. Returns (clean_plan, warnings)."""
        a = args if isinstance(args, dict) else {}
        f_in = a.get("filters") if isinstance(a.get("filters"), dict) else {}
        warns: list[str] = []
        f: dict = {}
        for k in ("title", "director", "actor"):
            if (s := self._str(f_in.get(k))):
                f[k] = s
        if f_in.get("actor_role") in ("lead", "any"):
            f["actor_role"] = f_in["actor_role"]
        for key in ("genres_any", "genres_all"):
            vals = f_in.get(key)
            if isinstance(vals, str):
                vals = [vals]
            if isinstance(vals, list):
                ok = [_GENRE_MAP[str(v).strip().lower()] for v in vals if str(v).strip().lower() in _GENRE_MAP]
                bad = [v for v in vals if str(v).strip().lower() not in _GENRE_MAP]
                if bad:
                    warns.append(f"dropped unknown genres {bad}")
                if ok:
                    f[key] = ok
        for key, lo, hi, cast in [("year_min", 1880, 2030, int), ("year_max", 1880, 2030, int),
                                  ("min_rating", 0, 10, float), ("max_rating", 0, 10, float),
                                  ("min_meta", 0, 100, float), ("max_meta", 0, 100, float),
                                  ("min_gross", 0, 1e10, float), ("max_gross", 0, 1e10, float),
                                  ("min_votes", 0, 1e8, int), ("max_votes", 0, 1e8, int),
                                  ("min_runtime", 0, 600, int), ("max_runtime", 0, 600, int)]:
            if (x := self._num(f_in.get(key), lo, hi, cast)) is not None:
                f[key] = x
        if (c := self._str(f_in.get("certificate"), 10)):
            f["certificate"] = c
        plan: dict = {"filters": f}
        sb = str(a.get("sort_by") or "").lower()
        if sb in SORTS:
            plan["sort_by"] = sb
        elif sb:
            warns.append(f"ignored sort_by={sb!r}")
        plan["ascending"] = bool(a.get("ascending", False))
        plan["limit"] = self._num(a.get("limit"), 1, 30, int) or 10
        if (sq := self._str(a.get("semantic_query"), 300)):
            plan["semantic_query"] = sq
        plan["need_overview"] = bool(a.get("need_overview", False))
        ag = a.get("aggregate")
        if isinstance(ag, dict) and ag.get("type") == "director_threshold":
            plan["aggregate"] = {"type": "director_threshold",
                                 "gross_threshold": self._num(ag.get("gross_threshold"), 0, 1e10) or 5e8,
                                 "min_count": self._num(ag.get("min_count"), 1, 10, int) or 2}
        return plan, warns

    # ---------------------------------------------------------------- output
    @staticmethod
    def ungrounded_titles(answer: str, allowed: set[str], all_titles: list[str], question: str = "") -> list[str]:
        """Dataset titles named in the answer that were NOT retrieved this turn (= possible hallucination)."""
        allowed_l = {t.lower() for t in allowed}
        text = answer
        for t in sorted(allowed, key=len, reverse=True):  # blank out legit mentions (and their substrings)
            text = text.replace(t, " ")
        bad = []
        for t in all_titles:
            if len(t) < 4 or t.lower() in allowed_l or t in question:
                continue
            if any(t in a for a in allowed):  # e.g. 'The Godfather' inside 'The Godfather: Part II'
                continue
            if re.search(rf"(?<![\w]){re.escape(t)}(?![\w])", text):
                bad.append(t)
        return bad

    @staticmethod
    def scrub_output(text: str) -> tuple[str, list[str]]:
        flags = []
        if _SECRET.search(text):
            text = _SECRET.sub("[REDACTED]", text)
            flags.append("secret_redacted")
        if _LEAK.search(text):
            text = SCOPE_MESSAGE
            flags.append("prompt_leak_blocked")
        return text, flags

    @staticmethod
    def fallback_answer(table, n: int = 5) -> str:
        """Deterministic, always-grounded answer built straight from retrieved rows."""
        if table is None or len(table) == 0:
            return "I couldn't find any movies in the dataset matching that."
        cols = list(table.columns)
        tcol = "Series_Title" if "Series_Title" in cols else cols[0]
        parts = []
        for _, r in table.head(n).iterrows():
            extra = []
            if "Year" in cols:
                extra.append(str(r["Year"]))
            if "IMDB_Rating" in cols:
                extra.append(f"IMDB {r['IMDB_Rating']}")
            parts.append(f"{r[tcol]} ({', '.join(extra)})" if extra else str(r[tcol]))
        return f"I found {len(table)} matching result(s). Top ones: " + "; ".join(parts) + "."


class RateLimiter:
    def __init__(self, max_calls: int = 10, per_seconds: int = 60):
        self.max, self.per, self.calls = max_calls, per_seconds, deque()

    def allow(self) -> bool:
        now = time.time()
        while self.calls and now - self.calls[0] > self.per:
            self.calls.popleft()
        if len(self.calls) >= self.max:
            return False
        self.calls.append(now)
        return True
