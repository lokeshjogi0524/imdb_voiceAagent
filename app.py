"""Streamlit voice + chat UI — Agentic RAG over IMDB Top-1000 (ChromaDB + Redis + guardrails).
Run:  streamlit run app.py"""
import hashlib
import io
import os

import streamlit as st
from dotenv import load_dotenv

from imdb_agent.agent import MovieAgent
from imdb_agent.cache import QACache, make_kv
from imdb_agent.data import load_movies
from imdb_agent.guardrails import Guardrails, RateLimiter
from imdb_agent.llm import DEFAULTS, LLM
from imdb_agent.vectorstore import VectorStore

load_dotenv()
st.set_page_config(page_title="IMDB Voice Agent", page_icon="🎬", layout="wide")

SAMPLES = [
    "When did The Matrix release?",
    "What are the top 5 movies of 2019 by meta score?",
    "Top 7 comedy movies between 2010-2020 by IMDB rating?",
    "Top horror movies with a meta score above 85 and IMDB rating above 8",
    "Top directors and their highest grossing movies with gross earnings of greater than 500M at least twice",
    "Top 10 movies with over 1M votes but lower gross earnings",
    "List of movies from the comedy genre where there is death or dead people involved",
    "Summarize the movie plots of Steven Spielberg's top-rated sci-fi movies",
    "List of movies before 1990 that have involvement of police in the plot",
    "List of Al Pacino movies that have grossed over $50M and IMDB rating of 8 or higher",
]
FLAG_LABELS = {"grounding_ok": "✅ answer grounded in retrieved data", "pii_redacted": "🔒 personal data redacted",
               "cached_to_redis": "💾 saved to Redis cache", "cached_in_memory": "💾 saved to in-memory cache",
               "input_blocked": "🛡️ blocked by input guardrail", "off_topic_refused": "🛡️ off-topic request refused",
               "arg_sanitised": "🧹 tool arguments sanitised", "grounding_fallback": "🛡️ hallucination caught → safe answer",
               "step_budget_exhausted": "⏱️ agent step budget reached"}


@st.cache_resource(show_spinner="Loading dataset & vector DB (first run indexes into ChromaDB)…")
def get_data():
    df = load_movies()
    return df, VectorStore(df)


@st.cache_resource(show_spinner=False)
def get_cache():
    kv, label, is_redis = make_kv()
    return QACache(kv, label, is_redis)


def speak(text: str) -> bytes | None:
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def flag_badges(flags):
    seen = []
    for f in flags:
        label = FLAG_LABELS.get(f.split(":")[0])
        if label and label not in seen:
            seen.append(label)
    return " · ".join(seen)


df, store = get_data()
cache = get_cache()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("limiter", RateLimiter(max_calls=10, per_seconds=60))
st.session_state.setdefault("mic_key", 0)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎬 IMDB Voice Agent")
    provider = st.selectbox("Free LLM provider", ["gemini", "groq"],
                            index=0 if os.getenv("LLM_PROVIDER", "gemini") == "gemini" else 1)
    env_key = os.getenv("GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY", "")
    api_key = st.text_input(f"{provider.title()} API key", value=env_key, type="password",
                            help="Free key: aistudio.google.com/apikey  |  console.groq.com/keys")
    model = st.text_input("Model", value=DEFAULTS[provider])
    tts_on = st.toggle("🔊 Speak answers", value=True)
    use_cache = st.toggle("⚡ Use Redis answer cache", value=True)
    st.divider()
    st.subheader("🎙️ Voice input")
    audio = st.audio_input("Click, ask, stop", key=f"mic_{st.session_state.mic_key}")
    st.divider()
    with st.expander("🧱 Infrastructure", expanded=False):
        st.markdown(f"**Vector DB:** {store.db}\n\n**Embeddings:** {store.backend}\n\n**Cache:** {cache.label}\n\n"
                    f"**Guardrails:** input · tool-args · grounding · rate-limit")
    pop = cache.popular(6)
    if pop:
        st.subheader("🔥 Popular (from cache)")
        for i, (pq, hits) in enumerate(pop):
            if st.button(f"{pq[:70]}  ·  {hits}×", use_container_width=True, key=f"pop_{i}"):
                st.session_state.pending = pq
    st.subheader("Try a sample question")
    for q in SAMPLES:
        if st.button(q, use_container_width=True, key=f"s_{q}"):
            st.session_state.pending = q
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.caption(f"Agentic RAG · {len(df)} movies · {store.db.split(' · ')[0]} · {cache.label.split(' (')[0]} · "
           f"LLM: {provider}/{model}")

# ------------------------------------------------------------- input handling
user_text = st.chat_input("Ask about movies (or use the mic in the sidebar)…")
if st.session_state.get("pending"):
    user_text, st.session_state.pending = st.session_state.pending, None

if not api_key and not (user_text and cache.popular(1)):
    st.info("👈 Add a free Gemini or Groq API key in the sidebar (or in a .env file) to start.")
    st.stop()
llm = LLM(provider, api_key, model) if api_key else None

if audio is not None and not user_text and llm:
    raw = audio.getvalue()
    h = hashlib.md5(raw).hexdigest()
    if h != st.session_state.get("last_audio"):
        st.session_state.last_audio = h
        with st.spinner("Transcribing…"):
            try:
                user_text = llm.transcribe(raw, "audio/wav")
            except Exception as e:
                st.error(f"Transcription failed: {e}")

if user_text and llm:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": user_text})
    if not st.session_state.limiter.allow():
        r = {"answer": "You're asking very quickly — please wait a few seconds and try again.", "table": None,
             "recs": None, "reasoning": None, "suggestions": None, "plan": None, "flags": ["rate_limited"],
             "trace": [], "cache": None, "status": "blocked", "clean_question": user_text}
    else:
        with st.spinner("Agent is thinking…"):
            try:
                r = MovieAgent(llm, df, store, cache).ask(user_text, history, use_cache=use_cache)
            except Exception as e:
                r = {"answer": f"Sorry, something went wrong: {e}", "table": None, "recs": None, "reasoning": None,
                     "suggestions": None, "plan": None, "flags": [], "trace": [], "cache": None,
                     "status": "error", "clean_question": user_text}
    st.session_state.messages[-1]["content"] = r.get("clean_question") or user_text  # PII-redacted text
    r["role"], r["content"] = "assistant", r["answer"]
    r["audio"] = speak(r["answer"]) if tts_on else None
    r["autoplay"] = True
    st.session_state.messages.append(r)
    st.session_state.mic_key += 1
    st.rerun()

# ------------------------------------------------------------------ render chat
if not st.session_state.messages:
    st.markdown("### Ask me anything about the IMDB Top 1000 🍿\nUse the **mic** or the **chat box**, "
                "or pick a sample question from the sidebar.")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] != "assistant":
            continue
        if m.get("cache"):
            st.caption(f"⚡ Served from Redis cache (match {m['cache']['score']:.0%}) — no LLM call needed")
        if m.get("audio"):
            st.audio(m["audio"], format="audio/mp3", autoplay=m.pop("autoplay", False))
        if m.get("table") is not None and len(m["table"]):
            st.dataframe(m["table"], hide_index=True, use_container_width=True)
        if m.get("reasoning"):
            st.markdown(f"**🧠 Reasoning:** {m['reasoning']}")
        if m.get("suggestions"):
            st.markdown(f"**💡 Suggestions:** {m['suggestions']}")
        if m.get("recs") is not None:
            with st.expander("🎞️ Similar movies you might like (with IMDB & Meta scores)"):
                st.dataframe(m["recs"], hide_index=True, use_container_width=True)
        if m.get("flags") and (b := flag_badges(m["flags"])):
            st.caption(b)
        if m.get("trace"):
            with st.expander(f"🤖 Agent trace ({len(m['trace'])} steps)"):
                for t in m["trace"]:
                    st.markdown(f"**{t['step']}. `{t['action']}`** — {t.get('thought', '')}")
                    if t.get("args"):
                        st.code(t["args"], language="json")
                    if t.get("obs"):
                        st.caption(f"↳ {t['obs']}")
