"""Thin provider abstraction over two FREE-tier LLM APIs: Google Gemini and Groq (Llama 3.3).
Also handles speech-to-text (Gemini audio understanding / Groq Whisper)."""
from __future__ import annotations
import json
import os
import time

DEFAULTS = {
    "gemini": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    "groq": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
}


class LLM:
    def __init__(self, provider: str, api_key: str, model: str | None = None):
        self.provider = provider.lower()
        self.model = model or DEFAULTS[self.provider]
        if not api_key:
            raise ValueError(f"Missing API key for {self.provider}")
        if self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=api_key)
        elif self.provider == "groq":
            from groq import Groq
            self.client = Groq(api_key=api_key)
        else:
            raise ValueError("provider must be 'gemini' or 'groq'")

    # ------------------------------------------------------------------ chat
    def generate(self, system: str, messages: list[dict], json_mode: bool = False,
                 temperature: float = 0.1) -> str:
        """messages: [{'role': 'user'|'assistant', 'content': str}, ...]"""
        last_err = None
        for attempt in range(3):
            try:
                return self._generate(system, messages, json_mode, temperature)
            except Exception as e:  # rate limits on free tiers -> back off and retry
                last_err = e
                if self.provider == "groq" and json_mode and self._is_json_refusal(e):
                    return json.dumps({"thought": "Provider returned a safety refusal.",
                                       "action": "refuse", "args": {}})
                if any(k in str(e).lower() for k in ("429", "rate", "quota", "503", "overloaded")):
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        raise last_err

    @staticmethod
    def _is_json_refusal(error: Exception) -> bool:
        body = getattr(error, "body", None)
        detail = body.get("error") if isinstance(body, dict) else None
        if not isinstance(detail, dict) or detail.get("code") != "json_validate_failed":
            return False
        generation = detail.get("failed_generation")
        if not isinstance(generation, str):
            return False
        text = generation.strip().lower()
        return text.startswith(("i'm sorry", "i am sorry", "i cannot", "i can't", "i’m sorry"))

    def _generate(self, system, messages, json_mode, temperature) -> str:
        if self.provider == "gemini":
            from google.genai import types
            contents = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                      parts=[types.Part(text=m["content"])]) for m in messages]
            cfg = types.GenerateContentConfig(
                system_instruction=system, temperature=temperature,
                response_mime_type="application/json" if json_mode else None)
            return self.client.models.generate_content(
                model=self.model, contents=contents, config=cfg).text or ""
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system}] + messages, **kwargs)
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------ speech→text
    def transcribe(self, audio_bytes: bytes, mime: str = "audio/wav") -> str:
        if self.provider == "gemini":
            from google.genai import types
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                          "Transcribe this audio exactly. Return only the transcript."])
            return (resp.text or "").strip()
        resp = self.client.audio.transcriptions.create(
            file=("question.wav", audio_bytes), model="whisper-large-v3-turbo")
        return resp.text.strip()
