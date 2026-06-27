"""
LLM layer using Groq (primary).
Falls back gracefully with clear error messages.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from time import sleep

import streamlit as st

from config import GROQ_API_KEY, GROQ_MODEL, LLM_MAX_TOKENS, LLM_RETRIES, LLM_TIMEOUT_SECONDS


@dataclass
class LLMResult:
    ok: bool
    text: str


@st.cache_resource(show_spinner=False)
def load_groq_client():
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        return Groq(api_key=GROQ_API_KEY)
    except ImportError:
        return None


class GroqAgentMixin:
    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = load_groq_client()
        return self._client

    def generate(self, prompt: str, system: str = "", fallback: str = "") -> LLMResult:
        if self.client is None:
            return LLMResult(False, "GROQ_API_KEY is missing. Add it to .env and restart.")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error = ""
        for attempt in range(LLM_RETRIES + 1):
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                self.client.chat.completions.create,
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.7,
            )
            try:
                response = future.result(timeout=LLM_TIMEOUT_SECONDS)
                text = response.choices[0].message.content or fallback
                return LLMResult(True, text.strip())
            except TimeoutError:
                future.cancel()
                last_error = f"LLM request timed out after {LLM_TIMEOUT_SECONDS}s."
            except Exception as exc:
                last_error = f"LLM request failed: {exc}"
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            if attempt < LLM_RETRIES:
                sleep(1.5 * (attempt + 1))

        return LLMResult(False, last_error or fallback)

    def stream_generate(self, prompt: str, system: str = ""):
        """Yields text chunks for streaming responses."""
        if self.client is None:
            yield "GROQ_API_KEY is missing."
            return

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.7,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            yield f"\n\n[Error: {exc}]"
