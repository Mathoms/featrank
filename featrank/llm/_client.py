"""Thin LLM client abstraction — wraps Groq and Ollama."""

from __future__ import annotations

from loguru import logger

from featrank.config import settings


class _LLMClient:
    def chat(self, system: str, user: str) -> str:
        raise NotImplementedError


class _GroqClient(_LLMClient):
    def __init__(self) -> None:
        from groq import Groq

        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.llm_model

    def chat(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""


class _OllamaClient(_LLMClient):
    def __init__(self) -> None:
        import requests as _requests

        self._requests = _requests
        self._base = settings.ollama_base_url
        self._model = settings.ollama_model

    def chat(self, system: str, user: str) -> str:
        resp = self._requests.post(
            f"{self._base}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


_client_instance: _LLMClient | None = None


def get_llm_client() -> _LLMClient:
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    provider = settings.llm_provider
    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Set it in .env or switch LLM_PROVIDER=ollama"
            )
        logger.info("[llm] Using Groq client")
        _client_instance = _GroqClient()
    elif provider == "ollama":
        logger.info(f"[llm] Using Ollama client ({settings.ollama_base_url})")
        _client_instance = _OllamaClient()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

    return _client_instance
