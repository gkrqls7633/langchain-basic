from __future__ import annotations

import os

from src.domain.llm import LLMProvider
from src.infrastructure.llm.fake_llm import FakeLLM
from src.infrastructure.llm.gemini_llm import GeminiLLM
from src.infrastructure.llm.ollama_llm import OllamaLLM
from src.infrastructure.logger import logger


def build_llm_provider() -> LLMProvider:
    """
    Composition helper: choose one LLM backend for the app.

    Supported:
    - Gemini (LangChain Google GenAI)
    - Local Ollama (LangChain Ollama)
    - Fallback FakeLLM (no keys)
    """
    backend = (os.getenv("LLM_BACKEND") or "").strip().lower()

    has_gemini_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    if backend == "ollama":
        return OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model_name=os.getenv("OLLAMA_MODEL", "lfm2.5-thinking:1.2b"),
        )

    if backend == "gemini" or (backend == "" and has_gemini_key):
        return GeminiLLM(model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

    if backend == "" and not has_gemini_key:
        logger.warning("No Gemini API key found. Falling back to Ollama.")
        return OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model_name=os.getenv("OLLAMA_MODEL", "lfm2.5-thinking:1.2b"),
        )

    logger.warning("Unknown LLM_BACKEND=%r. Falling back to FakeLLM.", backend)
    return FakeLLM()
