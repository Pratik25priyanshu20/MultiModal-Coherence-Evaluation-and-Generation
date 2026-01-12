import os

from src.llm.ollama_llm import OllamaLLM


def get_llm():
    """
    Unified LLM selector.
    Currently defaults to Ollama (local, free).
    """
    if os.getenv("USE_OLLAMA", "1") == "1":
        return OllamaLLM(model="qwen2:7b")

    raise RuntimeError(
        "No active LLM backend. Set USE_OLLAMA=1 or add Gemini later."
    )
