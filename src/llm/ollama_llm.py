from __future__ import annotations

import json
import requests
from typing import Any, Dict

from src.utils.json_repair import try_repair_json


class OllamaLLM:
    def __init__(self, model: str = "qwen2:7b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        wrapped_prompt = (
            "You are a JSON generation engine.\n"
            "You MUST return ONLY valid JSON.\n"
            "NO explanations. NO markdown. NO comments.\n\n"
            + prompt
        )

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": wrapped_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                },
            },
            timeout=120,
        )

        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()

        data = try_repair_json(raw_text)
        if data is None:
            raise ValueError(
                "[OllamaLLM] Could not recover JSON:\n"
                + raw_text[:500]
            )

        return data
