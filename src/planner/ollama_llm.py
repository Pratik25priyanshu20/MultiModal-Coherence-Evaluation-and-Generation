from __future__ import annotations

import json
import requests
import time
from typing import Any, Dict

from src.utils.json_repair import try_repair_json


class OllamaPlannerLLM:
    def __init__(
        self,
        model: str = "qwen2:7b",
        base_url: str = "http://localhost:11434",
        max_retries: int = 3,
    ):
        self.model = model
        self.url = f"{base_url}/api/generate"
        self.max_retries = max_retries
        print(f"Using Ollama model: {self.model}")

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generate JSON with retry logic for incomplete/truncated responses.
        """
        last_error = None
        last_raw = None

        for attempt in range(1, self.max_retries + 1):
            # Increase num_predict for longer responses (especially on retries)
            num_predict = 2000 if attempt == 1 else 3000
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": num_predict,
                    "temperature": 0.2,  # Lower temperature for more consistent JSON
                },
            }

            try:
                resp = requests.post(self.url, json=payload, timeout=180)
                resp.raise_for_status()
            except requests.RequestException as e:
                last_error = f"Request failed (attempt {attempt}): {e}"
                if attempt < self.max_retries:
                    time.sleep(1)  # Brief delay before retry
                    continue
                raise RuntimeError(f"Failed to get response from Ollama after {self.max_retries} attempts: {e}")

            raw_text = resp.json().get("response", "").strip()
            last_raw = raw_text

            # Try to repair and parse JSON
            data = try_repair_json(raw_text)

            if data is not None:
                return data

            # If repair failed, check if response looks truncated
            if attempt < self.max_retries:
                # Check for common truncation patterns
                if raw_text and (not raw_text.rstrip().endswith("}") or 
                                raw_text.count("{") > raw_text.count("}")):
                    last_error = f"JSON appears truncated (attempt {attempt}). Retrying with longer context..."
                    time.sleep(0.5)  # Brief delay before retry
                    continue

            last_error = f"Could not recover JSON (attempt {attempt})"

        # All retries failed
        raise ValueError(
            f"[OllamaPlannerLLM] Could not recover JSON after {self.max_retries} attempts.\n"
            f"Last error: {last_error}\n"
            f"Last raw output (first 800 chars):\n{last_raw[:800] if last_raw else 'None'}"
        )
