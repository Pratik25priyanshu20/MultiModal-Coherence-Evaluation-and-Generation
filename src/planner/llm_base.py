from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from transformers import pipeline

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class LLMConfig:
    model_name: str = "google/flan-t5-base"
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    max_retries: int = 3


class LocalLLM:
    """
    Local HF LLM wrapper. Uses text2text-generation (T5-style instruction models).
    Designed for stable structured JSON output.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.pipe = pipeline(
            "text2text-generation",
            model=self.config.model_name,
        )

    def generate(self, prompt: str) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            num_return_sequences=1,
        )[0]["generated_text"]
        return out.strip()


def extract_json(text: str) -> Optional[str]:
    """
    Extract the first JSON object from model output.
    Handles cases where the model wraps JSON in explanations.
    """
    text = text.strip()

    if "```" in text:
        blocks = re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if blocks:
            return blocks[0].strip()

    match = _JSON_RE.search(text)
    if not match:
        return None
    return match.group(0).strip()


def safe_json_loads(s: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return json.loads(s), None
    except Exception as exc:
        return None, str(exc)


def generate_validated_json(
    llm: LocalLLM,
    prompt: str,
    validator_fn,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Generate JSON with retries.
    validator_fn must raise on invalid JSON or invalid schema.
    """
    last_error = None
    last_raw = None

    for attempt in range(1, max_retries + 1):
        raw = llm.generate(prompt)
        last_raw = raw

        js = extract_json(raw)
        if js is None:
            last_error = f"No JSON found in output (attempt {attempt}). Raw: {raw[:200]}..."
            continue

        data, err = safe_json_loads(js)
        if data is None:
            last_error = f"JSON parse error (attempt {attempt}): {err}"
            continue

        try:
            validator_fn(data)
            return data
        except Exception as exc:
            last_error = f"Schema validation error (attempt {attempt}): {exc}"
            continue

    raise RuntimeError(
        "Failed to generate valid JSON after retries.\n"
        f"Last error: {last_error}\n"
        f"Last raw output: {last_raw}"
    )
