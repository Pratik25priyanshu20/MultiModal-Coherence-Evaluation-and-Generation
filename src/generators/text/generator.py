from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import json
import re
import urllib.request


@dataclass(frozen=True)
class TextGenResult:
    text: str


def _sanitize_text(s: str) -> str:
    """Remove common failure patterns (echoing rules, bullets, repetitions)."""
    s = s.strip()

    # Remove markdown/bullets
    s = re.sub(r"^\s*[-*•]\s+", "", s, flags=re.MULTILINE)

    # Remove obvious meta/instruction echoes
    bad_patterns = [
        r"(?i)\blength\s*:\s*\d+\s*[-–]\s*\d+\s*sentences\b.*",
        r"(?i)\brules\s*:\b.*",
        r"(?i)\bno bullet points\b.*",
        r"(?i)\bno repetition\b.*",
        r"(?i)\bno meta commentary\b.*",
        r"(?i)\bdescribe only\b.*",
    ]
    for pat in bad_patterns:
        s = re.sub(pat, "", s).strip()

    # Collapse whitespace
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)

    # If the model repeated the same line many times, de-dup consecutive duplicates
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    deduped = []
    for ln in lines:
        if not deduped or deduped[-1] != ln:
            deduped.append(ln)
    s = "\n".join(deduped).strip()

    return s


def _ollama_generate(
    prompt: str,
    model: str = "qwen2:7b",
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_predict: int = 180,
    host: str = "http://localhost:11434",
) -> str:
    """
    Calls Ollama local server: POST /api/generate
    """
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except Exception as e:
        raise RuntimeError(
            f"Ollama call failed. Is Ollama running on {host}? Error: {e}"
        ) from e


class TextGenerator:
    """
    Option A (recommended): Ollama text generator (instruction-following).
    Falls back to HF pipeline if use_ollama=False.
    """

    def __init__(
        self,
        use_ollama: bool = True,
        ollama_model: str = "qwen2:7b",
        ollama_host: str = "http://localhost:11434",
        max_new_tokens: int = 160,
        hf_model_name: str = "gpt2",
    ):
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.max_new_tokens = max_new_tokens
        self.hf_model_name = hf_model_name

        self._hf_pipe = None
        if not self.use_ollama:
            from transformers import pipeline

            self._hf_pipe = pipeline("text-generation", model=self.hf_model_name)

    def generate(self, prompt: str, deterministic: bool = True) -> TextGenResult:
        # This is the IMPORTANT part: we wrap your plan_text with strict generation rules.
        wrapped_prompt = """You are a concise descriptive writer.

Write a literal description of the same scene. Follow these rules:
- Write 3 to 5 natural sentences.
- No bullet points, no numbered lists.
- No repetition.
- No meta commentary (do not mention rules, prompts, or constraints).
- Focus on concrete visual details AND the likely audio ambience.

SCENE PLAN:
"""
        wrapped_prompt = f"{wrapped_prompt}{prompt}\n\nNow write the description:\n"

        if self.use_ollama:
            raw = _ollama_generate(
                prompt=wrapped_prompt,
                model=self.ollama_model,
                host=self.ollama_host,
                temperature=0.0 if deterministic else 0.7,
                top_p=1.0 if deterministic else 0.9,
                num_predict=max(self.max_new_tokens, 120),
            )
            clean = _sanitize_text(raw)

            # Last safety: if it comes out empty, return raw (better than nothing)
            return TextGenResult(text=clean if clean else raw)

        # HF fallback
        outputs = self._hf_pipe(
            wrapped_prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=not deterministic,
            temperature=0.0 if deterministic else 0.9,
            top_p=1.0 if deterministic else 0.95,
            num_return_sequences=1,
        )
        text = outputs[0]["generated_text"]
        text = _sanitize_text(text)
        return TextGenResult(text=text)


def generate_text(
    prompt: str,
    use_ollama: bool = True,
    deterministic: bool = True,
    ollama_model: str = "qwen2:7b",
    ollama_host: str = "http://localhost:11434",
    max_new_tokens: int = 160,
    hf_model_name: str = "gpt2",
) -> str:
    generator = TextGenerator(
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
        max_new_tokens=max_new_tokens,
        hf_model_name=hf_model_name,
    )
    return generator.generate(prompt, deterministic=deterministic).text
