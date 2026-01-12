import json
import re
from typing import Any


def _strip_code_fences(text: str) -> str:
    """
    Removes markdown code fences like ```json ... ``` or ``` ... ```.
    """
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _extract_first_json_object(text: str) -> str | None:
    """
    Extracts the first valid JSON object substring using brace counting.
    Works even if additional text exists after JSON.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return text[start:]


def _close_open_braces(text: str) -> str:
    """
    If JSON is truncated, add missing closing braces.
    """
    open_braces = text.count("{")
    close_braces = text.count("}")
    if close_braces < open_braces:
        text = text + ("}" * (open_braces - close_braces))
    return text


def _remove_trailing_commas(text: str) -> str:
    """
    Removes trailing commas before closing ] or }
    """
    return re.sub(r",\s*([}\]])", r"\1", text)


def _truncate_to_last_safe_boundary(text: str) -> str | None:
    """
    Truncates to the last comma outside of strings to drop incomplete tail data.
    Also handles cases where we're in the middle of a field value.
    """
    depth = 0
    in_str = False
    escape = False
    last_cut = None
    last_colon = None

    for idx, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == ":" and depth >= 1:
            last_colon = idx
        elif ch == "," and depth >= 1:
            last_cut = idx

    # If we found a comma, use that
    if last_cut is not None:
        return text[:last_cut]
    
    # If we found a colon but no comma, try truncating after the colon's value
    # This handles cases like "ligh" where we're mid-field
    if last_colon is not None:
        # Find the end of the current line or next quote
        rest = text[last_colon:]
        # Try to find end of current value
        for i, c in enumerate(rest[1:], 1):
            if c in ['\n', ',', '}']:
                return text[:last_colon + i]
    
    return None


def try_repair_json(text: str) -> dict[str, Any] | None:
    """
    Attempts to recover JSON from LLM output:
    - Strips code fences
    - Extracts first JSON object using brace counting
    - Repairs missing closing braces
    - Tries json.loads()
    """
    if not text:
        return None

    text = _strip_code_fences(text)

    candidate = _extract_first_json_object(text)
    if candidate is None:
        return None

    candidate = _close_open_braces(candidate)
    candidate = _remove_trailing_commas(candidate)

    try:
        return json.loads(candidate)
    except Exception:
        pass

    truncated = _truncate_to_last_safe_boundary(candidate)
    if truncated:
        truncated = _close_open_braces(truncated)
        truncated = _remove_trailing_commas(truncated)
        try:
            return json.loads(truncated)
        except Exception:
            return None

    return None
