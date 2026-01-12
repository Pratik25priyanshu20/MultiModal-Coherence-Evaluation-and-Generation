from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_run_metadata(log_path: str | Path, payload: Dict[str, Any]) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
