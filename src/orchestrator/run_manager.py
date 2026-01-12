from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    root: Path
    images_dir: Path
    audio_dir: Path
    logs_dir: Path


def create_run_paths(base_dir: str = "runs") -> RunPaths:
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    root = Path(base_dir) / run_id
    images_dir = root / "images"
    audio_dir = root / "audio"
    logs_dir = root / "logs"

    images_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_id=run_id,
        root=root,
        images_dir=images_dir,
        audio_dir=audio_dir,
        logs_dir=logs_dir,
    )
