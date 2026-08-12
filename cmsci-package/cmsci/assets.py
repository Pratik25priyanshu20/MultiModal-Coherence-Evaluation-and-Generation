"""
Asset resolution for the cmsci package.

The cMSCI scorer depends on a small set of trained artifacts (~9 MB total):
calibration statistics, the Ex-MCR projector, the cross-space bridge,
probabilistic adapters, and the negative-bank embedding indexes. They are
bundled with the package under ``cmsci/assets/``.

Resolution order:
    1. Explicit ``assets_dir`` argument (or ``CMSCI_ASSETS_DIR`` env var)
    2. The assets bundled with the installed package

Every asset is optional: a missing file degrades the engine gracefully
(e.g. no bridge -> si_a channel omitted, no calibration -> raw scores).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Relative layout inside the assets directory (bundled or user-provided).
ASSET_LAYOUT = {
    "calibration": "artifacts/cmsci_calibration.json",
    "coherence_stats": "artifacts/coherence_stats.json",
    "exmcr": "models/exmcr/ex_clap.pt",
    "bridge": "models/bridge/bridge_best.pt",
    "clip_adapter": "models/prob_adapters/clip_adapter.pt",
    "clap_adapter": "models/prob_adapters/clap_adapter.pt",
    "image_index": "data/embeddings/image_index.npz",
    "audio_index": "data/embeddings/audio_index.npz",
}

_BUNDLED_DIR = Path(__file__).resolve().parent / "assets"


@dataclass
class AssetPaths:
    """Resolved absolute paths for each asset (None when unavailable)."""

    calibration: Optional[Path] = None
    coherence_stats: Optional[Path] = None
    exmcr: Optional[Path] = None
    bridge: Optional[Path] = None
    clip_adapter: Optional[Path] = None
    clap_adapter: Optional[Path] = None
    image_index: Optional[Path] = None
    audio_index: Optional[Path] = None

    @property
    def missing(self) -> list:
        return [f.name for f in fields(self) if getattr(self, f.name) is None]


def _paths_from_dir(root: Path) -> AssetPaths:
    resolved = {}
    for name, rel in ASSET_LAYOUT.items():
        p = root / rel
        resolved[name] = p if p.exists() else None
    return AssetPaths(**resolved)


def resolve_assets(assets_dir: Optional[str] = None) -> AssetPaths:
    """
    Locate the cMSCI assets.

    Args:
        assets_dir: Directory containing the asset layout (see ASSET_LAYOUT).
            Defaults to ``CMSCI_ASSETS_DIR`` env var, then the bundled assets.

    Returns:
        AssetPaths with absolute paths (None entries for missing assets).
    """
    assets_dir = assets_dir or os.environ.get("CMSCI_ASSETS_DIR")
    root = Path(assets_dir).expanduser().resolve() if assets_dir else _BUNDLED_DIR
    paths = _paths_from_dir(root)
    if paths.missing:
        logger.warning(
            "Assets missing under %s: %s — affected features are disabled",
            root,
            paths.missing,
        )
    return paths
