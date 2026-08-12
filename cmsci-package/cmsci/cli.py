"""Command-line interface: ``cmsci --text "..." --image x.png --audio y.wav``."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="cmsci",
        description="Score the semantic coherence of a text/image/audio triple "
        "with the Calibrated Multimodal Semantic Coherence Index (cMSCI).",
    )
    parser.add_argument("--text", required=True, help="Text description")
    parser.add_argument("--image", default=None, help="Path to an image file")
    parser.add_argument("--audio", default=None, help="Path to an audio file")
    parser.add_argument("--domain", default="", help="Domain hint (nature/urban/water)")
    parser.add_argument("--assets-dir", default=None, help="Override assets directory")
    parser.add_argument("--no-negative-bank", action="store_true", help="Disable contrastive calibration")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    args = parser.parse_args(argv)

    if not args.image and not args.audio:
        parser.error("provide at least one of --image / --audio")

    from cmsci.api import CoherenceScorer

    scorer = CoherenceScorer(
        assets_dir=args.assets_dir,
        negative_bank=not args.no_negative_bank,
    )
    result = scorer.score(
        text=args.text, image=args.image, audio=args.audio, domain=args.domain
    )

    if args.json:
        json.dump(result.raw, sys.stdout, indent=2, default=str)
        print()
    else:
        print(f"cMSCI  : {result.cmsci}  (variant {result.variant})")
        print(f"MSCI   : {result.msci}  (legacy baseline)")
        print(f"st_i   : {result.st_i}  (text-image)")
        print(f"st_a   : {result.st_a}  (text-audio)")
        print(f"si_a   : {result.si_a}  (image-audio, bridge)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
