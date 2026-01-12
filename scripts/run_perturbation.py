from pathlib import Path
import json
import random

from src.pipeline.generate_and_evaluate import generate_and_evaluate


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
PROMPT = "A calm foggy forest at dawn with distant birds and soft wind"
OUT_DIR = "runs/perturbation"


# -------------------------------------------------
# Utility: load random mismatched assets
# -------------------------------------------------
def random_image(exclude: str) -> str:
    images = list(Path("data/processed/images").glob("*.png"))
    candidates = [p for p in images if str(p) != exclude]
    return str(random.choice(candidates))


def random_audio(exclude: str) -> str:
    audios = list(Path("data/wikimedia/audio").glob("*.wav"))
    candidates = [p for p in audios if str(p) != exclude]
    return str(random.choice(candidates))


# -------------------------------------------------
# Main experiment
# -------------------------------------------------
def main():
    results = []

    print("\n=== BASELINE RUN ===")
    baseline = generate_and_evaluate(
        prompt=PROMPT,
        out_dir=OUT_DIR,
        use_ollama=True,
    )

    results.append({
        "variant": "baseline",
        "scores": baseline.scores,
        "classification": baseline.coherence,
        "bundle": baseline.run_id,
    })

    # -------------------------------------------------
    # WRONG IMAGE
    # -------------------------------------------------
    print("\n=== PERTURBATION: WRONG IMAGE ===")
    wrong_img = random_image(baseline.image_path)

    img_variant = generate_and_evaluate(
        prompt=PROMPT,
        out_dir=OUT_DIR,
        use_ollama=True,
    )

    img_variant.image_path = wrong_img

    results.append({
        "variant": "wrong_image",
        "scores": img_variant.scores,
        "classification": img_variant.coherence,
        "bundle": img_variant.run_id,
    })

    # -------------------------------------------------
    # WRONG AUDIO
    # -------------------------------------------------
    print("\n=== PERTURBATION: WRONG AUDIO ===")
    wrong_aud = random_audio(baseline.audio_path)

    aud_variant = generate_and_evaluate(
        prompt=PROMPT,
        out_dir=OUT_DIR,
        use_ollama=True,
    )

    aud_variant.audio_path = wrong_aud

    results.append({
        "variant": "wrong_audio",
        "scores": aud_variant.scores,
        "classification": aud_variant.coherence,
        "bundle": aud_variant.run_id,
    })

    # -------------------------------------------------
    # WRONG TEXT
    # -------------------------------------------------
    print("\n=== PERTURBATION: WRONG TEXT ===")
    wrong_text_prompt = (
        "A loud futuristic cyberpunk nightclub with flashing neon lights "
        "and heavy electronic bass music"
    )

    txt_variant = generate_and_evaluate(
        prompt=wrong_text_prompt,
        out_dir=OUT_DIR,
        use_ollama=True,
    )

    results.append({
        "variant": "wrong_text",
        "scores": txt_variant.scores,
        "classification": txt_variant.coherence,
        "bundle": txt_variant.run_id,
    })

    # -------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------
    out = Path(OUT_DIR) / "perturbation_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== PERTURBATION EXPERIMENT COMPLETE ===")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()