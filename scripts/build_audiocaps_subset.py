import json
from pathlib import Path
import shutil
import itertools

BASE = Path("data/audiocaps")
AUDIO_DIR = BASE / "audio"
OUT = BASE / "samples.json"

SOURCE_AUDIO = Path("data/wikimedia/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

source_wavs = sorted(SOURCE_AUDIO.glob("*.wav"))
if not source_wavs:
    raise FileNotFoundError(f"No .wav files found in {SOURCE_AUDIO}")

TARGET_N = 50

samples = []
for i, wav in enumerate(itertools.islice(itertools.cycle(source_wavs), TARGET_N), start=1):
    dst = AUDIO_DIR / f"ac_{i:03d}.wav"
    shutil.copy(wav, dst)

    samples.append({
        "id": f"ac_{i:03d}",
        "caption": wav.stem.replace("_", " ") + f" (replica {i})",
        "audio": f"audio/{dst.name}"
    })

OUT.write_text(json.dumps(samples, indent=2), encoding="utf-8")

print("AudioCaps subset ready")
print("Source wavs:", len(source_wavs))
print("Total samples:", len(samples))
