from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

RUNS_PER_PROMPT = 3
SLEEP_BETWEEN_RUNS = 2

PROMPT_FILE = Path("scripts/prompts_batch.json")
PHASE2_CMD = ["python", "scripts/run_phase2_v1.py"]


def main() -> None:
    prompts = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))

    total_runs = 0
    for prompt in prompts:
        print(f"\n=== PROMPT: {prompt} ===")

        for i in range(RUNS_PER_PROMPT):
            print(f"Run {i + 1}/{RUNS_PER_PROMPT}")
            subprocess.run(
                PHASE2_CMD,
                env={
                    **os.environ,
                    "USER_PROMPT_OVERRIDE": prompt,
                    "USE_OLLAMA": "1",
                },
                check=True,
            )
            total_runs += 1
            time.sleep(SLEEP_BETWEEN_RUNS)

    print(f"\n✅ Completed {total_runs} runs")


if __name__ == "__main__":
    main()
