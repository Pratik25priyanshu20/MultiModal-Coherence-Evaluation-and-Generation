"""
Test script to verify fixes are working.

Runs a small evaluation to check if:
1. Prompts are more specific (check bundle.json)
2. Image retrieval finds better matches
3. Audio prompts are more detailed
4. Coherence scores improve
"""

import json
from pathlib import Path

from src.pipeline.generate_and_evaluate import generate_and_evaluate


def test_fixes():
    """Test fixes on a few sample prompts."""
    
    test_prompts = [
        "A quiet beach at night with gentle waves and distant wind",
        "A rainy neon-lit city street at night with reflections on wet pavement",
        "A calm forest at dawn with birdsong and soft mist",
    ]
    
    print("=" * 60)
    print("Testing Fixes")
    print("=" * 60)
    
    results = []
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Testing: {prompt}")
        print('='*60)
        
        bundle = generate_and_evaluate(
            prompt=prompt,
            out_dir="runs/fix_test",
            use_ollama=True,
            deterministic=True,
            seed=42,
        )
        
        # Check prompt quality
        meta = bundle.meta
        modality_prompts = meta.get("modality_prompts", {})
        
        print("\n📝 Generated Prompts:")
        print(f"  Image: {modality_prompts.get('image_prompt', '')[:100]}...")
        print(f"  Audio: {modality_prompts.get('audio_prompt', '')[:100]}...")
        
        print("\n📊 Scores:")
        print(f"  MSCI: {bundle.scores.get('msci', 'N/A')}")
        print(f"  st_i: {bundle.scores.get('st_i', 'N/A')}")
        print(f"  st_a: {bundle.scores.get('st_a', 'N/A')}")
        print(f"  si_a: {bundle.scores.get('si_a', 'N/A')}")
        
        print("\n🎯 Drift:")
        drift = bundle.semantic_drift
        print(f"  Text:  {drift.get('text', 'N/A')}")
        print(f"  Image: {drift.get('image', 'N/A')}")
        print(f"  Audio: {drift.get('audio', 'N/A')}")
        
        results.append({
            "prompt": prompt,
            "run_id": bundle.run_id,
            "scores": bundle.scores,
            "drift": bundle.semantic_drift,
            "image_prompt": modality_prompts.get("image_prompt", ""),
            "audio_prompt": modality_prompts.get("audio_prompt", ""),
        })
    
    # Save results
    out_file = Path("runs/fix_test/test_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    print(f"\nResults saved to: {out_file}")
    print("\nCheck the prompts in bundle.json files to verify:")
    print("  ✓ Image prompts are specific (include subjects, style, mood)")
    print("  ✓ Audio prompts are detailed (include sound sources, ambience)")
    print("  ✓ Scores are better than baseline (MSCI > 0.01)")


if __name__ == "__main__":
    test_fixes()
