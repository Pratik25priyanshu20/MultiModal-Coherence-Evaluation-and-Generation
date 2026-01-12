from src.generators.text.generator import generate_text
from src.planner.canonical_text import plan_to_text
from src.planner.semantic_planner import generate_semantic_plan

prompt = "A rainy neon-lit city street at night with reflections on wet pavement"

plan = generate_semantic_plan(prompt)
canonical_prompt = plan_to_text(plan)
text = generate_text(canonical_prompt, use_ollama=True)

print(plan)
print("\n--- Canonical Prompt ---")
print(canonical_prompt)
print("\n--- Generated Text ---")
print(text)
