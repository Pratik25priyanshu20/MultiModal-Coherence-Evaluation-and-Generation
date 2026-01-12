# (REFERENCE ONLY – do not import in runtime)

"""
Gold Sample Schema (v0)

{
  "id": str,
  "source": str,  # laion | wikimedia | audioset | musiccaps | bridged
  "domain": str,  # mixed_general
  "text_prompt": str,

  "semantic_plan": {
    ... merged SemanticPlan object ...
  },

  "image_path": str | null,
  "audio_path": str | null,

  "tags": {
    "scene_type": str,
    "mood": list[str],
    "environment": str
  },

  "human_criteria": {
    "entities_present": bool,
    "mood_matches": bool,
    "audio_matches_environment": bool,
    "no_major_contradictions": bool
  },

  "notes": str | null
}
"""
