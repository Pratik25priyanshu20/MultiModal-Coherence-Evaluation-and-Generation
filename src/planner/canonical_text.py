from src.planner.semantic_plan import SemanticPlan


def plan_to_text(plan: SemanticPlan) -> str:
    scene = plan.scene

    visual = ", ".join(plan.visual_elements)
    audio = ", ".join(plan.audio_elements)

    return (
        f"A {plan.mood} {scene.setting} scene at {scene.time} with {scene.weather} weather. "
        f"Visual elements include {visual}. "
        f"Audio elements include {audio}. "
        f"The motion is {plan.motion}."
    )
