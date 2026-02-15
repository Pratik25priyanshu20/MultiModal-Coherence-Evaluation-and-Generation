from pydantic import BaseModel, Field


class NarrativeOutput(BaseModel):
    visual_description: str = Field(
        ...,
        description="Concise visual description of the scene",
    )
    audio_description: str = Field(
        ...,
        description="Concise description of the audio environment",
    )
    combined_scene: str = Field(
        ...,
        description="Unified multimodal narrative combining visual and audio",
    )
    style_summary: str = Field(
        ...,
        description="High-level artistic style summary",
    )
    mood_summary: str = Field(
        ...,
        description="Overall emotional tone of the scene",
    )
