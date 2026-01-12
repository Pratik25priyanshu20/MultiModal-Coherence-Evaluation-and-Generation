from __future__ import annotations

AUDIOSET_LABEL_MAP = {
    "Rain": "gentle rainfall with distant ambience",
    "Thunder": "deep distant thunder with low rumble",
    "Wind": "soft wind moving through open space",
    "Bird": "natural birds chirping in the distance",
    "Traffic": "distant city traffic hum",
}


def labels_to_soundscape(labels: list[str]) -> str:
    parts = []
    for label in labels:
        if label in AUDIOSET_LABEL_MAP:
            parts.append(AUDIOSET_LABEL_MAP[label])
    if not parts:
        return "ambient environmental soundscape"
    return ", ".join(parts)
