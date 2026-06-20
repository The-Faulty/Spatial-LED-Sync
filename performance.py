from __future__ import annotations

from config import EngineConfig


RUNTIME_PROFILES: dict[str, dict[str, object]] = {
    "pi_zero": {
        "debug": False,
        "motion_algorithm": "frame_difference",
        "analysis_width": 96,
        "analysis_height": 54,
        "target_fps": 20,
        "wled_fps": 25,
        "frame_buffer_size": 2,
        "max_active_waves": 18,
        "front_ambient_blur": 0,
    },
    "pi_5": {
        "debug": False,
        "motion_algorithm": "frame_difference",
        "analysis_width": 160,
        "analysis_height": 90,
        "target_fps": 30,
        "wled_fps": 30,
        "frame_buffer_size": 3,
        "max_active_waves": 36,
    },
    "desktop_dev": {},
}


def apply_runtime_profile(config: EngineConfig, profile: str | None = None) -> None:
    selected = profile or config.runtime_profile
    for key, value in RUNTIME_PROFILES.get(selected, {}).items():
        setattr(config, key, value)
    config.runtime_profile = selected


def valid_runtime_profiles() -> set[str]:
    return set(RUNTIME_PROFILES)
