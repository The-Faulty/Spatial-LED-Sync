from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spatial_config import default_spatial_dict, parse_spatial_config, validate_spatial_config


DEFAULT_ENABLED_EFFECTS = {
    "front_ambient": True,
    "ambient_side_spill": True,
    "spill": True,
    "top_color_exit": True,
    "flash": True,
    "explosion": True,
    "camera_pan": True,
    "energy_trail": True,
    "shockwave": True,
    "directional_sweep": True,
    "lightning": True,
    "impact_pulse": True,
    "color_bloom": True,
    "flame_shimmer": True,
    "underwater": True,
    "portal_vortex": True,
    "scene_wipe": True,
    "ember_particles": True,
    "negative_wave": True,
}

TRIGGER_EFFECTS = {
    "spill",
    "top_color_exit",
    "flash",
    "explosion",
    "camera_pan",
    "energy_trail",
    "shockwave",
    "directional_sweep",
    "lightning",
    "impact_pulse",
    "color_bloom",
    "flame_shimmer",
    "underwater",
    "portal_vortex",
    "scene_wipe",
    "negative_wave",
}

DEFAULT_EFFECT_SENSITIVITY = {name: 0.5 for name in TRIGGER_EFFECTS}


@dataclass
class WallConfig:
    start: int
    end: int


@dataclass
class EngineConfig:
    hyperhdr_host: str = "127.0.0.1"
    hyperhdr_port: int = 19400
    hyperhdr_input_mode: str = "websocket"
    hyperhdr_ws_port: int = 8090
    hyperhdr_ws_path: str = "/json-rpc"
    hyperhdr_instance: int = 0
    hyperhdr_token: str = ""
    hyperhdr_frame_timeout: float = 2.0
    simulate_input: bool = True

    wled_ip: str = ""
    wled_timeout: float = 0.35
    wled_segment_id: int | None = None
    send_to_wled: bool = False
    wled_protocol: str = "ddp"
    wled_udp_port: int = 4048
    wled_delta_threshold: int = 3

    total_leds: int = 240
    front_wall: WallConfig = field(default_factory=lambda: WallConfig(80, 140))
    left_wall: WallConfig = field(default_factory=lambda: WallConfig(141, 199))
    rear_wall: WallConfig = field(default_factory=lambda: WallConfig(200, 39))
    right_wall: WallConfig = field(default_factory=lambda: WallConfig(40, 79))
    clockwise_order: list[str] = field(default_factory=lambda: ["front", "left", "rear", "right"])

    tv_center_led: int = 110
    tv_left_boundary: int = 95
    tv_right_boundary: int = 125

    target_fps: int = 90
    motion_analysis_fps: int = 15
    optical_flow_fps: int = 15
    frame_difference_fill_enabled: bool = False
    frame_difference_fill_fps: int = 0
    wled_fps: int = 90
    parallel_runtime: bool = True
    tv_frame_queue_size: int = 1
    analysis_queue_size: int = 1
    event_queue_size: int = 2
    frame_buffer_size: int = 4
    analysis_width: int = 192
    analysis_height: int = 108
    render_mode: str = "full_frame"
    edge_band_width: int = 192
    edge_band_height: int = 108
    edge_band_fraction: float = 0.12
    hybrid_full_width: int = 64
    hybrid_full_height: int = 36
    max_active_waves: int = 50

    normal_spill_radius: int = 20
    strong_spill_radius: int = 50
    level1_threshold: float = 0.30
    level2_threshold: float = 0.70
    room_fill_threshold: float = 0.88
    lighting_mode: str = "cinematic"
    side_major_threshold: float = 0.70
    front_ambient_min_brightness: float = 0.04
    front_ambient_intensity: float = 0.35
    front_ambient_coverage: float = 1.0
    front_ambient_source: str = "top_strip"
    front_ambient_top_height: float = 0.12
    front_ambient_blur: int = 5
    tv_image_blur: int = 0
    ambient_side_spill_base_intensity: float = 0.45
    ambient_side_spill_boost_intensity: float = 1.0
    enabled_effects: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_ENABLED_EFFECTS))
    effect_sensitivity: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_EFFECT_SENSITIVITY))

    wave_speed: float = 1.0
    wave_decay: float = 0.95
    wave_spread: float = 9.0
    wave_min_intensity: float = 0.01
    variable_event_intensity: bool = True
    color_velocity_speed_boost: float = 0.85
    color_velocity_decay_boost: float = 0.10
    top_color_exit_velocity_threshold: float = 0.20
    top_color_exit_motion_threshold: float = 0.12
    top_color_exit_coverage_threshold: float = 0.04

    brightness: float = 0.50
    gamma: float = 2.2
    saturation: float = 1.15
    color_smoothing: float = 0.35
    white_balance: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    motion_algorithm: str = "optical_flow"
    motion_history: int = 8
    min_motion: float = 0.015
    bottom_edge_enabled: bool = False
    bottom_edge_mode: str = "local"

    weights: dict[str, float] = field(default_factory=lambda: {
        "motion": 0.28,
        "direction": 0.18,
        "brightness": 0.15,
        "saturation": 0.12,
        "coverage": 0.14,
        "edge": 0.18,
        "change": 0.17,
    })

    explosion_brightness_spike: float = 0.22
    flash_changed_fraction: float = 0.48
    camera_pan_confidence: float = 0.68
    trail_decay_boost: float = 0.92

    runtime_profile: str = "desktop_dev"
    overload_policy: str = "adaptive_quality"
    effect_render_skip_policy: str = "tv_first"
    spatial_renderer_backend: str = "auto"
    spatial_priority_bands: int = 3
    spatial_near_budget_ratio: float = 1.0
    spatial_mid_budget_ratio: float = 0.75
    spatial_far_budget_ratio: float = 0.35
    spatial: dict[str, Any] = field(default_factory=lambda: default_spatial_dict(240))

    debug: bool = True
    debug_window_scale: float = 1.0
    log_level: str = "INFO"
    log_file: str = "cinematic_spill.log"

    @classmethod
    def load(cls, path: str | Path = "config.json") -> "EngineConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EngineConfig":
        data = dict(raw)
        for name in ("front_wall", "left_wall", "rear_wall", "right_wall"):
            if isinstance(data.get(name), dict):
                data[name] = WallConfig(**data[name])
        enabled_effects = dict(DEFAULT_ENABLED_EFFECTS)
        if isinstance(data.get("enabled_effects"), dict):
            enabled_effects.update({k: bool(v) for k, v in data["enabled_effects"].items() if k in DEFAULT_ENABLED_EFFECTS})
        data["enabled_effects"] = enabled_effects
        effect_sensitivity = dict(DEFAULT_EFFECT_SENSITIVITY)
        if isinstance(data.get("effect_sensitivity"), dict):
            for key, value in data["effect_sensitivity"].items():
                if key in TRIGGER_EFFECTS:
                    try:
                        effect_sensitivity[key] = float(value)
                    except (TypeError, ValueError):
                        effect_sensitivity[key] = value
        data["effect_sensitivity"] = effect_sensitivity
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        for name in ("front_wall", "left_wall", "rear_wall", "right_wall"):
            wall = result[name]
            result[name] = {"start": wall.start, "end": wall.end}
        return result

    def save(self, path: str | Path = "config.json") -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.total_leds <= 0:
            errors.append("total_leds must be greater than 0")
            return errors

        for name in ("front_wall", "left_wall", "rear_wall", "right_wall"):
            wall = getattr(self, name)
            if not 0 <= wall.start < self.total_leds:
                errors.append(f"{name}.start must be between 0 and {self.total_leds - 1}")
            if not 0 <= wall.end < self.total_leds:
                errors.append(f"{name}.end must be between 0 and {self.total_leds - 1}")

        expected = {"front", "left", "rear", "right"}
        if set(self.clockwise_order) != expected or len(self.clockwise_order) != 4:
            errors.append("clockwise_order must contain front, left, rear, and right exactly once")

        for name in ("tv_center_led", "tv_left_boundary", "tv_right_boundary"):
            value = getattr(self, name)
            if not 0 <= value < self.total_leds:
                errors.append(f"{name} must be between 0 and {self.total_leds - 1}")

        if not 1 <= self.hyperhdr_port <= 65535:
            errors.append("hyperhdr_port must be between 1 and 65535")
        if self.hyperhdr_input_mode not in {"websocket", "flatbuffers"}:
            errors.append("hyperhdr_input_mode must be websocket or flatbuffers")
        if not 1 <= self.hyperhdr_ws_port <= 65535:
            errors.append("hyperhdr_ws_port must be between 1 and 65535")
        if not self.hyperhdr_ws_path.startswith("/"):
            errors.append("hyperhdr_ws_path must start with /")
        if self.hyperhdr_instance < 0:
            errors.append("hyperhdr_instance must be zero or greater")
        if self.target_fps <= 0:
            errors.append("target_fps must be greater than 0")
        if self.motion_analysis_fps <= 0:
            errors.append("motion_analysis_fps must be greater than 0")
        if self.optical_flow_fps <= 0:
            errors.append("optical_flow_fps must be greater than 0")
        if self.frame_difference_fill_fps < 0:
            errors.append("frame_difference_fill_fps must be zero or greater")
        if not isinstance(self.frame_difference_fill_enabled, bool):
            errors.append("frame_difference_fill_enabled must be true or false")
        if self.wled_fps <= 0:
            errors.append("wled_fps must be greater than 0")
        if self.tv_frame_queue_size <= 0:
            errors.append("tv_frame_queue_size must be greater than 0")
        if self.analysis_queue_size <= 0:
            errors.append("analysis_queue_size must be greater than 0")
        if self.event_queue_size <= 0:
            errors.append("event_queue_size must be greater than 0")
        if self.effect_render_skip_policy not in {"tv_first", "none"}:
            errors.append("effect_render_skip_policy must be tv_first or none")
        if self.spatial_renderer_backend not in {"auto", "cpp", "numpy", "numba"}:
            errors.append("spatial_renderer_backend must be auto, cpp, numpy, or numba")
        if self.spatial_priority_bands <= 0:
            errors.append("spatial_priority_bands must be greater than 0")
        for name in ("spatial_near_budget_ratio", "spatial_mid_budget_ratio", "spatial_far_budget_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                errors.append(f"{name} must be between 0.0 and 1.0")
        if self.wled_protocol not in {"ddp", "json"}:
            errors.append("wled_protocol must be ddp or json")
        if not 1 <= self.wled_udp_port <= 65535:
            errors.append("wled_udp_port must be between 1 and 65535")
        if self.wled_delta_threshold < 0:
            errors.append("wled_delta_threshold must be zero or greater")
        if self.frame_buffer_size <= 0:
            errors.append("frame_buffer_size must be greater than 0")
        if self.analysis_width <= 0 or self.analysis_height <= 0:
            errors.append("analysis dimensions must be greater than 0")
        if self.render_mode not in {"full_frame", "edge_effects", "hybrid_edge_full"}:
            errors.append("render_mode must be full_frame, edge_effects, or hybrid_edge_full")
        if self.edge_band_width <= 0 or self.edge_band_height <= 0:
            errors.append("edge band dimensions must be greater than 0")
        if self.hybrid_full_width <= 0 or self.hybrid_full_height <= 0:
            errors.append("hybrid full-frame dimensions must be greater than 0")
        if not 0.01 <= self.edge_band_fraction <= 0.5:
            errors.append("edge_band_fraction must be between 0.01 and 0.5")
        if self.max_active_waves <= 0:
            errors.append("max_active_waves must be greater than 0")

        for name in ("level1_threshold", "level2_threshold", "room_fill_threshold", "brightness"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                errors.append(f"{name} must be between 0.0 and 1.0")
        for name in ("side_major_threshold", "front_ambient_min_brightness", "front_ambient_intensity", "front_ambient_coverage", "ambient_side_spill_base_intensity"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                errors.append(f"{name} must be between 0.0 and 1.0")
        if not 0.0 <= self.ambient_side_spill_boost_intensity <= 2.0:
            errors.append("ambient_side_spill_boost_intensity must be between 0.0 and 2.0")
        if self.front_ambient_source not in {"top_strip", "average"}:
            errors.append("front_ambient_source must be top_strip or average")
        if not 0.02 <= self.front_ambient_top_height <= 0.50:
            errors.append("front_ambient_top_height must be between 0.02 and 0.50")
        if self.front_ambient_blur < 0:
            errors.append("front_ambient_blur must be zero or greater")
        if self.tv_image_blur < 0:
            errors.append("tv_image_blur must be zero or greater")
        if self.lighting_mode not in {"cinematic", "front_ambient"}:
            errors.append("lighting_mode must be cinematic or front_ambient")
        if not isinstance(self.enabled_effects, dict):
            errors.append("enabled_effects must be an object")
        else:
            for name in DEFAULT_ENABLED_EFFECTS:
                if name not in self.enabled_effects:
                    errors.append(f"enabled_effects.{name} is required")
                elif not isinstance(self.enabled_effects[name], bool):
                    errors.append(f"enabled_effects.{name} must be true or false")
        if not isinstance(self.effect_sensitivity, dict):
            errors.append("effect_sensitivity must be an object")
        else:
            for name in TRIGGER_EFFECTS:
                if name not in self.effect_sensitivity:
                    errors.append(f"effect_sensitivity.{name} is required")
                    continue
                value = self.effect_sensitivity[name]
                if not isinstance(value, (int, float)):
                    errors.append(f"effect_sensitivity.{name} must be a number")
                elif not 0.0 <= float(value) <= 1.0:
                    errors.append(f"effect_sensitivity.{name} must be between 0.0 and 1.0")
        if self.level1_threshold > self.level2_threshold:
            errors.append("level1_threshold must be less than or equal to level2_threshold")
        if not isinstance(self.variable_event_intensity, bool):
            errors.append("variable_event_intensity must be true or false")
        if self.motion_algorithm not in {"optical_flow", "frame_difference"}:
            errors.append("motion_algorithm must be optical_flow or frame_difference")
        for name in ("color_velocity_speed_boost", "color_velocity_decay_boost"):
            if getattr(self, name) < 0.0:
                errors.append(f"{name} must be zero or greater")
        for name in ("top_color_exit_velocity_threshold", "top_color_exit_motion_threshold", "top_color_exit_coverage_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                errors.append(f"{name} must be between 0.0 and 1.0")
        if self.bottom_edge_mode not in {"local", "room"}:
            errors.append("bottom_edge_mode must be local or room")
        if self.runtime_profile not in {"pi_zero", "pi_5", "desktop_dev"}:
            errors.append("runtime_profile must be pi_zero, pi_5, or desktop_dev")
        if self.overload_policy not in {"adaptive_quality", "fixed_quality", "output_first"}:
            errors.append("overload_policy must be adaptive_quality, fixed_quality, or output_first")
        if len(self.white_balance) != 3 or any(v < 0 for v in self.white_balance):
            errors.append("white_balance must contain three non-negative values")
        spatial = parse_spatial_config(self.spatial, self.total_leds)
        has_spatial_wled = spatial.enabled and any(device.enabled and device.ip.strip() for device in spatial.devices)
        if self.send_to_wled and not self.wled_ip.strip() and not has_spatial_wled:
            errors.append("wled_ip is required when send_to_wled is enabled")
        errors.extend(validate_spatial_config(self.spatial, self.total_leds))
        return errors


def write_default_config(path: str | Path = "config.json") -> None:
    path = Path(path)
    if not path.exists():
        EngineConfig().save(path)
