from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from spatial_renderer import SpatialLightWave, _wave_kind_code


try:
    import _spatial_native as _native
except Exception:
    _native = None


def available() -> bool:
    return _native is not None


def unavailable_reason() -> str:
    if _native is not None:
        return ""
    try:
        import _spatial_native  # noqa: F401
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return "unknown import failure"


def _require_native() -> Any:
    if _native is None:
        raise RuntimeError(f"cpp_probe native module is not available: {unavailable_reason()}")
    return _native


def advance_active_waves(renderer: object, waves: list[object], dt: float) -> list[SpatialLightWave]:
    config = getattr(renderer, "config")
    active: list[SpatialLightWave] = []
    for source in waves:
        wave = replace(source)
        wave.age += dt
        wave.intensity *= wave.decay_rate ** (dt * 30.0)
        travelled = wave.speed * wave.age
        if wave.intensity < config.wave_min_intensity or travelled > wave.radius_limit + wave.spread * 3.0:
            continue
        active.append(wave)
    return active


def wave_probe_inputs(renderer: object, waves: list[object], dt: float = 1.0 / 60.0) -> dict[str, Any]:
    spatial_idx, positions, angles, _ = renderer._priority_render_selection()  # type: ignore[attr-defined]
    active = advance_active_waves(renderer, waves, dt)
    room = renderer.topology.spatial.room  # type: ignore[attr-defined]
    total = int(renderer.topology.total)  # type: ignore[attr-defined]
    if active:
        origins = np.stack([wave.origin.astype(np.float32, copy=False) for wave in active]).astype(np.float32, copy=False)
    else:
        origins = np.zeros((0, 3), dtype=np.float32)
    return {
        "total_leds": total,
        "spatial_idx": np.ascontiguousarray(spatial_idx, dtype=np.int32),
        "positions": np.ascontiguousarray(positions, dtype=np.float32),
        "angles": np.ascontiguousarray(angles, dtype=np.float32),
        "kind_codes": np.ascontiguousarray([_wave_kind_code(wave.kind) for wave in active], dtype=np.int32),
        "colors": np.ascontiguousarray([wave.color for wave in active], dtype=np.float32).reshape(-1, 3) / 255.0,
        "intensities": np.ascontiguousarray([wave.intensity for wave in active], dtype=np.float32),
        "origins": origins,
        "speeds": np.ascontiguousarray([wave.speed for wave in active], dtype=np.float32),
        "ages": np.ascontiguousarray([wave.age for wave in active], dtype=np.float32),
        "radius_limits": np.ascontiguousarray([wave.radius_limit for wave in active], dtype=np.float32),
        "spreads": np.ascontiguousarray([wave.spread for wave in active], dtype=np.float32),
        "widths": np.ascontiguousarray([wave.width for wave in active], dtype=np.float32),
        "direction_hints": np.ascontiguousarray([wave.direction_hint for wave in active], dtype=np.int32),
        "phases": np.ascontiguousarray([wave.phase for wave in active], dtype=np.float32),
        "room_diagonal": float(renderer._room_diagonal),  # type: ignore[attr-defined]
        "room_width": float(room.width),
        "room_depth": float(room.depth),
    }


def _wave_args(inputs: dict[str, Any]) -> tuple[Any, ...]:
    return (
        inputs["total_leds"],
        inputs["spatial_idx"],
        inputs["positions"],
        inputs["angles"],
        inputs["kind_codes"],
        inputs["colors"],
        inputs["intensities"],
        inputs["origins"],
        inputs["speeds"],
        inputs["ages"],
        inputs["radius_limits"],
        inputs["spreads"],
        inputs["widths"],
        inputs["direction_hints"],
        inputs["phases"],
        inputs["room_diagonal"],
        inputs["room_width"],
        inputs["room_depth"],
    )


def render_waves_probe(inputs: dict[str, Any]) -> np.ndarray:
    return _require_native().render_waves_probe(*_wave_args(inputs))


def post_process_probe(
    leds: np.ndarray,
    previous_leds: np.ndarray,
    color_smoothing: float,
    saturation: float,
    white_balance: tuple[float, float, float] | list[float],
    brightness: float,
    gamma: float,
) -> np.ndarray:
    return _require_native().post_process_probe(
        np.ascontiguousarray(leds, dtype=np.float32),
        np.ascontiguousarray(previous_leds, dtype=np.float32),
        float(color_smoothing),
        float(saturation),
        np.ascontiguousarray(white_balance, dtype=np.float32),
        float(brightness),
        float(gamma),
    )


def full_probe(inputs: dict[str, Any], renderer: object) -> np.ndarray:
    config = getattr(renderer, "config")
    previous_leds = np.ascontiguousarray(getattr(renderer, "previous_leds"), dtype=np.float32)
    return _require_native().full_probe(
        *_wave_args(inputs),
        previous_leds,
        float(config.color_smoothing),
        float(config.saturation),
        np.ascontiguousarray(config.white_balance, dtype=np.float32),
        float(config.brightness),
        float(config.gamma),
    )
