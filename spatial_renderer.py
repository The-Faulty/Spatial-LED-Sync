from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol

import cv2
import numpy as np

from config import EngineConfig
from event_detector import LightEvent
from event_mixer import should_duck_wave
from spatial_topology import SpatialRoomTopology


class SpatialRenderer(Protocol):
    waves: list[object]
    accepts_front_ambient_strip_sample: bool

    def add_events(self, events: list[LightEvent]) -> None:
        ...

    def duck_lower_priority_waves(self, primary_kind: str, factor: float = 0.35) -> None:
        ...

    def set_tv_frame(self, frame_bgr: np.ndarray | None) -> None:
        ...

    def set_front_ambient_strip(self, colors: np.ndarray, intensity: float) -> None:
        ...

    def set_ambient_spill_scene_boost(self, score: float, color: tuple[int, int, int]) -> None:
        ...

    def front_ambient_led_count(self) -> int:
        ...

    def set_spatial_priority_budget(self, max_bands: int | None) -> None:
        ...

    def step(self, dt: float) -> np.ndarray:
        ...


@dataclass
class SpatialLightWave:
    color: tuple[int, int, int]
    intensity: float
    origin: np.ndarray
    speed: float
    decay_rate: float
    spread: float
    age: float
    radius_limit: float
    kind: str
    color_velocity: float = 0.0
    direction_hint: int = 0
    width: float = 1.0
    pulse_count: int = 1
    phase: float = 0.0
    secondary: bool = False


WAVE_KIND_CODES = {
    "flash": 1,
    "explosion": 1,
    "impact_pulse": 2,
    "shockwave": 3,
    "directional_sweep": 4,
    "scene_wipe": 4,
    "energy_trail": 5,
    "flame_shimmer": 6,
    "underwater": 7,
    "portal_vortex": 8,
    "color_bloom": 9,
    "lightning": 10,
    "ember_particles": 11,
    "negative_wave": 12,
}

SPATIAL_RENDERER_BACKEND_PROBE_VERSION = "cpp-native-1"


def _wave_kind_code(kind: str) -> int:
    return WAVE_KIND_CODES.get(kind, 0)


def _numba_render_kernel():
    try:
        from numba import njit
    except Exception:
        return None

    @njit(cache=True, fastmath=True)
    def render_kernel(
        leds,
        spatial_idx,
        positions,
        angles,
        kind_codes,
        colors,
        intensities,
        origins,
        speeds,
        ages,
        radius_limits,
        spreads,
        widths,
        direction_hints,
        phases,
        room_diagonal,
        room_width,
        room_depth,
    ):
        for wave_index in range(kind_codes.shape[0]):
            kind = kind_codes[wave_index]
            intensity = intensities[wave_index]
            origin_x = origins[wave_index, 0]
            origin_y = origins[wave_index, 1]
            origin_z = origins[wave_index, 2]
            travelled = speeds[wave_index] * ages[wave_index]
            radius_limit = radius_limits[wave_index]
            spread = spreads[wave_index]
            width = widths[wave_index]
            direction_hint = direction_hints[wave_index]
            phase = phases[wave_index]
            travel_fade = max(0.0, 1.0 - travelled / max(0.1, radius_limit))
            contribution_scale = intensity * (0.2 + 0.8 * travel_fade)
            red = colors[wave_index, 0] * contribution_scale
            green = colors[wave_index, 1] * contribution_scale
            blue = colors[wave_index, 2] * contribution_scale

            for local_index in range(spatial_idx.shape[0]):
                dx = positions[local_index, 0] - origin_x
                dy = positions[local_index, 1] - origin_y
                dz = positions[local_index, 2] - origin_z
                distances_sq = dx * dx + dy * dy + dz * dz
                envelope = 0.0

                if kind == 1:
                    fill_radius = max(spread, travelled + spread)
                    envelope = math.exp(-distances_sq / (2.0 * fill_radius * fill_radius))
                elif kind == 2:
                    fill_radius = max(room_diagonal, travelled + spread)
                    local = math.exp(-distances_sq / (2.0 * fill_radius * fill_radius))
                    collapse = max(0.0, 1.0 - ages[wave_index] / max(0.2, width))
                    envelope = max(local, 0.45 * collapse)
                elif kind == 3:
                    distance = math.sqrt(distances_sq)
                    ring_width = max(spread * 0.45, width * 0.08)
                    ring = math.exp(-((distance - travelled) * (distance - travelled)) / (2.0 * ring_width * ring_width))
                    room_flash = max(0.0, 1.0 - travelled / max(0.1, radius_limit)) * 0.32
                    if direction_hint != 0:
                        sign = 1.0 if direction_hint > 0 else -1.0
                        projection = dx * sign
                        forward = min(max((projection / max(distance, 0.001) + 1.0) * 0.5, 0.0), 1.0)
                        gate = min(max(forward * 1.35, 0.0), 1.0)
                        ring *= gate
                        room_flash *= 0.35 + 0.65 * gate
                    envelope = max(ring, room_flash)
                elif kind == 4:
                    axis = positions[local_index, 0]
                    span = max(max(room_width, room_depth), 0.1)
                    front = (travelled / max(0.1, radius_limit)) * span
                    if direction_hint < 0:
                        front = span - front
                    sweep_width = max(0.12, width * 0.25)
                    envelope = math.exp(-((axis - front) * (axis - front)) / (2.0 * sweep_width * sweep_width))
                elif kind == 5:
                    distance = math.sqrt(distances_sq)
                    head = math.exp(-((distance - travelled) * (distance - travelled)) / (2.0 * spread * spread))
                    tail_distance = distance - max(0.0, travelled - spread * 3.0)
                    tail_spread = spread * 2.2
                    tail = math.exp(-(tail_distance * tail_distance) / (2.0 * tail_spread * tail_spread)) * 0.45
                    envelope = max(head, tail)
                elif kind == 6:
                    noise = 0.65 + 0.35 * math.sin(positions[local_index, 0] * 7.0 + positions[local_index, 2] * 5.0 + ages[wave_index] * 18.0 + phase * 6.28)
                    local_spread = max(spread * 2.4, 0.1)
                    envelope = math.exp(-distances_sq / (2.0 * local_spread * local_spread)) * noise
                elif kind == 7:
                    distance = math.sqrt(distances_sq)
                    ripple_base = math.sin(distance * 6.0 - ages[wave_index] * 4.0 + phase * 6.28)
                    ripple = 0.45 + 0.55 * ripple_base * ripple_base
                    local_spread = max(spread * 5.0, 0.1)
                    envelope = math.exp(-distances_sq / (2.0 * local_spread * local_spread)) * ripple
                elif kind == 8:
                    distance = math.sqrt(distances_sq)
                    turns = max(1, direction_hint)
                    spiral = 0.5 + 0.5 * math.sin(angles[local_index] * 3.0 * turns + distance * 4.0 - ages[wave_index] * 6.0)
                    vortex_spread = spread * 2.5
                    drift = distance - travelled * 0.45
                    envelope = math.exp(-(drift * drift) / (2.0 * vortex_spread * vortex_spread)) * spiral
                elif kind == 9 or kind == 10 or kind == 11:
                    bloom_spread = spread * 4.0 if kind == 9 else spread
                    local_spread = max(bloom_spread, 0.1)
                    envelope = math.exp(-distances_sq / (2.0 * local_spread * local_spread))
                    if kind == 10 and intensity >= 0.72:
                        strobe = 1.0 if int(ages[wave_index] * 28.0 + phase * 7.0) % 2 == 0 else 0.0
                        envelope = max(envelope, strobe * intensity * 0.55)
                else:
                    distance = math.sqrt(distances_sq)
                    envelope = math.exp(-((distance - travelled) * (distance - travelled)) / (2.0 * spread * spread))

                led_index = spatial_idx[local_index]
                if kind == 12:
                    dim = envelope * intensity * 0.65
                    leds[led_index, 0] -= dim
                    leds[led_index, 1] -= dim
                    leds[led_index, 2] -= dim
                else:
                    leds[led_index, 0] += envelope * red
                    leds[led_index, 1] += envelope * green
                    leds[led_index, 2] += envelope * blue

    return render_kernel


_NUMBA_RENDER_KERNEL = None


class VectorizedSpatialRenderer:
    def __init__(self, config: EngineConfig, topology: SpatialRoomTopology):
        self.config = config
        self.topology = topology
        self.render_backend = "numpy"
        self.render_backend_status = "ready"
        self.last_render_ms = 0.0
        self.last_readback_ms = 0.0
        self.accepts_front_ambient_strip_sample = True
        self.waves: list[SpatialLightWave] = []
        self.previous_leds = np.zeros((topology.total, 3), dtype=np.float32)
        self._working_leds = np.zeros((topology.total, 3), dtype=np.float32)
        self.tv_frame: np.ndarray | None = None
        self.front_ambient_color = np.zeros(3, dtype=np.float32)
        self.front_ambient_strip_colors: np.ndarray | None = None
        self.front_ambient_intensity = 0.0
        self._front_ambient_scene_luma = 0.0
        self._ambient_side_spill_boost = 0.0
        self._ambient_side_spill_scene_color = np.zeros(3, dtype=np.float32)
        self._spatial_mask = np.array([mode in {"spatial", "blend"} for mode in topology.sync_modes], dtype=bool)
        self._spatial_indices = np.nonzero(self._spatial_mask)[0].astype(np.int32)
        self._spatial_positions = topology.positions[self._spatial_indices]
        self._tv_mask = np.array([mode in {"tv_image", "blend"} for mode in topology.sync_modes], dtype=bool)
        self._blend = topology.strip_blend.reshape(-1, 1)
        self._room_center = topology.room_center()
        self._room_diagonal = topology.room_diagonal()
        centered = self._spatial_positions - self._room_center.reshape(1, 3)
        self._spatial_angles = np.arctan2(centered[:, 2], centered[:, 0]) if centered.size else np.zeros(0, dtype=np.float32)
        self._spatial_priority_bands = self._build_spatial_priority_bands()
        self._spatial_priority_budget: int | None = None
        self.last_spatial_priority_band_skips = 0

    def add_events(self, events: list[LightEvent]) -> None:
        diagonal = max(0.1, self.topology.room_diagonal())
        for event in events:
            brightness_gain = self._event_brightness_gain(event)
            if event.kind == "front_ambient":
                color = np.array(event.color, dtype=np.float32) / 255.0
                self.front_ambient_color = self.front_ambient_color * 0.85 + color * 0.15
                self.front_ambient_intensity = max(self.front_ambient_intensity * 0.8, brightness_gain)
                continue
            color_velocity = float(np.clip(event.color_velocity, 0.0, 1.0))
            speed = diagonal * 0.18 * self.config.wave_speed * (0.55 + event.intensity * 1.35)
            speed *= 1.0 + color_velocity * self.config.color_velocity_speed_boost
            spread = max(0.05, diagonal * 0.025 * (0.65 + event.intensity) * self.config.wave_spread / 9.0)
            decay = float(np.clip(self.config.wave_decay - color_velocity * self.config.color_velocity_decay_boost, 0.50, 0.999))
            if event.kind == "flash":
                speed *= 2.2
                spread *= 2.0
                decay *= 0.90
            elif event.kind in {"shockwave", "scene_wipe", "directional_sweep"}:
                speed *= 1.55
                spread *= max(0.7, event.width)
                decay *= 0.95
            elif event.kind in {"color_bloom", "underwater"}:
                speed *= 0.25
                spread *= 3.0
                decay = min(max(decay, 0.955), 0.975)
            elif event.kind in {"lightning", "impact_pulse"}:
                speed *= 2.5
                spread *= 2.5
                decay *= 0.78
            elif event.kind == "energy_trail":
                speed *= 1.7
                spread *= 0.75
                decay *= 0.90
            elif event.kind == "ember_particles":
                speed *= 0.55
                spread = max(spread * 0.75, diagonal * 0.06)
                decay *= 0.88
            radius_limit = self._radius_limit(event.intensity, event.kind)
            origin = self.topology.room_center() if event.kind in {"flash", "explosion"} else self.topology.event_origin_point(event.edge)
            count = max(1, int(event.pulse_count if event.kind in {"ember_particles", "lightning"} else 1))
            for pulse in range(count):
                pulse_phase = event.phase + pulse / max(1, count)
                offset = np.zeros(3, dtype=np.float32)
                if event.kind in {"ember_particles", "lightning"}:
                    angle = pulse_phase * np.pi * 2.0
                    offset = np.array([np.cos(angle), 0.25 * np.sin(angle * 1.7), np.sin(angle)], dtype=np.float32) * diagonal * 0.12
                self.waves.append(
                    SpatialLightWave(
                        color=event.color,
                        intensity=brightness_gain * (1.0 - pulse * 0.035),
                        origin=origin + offset,
                        speed=speed * (0.85 + pulse_phase * 0.35),
                        decay_rate=decay,
                        spread=spread,
                        age=0.0,
                        radius_limit=radius_limit,
                        kind=event.kind,
                        color_velocity=color_velocity,
                        direction_hint=event.direction_hint,
                        width=max(0.05, event.width),
                        pulse_count=count,
                        phase=pulse_phase,
                        secondary=event.secondary,
                    )
                )
        if len(self.waves) > self.config.max_active_waves:
            self.waves = sorted(self.waves, key=lambda wave: wave.intensity, reverse=True)[: self.config.max_active_waves]

    def duck_lower_priority_waves(self, primary_kind: str, factor: float = 0.35) -> None:
        for wave in self.waves:
            if should_duck_wave(primary_kind, wave.kind):
                wave.intensity *= factor

    def set_tv_frame(self, frame_bgr: np.ndarray | None) -> None:
        self.tv_frame = frame_bgr

    def set_front_ambient_strip(self, colors: np.ndarray, intensity: float) -> None:
        if colors.size:
            target = np.clip(colors.astype(np.float32), 0.0, 1.0)
            self._update_ambient_side_spill_boost(target, intensity)
            if self.front_ambient_strip_colors is None or self.front_ambient_strip_colors.shape != target.shape:
                self.front_ambient_strip_colors = target
            else:
                self.front_ambient_strip_colors = self.front_ambient_strip_colors * 0.70 + target * 0.30
            self.front_ambient_color = np.mean(self.front_ambient_strip_colors, axis=0)
            self.front_ambient_intensity = max(self.front_ambient_intensity * 0.8, float(np.clip(intensity, 0.0, 1.0)))

    def set_ambient_spill_scene_boost(self, score: float, color: tuple[int, int, int]) -> None:
        score = float(np.clip(score, 0.0, 1.0))
        target_color = np.array(color, dtype=np.float32) / 255.0
        if score <= 0.0 or float(np.max(target_color)) <= 0.001:
            self._ambient_side_spill_boost *= 0.92
            return
        self._ambient_side_spill_boost = max(self._ambient_side_spill_boost * 0.88, score)
        if float(np.max(self._ambient_side_spill_scene_color)) <= 0.001:
            self._ambient_side_spill_scene_color = target_color
        else:
            self._ambient_side_spill_scene_color = self._ambient_side_spill_scene_color * 0.65 + target_color * 0.35

    def front_ambient_led_count(self) -> int:
        return max(1, int(self.topology.front_ambient_indices().size))

    def set_spatial_priority_budget(self, max_bands: int | None) -> None:
        if max_bands is None:
            self._spatial_priority_budget = None
        else:
            self._spatial_priority_budget = max(1, min(int(max_bands), len(self._spatial_priority_bands)))

    def step(self, dt: float) -> np.ndarray:
        leds = self._working_leds
        leds.fill(0.0)
        render_started = time.perf_counter()
        self._render_spatial_waves(leds, dt)
        self.last_render_ms = (time.perf_counter() - render_started) * 1000.0
        self.last_readback_ms = 0.0
        self._apply_tv_sync(leds)
        self._render_front_ambient(leds, dt)
        leds = np.clip(leds, 0.0, 1.0)
        alpha = self.config.color_smoothing
        leds *= 1.0 - alpha
        leds += self.previous_leds * alpha
        self.previous_leds[...] = leds
        return self._post_process(leds)

    def _render_spatial_waves(self, leds: np.ndarray, dt: float) -> None:
        if self.topology.total == 0 or self._spatial_indices.size == 0:
            return
        active: list[SpatialLightWave] = []
        spatial_idx, positions, angles, skipped = self._priority_render_selection()
        self.last_spatial_priority_band_skips = skipped
        if spatial_idx.size == 0:
            return
        for wave in self.waves:
            wave.age += dt
            wave.intensity *= wave.decay_rate ** (dt * 30.0)
            travelled = wave.speed * wave.age
            if wave.intensity < self.config.wave_min_intensity or travelled > wave.radius_limit + wave.spread * 3.0:
                continue
            active.append(wave)
            offset = positions - wave.origin.reshape(1, 3)
            distances_sq = np.einsum("ij,ij->i", offset, offset)
            if wave.kind in {"flash", "explosion"}:
                fill_radius = max(wave.spread, travelled + wave.spread)
                envelope = np.exp(-distances_sq / (2.0 * fill_radius**2))
            elif wave.kind == "impact_pulse":
                fill_radius = max(self._room_diagonal, travelled + wave.spread)
                local = np.exp(-distances_sq / (2.0 * fill_radius**2))
                collapse = max(0.0, 1.0 - wave.age / max(0.2, wave.width))
                envelope = np.maximum(local, 0.45 * collapse)
            elif wave.kind == "shockwave":
                distances = np.sqrt(distances_sq)
                ring_width = max(wave.spread * 0.45, wave.width * 0.08)
                ring = np.exp(-((distances - travelled) ** 2) / (2.0 * ring_width**2))
                room_flash = max(0.0, 1.0 - travelled / max(0.1, wave.radius_limit)) * 0.32
                if wave.direction_hint:
                    direction = np.array([float(np.sign(wave.direction_hint)), 0.0, 0.0], dtype=np.float32)
                    projection = offset @ direction
                    forward = np.clip((projection / np.maximum(distances, 0.001) + 1.0) * 0.5, 0.0, 1.0)
                    gate = np.clip(forward * 1.35, 0.0, 1.0)
                    ring *= gate
                    room_flash *= 0.35 + 0.65 * gate
                envelope = np.maximum(ring, room_flash)
            elif wave.kind in {"directional_sweep", "scene_wipe"}:
                axis = positions[:, 0] if abs(wave.direction_hint) >= 0 else positions[:, 2]
                room = self.topology.spatial.room
                span = max(room.width, room.depth, 0.1)
                front = (travelled / max(0.1, wave.radius_limit)) * span
                if wave.direction_hint < 0:
                    front = span - front
                envelope = np.exp(-((axis - front) ** 2) / (2.0 * max(0.12, wave.width * 0.25) ** 2))
            elif wave.kind == "energy_trail":
                distances = np.sqrt(distances_sq)
                head = np.exp(-((distances - travelled) ** 2) / (2.0 * wave.spread**2))
                tail = np.exp(-((distances - max(0.0, travelled - wave.spread * 3.0)) ** 2) / (2.0 * (wave.spread * 2.2) ** 2)) * 0.45
                envelope = np.maximum(head, tail)
            elif wave.kind == "flame_shimmer":
                noise = 0.65 + 0.35 * np.sin((positions[:, 0] * 7.0 + positions[:, 2] * 5.0 + wave.age * 18.0 + wave.phase * 6.28))
                envelope = np.exp(-distances_sq / (2.0 * max(wave.spread * 2.4, 0.1) ** 2)) * noise
            elif wave.kind == "underwater":
                distances = np.sqrt(distances_sq)
                ripple = 0.45 + 0.55 * np.sin(distances * 6.0 - wave.age * 4.0 + wave.phase * 6.28) ** 2
                envelope = np.exp(-distances_sq / (2.0 * max(wave.spread * 5.0, 0.1) ** 2)) * ripple
            elif wave.kind == "portal_vortex":
                distances = np.sqrt(distances_sq)
                spiral = 0.5 + 0.5 * np.sin(angles * 3.0 * max(1, wave.direction_hint) + distances * 4.0 - wave.age * 6.0)
                envelope = np.exp(-((distances - travelled * 0.45) ** 2) / (2.0 * (wave.spread * 2.5) ** 2)) * spiral
            elif wave.kind in {"color_bloom", "lightning", "ember_particles"}:
                envelope = np.exp(-distances_sq / (2.0 * max(wave.spread * (4.0 if wave.kind == "color_bloom" else 1.0), 0.1) ** 2))
                if wave.kind == "lightning" and wave.intensity >= 0.72:
                    strobe = 1.0 if int(wave.age * 28.0 + wave.phase * 7.0) % 2 == 0 else 0.0
                    envelope = np.maximum(envelope, strobe * wave.intensity * 0.55)
            else:
                distances = np.sqrt(distances_sq)
                envelope = np.exp(-((distances - travelled) ** 2) / (2.0 * wave.spread**2))
            travel_fade = max(0.0, 1.0 - travelled / max(0.1, wave.radius_limit))
            contribution = (np.array(wave.color, dtype=np.float32) / 255.0) * wave.intensity * (0.2 + 0.8 * travel_fade)
            if wave.kind == "negative_wave":
                leds[spatial_idx] -= envelope.reshape(-1, 1) * wave.intensity * 0.65
            else:
                leds[spatial_idx] += envelope.reshape(-1, 1) * contribution
        self.waves = active

    def _build_spatial_priority_bands(self) -> list[np.ndarray]:
        if self._spatial_indices.size == 0:
            return []
        tv = self.topology.spatial.tv
        tv_center = self.topology.wall_point(tv.wall, tv.center_u, tv.center_v).reshape(1, 3)
        distances = np.linalg.norm(self._spatial_positions - tv_center, axis=1)
        ordered_local = np.argsort(distances).astype(np.int32)
        band_count = max(1, int(self.config.spatial_priority_bands))
        return [band.astype(np.int32) for band in np.array_split(ordered_local, band_count) if band.size]

    def _priority_render_selection(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if not self._spatial_priority_bands:
            return self._spatial_indices, self._spatial_positions, self._spatial_angles, 0
        max_bands = self._spatial_priority_budget or len(self._spatial_priority_bands)
        max_bands = max(1, min(max_bands, len(self._spatial_priority_bands)))
        selected_local = np.concatenate(self._spatial_priority_bands[:max_bands])
        skipped = max(0, len(self._spatial_priority_bands) - max_bands)
        return (
            self._spatial_indices[selected_local],
            self._spatial_positions[selected_local],
            self._spatial_angles[selected_local],
            skipped,
        )

    def _render_front_ambient(self, leds: np.ndarray, dt: float) -> None:
        if self.front_ambient_intensity <= self.config.wave_min_intensity:
            return
        ambient = self._front_ambient_led_colors()
        ambient_mask = np.any(ambient > 0.0, axis=1) & self._spatial_mask
        leds[ambient_mask] += ambient[ambient_mask]
        if self.config.enabled_effects.get("ambient_side_spill", True):
            spill_source = ambient
            if self._ambient_side_spill_boost > 0.05 and float(np.max(self._ambient_side_spill_scene_color)) > 0.001:
                spill_source = self._ambient_with_scene_boost_color(ambient)
            spill = self.topology.ambient_extension_from_leds(
                spill_source,
                strength_boost=self._ambient_side_spill_boost,
                reach_boost=self._ambient_side_spill_boost,
            )
            spill_mask = np.any(spill > 0.0, axis=1)
            leds[spill_mask] = np.maximum(leds[spill_mask], spill[spill_mask])
            self._ambient_side_spill_boost *= 0.94 ** (dt * 30.0)
        self.front_ambient_intensity *= 0.92 ** (dt * 30.0)

    def _ambient_with_scene_boost_color(self, ambient: np.ndarray) -> np.ndarray:
        tinted = ambient.copy()
        idx = self.topology.front_ambient_indices()
        if idx.size == 0:
            return tinted
        active = idx[np.any(tinted[idx] > 0.0, axis=1)]
        if active.size == 0:
            return tinted
        level = np.max(tinted[active], axis=1, keepdims=True)
        scene = self._ambient_side_spill_scene_color.reshape(1, 3)
        blend = min(0.90, self._ambient_side_spill_boost * 1.05)
        tinted[active] = tinted[active] * (1.0 - blend) + scene * np.maximum(level, 0.08) * blend
        return tinted

    def _update_ambient_side_spill_boost(self, colors: np.ndarray, intensity: float) -> None:
        luminance = colors[:, 0] * 0.2126 + colors[:, 1] * 0.7152 + colors[:, 2] * 0.0722
        mean_luma = float(np.mean(luminance)) * float(np.clip(intensity, 0.0, 1.0))
        previous_luma = self._front_ambient_scene_luma
        self._front_ambient_scene_luma = previous_luma * 0.78 + mean_luma * 0.22
        if mean_luma <= self.config.front_ambient_min_brightness:
            self._ambient_side_spill_boost *= 0.82
            return

        color_mean = np.mean(colors, axis=0)
        color_level = float(np.max(color_mean))
        if color_level <= 0.001:
            self._ambient_side_spill_boost *= 0.82
            return

        chroma = colors / np.maximum(np.max(colors, axis=1, keepdims=True), 0.001)
        uniformity = 1.0 - float(np.mean(np.std(chroma, axis=0)))
        saturation = (float(np.max(color_mean)) - float(np.min(color_mean))) / max(color_level, 0.001)
        rising = max(0.0, mean_luma - previous_luma)
        uniform_factor = np.clip((uniformity - 0.55) / 0.45, 0.0, 1.0)
        rise_factor = np.clip(rising / 0.045, 0.0, 1.0)
        brightness_factor = np.clip((mean_luma - self.config.front_ambient_min_brightness) / 0.45, 0.0, 1.0)
        color_factor = np.clip(0.35 + saturation * 0.65, 0.0, 1.0)
        target_boost = float(uniform_factor * rise_factor * brightness_factor * color_factor)
        self._ambient_side_spill_boost = max(self._ambient_side_spill_boost * 0.82, target_boost)

    def _front_ambient_led_colors(self) -> np.ndarray:
        colors = np.zeros((self.topology.total, 3), dtype=np.float32)
        idx = self.topology.front_ambient_indices()
        if idx.size == 0:
            return colors
        if self.front_ambient_strip_colors is not None and self.config.front_ambient_source == "top_strip":
            sampled = self._resampled_front_ambient_colors(idx.size)
            colors[idx] = sampled * self.front_ambient_intensity
        else:
            colors[idx] = self.front_ambient_color * self.front_ambient_intensity
        return colors

    def _resampled_front_ambient_colors(self, count: int) -> np.ndarray:
        assert self.front_ambient_strip_colors is not None
        if len(self.front_ambient_strip_colors) == count:
            return self.front_ambient_strip_colors
        sampled = cv2.resize(self.front_ambient_strip_colors.reshape(1, -1, 3), (count, 1), interpolation=cv2.INTER_AREA)
        return sampled.reshape(count, 3)

    def _apply_tv_sync(self, leds: np.ndarray) -> None:
        if not np.any(self._tv_mask):
            return
        tv_colors = self.topology.tv_sample_colors(self._tv_sample_frame())
        tv_only = np.array([mode == "tv_image" for mode in self.topology.sync_modes], dtype=bool)
        blend = np.array([mode == "blend" for mode in self.topology.sync_modes], dtype=bool)
        leds[tv_only] = tv_colors[tv_only]
        leds[blend] = leds[blend] * (1.0 - self._blend[blend]) + tv_colors[blend] * self._blend[blend]

    def _tv_sample_frame(self) -> np.ndarray | None:
        if self.tv_frame is None:
            return None
        blur = int(max(0, getattr(self.config, "tv_image_blur", 0)))
        if blur <= 0 or min(self.tv_frame.shape[:2]) < 3:
            return self.tv_frame
        kernel = max(3, blur)
        if kernel % 2 == 0:
            kernel += 1
        min_dim = min(self.tv_frame.shape[0], self.tv_frame.shape[1])
        max_kernel = min_dim if min_dim % 2 == 1 else min_dim - 1
        max_kernel = max(3, max_kernel)
        kernel = min(kernel, max_kernel)
        try:
            return cv2.GaussianBlur(self.tv_frame, (kernel, kernel), 0)
        except cv2.error:
            blur_w = max(1, min(kernel, self.tv_frame.shape[1]))
            blur_h = max(1, min(kernel, self.tv_frame.shape[0]))
            return cv2.blur(self.tv_frame, (blur_w, blur_h))

    def _radius_limit(self, intensity: float, kind: str) -> float:
        diagonal = max(0.1, self.topology.room_diagonal())
        if kind in {"flash", "explosion", "impact_pulse", "shockwave", "color_bloom", "underwater", "portal_vortex", "negative_wave", "ember_particles"} or intensity >= self.config.room_fill_threshold:
            return diagonal
        if kind in {"directional_sweep", "scene_wipe"}:
            return diagonal * 0.9
        if intensity >= self.config.level2_threshold:
            return diagonal * 0.65
        if intensity >= self.config.level1_threshold:
            return diagonal * 0.35
        return diagonal * 0.2

    def _event_brightness_gain(self, event: LightEvent) -> float:
        intensity = float(np.clip(event.intensity, 0.0, 1.0))
        if event.kind in {"front_ambient", "color_bloom", "underwater"}:
            return intensity
        if not self.config.variable_event_intensity:
            return 1.0
        return float(np.clip(intensity**0.7, 0.0, 1.0))

    def _post_process(self, leds: np.ndarray) -> np.ndarray:
        np.clip(leds, 0, 1, out=leds)
        leds = leds.astype(np.float32, copy=False)
        if abs(float(self.config.saturation) - 1.0) > 0.001:
            hsv = cv2.cvtColor(leds.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.config.saturation, 0, 1)
            leds = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)
        wb = np.array(self.config.white_balance, dtype=np.float32)
        leds = np.clip(leds * wb * self.config.brightness, 0, 1)
        gamma = max(0.1, self.config.gamma)
        if abs(gamma - 1.0) > 0.001:
            leds = np.power(leds, 1.0 / gamma)
        return np.clip(leds * 255.0, 0, 255).astype(np.uint8)


class NumbaSpatialRenderer(VectorizedSpatialRenderer):
    def __init__(self, config: EngineConfig, topology: SpatialRoomTopology):
        super().__init__(config, topology)
        self.render_backend = "numba"
        self.render_backend_status = "ready"
        self._kernel = self._load_kernel()
        self._warm_numba_kernel()

    @staticmethod
    def available() -> bool:
        return _numba_render_kernel() is not None

    def _load_kernel(self):
        global _NUMBA_RENDER_KERNEL
        if _NUMBA_RENDER_KERNEL is None:
            _NUMBA_RENDER_KERNEL = _numba_render_kernel()
        if _NUMBA_RENDER_KERNEL is None:
            raise RuntimeError("numba is not available")
        return _NUMBA_RENDER_KERNEL

    def _warm_numba_kernel(self) -> None:
        leds = np.zeros((1, 3), dtype=np.float32)
        self._kernel(
            leds,
            np.array([0], dtype=np.int32),
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([0.1], dtype=np.float32),
            np.array([5.0], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([0], dtype=np.int32),
            np.array([0.0], dtype=np.float32),
            5.0,
            4.0,
            3.0,
        )
        if leds.shape != (1, 3) or float(leds.max()) <= 0.0:
            raise RuntimeError("numba renderer self-test failed")

    def _render_spatial_waves(self, leds: np.ndarray, dt: float) -> None:
        if self.topology.total == 0 or self._spatial_indices.size == 0:
            return
        spatial_idx, positions, angles, skipped = self._priority_render_selection()
        self.last_spatial_priority_band_skips = skipped
        if spatial_idx.size == 0:
            return

        active: list[SpatialLightWave] = []
        for wave in self.waves:
            wave.age += dt
            wave.intensity *= wave.decay_rate ** (dt * 30.0)
            travelled = wave.speed * wave.age
            if wave.intensity < self.config.wave_min_intensity or travelled > wave.radius_limit + wave.spread * 3.0:
                continue
            active.append(wave)
        self.waves = active
        if not active:
            return

        kind_codes = np.array([_wave_kind_code(wave.kind) for wave in active], dtype=np.int32)
        colors = np.array([wave.color for wave in active], dtype=np.float32) / 255.0
        intensities = np.array([wave.intensity for wave in active], dtype=np.float32)
        origins = np.stack([wave.origin.astype(np.float32, copy=False) for wave in active]).astype(np.float32, copy=False)
        speeds = np.array([wave.speed for wave in active], dtype=np.float32)
        ages = np.array([wave.age for wave in active], dtype=np.float32)
        radius_limits = np.array([wave.radius_limit for wave in active], dtype=np.float32)
        spreads = np.array([wave.spread for wave in active], dtype=np.float32)
        widths = np.array([wave.width for wave in active], dtype=np.float32)
        direction_hints = np.array([wave.direction_hint for wave in active], dtype=np.int32)
        phases = np.array([wave.phase for wave in active], dtype=np.float32)
        room = self.topology.spatial.room
        self._kernel(
            leds,
            spatial_idx.astype(np.int32, copy=False),
            positions.astype(np.float32, copy=False),
            angles.astype(np.float32, copy=False),
            kind_codes,
            colors,
            intensities,
            origins,
            speeds,
            ages,
            radius_limits,
            spreads,
            widths,
            direction_hints,
            phases,
            float(self._room_diagonal),
            float(room.width),
            float(room.depth),
        )


class CppSpatialRenderer(VectorizedSpatialRenderer):
    def __init__(self, config: EngineConfig, topology: SpatialRoomTopology):
        super().__init__(config, topology)
        self.render_backend = "cpp"
        self.render_backend_status = "ready"
        self._native = self._load_native()
        self._sync_codes = np.array([1 if mode == "tv_image" else 2 if mode == "blend" else 0 for mode in topology.sync_modes], dtype=np.int32)
        self._blend_flat = topology.strip_blend.astype(np.float32, copy=True)
        self._tv_sample_kind, self._tv_sample_a, self._tv_sample_b = self._build_tv_sample_plan()
        self._ambient_ext_indices, self._ambient_ext_parent, self._ambient_ext_fade, self._ambient_ext_strength = self._build_ambient_extension_plan()
        self._cpp_priority_cache: dict[int | None, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
        self._allocate_wave_buffers()
        self._empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        self._empty_float_rgb = np.zeros((0, 3), dtype=np.float32)
        self._native_renderer = self._native.Renderer(self._native_renderer_config())
        self._warm_cpp_renderer()

    @staticmethod
    def available() -> bool:
        try:
            import _spatial_native  # noqa: F401
        except Exception:
            return False
        return True

    def _load_native(self):
        try:
            import _spatial_native
        except Exception as exc:
            raise RuntimeError(f"native C++ renderer is not available: {exc}") from exc
        return _spatial_native

    def _warm_cpp_renderer(self) -> None:
        saved = self.get_wave_snapshot()
        self.set_wave_snapshot([
            SpatialLightWave(
                color=(255, 0, 0),
                intensity=1.0,
                origin=self._room_center.copy(),
                speed=1.0,
                decay_rate=0.95,
                spread=0.5,
                age=0.0,
                radius_limit=max(1.0, self._room_diagonal),
                kind="explosion",
            )
        ])
        out = self.step(1.0 / 60.0)
        self.set_wave_snapshot(saved)
        self.previous_leds.fill(0.0)
        if out.shape != (self.topology.total, 3):
            raise RuntimeError("native C++ renderer self-test failed")

    def _native_renderer_config(self) -> dict:
        spatial_idx, positions, angles, _ = self._cpp_priority_selection()
        full_spatial_idx = np.ascontiguousarray(self._spatial_indices, dtype=np.int32)
        full_positions = np.ascontiguousarray(self._spatial_positions, dtype=np.float32)
        full_angles = np.ascontiguousarray(self._spatial_angles, dtype=np.float32)
        spatial_band = np.zeros(full_spatial_idx.shape[0], dtype=np.int32)
        if self._spatial_priority_bands:
            for band_index, local_indices in enumerate(self._spatial_priority_bands):
                spatial_band[np.asarray(local_indices, dtype=np.int32)] = band_index
        return {
            "total_leds": int(self.topology.total),
            "max_active_waves": int(self.config.max_active_waves),
            "room_diagonal": float(self._room_diagonal),
            "room_width": float(self.topology.spatial.room.width),
            "room_depth": float(self.topology.spatial.room.depth),
            "wave_speed": float(self.config.wave_speed),
            "wave_decay": float(self.config.wave_decay),
            "wave_spread": float(self.config.wave_spread),
            "wave_min_intensity": float(self.config.wave_min_intensity),
            "variable_event_intensity": bool(self.config.variable_event_intensity),
            "color_velocity_speed_boost": float(self.config.color_velocity_speed_boost),
            "color_velocity_decay_boost": float(self.config.color_velocity_decay_boost),
            "room_fill_threshold": float(self.config.room_fill_threshold),
            "level1_threshold": float(self.config.level1_threshold),
            "level2_threshold": float(self.config.level2_threshold),
            "color_smoothing": float(self.config.color_smoothing),
            "saturation": float(self.config.saturation),
            "brightness": float(self.config.brightness),
            "gamma": float(self.config.gamma),
            "white_balance": np.ascontiguousarray(self.config.white_balance, dtype=np.float32),
            "ambient_base_intensity": float(self.config.ambient_side_spill_base_intensity),
            "ambient_side_spill_enabled": bool(self.config.enabled_effects.get("ambient_side_spill", True)),
            "spatial_idx": full_spatial_idx,
            "positions": full_positions,
            "angles": full_angles,
            "spatial_band": np.ascontiguousarray(spatial_band, dtype=np.int32),
            "spatial_band_count": int(max(1, len(self._spatial_priority_bands))),
            "sync_codes": self._sync_codes,
            "blend": self._blend_flat,
            "tv_sample_kind": self._tv_sample_kind,
            "tv_sample_a": self._tv_sample_a,
            "tv_sample_b": self._tv_sample_b,
            "front_indices": np.ascontiguousarray(self.topology.front_ambient_indices(), dtype=np.int32),
            "ambient_ext_indices": self._ambient_ext_indices,
            "ambient_ext_parent": self._ambient_ext_parent,
            "ambient_ext_fade": self._ambient_ext_fade,
            "ambient_ext_strength": self._ambient_ext_strength,
        }

    def _allocate_wave_buffers(self) -> None:
        count = max(1, int(self.config.max_active_waves))
        self._wave_kind_buf = np.zeros(count, dtype=np.int32)
        self._wave_color_buf = np.zeros((count, 3), dtype=np.float32)
        self._wave_intensity_buf = np.zeros(count, dtype=np.float32)
        self._wave_origin_buf = np.zeros((count, 3), dtype=np.float32)
        self._wave_speed_buf = np.zeros(count, dtype=np.float32)
        self._wave_age_buf = np.zeros(count, dtype=np.float32)
        self._wave_radius_buf = np.zeros(count, dtype=np.float32)
        self._wave_spread_buf = np.zeros(count, dtype=np.float32)
        self._wave_width_buf = np.zeros(count, dtype=np.float32)
        self._wave_direction_buf = np.zeros(count, dtype=np.int32)
        self._wave_phase_buf = np.zeros(count, dtype=np.float32)

    def _build_tv_sample_plan(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kind = np.zeros(self.topology.total, dtype=np.int32)
        a = np.zeros(self.topology.total, dtype=np.float32)
        b = np.zeros(self.topology.total, dtype=np.float32)
        role_codes = {"top": 1, "bottom": 2, "left": 3, "right": 4}
        for strip in self.topology.spatial.strips:
            role = self.topology.effective_tv_role_for_strip(strip.id)
            if role == "none" or strip.sync_mode not in {"tv_image", "blend"}:
                continue
            start, end = self.topology.strip_ranges.get(strip.id, (0, 0))
            idx = np.arange(start, end, dtype=np.int32)
            if idx.size == 0:
                continue
            fill_idx = self.topology._tv_fill_indices(idx, role, strip)
            if fill_idx.size == 0:
                continue
            progress = self.topology._tv_edge_progress(fill_idx, role, strip)
            sample_role = self.topology._mirrored_sample_role(role)
            if role in {"top", "bottom"} and self.topology.spatial.tv.mirror_horizontal:
                progress = 1.0 - progress
            kind[fill_idx] = role_codes.get(sample_role if role in {"left", "right"} else role, 0)
            a[fill_idx] = progress.astype(np.float32, copy=False)

        roles = np.array(self.topology.effective_tv_roles, dtype=object)
        tv_capable = np.array([mode in {"tv_image", "blend"} for mode in self.topology.sync_modes], dtype=bool)
        unmapped = np.nonzero((roles == "none") & tv_capable & self.topology._same_wall_tv_mask())[0].astype(np.int32)
        if unmapped.size:
            tv = self.topology.spatial.tv
            left = tv.center_u - tv.width / 2.0
            bottom = tv.center_v - tv.height / 2.0
            a[unmapped] = np.clip((self.topology.wall_u[unmapped] - left) / max(0.001, tv.width), 0.0, 1.0)
            b[unmapped] = np.clip((self.topology.wall_v[unmapped] - bottom) / max(0.001, tv.height), 0.0, 1.0)
            kind[unmapped] = 5
        return kind, a, b

    def _build_ambient_extension_plan(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ext_indices: list[np.ndarray] = []
        parent_indices: list[np.ndarray] = []
        fades: list[np.ndarray] = []
        strengths: list[np.ndarray] = []
        for strip in self.topology.spatial.strips:
            parent_id = strip.extends_strip_id.strip()
            if strip.sync_mode != "spatial" or not parent_id or strip.extension_mode == "effects_only":
                continue
            start, end = self.topology.strip_ranges.get(strip.id, (0, 0))
            parent_start, parent_end = self.topology.strip_ranges.get(parent_id, (0, 0))
            idx = np.arange(start, end, dtype=np.int32)
            parent_idx = np.arange(parent_start, parent_end, dtype=np.int32)
            if idx.size == 0 or parent_idx.size == 0:
                continue
            edge_offset = self.topology._nearest_parent_edge_offset(idx, parent_idx)
            parent_edge = int(parent_idx[edge_offset])
            ext_indices.append(idx)
            parent_indices.append(np.full(idx.size, parent_edge, dtype=np.int32))
            fades.append(self.topology._fade_from_parent_edge(idx, parent_edge, strip).astype(np.float32, copy=False))
            strengths.append(np.full(idx.size, float(np.clip(strip.extension_strength, 0.0, 1.0)), dtype=np.float32))
        if not ext_indices:
            return (
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        return (
            np.ascontiguousarray(np.concatenate(ext_indices), dtype=np.int32),
            np.ascontiguousarray(np.concatenate(parent_indices), dtype=np.int32),
            np.ascontiguousarray(np.concatenate(fades), dtype=np.float32),
            np.ascontiguousarray(np.concatenate(strengths), dtype=np.float32),
        )

    def add_events(self, events: list[LightEvent]) -> None:
        payload = []
        for event in events:
            if event.kind in {"flash", "explosion"}:
                origin = self.topology.room_center()
            else:
                origin = self.topology.event_origin_point(event.edge)
            payload.append({
                "kind": event.kind,
                "intensity": float(event.intensity),
                "r": int(event.color[0]),
                "g": int(event.color[1]),
                "b": int(event.color[2]),
                "origin_x": float(origin[0]),
                "origin_y": float(origin[1]),
                "origin_z": float(origin[2]),
                "color_velocity": float(event.color_velocity),
                "direction_hint": int(event.direction_hint),
                "width": float(event.width),
                "pulse_count": int(event.pulse_count),
                "phase": float(event.phase),
                "secondary": bool(event.secondary),
            })
        if payload:
            self._native_renderer.add_events(payload)
        self.waves = self.get_wave_snapshot()

    def duck_lower_priority_waves(self, primary_kind: str, factor: float = 0.35) -> None:
        self._native_renderer.duck_lower_priority_waves(primary_kind, float(factor))
        self.waves = self.get_wave_snapshot()

    def set_tv_frame(self, frame_bgr: np.ndarray | None) -> None:
        self.tv_frame = frame_bgr
        self._native_renderer.set_tv_frame(None if frame_bgr is None else np.ascontiguousarray(self._tv_sample_frame(), dtype=np.uint8))

    def set_front_ambient_strip(self, colors: np.ndarray, intensity: float) -> None:
        if colors.size:
            self._native_renderer.set_front_ambient_strip(np.ascontiguousarray(np.clip(colors.astype(np.float32), 0.0, 1.0), dtype=np.float32), float(intensity))

    def set_ambient_spill_scene_boost(self, score: float, color: tuple[int, int, int]) -> None:
        self._native_renderer.set_ambient_spill_scene_boost(float(score), tuple(int(component) for component in color))

    def set_spatial_priority_budget(self, max_bands: int | None) -> None:
        super().set_spatial_priority_budget(max_bands)
        self._native_renderer.set_spatial_priority_budget(None if max_bands is None else int(max_bands))

    def get_wave_snapshot(self) -> list[SpatialLightWave]:
        waves: list[SpatialLightWave] = []
        for item in self._native_renderer.get_waves():
            waves.append(
                SpatialLightWave(
                    color=(int(item["r"]), int(item["g"]), int(item["b"])),
                    intensity=float(item["intensity"]),
                    origin=np.array([float(item["origin_x"]), float(item["origin_y"]), float(item["origin_z"])], dtype=np.float32),
                    speed=float(item["speed"]),
                    decay_rate=float(item["decay_rate"]),
                    spread=float(item["spread"]),
                    age=float(item["age"]),
                    radius_limit=float(item["radius_limit"]),
                    kind=str(item["kind"]),
                    color_velocity=float(item["color_velocity"]),
                    direction_hint=int(item["direction_hint"]),
                    width=float(item["width"]),
                    pulse_count=int(item["pulse_count"]),
                    phase=float(item["phase"]),
                    secondary=bool(item["secondary"]),
                )
            )
        return waves

    def set_wave_snapshot(self, waves: list[object]) -> None:
        payload = []
        for wave in waves:
            payload.append({
                "kind": str(wave.kind),
                "r": int(wave.color[0]),
                "g": int(wave.color[1]),
                "b": int(wave.color[2]),
                "intensity": float(wave.intensity),
                "origin_x": float(wave.origin[0]),
                "origin_y": float(wave.origin[1]),
                "origin_z": float(wave.origin[2]),
                "speed": float(wave.speed),
                "decay_rate": float(wave.decay_rate),
                "spread": float(wave.spread),
                "age": float(wave.age),
                "radius_limit": float(wave.radius_limit),
                "color_velocity": float(wave.color_velocity),
                "direction_hint": int(wave.direction_hint),
                "width": float(wave.width),
                "pulse_count": int(wave.pulse_count),
                "phase": float(wave.phase),
                "secondary": bool(wave.secondary),
            })
        self._native_renderer.set_waves(payload)
        self.waves = self.get_wave_snapshot()

    def _cpp_priority_selection(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        key = self._spatial_priority_budget
        cached = self._cpp_priority_cache.get(key)
        if cached is not None:
            return cached
        spatial_idx, positions, angles, skipped = self._priority_render_selection()
        cached = (
            np.ascontiguousarray(spatial_idx, dtype=np.int32),
            np.ascontiguousarray(positions, dtype=np.float32),
            np.ascontiguousarray(angles, dtype=np.float32),
            skipped,
        )
        self._cpp_priority_cache[key] = cached
        return cached

    def _active_wave_arrays(self, dt: float) -> tuple[list[SpatialLightWave], tuple[np.ndarray, ...]]:
        active: list[SpatialLightWave] = []
        for wave in self.waves:
            wave.age += dt
            wave.intensity *= wave.decay_rate ** (dt * 30.0)
            travelled = wave.speed * wave.age
            if wave.intensity < self.config.wave_min_intensity or travelled > wave.radius_limit + wave.spread * 3.0:
                continue
            active.append(wave)
        self.waves = active
        active_count = len(active)
        if active_count > self._wave_kind_buf.shape[0]:
            self._allocate_wave_buffers()
        for index, wave in enumerate(active):
            self._wave_kind_buf[index] = _wave_kind_code(wave.kind)
            self._wave_color_buf[index, 0] = wave.color[0] / 255.0
            self._wave_color_buf[index, 1] = wave.color[1] / 255.0
            self._wave_color_buf[index, 2] = wave.color[2] / 255.0
            self._wave_intensity_buf[index] = wave.intensity
            self._wave_origin_buf[index] = wave.origin
            self._wave_speed_buf[index] = wave.speed
            self._wave_age_buf[index] = wave.age
            self._wave_radius_buf[index] = wave.radius_limit
            self._wave_spread_buf[index] = wave.spread
            self._wave_width_buf[index] = wave.width
            self._wave_direction_buf[index] = wave.direction_hint
            self._wave_phase_buf[index] = wave.phase
        return active, (
            self._wave_kind_buf[:active_count],
            self._wave_color_buf[:active_count],
            self._wave_intensity_buf[:active_count],
            self._wave_origin_buf[:active_count],
            self._wave_speed_buf[:active_count],
            self._wave_age_buf[:active_count],
            self._wave_radius_buf[:active_count],
            self._wave_spread_buf[:active_count],
            self._wave_width_buf[:active_count],
            self._wave_direction_buf[:active_count],
            self._wave_phase_buf[:active_count],
        )

    def step(self, dt: float) -> np.ndarray:
        render_started = time.perf_counter()
        result = self._native_renderer.step(float(dt))
        self.last_render_ms = (time.perf_counter() - render_started) * 1000.0
        self.last_readback_ms = 0.0
        return np.asarray(result, dtype=np.uint8)


def create_spatial_renderer(config: EngineConfig, topology: SpatialRoomTopology) -> SpatialRenderer:
    requested = getattr(config, "spatial_renderer_backend", "auto")
    candidates = ["cpp", "numba", "numpy"] if requested == "auto" else [requested]
    last_error = ""
    for candidate in candidates:
        try:
            if candidate == "cpp":
                return CppSpatialRenderer(config, topology)
            if candidate == "numba":
                return NumbaSpatialRenderer(config, topology)
            if candidate == "numpy":
                return VectorizedSpatialRenderer(config, topology)
        except Exception as exc:
            last_error = f"{candidate}: {exc}"
            if requested != "auto":
                renderer = VectorizedSpatialRenderer(config, topology)
                renderer.render_backend_status = f"fallback from {last_error}"
                return renderer
    renderer = VectorizedSpatialRenderer(config, topology)
    if last_error:
        renderer.render_backend_status = f"fallback from {last_error}"
    return renderer
