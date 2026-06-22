from __future__ import annotations

from dataclasses import dataclass
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


class VectorizedSpatialRenderer:
    def __init__(self, config: EngineConfig, topology: SpatialRoomTopology):
        self.config = config
        self.topology = topology
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
            if event.kind == "front_ambient":
                color = np.array(event.color, dtype=np.float32) / 255.0
                self.front_ambient_color = self.front_ambient_color * 0.85 + color * 0.15
                self.front_ambient_intensity = max(self.front_ambient_intensity * 0.8, event.intensity)
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
                        intensity=event.intensity * (1.0 - pulse * 0.035),
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
        self._render_spatial_waves(leds, dt)
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
