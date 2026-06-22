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

    def front_ambient_led_count(self) -> int:
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
        self.accepts_front_ambient_strip_sample = False
        self.waves: list[SpatialLightWave] = []
        self.previous_leds = np.zeros((topology.total, 3), dtype=np.float32)
        self.tv_frame: np.ndarray | None = None
        self.front_ambient_color = np.zeros(3, dtype=np.float32)
        self.front_ambient_intensity = 0.0
        self._spatial_mask = np.array([mode in {"spatial", "blend"} for mode in topology.sync_modes], dtype=bool)
        self._tv_mask = np.array([mode in {"tv_image", "blend"} for mode in topology.sync_modes], dtype=bool)
        self._blend = topology.strip_blend.reshape(-1, 1)

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
            self.front_ambient_color = np.mean(colors.astype(np.float32), axis=0)
            self.front_ambient_intensity = max(self.front_ambient_intensity * 0.8, float(np.clip(intensity, 0.0, 1.0)))

    def front_ambient_led_count(self) -> int:
        return max(1, int(np.count_nonzero(self._tv_mask)))

    def step(self, dt: float) -> np.ndarray:
        leds = np.zeros((self.topology.total, 3), dtype=np.float32)
        self._render_spatial_waves(leds, dt)
        self._apply_tv_sync(leds)
        leds = np.clip(leds, 0.0, 1.0)
        alpha = self.config.color_smoothing
        leds = self.previous_leds * alpha + leds * (1.0 - alpha)
        self.previous_leds = leds
        return self._post_process(leds)

    def _render_spatial_waves(self, leds: np.ndarray, dt: float) -> None:
        if self.topology.total == 0:
            return
        active: list[SpatialLightWave] = []
        positions = self.topology.positions
        for wave in self.waves:
            wave.age += dt
            wave.intensity *= wave.decay_rate ** (dt * 30.0)
            travelled = wave.speed * wave.age
            if wave.intensity < self.config.wave_min_intensity or travelled > wave.radius_limit + wave.spread * 3.0:
                continue
            active.append(wave)
            distances = np.linalg.norm(positions - wave.origin.reshape(1, 3), axis=1)
            if wave.kind in {"flash", "explosion", "impact_pulse"}:
                fill_radius = max(wave.spread, travelled + wave.spread)
                envelope = np.exp(-(distances**2) / (2.0 * fill_radius**2))
            elif wave.kind == "shockwave":
                ring_width = max(wave.spread * 0.45, wave.width * 0.08)
                envelope = np.exp(-((distances - travelled) ** 2) / (2.0 * ring_width**2))
            elif wave.kind in {"directional_sweep", "scene_wipe"}:
                axis = positions[:, 0] if abs(wave.direction_hint) >= 0 else positions[:, 2]
                room = self.topology.spatial.room
                span = max(room.width, room.depth, 0.1)
                front = (travelled / max(0.1, wave.radius_limit)) * span
                if wave.direction_hint < 0:
                    front = span - front
                envelope = np.exp(-((axis - front) ** 2) / (2.0 * max(0.12, wave.width * 0.25) ** 2))
            elif wave.kind == "energy_trail":
                head = np.exp(-((distances - travelled) ** 2) / (2.0 * wave.spread**2))
                tail = np.exp(-((distances - max(0.0, travelled - wave.spread * 3.0)) ** 2) / (2.0 * (wave.spread * 2.2) ** 2)) * 0.45
                envelope = np.maximum(head, tail)
            elif wave.kind == "flame_shimmer":
                noise = 0.65 + 0.35 * np.sin((positions[:, 0] * 7.0 + positions[:, 2] * 5.0 + wave.age * 18.0 + wave.phase * 6.28))
                envelope = np.exp(-(distances**2) / (2.0 * max(wave.spread * 2.4, 0.1) ** 2)) * noise
            elif wave.kind == "underwater":
                ripple = 0.45 + 0.55 * np.sin(distances * 6.0 - wave.age * 4.0 + wave.phase * 6.28) ** 2
                envelope = np.exp(-(distances**2) / (2.0 * max(wave.spread * 5.0, 0.1) ** 2)) * ripple
            elif wave.kind == "portal_vortex":
                centered = positions - self.topology.room_center().reshape(1, 3)
                angles = np.arctan2(centered[:, 2], centered[:, 0])
                spiral = 0.5 + 0.5 * np.sin(angles * 3.0 * max(1, wave.direction_hint) + distances * 4.0 - wave.age * 6.0)
                envelope = np.exp(-((distances - travelled * 0.45) ** 2) / (2.0 * (wave.spread * 2.5) ** 2)) * spiral
            elif wave.kind in {"color_bloom", "lightning", "ember_particles"}:
                envelope = np.exp(-(distances**2) / (2.0 * max(wave.spread * (4.0 if wave.kind == "color_bloom" else 1.0), 0.1) ** 2))
            else:
                envelope = np.exp(-((distances - travelled) ** 2) / (2.0 * wave.spread**2))
            travel_fade = max(0.0, 1.0 - travelled / max(0.1, wave.radius_limit))
            contribution = (np.array(wave.color, dtype=np.float32) / 255.0) * wave.intensity * (0.2 + 0.8 * travel_fade)
            if wave.kind == "negative_wave":
                leds[self._spatial_mask] -= envelope[self._spatial_mask].reshape(-1, 1) * wave.intensity * 0.65
            else:
                leds[self._spatial_mask] += envelope[self._spatial_mask].reshape(-1, 1) * contribution
        self.waves = active
        if self.front_ambient_intensity > self.config.wave_min_intensity:
            leds[self._tv_mask] += self.front_ambient_color * self.front_ambient_intensity
            self.front_ambient_intensity *= 0.92 ** (dt * 30.0)

    def _apply_tv_sync(self, leds: np.ndarray) -> None:
        if not np.any(self._tv_mask):
            return
        tv_colors = self.topology.tv_sample_colors(self.tv_frame)
        tv_only = np.array([mode == "tv_image" for mode in self.topology.sync_modes], dtype=bool)
        blend = np.array([mode == "blend" for mode in self.topology.sync_modes], dtype=bool)
        leds[tv_only] = tv_colors[tv_only]
        leds[blend] = leds[blend] * (1.0 - self._blend[blend]) + tv_colors[blend] * self._blend[blend]
        if self.config.enabled_effects.get("ambient_side_spill", True):
            ambient = self.topology.ambient_extension_colors(self.tv_frame)
            ambient_mask = np.any(ambient > 0.0, axis=1)
            leds[ambient_mask] = np.maximum(leds[ambient_mask], ambient[ambient_mask])

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
        hsv = cv2.cvtColor(np.clip(leds.reshape(1, -1, 3), 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.config.saturation, 0, 1)
        leds = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)
        wb = np.array(self.config.white_balance, dtype=np.float32)
        leds = np.clip(leds * wb * self.config.brightness, 0, 1)
        gamma = max(0.1, self.config.gamma)
        leds = np.power(leds, 1.0 / gamma)
        return np.clip(leds * 255.0, 0, 255).astype(np.uint8)
