from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from config import EngineConfig
from event_detector import LightEvent
from spatial_topology import SpatialRoomTopology


class SpatialRenderer(Protocol):
    waves: list[object]

    def add_events(self, events: list[LightEvent]) -> None:
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


class VectorizedSpatialRenderer:
    def __init__(self, config: EngineConfig, topology: SpatialRoomTopology):
        self.config = config
        self.topology = topology
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
            radius_limit = self._radius_limit(event.intensity, event.kind)
            origin = self.topology.room_center() if event.kind in {"flash", "explosion"} else self.topology.event_origin_point(event.edge)
            self.waves.append(
                SpatialLightWave(
                    color=event.color,
                    intensity=event.intensity,
                    origin=origin,
                    speed=speed,
                    decay_rate=decay,
                    spread=spread,
                    age=0.0,
                    radius_limit=radius_limit,
                    kind=event.kind,
                    color_velocity=color_velocity,
                )
            )
        if len(self.waves) > self.config.max_active_waves:
            self.waves = sorted(self.waves, key=lambda wave: wave.intensity, reverse=True)[: self.config.max_active_waves]

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
            if wave.kind in {"flash", "explosion"}:
                fill_radius = max(wave.spread, travelled + wave.spread)
                envelope = np.exp(-(distances**2) / (2.0 * fill_radius**2))
            else:
                envelope = np.exp(-((distances - travelled) ** 2) / (2.0 * wave.spread**2))
            travel_fade = max(0.0, 1.0 - travelled / max(0.1, wave.radius_limit))
            contribution = (np.array(wave.color, dtype=np.float32) / 255.0) * wave.intensity * (0.2 + 0.8 * travel_fade)
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

    def _radius_limit(self, intensity: float, kind: str) -> float:
        diagonal = max(0.1, self.topology.room_diagonal())
        if kind in {"flash", "explosion"} or intensity >= self.config.room_fill_threshold:
            return diagonal
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
