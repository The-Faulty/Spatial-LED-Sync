from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import EngineConfig
from event_detector import LightEvent
from event_mixer import should_duck_wave
from topology import RoomTopology


@dataclass
class LightWave:
    color: tuple[int, int, int]
    intensity: float
    position: float
    velocity: float
    decay_rate: float
    spread_rate: float
    age: float
    direction: int
    radius_limit: float
    kind: str = "spill"
    color_velocity: float = 0.0
    width: float = 1.0
    pulse_count: int = 1
    phase: float = 0.0
    secondary: bool = False


class WaveEngine:
    def __init__(self, config: EngineConfig, topology: RoomTopology):
        self.config = config
        self.topology = topology
        self.accepts_front_ambient_strip_sample = True
        self.waves: list[LightWave] = []
        self.previous_leds = np.zeros((config.total_leds, 3), dtype=np.float32)
        self.front_ambient_color = np.zeros(3, dtype=np.float32)
        self.front_ambient_strip_colors: np.ndarray | None = None
        self.front_ambient_intensity = 0.0
        self._front_ambient_led_cache: list[int] | None = None
        self._front_ambient_led_set_cache: set[int] | None = None

    def add_events(self, events: list[LightEvent]) -> None:
        for event in events:
            if event.kind == "front_ambient":
                target = np.array(event.color, dtype=np.float32) / 255.0
                if self.front_ambient_strip_colors is None or self.config.front_ambient_source == "average":
                    self.front_ambient_color = self.front_ambient_color * 0.80 + target * 0.20
                self.front_ambient_intensity = max(self.front_ambient_intensity * 0.85, event.intensity)
                continue

            for position, direction, radius in self._wave_specs(event):
                color_velocity = float(np.clip(event.color_velocity, 0.0, 1.0))
                speed = self.config.wave_speed * (0.55 + event.intensity * 1.35)
                speed *= 1.0 + color_velocity * self.config.color_velocity_speed_boost
                spread = self.config.wave_spread * (0.65 + event.intensity)
                decay = self._decay_for_color_velocity(color_velocity)
                if event.kind == "flash":
                    speed *= 2.2
                    spread *= 2.0
                    decay *= 0.90
                elif event.kind in {"shockwave", "directional_sweep", "scene_wipe"}:
                    speed *= 1.55
                    spread *= max(0.8, event.width)
                    decay *= 0.95
                elif event.kind in {"color_bloom", "underwater"}:
                    speed *= 0.20
                    spread *= 3.5
                    decay = min(max(decay, 0.955), 0.975)
                elif event.kind in {"lightning", "impact_pulse"}:
                    speed *= 2.4
                    spread *= 2.3
                    decay *= 0.78
                elif event.kind == "energy_trail":
                    speed *= 1.7
                    spread *= 0.75
                    decay *= 0.90
                elif event.kind == "ember_particles":
                    speed *= 0.65
                    spread = max(spread * 0.75, 3.0)
                    decay *= 0.88
                count = max(1, int(event.pulse_count if event.kind in {"ember_particles", "lightning"} else 1))
                for pulse in range(count):
                    pulse_phase = event.phase + pulse / max(1, count)
                    self.waves.append(
                        LightWave(
                            color=event.color,
                            intensity=event.intensity * (1.0 - pulse * 0.035),
                            position=(float(position) + pulse_phase * self.config.total_leds * 0.2) % self.config.total_leds,
                            velocity=speed * (0.85 + pulse_phase * 0.35),
                            decay_rate=decay,
                            spread_rate=spread,
                            age=0.0,
                            direction=direction,
                            radius_limit=radius,
                            kind=event.kind,
                            color_velocity=color_velocity,
                            width=max(0.05, event.width),
                            pulse_count=count,
                            phase=pulse_phase,
                            secondary=event.secondary,
                        )
                    )
        if len(self.waves) > self.config.max_active_waves:
            self.waves = sorted(self.waves, key=lambda w: w.intensity, reverse=True)[: self.config.max_active_waves]

    def duck_lower_priority_waves(self, primary_kind: str, factor: float = 0.35) -> None:
        for wave in self.waves:
            if should_duck_wave(primary_kind, wave.kind):
                wave.intensity *= factor

    def set_tv_frame(self, frame_bgr: np.ndarray | None) -> None:
        return None

    def set_front_ambient_strip(self, colors: np.ndarray, intensity: float) -> None:
        if colors.size == 0:
            return
        target = np.clip(colors.astype(np.float32), 0.0, 1.0)
        if self.front_ambient_strip_colors is None or self.front_ambient_strip_colors.shape != target.shape:
            self.front_ambient_strip_colors = target
        else:
            self.front_ambient_strip_colors = self.front_ambient_strip_colors * 0.70 + target * 0.30
        self.front_ambient_color = np.mean(self.front_ambient_strip_colors, axis=0)
        self.front_ambient_intensity = max(self.front_ambient_intensity * 0.80, float(np.clip(intensity, 0.0, 1.0)))

    def front_ambient_led_count(self) -> int:
        return len(self._front_ambient_leds())

    def _decay_for_color_velocity(self, color_velocity: float) -> float:
        base_decay = self.config.wave_decay
        fade_boost = color_velocity * self.config.color_velocity_decay_boost
        return float(np.clip(base_decay - fade_boost, 0.50, 0.999))

    def _wave_specs(self, event: LightEvent) -> list[tuple[int, int, float]]:
        if self.config.lighting_mode == "front_ambient" and event.edge == "top":
            return self._front_ambient_boundary_specs(event)

        origin = self.topology.origin_for_edge(event.edge, event.intensity)
        if event.kind in ("explosion", "flash", "impact_pulse", "shockwave", "color_bloom", "underwater", "portal_vortex", "negative_wave", "ember_particles"):
            radius = self.config.total_leds / 2
            directions = (-1, 1)
        elif event.kind in {"camera_pan", "directional_sweep", "scene_wipe", "energy_trail"} and event.direction_hint:
            radius = self.config.total_leds / 2 if event.intensity > 0.55 else self.config.strong_spill_radius
            directions = (event.direction_hint,)
        else:
            radius = origin.radius
            directions = origin.directions
        return [(origin.led, direction, radius) for direction in directions]

    def _front_ambient_boundary_specs(self, event: LightEvent) -> list[tuple[int, int, float]]:
        if event.kind in ("explosion", "flash", "impact_pulse", "shockwave", "color_bloom", "underwater", "portal_vortex", "negative_wave") or event.intensity >= self.config.room_fill_threshold:
            radius = self.config.total_leds / 2
        elif event.intensity >= self.config.level2_threshold:
            radius = self.config.strong_spill_radius
        else:
            radius = self.config.normal_spill_radius

        left_direction = self.topology.direction_from_to(self.config.tv_left_boundary, "left")
        right_direction = self.topology.direction_from_to(self.config.tv_right_boundary, "right")
        if event.kind == "camera_pan" and event.direction_hint:
            if event.direction_hint > 0:
                return [(self.config.tv_right_boundary, right_direction, radius)]
            return [(self.config.tv_left_boundary, left_direction, radius)]
        return [
            (self.config.tv_left_boundary, left_direction, radius),
            (self.config.tv_right_boundary, right_direction, radius),
        ]

    def step(self, dt: float) -> np.ndarray:
        leds = np.zeros((self.config.total_leds, 3), dtype=np.float32)
        self._render_front_ambient(leds, dt)
        active: list[LightWave] = []
        for wave in self.waves:
            wave.age += dt
            wave.position = (wave.position + wave.direction * wave.velocity * dt * 30.0) % self.config.total_leds
            wave.intensity *= wave.decay_rate ** (dt * 30.0)
            travelled = abs(wave.velocity * wave.age * 30.0)
            if wave.intensity < self.config.wave_min_intensity or travelled > wave.radius_limit + wave.spread_rate * 2:
                continue
            active.append(wave)
            self._render_wave(leds, wave, travelled)
        self.waves = active
        leds = np.clip(leds, 0.0, 1.0)
        alpha = self.config.color_smoothing
        leds = self.previous_leds * alpha + leds * (1.0 - alpha)
        self.previous_leds = leds
        return self._post_process(leds)

    def _render_front_ambient(self, leds: np.ndarray, dt: float) -> None:
        if self.front_ambient_intensity <= self.config.wave_min_intensity:
            return
        ambient_leds = self._front_ambient_leds()
        if not ambient_leds:
            return
        if self.front_ambient_strip_colors is not None and self.config.front_ambient_source == "top_strip":
            colors = self._resampled_strip_colors(len(ambient_leds))
            for led, color in zip(ambient_leds, colors):
                leds[led] += color * self.front_ambient_intensity
        else:
            center = self.config.tv_center_led
            max_dist = max(1, max(self.topology.circular_distance(center, led) for led in ambient_leds))
            for led in ambient_leds:
                dist = self.topology.circular_distance(center, led) / max_dist
                envelope = 0.35 + 0.65 * np.exp(-(dist ** 2) / 0.55)
                leds[led] += self.front_ambient_color * self.front_ambient_intensity * envelope
        self.front_ambient_intensity *= 0.92 ** (dt * 30.0)

    def _resampled_strip_colors(self, count: int) -> np.ndarray:
        assert self.front_ambient_strip_colors is not None
        if len(self.front_ambient_strip_colors) == count:
            return self.front_ambient_strip_colors
        sampled = cv2.resize(self.front_ambient_strip_colors.reshape(1, -1, 3), (count, 1), interpolation=cv2.INTER_AREA)
        return sampled.reshape(count, 3)

    def _front_ambient_leds(self) -> list[int]:
        if self._front_ambient_led_cache is not None:
            return self._front_ambient_led_cache
        front_leds = self.topology.range_inclusive(self.config.front_wall.start, self.config.front_wall.end)
        if not front_leds:
            self._front_ambient_led_cache = []
            return self._front_ambient_led_cache
        try:
            left_idx = front_leds.index(self.config.tv_left_boundary)
            right_idx = front_leds.index(self.config.tv_right_boundary)
            front_leds.index(self.config.tv_center_led)
        except ValueError:
            self._front_ambient_led_cache = front_leds
            return self._front_ambient_led_cache

        forward = self._wall_path(front_leds, left_idx, right_idx, 1)
        backward = self._wall_path(front_leds, left_idx, right_idx, -1)
        if self.config.tv_center_led in forward and self.config.tv_center_led not in backward:
            self._front_ambient_led_cache = forward
            return self._front_ambient_led_cache
        if self.config.tv_center_led in backward and self.config.tv_center_led not in forward:
            self._front_ambient_led_cache = backward
            return self._front_ambient_led_cache
        self._front_ambient_led_cache = forward if len(forward) <= len(backward) else backward
        return self._front_ambient_led_cache

    @staticmethod
    def _wall_path(wall_leds: list[int], start_idx: int, end_idx: int, step: int) -> list[int]:
        path = [wall_leds[start_idx]]
        idx = start_idx
        guard = 0
        while idx != end_idx and guard <= len(wall_leds):
            idx = (idx + step) % len(wall_leds)
            path.append(wall_leds[idx])
            guard += 1
        return path

    def _render_wave(self, leds: np.ndarray, wave: LightWave, travelled: float) -> None:
        rgb = np.array(wave.color, dtype=np.float32) / 255.0
        for led in range(self.config.total_leds):
            if self._is_reserved_front_ambient_led(led):
                continue
            dist = self.topology.signed_distance(wave.position, led, wave.direction)
            reverse_dist = self.topology.signed_distance(wave.position, led, -wave.direction)
            bidirectional = wave.kind in {"flash", "explosion", "impact_pulse", "shockwave", "color_bloom", "underwater", "portal_vortex", "negative_wave"}
            dist = min(dist, reverse_dist if bidirectional else dist)
            spread_scale = 4.5 if wave.kind in {"color_bloom", "underwater"} else 2.8
            if dist > wave.spread_rate * spread_scale:
                continue
            if wave.kind == "shockwave":
                center = travelled
                ring_width = max(1.0, wave.spread_rate * 0.45)
                envelope = np.exp(-((dist - center) ** 2) / (2.0 * ring_width**2))
            elif wave.kind in {"flame_shimmer", "underwater", "portal_vortex"}:
                shimmer = 0.55 + 0.45 * np.sin(led * 0.37 + wave.age * (14.0 if wave.kind == "flame_shimmer" else 4.0) + wave.phase * 6.28) ** 2
                envelope = np.exp(-(dist ** 2) / (2.0 * max(1.0, wave.spread_rate * spread_scale / 2.0) ** 2)) * shimmer
            elif wave.kind == "energy_trail":
                head = np.exp(-(dist ** 2) / (2.0 * max(1.0, wave.spread_rate) ** 2))
                tail = np.exp(-((dist - wave.spread_rate * 3.0) ** 2) / (2.0 * max(1.0, wave.spread_rate * 2.2) ** 2)) * 0.45
                envelope = max(head, tail)
            else:
                envelope = np.exp(-(dist ** 2) / (2.0 * max(1.0, wave.spread_rate) ** 2))
            travel_fade = max(0.0, 1.0 - travelled / max(1.0, wave.radius_limit))
            contribution = rgb * wave.intensity * envelope * (0.25 + 0.75 * travel_fade)
            if wave.kind == "negative_wave":
                leds[led] -= wave.intensity * envelope * 0.65
            else:
                leds[led] += contribution

    def _is_reserved_front_ambient_led(self, led: int) -> bool:
        if self.config.lighting_mode != "front_ambient":
            return False
        if self.topology.wall_for_led(led) != "front":
            return False
        if self._front_ambient_led_set_cache is None:
            self._front_ambient_led_set_cache = set(self._front_ambient_leds())
        return led in self._front_ambient_led_set_cache

    def _post_process(self, leds: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(np.clip(leds.reshape(1, -1, 3), 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.config.saturation, 0, 1)
        leds = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(-1, 3)
        wb = np.array(self.config.white_balance, dtype=np.float32)
        leds = np.clip(leds * wb * self.config.brightness, 0, 1)
        gamma = max(0.1, self.config.gamma)
        leds = np.power(leds, 1.0 / gamma)
        return np.clip(leds * 255.0, 0, 255).astype(np.uint8)
