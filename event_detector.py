from __future__ import annotations

from dataclasses import dataclass

from config import EngineConfig
from motion_detector import MotionAnalysis


@dataclass
class LightEvent:
    edge: str
    intensity: float
    color: tuple[int, int, int]
    kind: str
    direction_hint: int = 0
    color_velocity: float = 0.0
    effect_id: str = ""
    origin_hint: str = ""
    width: float = 1.0
    duration: float = 1.0
    pulse_count: int = 1
    phase: float = 0.0
    secondary: bool = False
    primary: bool = False
    suppressed_by: str = ""


class EventDetector:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.previous_brightness = 0.0
        self.frame_index = 0
        self.last_emit_frame: dict[str, int] = {}

    def detect(self, analysis: MotionAnalysis) -> list[LightEvent]:
        self.frame_index += 1
        events: list[LightEvent] = []
        if self.config.lighting_mode == "front_ambient" and self._effect_enabled("front_ambient"):
            ambient = self._front_ambient_event(analysis)
            if ambient:
                events.append(ambient)

        if self._effect_enabled("spill"):
            for edge, activity in analysis.edge_activity.items():
                if edge == "bottom" and not self.config.bottom_edge_enabled:
                    continue
                if activity.toward_edge <= 0.12 and activity.coverage <= 0.08:
                    continue
                intensity = self._score(analysis, edge)
                if self.config.lighting_mode == "front_ambient" and edge in {"left", "right", "bottom"} and intensity < self.config.side_major_threshold:
                    continue
                if intensity >= self.config.level1_threshold:
                    events.append(LightEvent(edge, intensity, activity.color, "spill", color_velocity=activity.color_velocity, effect_id="spill"))

        if self._effect_enabled("top_color_exit"):
            events.extend(self._top_color_exit_events(analysis))

        if self._effect_enabled("energy_trail"):
            events.extend(self._energy_trail_events(analysis))

        if self._effect_enabled("flash") and self._is_flash(analysis):
            intensity = min(1.0, analysis.changed_fraction + analysis.brightness)
            events.append(LightEvent("top", intensity, analysis.dominant_color, "flash", color_velocity=analysis.color_velocity, effect_id="flash", width=1.6, duration=0.65))

        if self._effect_enabled("shockwave") and self._is_shockwave(analysis):
            events.append(LightEvent("top", min(1.0, analysis.changed_fraction * 1.4), analysis.dominant_color, "shockwave", color_velocity=analysis.color_velocity, effect_id="shockwave", width=0.22, duration=1.4))

        if self._effect_enabled("explosion") and self._is_explosion(analysis):
            events.append(LightEvent("top", 1.0, analysis.dominant_color, "explosion", color_velocity=analysis.color_velocity, effect_id="explosion", width=2.0, duration=1.2))
            if self._effect_enabled("ember_particles"):
                events.append(LightEvent("top", 0.75, analysis.dominant_color, "ember_particles", color_velocity=analysis.color_velocity, effect_id="ember_particles", pulse_count=10, secondary=True))

        if self._effect_enabled("lightning") and self._is_lightning(analysis):
            events.append(LightEvent("top", min(1.0, analysis.brightness + analysis.changed_fraction), (235, 245, 255), "lightning", color_velocity=analysis.color_velocity, effect_id="lightning", pulse_count=5, duration=0.45))
            if self._effect_enabled("ember_particles"):
                events.append(LightEvent("top", 0.35, (255, 180, 80), "ember_particles", effect_id="ember_particles", pulse_count=5, secondary=True))

        if self._effect_enabled("impact_pulse") and self._is_impact_pulse(analysis):
            events.append(LightEvent("top", min(1.0, analysis.rate_of_change * 3.0 + analysis.changed_fraction), analysis.dominant_color, "impact_pulse", effect_id="impact_pulse", width=2.2, duration=0.55))

        if self._effect_enabled("negative_wave") and self._is_negative_wave(analysis):
            events.append(LightEvent("top", min(1.0, self.previous_brightness - analysis.brightness), (0, 0, 0), "negative_wave", effect_id="negative_wave", width=1.5, duration=0.9))

        if self._effect_enabled("color_bloom") and self._can_emit("color_bloom", 50) and self._is_color_bloom(analysis):
            self._mark_emitted("color_bloom")
            events.append(LightEvent("top", min(0.65, analysis.saturation * analysis.brightness), analysis.dominant_color, "color_bloom", effect_id="color_bloom", width=2.0, duration=2.5))

        if self._effect_enabled("flame_shimmer") and self._can_emit("flame_shimmer", 18) and self._is_flame_shimmer(analysis):
            self._mark_emitted("flame_shimmer")
            events.append(LightEvent("top", min(1.0, analysis.saturation * 0.6 + analysis.rate_of_change * 2.0), analysis.dominant_color, "flame_shimmer", effect_id="flame_shimmer", pulse_count=4, duration=0.8))
            if self._effect_enabled("ember_particles"):
                events.append(LightEvent("top", 0.45, analysis.dominant_color, "ember_particles", effect_id="ember_particles", pulse_count=7, secondary=True))

        if self._effect_enabled("underwater") and self._can_emit("underwater", 55) and self._is_underwater(analysis):
            self._mark_emitted("underwater")
            events.append(LightEvent("top", min(0.55, analysis.brightness + analysis.saturation * 0.35), analysis.dominant_color, "underwater", effect_id="underwater", width=1.2, duration=2.0, phase=analysis.color_velocity))

        if self._effect_enabled("directional_sweep"):
            sweep = self._directional_sweep(analysis)
            if sweep:
                events.append(LightEvent("top", sweep[1], analysis.dominant_color, "directional_sweep", direction_hint=sweep[0], color_velocity=analysis.color_velocity, effect_id="directional_sweep", width=1.8, duration=1.1))

        if self._effect_enabled("scene_wipe"):
            wipe = self._scene_wipe(analysis)
            if wipe:
                events.append(LightEvent("top", wipe[1], analysis.dominant_color, "scene_wipe", direction_hint=wipe[0], color_velocity=analysis.color_velocity, effect_id="scene_wipe", width=1.4, duration=1.0))

        if self._effect_enabled("portal_vortex") and self._is_portal_vortex(analysis):
            direction = 1 if analysis.dominant_flow[0] >= 0 else -1
            events.append(LightEvent("top", min(1.0, analysis.flow_confidence + analysis.changed_fraction), analysis.dominant_color, "portal_vortex", direction_hint=direction, color_velocity=analysis.color_velocity, effect_id="portal_vortex", width=1.0, duration=1.8))

        pan = self._camera_pan(analysis)
        if self._effect_enabled("camera_pan") and pan:
            events.append(LightEvent("top", pan[1], analysis.dominant_color, "camera_pan", direction_hint=pan[0], color_velocity=analysis.color_velocity, effect_id="camera_pan", width=1.3, duration=1.1))

        self.previous_brightness = analysis.brightness
        return sorted(events, key=lambda e: e.intensity, reverse=True)

    def _effect_enabled(self, kind: str) -> bool:
        return self.config.enabled_effects.get(kind, True)

    def _can_emit(self, kind: str, cooldown_frames: int) -> bool:
        return self.frame_index - self.last_emit_frame.get(kind, -cooldown_frames) >= cooldown_frames

    def _mark_emitted(self, kind: str) -> None:
        self.last_emit_frame[kind] = self.frame_index

    def _front_ambient_event(self, analysis: MotionAnalysis) -> LightEvent | None:
        if analysis.brightness < self.config.front_ambient_min_brightness:
            return None
        base = analysis.brightness * 0.70 + analysis.saturation * 0.30
        intensity = max(0.0, min(1.0, base * self.config.front_ambient_intensity))
        if intensity <= 0.01:
            return None
        return LightEvent("top", intensity, analysis.dominant_color, "front_ambient", color_velocity=analysis.color_velocity, effect_id="front_ambient")

    def _score(self, analysis: MotionAnalysis, edge: str) -> float:
        a = analysis.edge_activity[edge]
        w = self.config.weights
        weighted = (
            w["motion"] * a.magnitude
            + w["direction"] * a.toward_edge
            + w["brightness"] * analysis.brightness
            + w["saturation"] * analysis.saturation
            + w["coverage"] * a.coverage
            + w["edge"] * a.confidence
            + w["change"] * min(1.0, analysis.rate_of_change * 4.0 + analysis.changed_fraction + a.color_velocity * 0.65)
        )
        return max(0.0, min(1.0, weighted / max(0.001, sum(w.values())) * 1.55))

    def _top_color_exit_events(self, analysis: MotionAnalysis) -> list[LightEvent]:
        events: list[LightEvent] = []
        for edge, activity in analysis.top_corner_activity.items():
            color_exit_velocity = max(activity.color_velocity, activity.color_motion_velocity)
            if color_exit_velocity < self.config.top_color_exit_velocity_threshold:
                continue
            if activity.magnitude < self.config.top_color_exit_motion_threshold:
                continue
            if activity.coverage < self.config.top_color_exit_coverage_threshold:
                continue
            if activity.toward_edge < 0.22:
                continue

            intensity = (
                color_exit_velocity * 0.40
                + activity.toward_edge * 0.25
                + activity.magnitude * 0.18
                + activity.coverage * 0.10
                + analysis.saturation * 0.07
            )
            intensity = max(0.0, min(1.0, intensity * 1.45))
            if self.config.lighting_mode == "front_ambient" and intensity < self.config.side_major_threshold:
                continue
            if intensity < min(self.config.level1_threshold, self.config.side_major_threshold):
                continue
            events.append(
                LightEvent(
                    edge=edge,
                    intensity=intensity,
                    color=activity.color,
                    kind="spill",
                    direction_hint=-1 if edge == "left" else 1,
                    color_velocity=color_exit_velocity,
                    effect_id="top_color_exit",
                    width=0.7,
                )
            )
        return events

    def _energy_trail_events(self, analysis: MotionAnalysis) -> list[LightEvent]:
        events: list[LightEvent] = []
        for edge, activity in analysis.edge_activity.items():
            if activity.coverage < 0.05 or activity.magnitude < 0.18:
                continue
            if activity.toward_edge < 0.18 and activity.color_motion_velocity < 0.25:
                continue
            if max(activity.color_velocity, activity.color_motion_velocity, analysis.color_velocity) < 0.12:
                continue
            intensity = min(1.0, activity.magnitude * 0.45 + activity.coverage * 0.25 + activity.toward_edge * 0.25 + activity.color_motion_velocity * 0.35)
            if intensity >= self.config.level1_threshold:
                events.append(LightEvent(edge, intensity, activity.color, "energy_trail", color_velocity=activity.color_motion_velocity, effect_id="energy_trail", width=0.6, duration=1.3))
        return events

    def _is_flash(self, analysis: MotionAnalysis) -> bool:
        return analysis.changed_fraction >= self.config.flash_changed_fraction and analysis.brightness > 0.35

    def _is_shockwave(self, analysis: MotionAnalysis) -> bool:
        return analysis.changed_fraction >= self.config.flash_changed_fraction * 0.75 and analysis.rate_of_change > 0.08

    def _is_explosion(self, analysis: MotionAnalysis) -> bool:
        brightness_spike = analysis.brightness - self.previous_brightness
        return (
            brightness_spike >= self.config.explosion_brightness_spike
            and analysis.saturation > 0.35
            and analysis.changed_fraction > 0.22
        )

    def _is_lightning(self, analysis: MotionAnalysis) -> bool:
        return analysis.brightness > 0.68 and analysis.changed_fraction > 0.18 and analysis.saturation < 0.45

    def _is_impact_pulse(self, analysis: MotionAnalysis) -> bool:
        return analysis.rate_of_change > 0.10 and analysis.changed_fraction > 0.12

    def _is_negative_wave(self, analysis: MotionAnalysis) -> bool:
        return (self.previous_brightness - analysis.brightness) > 0.16 and analysis.changed_fraction > 0.18

    def _is_color_bloom(self, analysis: MotionAnalysis) -> bool:
        return analysis.saturation > 0.68 and analysis.brightness > 0.24 and 0.01 <= analysis.changed_fraction < 0.28

    def _is_flame_shimmer(self, analysis: MotionAnalysis) -> bool:
        r, g, b = analysis.dominant_color
        warm = r > 140 and g > 45 and b < 110 and r >= g
        return warm and analysis.saturation > 0.42 and (analysis.rate_of_change > 0.025 or analysis.flow_confidence > 0.05)

    def _is_underwater(self, analysis: MotionAnalysis) -> bool:
        r, g, b = analysis.dominant_color
        cool = b > r * 1.25 and (g > r * 0.9 or b > 120)
        return cool and analysis.brightness > 0.12 and analysis.flow_confidence < 0.35 and 0.005 <= analysis.changed_fraction < 0.22

    def _is_portal_vortex(self, analysis: MotionAnalysis) -> bool:
        dx, dy = analysis.dominant_flow
        return analysis.flow_confidence > 0.34 and abs(dx) > 0.6 and abs(dy) > 0.25 and analysis.saturation > 0.35

    def _directional_sweep(self, analysis: MotionAnalysis) -> tuple[int, float] | None:
        dx, dy = analysis.dominant_flow
        dominant = dx if abs(dx) >= abs(dy) else dy
        if abs(dominant) < 0.75 or analysis.flow_confidence < 0.30:
            return None
        return (1 if dominant > 0 else -1, min(1.0, analysis.flow_confidence * 0.8 + abs(dominant) / 8.0))

    def _scene_wipe(self, analysis: MotionAnalysis) -> tuple[int, float] | None:
        dx, dy = analysis.dominant_flow
        if analysis.changed_fraction < 0.32 or analysis.color_velocity < 0.18:
            return None
        dominant = dx if abs(dx) >= abs(dy) else dy
        direction = 1 if dominant >= 0 else -1
        return (direction, min(1.0, analysis.changed_fraction * 0.65 + analysis.color_velocity * 0.55))

    def _camera_pan(self, analysis: MotionAnalysis) -> tuple[int, float] | None:
        dx, dy = analysis.dominant_flow
        horizontal = abs(dx)
        vertical = abs(dy)
        if horizontal <= vertical * 1.7:
            return None
        confidence = min(1.0, analysis.flow_confidence + horizontal / 6.0)
        if confidence < self.config.camera_pan_confidence:
            return None
        return (1 if dx > 0 else -1, confidence * 0.65)
