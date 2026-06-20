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


class EventDetector:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.previous_brightness = 0.0

    def detect(self, analysis: MotionAnalysis) -> list[LightEvent]:
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
                    events.append(LightEvent(edge, intensity, activity.color, "spill", color_velocity=activity.color_velocity))
            events.extend(self._top_color_exit_events(analysis))

        if self._effect_enabled("flash") and self._is_flash(analysis):
            events.append(LightEvent("top", min(1.0, analysis.changed_fraction + analysis.brightness), analysis.dominant_color, "flash", color_velocity=analysis.color_velocity))

        if self._effect_enabled("explosion") and self._is_explosion(analysis):
            events.append(LightEvent("top", 1.0, analysis.dominant_color, "explosion", color_velocity=analysis.color_velocity))

        pan = self._camera_pan(analysis)
        if self._effect_enabled("camera_pan") and pan:
            events.append(LightEvent("top", pan[1], analysis.dominant_color, "camera_pan", direction_hint=pan[0], color_velocity=analysis.color_velocity))

        self.previous_brightness = analysis.brightness
        return sorted(events, key=lambda e: e.intensity, reverse=True)

    def _effect_enabled(self, kind: str) -> bool:
        return self.config.enabled_effects.get(kind, True)

    def _front_ambient_event(self, analysis: MotionAnalysis) -> LightEvent | None:
        if analysis.brightness < self.config.front_ambient_min_brightness:
            return None
        base = analysis.brightness * 0.70 + analysis.saturation * 0.30
        intensity = max(0.0, min(1.0, base * self.config.front_ambient_intensity))
        if intensity <= 0.01:
            return None
        return LightEvent("top", intensity, analysis.dominant_color, "front_ambient", color_velocity=analysis.color_velocity)

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
                )
            )
        return events

    def _is_flash(self, analysis: MotionAnalysis) -> bool:
        return analysis.changed_fraction >= self.config.flash_changed_fraction and analysis.brightness > 0.35

    def _is_explosion(self, analysis: MotionAnalysis) -> bool:
        brightness_spike = analysis.brightness - self.previous_brightness
        return (
            brightness_spike >= self.config.explosion_brightness_spike
            and analysis.saturation > 0.35
            and analysis.changed_fraction > 0.22
        )

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
