from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import EngineConfig
from frame_processor import FrameProcessor


@dataclass(frozen=True)
class AnalysisRequirements:
    color_stats: bool = True
    frame_difference: bool = True
    edge_activity: bool = True
    top_corner_activity: bool = True
    optical_flow: bool = True
    retain_flow_debug: bool = True

    @property
    def any_analysis(self) -> bool:
        return self.color_stats or self.frame_difference or self.edge_activity or self.top_corner_activity or self.optical_flow


@dataclass
class EdgeMotion:
    magnitude: float = 0.0
    confidence: float = 0.0
    toward_edge: float = 0.0
    parallel: float = 0.0
    coverage: float = 0.0
    color: tuple[int, int, int] = (0, 0, 0)
    color_velocity: float = 0.0
    color_motion_velocity: float = 0.0


@dataclass
class MotionAnalysis:
    brightness: float = 0.0
    saturation: float = 0.0
    dominant_color: tuple[int, int, int] = (0, 0, 0)
    color_velocity: float = 0.0
    luminous_color_coverage: float = 0.0
    luminous_color_growth: float = 0.0
    luminous_brightness_growth: float = 0.0
    luminous_bloom_color: tuple[int, int, int] = (0, 0, 0)
    ambient_spill_boost_score: float = 0.0
    changed_fraction: float = 0.0
    rate_of_change: float = 0.0
    shockwave_line_strength: float = 0.0
    shockwave_line_axis: str = ""
    edge_activity: dict[str, EdgeMotion] = field(default_factory=dict)
    top_corner_activity: dict[str, EdgeMotion] = field(default_factory=dict)
    dominant_flow: tuple[float, float] = (0.0, 0.0)
    flow_confidence: float = 0.0
    scene_change: bool = False
    flow_vectors: np.ndarray | None = None
    heatmap: np.ndarray | None = None


class MotionDetector:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.previous_gray: np.ndarray | None = None
        self.previous_optical_gray: np.ndarray | None = None
        self.previous_difference_gray: np.ndarray | None = None
        self.history: deque[MotionAnalysis] = deque(maxlen=config.motion_history)

    def analyze(self, frame: np.ndarray, requirements: AnalysisRequirements | None = None) -> MotionAnalysis:
        requirements = requirements or AnalysisRequirements()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if requirements.frame_difference or requirements.optical_flow else None
        stats = FrameProcessor.color_stats(frame) if requirements.color_stats else {"brightness": 0.0, "saturation": 0.0, "color": (0, 0, 0)}
        previous_gray = self._previous_gray_for_requirements(requirements)
        if previous_gray is None:
            if gray is not None:
                self._store_previous_gray(gray, requirements)
            bloom = self._luminous_bloom_metrics(frame, None) if requirements.color_stats else {}
            analysis = MotionAnalysis(
                brightness=float(stats["brightness"]),
                saturation=float(stats["saturation"]),
                dominant_color=stats["color"],  # type: ignore[arg-type]
                **bloom,
            )
            self.history.append(analysis)
            return analysis

        if gray is None or previous_gray is None:
            analysis = MotionAnalysis()
        elif requirements.optical_flow and self.config.motion_algorithm != "frame_difference":
            analysis = self._optical_flow(frame, gray, previous_gray, requirements)
        elif requirements.frame_difference or requirements.edge_activity or requirements.top_corner_activity:
            analysis = self._frame_difference(frame, gray, previous_gray, requirements)
        else:
            analysis = MotionAnalysis()

        prev = self.history[-1] if self.history else None
        analysis.brightness = float(stats["brightness"])
        analysis.saturation = float(stats["saturation"])
        analysis.dominant_color = stats["color"]  # type: ignore[assignment]
        analysis.rate_of_change = abs(analysis.brightness - prev.brightness) if prev else 0.0
        analysis.color_velocity = self._color_velocity(analysis.dominant_color, prev.dominant_color) if prev else 0.0
        if requirements.color_stats:
            bloom = self._luminous_bloom_metrics(frame, prev)
            analysis.luminous_color_coverage = float(bloom["luminous_color_coverage"])
            analysis.luminous_color_growth = float(bloom["luminous_color_growth"])
            analysis.luminous_brightness_growth = float(bloom["luminous_brightness_growth"])
            analysis.luminous_bloom_color = bloom["luminous_bloom_color"]  # type: ignore[assignment]
            analysis.ambient_spill_boost_score = float(bloom["ambient_spill_boost_score"])
        if prev:
            self._apply_edge_color_velocity(analysis, prev)
        analysis.scene_change = analysis.changed_fraction > self.config.flash_changed_fraction
        if gray is not None:
            self._store_previous_gray(gray, requirements)
        self.history.append(analysis)
        return analysis

    def _previous_gray_for_requirements(self, requirements: AnalysisRequirements) -> np.ndarray | None:
        if requirements.optical_flow and self.config.motion_algorithm != "frame_difference":
            return self.previous_optical_gray
        if requirements.frame_difference or requirements.edge_activity or requirements.top_corner_activity:
            return self.previous_difference_gray
        return self.previous_gray

    def _store_previous_gray(self, gray: np.ndarray, requirements: AnalysisRequirements) -> None:
        self.previous_gray = gray
        if requirements.optical_flow and self.config.motion_algorithm != "frame_difference":
            self.previous_optical_gray = gray
        if requirements.frame_difference or requirements.edge_activity or requirements.top_corner_activity:
            self.previous_difference_gray = gray

    def _optical_flow(self, frame: np.ndarray, gray: np.ndarray, previous_gray: np.ndarray, requirements: AnalysisRequirements) -> MotionAnalysis:
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=17,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        heatmap = np.clip(mag / 8.0, 0.0, 1.0)
        changed = cv2.absdiff(gray, previous_gray)
        analysis = MotionAnalysis(
            changed_fraction=float(np.mean(changed > 22)),
            dominant_flow=(float(np.mean(flow[:, :, 0])), float(np.mean(flow[:, :, 1]))),
            flow_confidence=float(np.clip(np.mean(mag) / 4.0, 0.0, 1.0)),
            flow_vectors=flow if requirements.retain_flow_debug else None,
            heatmap=heatmap if requirements.retain_flow_debug else None,
        )
        line_strength, line_axis = self._shockwave_line_metrics(changed > 22)
        analysis.shockwave_line_strength = line_strength
        analysis.shockwave_line_axis = line_axis
        if requirements.edge_activity:
            analysis.edge_activity = self._edge_motion_from_flow(frame, flow, mag)
        if requirements.top_corner_activity:
            analysis.top_corner_activity = self._top_corner_motion_from_flow(frame, flow, mag)
        return analysis

    def _frame_difference(self, frame: np.ndarray, gray: np.ndarray, previous_gray: np.ndarray, requirements: AnalysisRequirements) -> MotionAnalysis:
        diff = cv2.absdiff(gray, previous_gray)
        _, mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(mask)
        h, w = gray.shape
        prev_center = np.array([w / 2, h / 2], dtype=np.float32)
        if moments["m00"] > 0:
            center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)
        else:
            center = prev_center
        vector = (center - prev_center) / np.array([w, h], dtype=np.float32)
        heatmap = (mask / 255.0).astype(np.float32)
        pseudo_flow = np.zeros((h, w, 2), dtype=np.float32)
        pseudo_flow[:, :, 0] = vector[0] * 12.0
        pseudo_flow[:, :, 1] = vector[1] * 12.0
        mag = np.sqrt(pseudo_flow[:, :, 0] ** 2 + pseudo_flow[:, :, 1] ** 2) * heatmap
        analysis = MotionAnalysis(
            changed_fraction=float(np.mean(mask > 0)),
            dominant_flow=(float(vector[0]), float(vector[1])),
            flow_confidence=float(np.clip(np.mean(mask > 0) * 3.0, 0.0, 1.0)),
            flow_vectors=pseudo_flow if requirements.retain_flow_debug else None,
            heatmap=heatmap if requirements.retain_flow_debug else None,
        )
        line_strength, line_axis = self._shockwave_line_metrics(mask > 0)
        analysis.shockwave_line_strength = line_strength
        analysis.shockwave_line_axis = line_axis
        if requirements.edge_activity:
            analysis.edge_activity = self._edge_motion_from_flow(frame, pseudo_flow, mag)
        if requirements.top_corner_activity:
            analysis.top_corner_activity = self._top_corner_motion_from_flow(frame, pseudo_flow, mag)
        return analysis

    def _edge_motion_from_flow(self, frame: np.ndarray, flow: np.ndarray, mag: np.ndarray) -> dict[str, EdgeMotion]:
        h, w = frame.shape[:2]
        edge_h = max(6, h // 6)
        edge_w = max(6, w // 6)
        slices = {
            "top": (slice(0, edge_h), slice(0, w), np.array([0.0, -1.0])),
            "left": (slice(0, h), slice(0, edge_w), np.array([-1.0, 0.0])),
            "right": (slice(0, h), slice(w - edge_w, w), np.array([1.0, 0.0])),
            "bottom": (slice(h - edge_h, h), slice(0, w), np.array([0.0, 1.0])),
        }
        result: dict[str, EdgeMotion] = {}
        for edge, (ys, xs, outward) in slices.items():
            result[edge] = self._motion_for_slice(frame, flow, mag, ys, xs, outward)
        return result

    def _top_corner_motion_from_flow(self, frame: np.ndarray, flow: np.ndarray, mag: np.ndarray) -> dict[str, EdgeMotion]:
        h, w = frame.shape[:2]
        edge_h = max(6, h // 6)
        edge_w = max(8, w // 5)
        return {
            "left": self._motion_for_slice(frame, flow, mag, slice(0, edge_h), slice(0, edge_w), np.array([-1.0, 0.0])),
            "right": self._motion_for_slice(frame, flow, mag, slice(0, edge_h), slice(w - edge_w, w), np.array([1.0, 0.0])),
        }

    def _motion_for_slice(
        self,
        frame: np.ndarray,
        flow: np.ndarray,
        mag: np.ndarray,
        ys: slice,
        xs: slice,
        outward: np.ndarray,
    ) -> EdgeMotion:
        edge_flow = flow[ys, xs]
        edge_mag = mag[ys, xs]
        mean_mag = float(np.clip(np.mean(edge_mag) / 5.0, 0.0, 1.0))
        coverage = float(np.mean(edge_mag > max(0.25, self.config.min_motion * 20.0)))
        vectors = edge_flow.reshape(-1, 2)
        norms = np.linalg.norm(vectors, axis=1) + 1e-6
        unit = vectors / norms[:, None]
        dot = unit @ outward
        toward = float(np.clip(np.average(np.maximum(dot, 0.0), weights=norms), 0.0, 1.0))
        parallel = float(np.clip(np.average(1.0 - np.abs(dot), weights=norms), 0.0, 1.0))
        confidence = float(np.clip(mean_mag * 0.65 + coverage * 0.35, 0.0, 1.0))
        region = frame[ys, xs]
        color = FrameProcessor.dominant_color(region)
        color_motion_velocity = self._color_motion_velocity(region, edge_mag, dot)
        return EdgeMotion(mean_mag, confidence, toward, parallel, coverage, color, color_motion_velocity=color_motion_velocity)

    def _color_motion_velocity(self, region: np.ndarray, mag: np.ndarray, dot: np.ndarray) -> float:
        if region.size == 0:
            return 0.0
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].reshape(-1).astype(np.float32) / 255.0
        brightness = hsv[:, :, 2].reshape(-1).astype(np.float32) / 255.0
        color_weight = np.clip(saturation * 0.65 + brightness * 0.35, 0.0, 1.0)
        outward = np.maximum(dot, 0.0)
        motion = np.clip(mag.reshape(-1).astype(np.float32) / 5.0, 0.0, 1.0)
        weighted = motion * outward * color_weight
        active = weighted[weighted > max(0.03, self.config.min_motion)]
        if active.size == 0:
            return 0.0
        return float(np.clip(np.mean(active) * 2.2, 0.0, 1.0))

    @staticmethod
    def _color_velocity(current: tuple[int, int, int], previous: tuple[int, int, int]) -> float:
        current_rgb = np.array(current, dtype=np.float32)
        previous_rgb = np.array(previous, dtype=np.float32)
        distance = float(np.linalg.norm(current_rgb - previous_rgb))
        return float(np.clip(distance / (255.0 * np.sqrt(3.0)), 0.0, 1.0))

    def _apply_edge_color_velocity(self, analysis: MotionAnalysis, previous: MotionAnalysis) -> None:
        for edge, activity in analysis.edge_activity.items():
            previous_activity = previous.edge_activity.get(edge)
            if previous_activity is None:
                activity.color_velocity = analysis.color_velocity
            else:
                activity.color_velocity = self._color_velocity(activity.color, previous_activity.color)

    def _shockwave_line_metrics(self, changed: np.ndarray) -> tuple[float, str]:
        if changed.size == 0:
            return 0.0, ""
        changed_fraction = float(np.mean(changed))
        if changed_fraction >= 0.68:
            return 0.0, ""
        row_score = self._line_axis_score(np.mean(changed, axis=1), changed_fraction)
        col_score = self._line_axis_score(np.mean(changed, axis=0), changed_fraction)
        if row_score >= col_score and row_score > 0.0:
            return row_score, "horizontal"
        if col_score > 0.0:
            return col_score, "vertical"
        return 0.0, ""

    @staticmethod
    def _line_axis_score(fractions: np.ndarray, changed_fraction: float) -> float:
        if fractions.size == 0:
            return 0.0
        active = fractions >= 0.42
        if not np.any(active):
            return 0.0
        best = 0.0
        start: int | None = None
        for idx, is_active in enumerate(np.append(active, False)):
            if is_active and start is None:
                start = idx
            elif not is_active and start is not None:
                run = fractions[start:idx]
                width_ratio = run.size / max(1, fractions.size)
                if 0.015 <= width_ratio <= 0.26:
                    peak = float(np.max(run))
                    narrow_score = float(np.clip((0.30 - width_ratio) / 0.285, 0.0, 1.0))
                    contrast = float(np.clip((peak - changed_fraction) / 0.42, 0.0, 1.0))
                    best = max(best, peak * (0.45 + 0.35 * narrow_score + 0.20 * contrast))
                start = None
        return float(np.clip(best, 0.0, 1.0))

    def _luminous_bloom_metrics(self, frame: np.ndarray, previous: MotionAnalysis | None) -> dict[str, float | tuple[int, int, int]]:
        rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]
        luminance = r * 0.2126 + g * 0.7152 + b * 0.0722
        cool_bias = ((g + b) * 0.5) - r
        cool = (
            (luminance >= 0.38)
            & (b >= r * 1.04)
            & (g >= r * 0.82)
            & (cool_bias >= 0.035)
            & ((b >= 0.32) | (g >= 0.34))
        )
        coverage = float(np.mean(cool))
        brightness = float(np.mean(luminance))
        prev_coverage = previous.luminous_color_coverage if previous else coverage
        prev_brightness = previous.brightness if previous else brightness
        coverage_growth = max(0.0, coverage - prev_coverage)
        brightness_growth = max(0.0, brightness - prev_brightness)

        if np.any(cool):
            weights = luminance[cool].reshape(-1, 1)
            pixels = rgb[cool]
            color = np.average(pixels, axis=0, weights=np.maximum(weights[:, 0], 0.001))
            bloom_color = tuple(int(value) for value in np.clip(np.rint(color * 255.0), 0, 255))
            color_mean = color
        else:
            bloom_color = (0, 0, 0)
            color_mean = np.zeros(3, dtype=np.float32)

        color_level = float(np.max(color_mean))
        if color_level > 0.001:
            cyan_bias = ((float(color_mean[1]) + float(color_mean[2])) * 0.5 - float(color_mean[0])) / color_level
        else:
            cyan_bias = 0.0
        cool_factor = float(np.clip((cyan_bias - 0.03) / 0.20, 0.0, 1.0))
        coverage_factor = float(np.clip((coverage - 0.20) / 0.50, 0.0, 1.0))
        brightness_factor = float(np.clip((brightness - 0.30) / 0.45, 0.0, 1.0))
        growth_factor = float(np.clip(max(coverage_growth / 0.22, brightness_growth / 0.22), 0.0, 1.0))
        stable_factor = 0.35 if coverage >= 0.28 and brightness >= 0.35 else 0.0
        boost_score = coverage_factor * brightness_factor * cool_factor * max(stable_factor, growth_factor)
        if coverage < 0.28 or brightness < 0.35:
            boost_score *= 0.35

        return {
            "luminous_color_coverage": coverage,
            "luminous_color_growth": coverage_growth,
            "luminous_brightness_growth": brightness_growth,
            "luminous_bloom_color": bloom_color,
            "ambient_spill_boost_score": float(np.clip(boost_score, 0.0, 1.0)),
        }
        for edge, activity in analysis.top_corner_activity.items():
            previous_activity = previous.top_corner_activity.get(edge)
            if previous_activity is None:
                activity.color_velocity = analysis.color_velocity
            else:
                activity.color_velocity = self._color_velocity(activity.color, previous_activity.color)
