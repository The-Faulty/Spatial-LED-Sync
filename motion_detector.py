from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import EngineConfig
from frame_processor import FrameProcessor


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
    changed_fraction: float = 0.0
    rate_of_change: float = 0.0
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
        self.history: deque[MotionAnalysis] = deque(maxlen=config.motion_history)

    def analyze(self, frame: np.ndarray) -> MotionAnalysis:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stats = FrameProcessor.color_stats(frame)
        if self.previous_gray is None:
            self.previous_gray = gray
            analysis = MotionAnalysis(
                brightness=float(stats["brightness"]),
                saturation=float(stats["saturation"]),
                dominant_color=stats["color"],  # type: ignore[arg-type]
            )
            self.history.append(analysis)
            return analysis

        if self.config.motion_algorithm == "frame_difference":
            analysis = self._frame_difference(frame, gray)
        else:
            analysis = self._optical_flow(frame, gray)

        prev = self.history[-1] if self.history else None
        analysis.brightness = float(stats["brightness"])
        analysis.saturation = float(stats["saturation"])
        analysis.dominant_color = stats["color"]  # type: ignore[assignment]
        analysis.rate_of_change = abs(analysis.brightness - prev.brightness) if prev else 0.0
        analysis.color_velocity = self._color_velocity(analysis.dominant_color, prev.dominant_color) if prev else 0.0
        if prev:
            self._apply_edge_color_velocity(analysis, prev)
        analysis.scene_change = analysis.changed_fraction > self.config.flash_changed_fraction
        self.previous_gray = gray
        self.history.append(analysis)
        return analysis

    def _optical_flow(self, frame: np.ndarray, gray: np.ndarray) -> MotionAnalysis:
        flow = cv2.calcOpticalFlowFarneback(
            self.previous_gray,
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
        changed = cv2.absdiff(gray, self.previous_gray)
        analysis = MotionAnalysis(
            changed_fraction=float(np.mean(changed > 22)),
            dominant_flow=(float(np.mean(flow[:, :, 0])), float(np.mean(flow[:, :, 1]))),
            flow_confidence=float(np.clip(np.mean(mag) / 4.0, 0.0, 1.0)),
            flow_vectors=flow,
            heatmap=heatmap,
        )
        analysis.edge_activity = self._edge_motion_from_flow(frame, flow, mag)
        analysis.top_corner_activity = self._top_corner_motion_from_flow(frame, flow, mag)
        return analysis

    def _frame_difference(self, frame: np.ndarray, gray: np.ndarray) -> MotionAnalysis:
        diff = cv2.absdiff(gray, self.previous_gray)
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
            edge_activity=self._edge_motion_from_flow(frame, pseudo_flow, mag),
            flow_vectors=pseudo_flow,
            heatmap=heatmap,
        )
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
        for edge, activity in analysis.top_corner_activity.items():
            previous_activity = previous.top_corner_activity.get(edge)
            if previous_activity is None:
                activity.color_velocity = analysis.color_velocity
            else:
                activity.color_velocity = self._color_velocity(activity.color, previous_activity.color)
