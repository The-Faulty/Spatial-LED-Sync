from __future__ import annotations

import cv2
import numpy as np

from config import EngineConfig
from motion_detector import MotionAnalysis


class Visualizer:
    def __init__(self, config: EngineConfig):
        self.config = config

    def show(self, frame: np.ndarray, analysis: MotionAnalysis, leds: np.ndarray, active_waves: int) -> bool:
        h, w = frame.shape[:2]
        canvas = np.zeros((h * 2 + 80, w * 2, 3), dtype=np.uint8)
        canvas[:h, :w] = frame
        canvas[:h, w:w * 2] = self._flow_view(frame, analysis)
        canvas[h:h * 2, :w] = self._heatmap_view(frame, analysis)
        canvas[h:h * 2, w:w * 2] = self._led_strip_view(leds, w, h)
        self._draw_zones(canvas[:h, :w])
        cv2.putText(
            canvas,
            f"intensity edges: {self._edge_text(analysis)}  waves: {active_waves}",
            (10, h * 2 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"brightness={analysis.brightness:.2f} saturation={analysis.saturation:.2f} changed={analysis.changed_fraction:.2f}",
            (10, h * 2 + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (190, 220, 255),
            1,
            cv2.LINE_AA,
        )
        if self.config.debug_window_scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.config.debug_window_scale, fy=self.config.debug_window_scale)
        cv2.imshow("HyperHDR Cinematic Light Spill Engine", canvas)
        return cv2.waitKey(1) & 0xFF != ord("q")

    def _flow_view(self, frame: np.ndarray, analysis: MotionAnalysis) -> np.ndarray:
        view = frame.copy()
        flow = analysis.flow_vectors
        if flow is None:
            return view
        step = 12
        h, w = frame.shape[:2]
        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                dx, dy = flow[y, x]
                end = (int(x + dx * 2), int(y + dy * 2))
                cv2.arrowedLine(view, (x, y), end, (0, 255, 180), 1, tipLength=0.35)
        return view

    def _heatmap_view(self, frame: np.ndarray, analysis: MotionAnalysis) -> np.ndarray:
        if analysis.heatmap is None:
            return frame.copy()
        heat = np.clip(analysis.heatmap * 255, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
        return cv2.addWeighted(frame, 0.35, heat, 0.65, 0)

    @staticmethod
    def _led_strip_view(leds: np.ndarray, width: int, height: int) -> np.ndarray:
        strip = np.zeros((height, width, 3), dtype=np.uint8)
        count = len(leds)
        for i, color in enumerate(leds):
            x0 = int(i * width / count)
            x1 = int((i + 1) * width / count) + 1
            strip[:, x0:x1] = color[::-1]
        return strip

    @staticmethod
    def _draw_zones(frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        edge_h = max(6, h // 6)
        edge_w = max(6, w // 6)
        color = (80, 220, 255)
        cv2.rectangle(frame, (0, 0), (w - 1, edge_h), color, 1)
        cv2.rectangle(frame, (0, 0), (edge_w, h - 1), color, 1)
        cv2.rectangle(frame, (w - edge_w, 0), (w - 1, h - 1), color, 1)
        cv2.rectangle(frame, (0, h - edge_h), (w - 1, h - 1), color, 1)

    @staticmethod
    def _edge_text(analysis: MotionAnalysis) -> str:
        return " ".join(f"{k}:{v.confidence:.2f}/{v.toward_edge:.2f}" for k, v in analysis.edge_activity.items())
