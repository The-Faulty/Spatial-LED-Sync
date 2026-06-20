from __future__ import annotations

import cv2
import numpy as np

from config import EngineConfig


class FrameProcessor:
    def __init__(self, config: EngineConfig):
        self.config = config

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("empty frame")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return cv2.resize(
            frame,
            (self.config.analysis_width, self.config.analysis_height),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def zones(frame: np.ndarray) -> dict[str, np.ndarray]:
        h, w = frame.shape[:2]
        edge_h = max(6, h // 6)
        edge_w = max(6, w // 6)
        return {
            "center": frame[edge_h:h - edge_h, edge_w:w - edge_w],
            "top": frame[:edge_h, :],
            "left": frame[:, :edge_w],
            "right": frame[:, w - edge_w:],
            "bottom": frame[h - edge_h:, :],
        }

    @staticmethod
    def color_stats(frame: np.ndarray) -> dict[str, float | tuple[int, int, int]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = float(np.mean(hsv[:, :, 2]) / 255.0)
        saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
        pixels = frame.reshape(-1, 3).astype(np.float32)
        weights = hsv[:, :, 1].reshape(-1).astype(np.float32) + hsv[:, :, 2].reshape(-1).astype(np.float32)
        if float(weights.sum()) <= 0:
            color = np.mean(pixels, axis=0)
        else:
            color = np.average(pixels, axis=0, weights=weights)
        b, g, r = np.clip(color, 0, 255).astype(np.uint8)
        return {"brightness": brightness, "saturation": saturation, "color": (int(r), int(g), int(b))}

    @staticmethod
    def dominant_color(frame: np.ndarray) -> tuple[int, int, int]:
        return FrameProcessor.color_stats(frame)["color"]  # type: ignore[return-value]

    def top_strip_colors(self, frame: np.ndarray, count: int) -> tuple[np.ndarray, float]:
        count = max(1, int(count))
        height = max(1, int(frame.shape[0] * self.config.front_ambient_top_height))
        height = min(height, frame.shape[0])
        top_band = np.ascontiguousarray(frame[:height, :])
        if top_band.size == 0:
            return np.zeros((count, 3), dtype=np.float32), 0.0
        if self.config.front_ambient_blur > 0 and min(top_band.shape[:2]) >= 3:
            kernel = max(3, int(self.config.front_ambient_blur))
            if kernel % 2 == 0:
                kernel += 1
            min_dim = min(top_band.shape[0], top_band.shape[1])
            max_kernel = min_dim if min_dim % 2 == 1 else min_dim - 1
            max_kernel = max(3, max_kernel)
            kernel = min(kernel, max_kernel)
            try:
                top_band = cv2.GaussianBlur(top_band, (kernel, kernel), 0)
            except cv2.error:
                blur_w = max(1, min(kernel, top_band.shape[1]))
                blur_h = max(1, min(kernel, top_band.shape[0]))
                top_band = cv2.blur(top_band, (blur_w, blur_h))
        sampled = cv2.resize(top_band, (count, 1), interpolation=cv2.INTER_AREA).reshape(count, 3)
        rgb = sampled[:, ::-1].astype(np.float32) / 255.0
        brightness = float(self.color_stats(top_band)["brightness"])
        if brightness < self.config.front_ambient_min_brightness:
            intensity = 0.0
        else:
            intensity = max(0.0, min(1.0, self.config.front_ambient_intensity))
        return rgb, intensity
