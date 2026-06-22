from __future__ import annotations

import logging
import base64
import json
import socket
import struct
import threading
import time
from queue import Empty, Queue

import cv2
import numpy as np
import websocket

from config import EngineConfig


SIMULATION_EFFECT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("front_ambient", "Front ambient"),
    ("ambient_side_spill", "Ambient side spill"),
    ("ambient_side_spill_boost", "Ambient spill boost"),
    ("spill", "Motion spill"),
    ("top_color_exit", "Top color exit"),
    ("flash", "Flash"),
    ("explosion", "Explosion"),
    ("camera_pan", "Camera pan"),
    ("energy_trail", "Energy trail"),
    ("shockwave", "Shockwave"),
    ("directional_sweep", "Directional sweep"),
    ("lightning", "Lightning"),
    ("impact_pulse", "Impact pulse"),
    ("color_bloom", "Color bloom"),
    ("flame_shimmer", "Flame shimmer"),
    ("underwater", "Underwater"),
    ("portal_vortex", "Portal vortex"),
    ("scene_wipe", "Scene wipe"),
    ("ember_particles", "Ember particles"),
    ("negative_wave", "Negative wave"),
)


class HyperHDRClient:
    """Receives full image frames from HyperHDR or produces a simulation stream.

    HyperHDR/Hyperion FlatBuffers deployments differ by version and generated
    schema package. This client supports length-prefixed image payloads and an
    optional generated decoder hook while keeping the rest of the engine usable
    in simulation mode.
    """

    def __init__(self, config: EngineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger.getChild("hyperhdr")
        self.frames: Queue[np.ndarray] = Queue(maxsize=config.frame_buffer_size)
        self.simulation_commands: Queue[str] = Queue()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.connected = False

    def start(self) -> None:
        if self.config.simulate_input:
            target = self._simulate_loop
        elif self.config.hyperhdr_input_mode == "flatbuffers":
            target = self._flatbuffers_loop
        else:
            target = self._websocket_loop
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def get_frame(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self.frames.get(timeout=timeout)
        except Empty:
            return None

    def trigger_simulation_effect(self, effect: str) -> None:
        self.simulation_commands.put(effect)

    def _put_frame(self, frame: np.ndarray) -> None:
        if self.frames.full():
            try:
                self.frames.get_nowait()
            except Empty:
                pass
        self.frames.put_nowait(frame)

    def _websocket_loop(self) -> None:
        url = f"ws://{self.config.hyperhdr_host}:{self.config.hyperhdr_ws_port}{self.config.hyperhdr_ws_path}"
        subscribe = {
            "command": "ledcolors",
            "subcommand": "imagestream-start",
            "tan": 1,
        }
        unsubscribe = {
            "command": "ledcolors",
            "subcommand": "imagestream-stop",
            "tan": 2,
        }
        while not self.stop_event.is_set():
            ws: websocket.WebSocket | None = None
            try:
                ws = websocket.create_connection(url, timeout=4.0)
                ws.settimeout(self.config.hyperhdr_frame_timeout)
                self.connected = True
                self.logger.info("Connected to HyperHDR WebSocket live-video stream at %s", url)
                if self.config.hyperhdr_token.strip():
                    ws.send(json.dumps({
                        "command": "authorize",
                        "subcommand": "login",
                        "token": self.config.hyperhdr_token.strip(),
                        "tan": 100,
                    }))
                ws.send(json.dumps(subscribe))
                last_frame = time.monotonic()
                while not self.stop_event.is_set():
                    message = ws.recv()
                    if isinstance(message, bytes):
                        frame = self._decode_image_bytes(message)
                        if frame is not None:
                            last_frame = time.monotonic()
                            self._put_frame(frame)
                    elif isinstance(message, str):
                        frame = self._handle_websocket_text(message)
                        if frame is not None:
                            last_frame = time.monotonic()
                            self._put_frame(frame)
                    if time.monotonic() - last_frame > max(5.0, self.config.hyperhdr_frame_timeout * 3):
                        self.logger.warning("No HyperHDR live-video frames received recently; check that live video is authorized and an input source is active")
                        last_frame = time.monotonic()
            except Exception as exc:
                self.connected = False
                self.logger.warning("HyperHDR WebSocket connection lost: %s", exc)
                time.sleep(1.0)
            finally:
                if ws is not None:
                    try:
                        ws.send(json.dumps(unsubscribe))
                    except Exception:
                        pass
                    try:
                        ws.close()
                    except Exception:
                        pass

    def _flatbuffers_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                with socket.create_connection((self.config.hyperhdr_host, self.config.hyperhdr_port), timeout=4.0) as sock:
                    sock.settimeout(self.config.hyperhdr_frame_timeout)
                    self.connected = True
                    self.logger.info("Connected to HyperHDR FlatBuffers server")
                    while not self.stop_event.is_set():
                        try:
                            header = self._recv_exact(sock, 4)
                        except socket.timeout as exc:
                            raise TimeoutError(
                                "no FlatBuffers bytes received; HyperHDR port 19400 is usually an input server. "
                                "Use hyperhdr_input_mode=websocket for live-video frames."
                            ) from exc
                        if not header:
                            raise ConnectionError("FlatBuffers stream closed")
                        size = struct.unpack("<I", header)[0]
                        if size <= 0 or size > 50_000_000:
                            raise ValueError(f"invalid FlatBuffers frame size {size}")
                        payload = self._recv_exact(sock, size)
                        if not payload:
                            raise ConnectionError("incomplete FlatBuffers payload")
                        frame = self._decode_payload(payload)
                        if frame is not None:
                            self._put_frame(frame)
            except Exception as exc:
                self.connected = False
                self.logger.warning("HyperHDR connection lost: %s", exc)
                time.sleep(1.0)

    @staticmethod
    def _decode_image_bytes(payload: bytes) -> np.ndarray | None:
        data = np.frombuffer(payload, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def _handle_websocket_text(self, message: str) -> np.ndarray | None:
        direct = self._decode_base64_image(message)
        if direct is not None:
            return direct
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            self.logger.debug("Non-JSON WebSocket text message: %s", message[:120])
            return None
        if data.get("success") is False:
            self.logger.warning("HyperHDR WebSocket error for %s: %s", data.get("command"), data.get("error", "unknown"))
        else:
            self.logger.debug("HyperHDR WebSocket message: %s", data.get("command", "unknown"))
        return self._extract_json_image(data)

    @classmethod
    def _extract_json_image(cls, data: object) -> np.ndarray | None:
        if isinstance(data, dict):
            for key in ("image", "frame", "data", "jpeg", "jpg", "payload"):
                value = data.get(key)
                if isinstance(value, str):
                    frame = cls._decode_base64_image(value)
                    if frame is not None:
                        return frame
            for value in data.values():
                frame = cls._extract_json_image(value)
                if frame is not None:
                    return frame
        elif isinstance(data, list):
            for value in data:
                frame = cls._extract_json_image(value)
                if frame is not None:
                    return frame
        return None

    @staticmethod
    def _decode_base64_image(value: str) -> np.ndarray | None:
        text = value.strip()
        if not text:
            return None
        if text.startswith("data:image"):
            _, _, text = text.partition(",")
        if len(text) < 64:
            return None
        try:
            payload = base64.b64decode(text, validate=False)
        except Exception:
            return None
        return HyperHDRClient._decode_image_bytes(payload)

    @staticmethod
    def _recv_exact(sock: socket.socket, count: int) -> bytes | None:
        chunks: list[bytes] = []
        remaining = count
        while remaining:
            chunk = sock.recv(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _decode_payload(self, payload: bytes) -> np.ndarray | None:
        decoded = self._decode_with_optional_schema(payload)
        if decoded is not None:
            return decoded
        return self._decode_raw_rgb_guess(payload)

    def _decode_with_optional_schema(self, payload: bytes) -> np.ndarray | None:
        try:
            from hyperhdr_flatbuf import Image  # type: ignore

            image = Image.Image.GetRootAsImage(payload, 0)
            width = int(image.Width())
            height = int(image.Height())
            data = image.DataAsNumpy()
            rgb = np.asarray(data, dtype=np.uint8).reshape((height, width, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _decode_raw_rgb_guess(self, payload: bytes) -> np.ndarray | None:
        # Fallback for streams that wrap raw RGB bytes with a small FlatBuffers header.
        # It searches for a plausible RGB tail matching common capture resolutions.
        common = [(1920, 1080), (1280, 720), (960, 540), (640, 360), (320, 180)]
        for width, height in common:
            size = width * height * 3
            if len(payload) >= size:
                raw = payload[-size:]
                arr = np.frombuffer(raw, dtype=np.uint8)
                if arr.size == size:
                    rgb = arr.reshape((height, width, 3))
                    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.logger.debug("Unable to decode FlatBuffers payload of %d bytes", len(payload))
        return None

    def _simulate_loop(self) -> None:
        self.logger.info("Running simulated HyperHDR frame stream")
        width, height = 640, 360
        active_effect: str | None = None
        looping_effect: str | None = None
        effect_started = 0.0
        effect_duration = 1.6
        frame_interval = 1.0 / max(1, self.config.target_fps)
        while not self.stop_event.is_set():
            try:
                command = self.simulation_commands.get_nowait()
                if command == "idle":
                    active_effect = None
                    looping_effect = None
                    self.logger.info("Stopped simulation effect loop")
                    continue
                if command.startswith("loop:"):
                    active_effect = command.split(":", 1)[1]
                    looping_effect = active_effect
                else:
                    active_effect = command
                    looping_effect = None
                effect_started = time.monotonic()
                effect_duration = self._simulation_duration(active_effect)
                self.logger.info("Triggered simulation effect: %s", active_effect)
            except Empty:
                pass

            progress = 0.0
            if active_effect:
                progress = (time.monotonic() - effect_started) / effect_duration
                if progress >= 1.0:
                    if looping_effect:
                        progress %= 1.0
                        effect_started = time.monotonic() - progress * effect_duration
                    else:
                        active_effect = None
                        progress = 0.0

            frame = self._simulation_frame(width, height, active_effect, progress)
            self._put_frame(frame)
            time.sleep(frame_interval)

    @staticmethod
    def _simulation_duration(effect: str) -> float:
        return {
            "front_ambient": 2.0,
            "ambient_side_spill": 2.0,
            "ambient_side_spill_boost": 2.4,
            "spill": 1.4,
            "top_color_exit": 1.2,
            "flash": 0.65,
            "explosion": 1.3,
            "shockwave": 1.5,
            "camera_pan": 1.8,
            "pan_left": 1.8,
            "pan_right": 1.8,
            "left_exit": 1.4,
            "right_exit": 1.4,
            "top_spill": 1.2,
            "top_color_exit_left": 1.2,
            "top_color_exit_right": 1.2,
            "energy_trail": 1.6,
            "directional_sweep": 1.4,
            "lightning": 0.8,
            "impact_pulse": 0.9,
            "color_bloom": 2.2,
            "flame_shimmer": 1.6,
            "underwater": 2.4,
            "portal_vortex": 1.8,
            "scene_wipe": 1.2,
            "ember_particles": 1.4,
            "negative_wave": 1.0,
        }.get(effect, 1.2)

    def _simulation_frame(self, width: int, height: int, effect: str | None, progress: float) -> np.ndarray:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (6, 5, 4)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (10, 8, 7), 3)
        if effect is None:
            cv2.putText(frame, "Simulation idle - use test pattern buttons", (34, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 2, cv2.LINE_AA)
            return cv2.flip(frame, 1)

        p = float(np.clip(progress, 0.0, 1.0))
        if effect == "front_ambient":
            frame[:] = (8, 8, 10)
            top_h = max(18, height // 7)
            for x in range(width):
                mix = x / max(1, width - 1)
                color = (int(220 * (1.0 - mix)), int(70 + 120 * mix), int(80 + 140 * mix))
                cv2.line(frame, (x, 0), (x, top_h), color, 1)
            cv2.GaussianBlur(frame, (0, 0), 8, dst=frame)
        elif effect == "ambient_side_spill":
            frame[:] = (5, 5, 7)
            top_h = max(18, height // 8)
            frame[:top_h, : width // 2] = (20, 80, 255)
            frame[:top_h, width // 2 :] = (255, 90, 25)
            cv2.rectangle(frame, (0, 0), (width - 1, top_h), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "edge color spill", (30, top_h + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 180), 2, cv2.LINE_AA)
        elif effect == "ambient_side_spill_boost":
            ramp = 0.18 + 0.82 * np.sin(p * np.pi / 2.0) ** 1.7
            pulse = 0.85 + 0.15 * np.sin(p * np.pi * 4.0)
            level = float(np.clip(ramp * pulse, 0.0, 1.0))
            color = np.array([22, 105, 255], dtype=np.float32) * level
            frame[:] = np.clip(color, 0, 255).astype(np.uint8)
            top_h = max(20, height // 6)
            frame[:top_h, :] = np.clip(color * 1.12, 0, 255).astype(np.uint8)
            cv2.rectangle(frame, (0, 0), (width - 1, top_h), (255, 230, 180), 2, cv2.LINE_AA)
            cv2.putText(frame, "solid color intensity ramp", (30, top_h + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 210, 190), 2, cv2.LINE_AA)
        elif effect == "flash":
            level = int(235 * max(0.0, 1.0 - p) + 16)
            frame[:] = (level, level, min(255, level + 15))
        elif effect == "explosion":
            radius = int(30 + p * max(width, height) * 0.75)
            cv2.circle(frame, (width // 2, height // 2), radius, (20, 135, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (width // 2, height // 2), max(1, radius // 2), (30, 235, 255), -1, cv2.LINE_AA)
        elif effect == "shockwave":
            radius = int(20 + p * max(width, height) * 0.65)
            cv2.circle(frame, (width // 2, height // 2), radius, (245, 245, 255), 16, cv2.LINE_AA)
            cv2.circle(frame, (width // 2, height // 2), max(1, radius - 28), (80, 80, 180), 6, cv2.LINE_AA)
        elif effect in {"spill", "left_exit"}:
            x = int(width * (0.55 - p * 0.75))
            self._draw_moving_orb(frame, x, height // 2, (255, 80, 40), trail_direction=1, progress=p)
        elif effect == "right_exit":
            x = int(width * (0.45 + p * 0.75))
            self._draw_moving_orb(frame, x, height // 2, (30, 180, 255), trail_direction=-1, progress=p)
        elif effect == "top_spill":
            y = int(height * (0.62 - p * 0.82))
            self._draw_moving_orb(frame, width // 2, y, (120, 255, 200), trail_direction=0, progress=p)
        elif effect == "top_color_exit_left":
            x = int(width * (0.72 - p * 0.92))
            y = max(20, height // 10)
            self._draw_moving_orb(frame, x, y, (255, 60, 180), trail_direction=1, progress=p)
        elif effect in {"top_color_exit", "top_color_exit_right"}:
            x = int(width * (0.28 + p * 0.92))
            y = max(20, height // 10)
            self._draw_moving_orb(frame, x, y, (40, 210, 255), trail_direction=-1, progress=p)
        elif effect == "pan_left":
            shift = int(p * width)
            self._draw_pan_bands(frame, -shift)
        elif effect in {"camera_pan", "pan_right"}:
            shift = int(p * width)
            self._draw_pan_bands(frame, shift)
        elif effect == "energy_trail":
            x = int(width * (0.20 + p * 0.95))
            y = int(height * (0.48 + 0.12 * np.sin(p * np.pi * 2)))
            for i in range(8):
                fade = 1.0 - i / 8
                cv2.circle(frame, (x - i * 34, y), int(30 * fade), (int(240 * fade), int(80 * fade), 255), -1, cv2.LINE_AA)
        elif effect == "directional_sweep":
            shift = int(p * width * 1.2)
            self._draw_pan_bands(frame, shift)
            cv2.rectangle(frame, (shift - 90, 0), (shift + 120, height), (220, 120, 40), -1)
        elif effect == "lightning":
            frame[:] = (12, 10, 16)
            for i in range(5):
                x = int(width * (0.15 + i * 0.18 + 0.04 * np.sin(p * 20 + i)))
                cv2.line(frame, (x, 0), (x + int(80 * np.sin(i + p * 8)), height), (255, 245, 230), 8, cv2.LINE_AA)
            if int(p * 16) % 2 == 0:
                frame[:] = np.maximum(frame, 190)
        elif effect == "impact_pulse":
            level = int(35 + 190 * np.exp(-p * 5.0))
            frame[:] = (level, min(255, level + 20), min(255, level + 35))
            cv2.circle(frame, (width // 2, height // 2), int(80 + p * 260), (40, 180, 255), 18, cv2.LINE_AA)
        elif effect == "color_bloom":
            hue = int(120 + 80 * np.sin(p * np.pi))
            frame[:] = (hue, 30, 190)
            cv2.GaussianBlur(frame, (0, 0), 12, dst=frame)
        elif effect == "flame_shimmer":
            frame[:] = (8, 12, 28)
            for i in range(18):
                x = int((i / 18) * width)
                y = int(height * (0.65 + 0.18 * np.sin(p * 18 + i)))
                cv2.circle(frame, (x, y), 70, (20, 90 + i * 5 % 120, 255), -1, cv2.LINE_AA)
            cv2.GaussianBlur(frame, (0, 0), 8, dst=frame)
        elif effect == "underwater":
            frame[:] = (120, 60, 8)
            for i in range(9):
                y = int(height * (i / 9) + np.sin(p * 8 + i) * 18)
                cv2.line(frame, (0, y), (width, y + int(np.sin(p * 6 + i) * 60)), (210, 170, 40), 5, cv2.LINE_AA)
            cv2.GaussianBlur(frame, (0, 0), 5, dst=frame)
        elif effect == "portal_vortex":
            center = (width // 2, height // 2)
            for i in range(26):
                angle = p * np.pi * 6 + i * 0.45
                radius = 12 + i * 11
                x = int(center[0] + np.cos(angle) * radius)
                y = int(center[1] + np.sin(angle) * radius * 0.55)
                cv2.circle(frame, (x, y), 26, (180, 40 + i * 7 % 180, 255), -1, cv2.LINE_AA)
        elif effect == "scene_wipe":
            x = int(width * p)
            frame[:, :x] = (220, 70, 30)
            frame[:, x:] = (20, 180, 230)
        elif effect == "ember_particles":
            frame[:] = (4, 8, 16)
            for i in range(24):
                x = int(width * ((i * 37 % 101) / 100))
                y = int(height * (1.0 - p) + (i * 19 % height) * 0.6)
                cv2.circle(frame, (x, y), 6 + i % 9, (20, 120 + i * 5 % 120, 255), -1, cv2.LINE_AA)
        elif effect == "negative_wave":
            level = int(150 * max(0.0, 1.0 - p))
            frame[:] = (level, level, level)
            cv2.rectangle(frame, (0, 0), (int(width * p), height), (0, 0, 0), -1)
        return cv2.flip(frame, 1)

    @staticmethod
    def _draw_moving_orb(frame: np.ndarray, x: int, y: int, color: tuple[int, int, int], trail_direction: int, progress: float) -> None:
        for i in range(7):
            fade = 1.0 - i / 7
            tx = x + trail_direction * i * 28
            ty = y + int(np.sin((progress + i * 0.05) * np.pi) * 10)
            radius = max(4, int(38 * fade))
            trail_color = tuple(int(c * fade) for c in color)
            cv2.circle(frame, (tx, ty), radius, trail_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 42, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x - 10, y - 12), 16, (255, 255, 255), -1, cv2.LINE_AA)

    @staticmethod
    def _draw_pan_bands(frame: np.ndarray, shift: int) -> None:
        height, width = frame.shape[:2]
        for i in range(-2, 8):
            x0 = (i * 150 + shift) % (width + 300) - 150
            color = (35 + i * 18 % 80, 80 + i * 30 % 140, 150 + i * 25 % 105)
            cv2.rectangle(frame, (x0, 0), (x0 + 90, height), color, -1)
        cv2.GaussianBlur(frame, (0, 0), 2, dst=frame)
