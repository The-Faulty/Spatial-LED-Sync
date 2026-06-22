from __future__ import annotations

import logging
import socket
import time

import numpy as np
import requests

from config import EngineConfig
from spatial_topology import SpatialRoomTopology


class WLEDOutput:
    DDP_PORT = 4048
    DDP_DATA_TYPE_RGB = 0x01
    DDP_MAX_DATA_BYTES = 1440

    def __init__(self, config: EngineConfig, logger: logging.Logger, spatial_topology: SpatialRoomTopology | None = None):
        self.config = config
        self.logger = logger.getChild("wled")
        self.session = requests.Session()
        self.device_sessions: dict[str, requests.Session] = {}
        self.udp_socket: socket.socket | None = None
        self.last_send = 0.0
        self.spatial_topology = spatial_topology
        self.enabled = bool(config.send_to_wled and (config.wled_ip or self._has_spatial_devices()))
        self.last_leds: np.ndarray | None = None
        self.last_device_leds: dict[str, np.ndarray] = {}
        self.skip_count = 0

    def send(self, leds: np.ndarray) -> bool:
        if not self.enabled:
            return False
        now = time.monotonic()
        min_interval = 1.0 / max(1, self.config.wled_fps)
        if now - self.last_send < min_interval:
            return False

        if self.spatial_topology is not None:
            return self._send_spatial(leds)

        colors_uint8 = leds.astype(np.uint8, copy=False)
        if not self._dirty_enough(colors_uint8, self.last_leds):
            self.skip_count += 1
            return False
        self.last_send = now
        if self.config.wled_protocol == "ddp":
            try:
                self._send_ddp(self.config.wled_ip, colors_uint8)
                self.last_leds = colors_uint8.copy()
                return True
            except OSError as exc:
                self.logger.warning("WLED DDP send failed: %s", exc)
                return False
        if self._send_json(self.config.wled_ip, colors_uint8, self.config.wled_segment_id):
            self.last_leds = colors_uint8.copy()
            return True
        return False

    def _send_json(self, ip: str, colors_uint8: np.ndarray, segment_id: int | None, session_id: str | None = None) -> bool:
        colors = colors_uint8.astype(int).tolist()
        segment: dict[str, object] = {"i": colors}
        if segment_id is not None:
            segment["id"] = segment_id
        payload = {"bri": int(max(0.0, min(1.0, self.config.brightness)) * 255), "seg": [segment]}
        url = f"http://{ip}/json/state"
        session = self.session if session_id is None else self.device_sessions.setdefault(session_id, requests.Session())
        try:
            response = session.post(url, json=payload, timeout=self.config.wled_timeout)
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            self.logger.warning("WLED send failed: %s", exc)
            session.close()
            if session_id is None:
                self.session = requests.Session()
            else:
                self.device_sessions[session_id] = requests.Session()
            return False

    def _has_spatial_devices(self) -> bool:
        if self.spatial_topology is None:
            return False
        return any(device.enabled and device.ip.strip() for device in self.spatial_topology.spatial.devices)

    def _send_spatial(self, leds: np.ndarray) -> bool:
        assert self.spatial_topology is not None
        sent_any = False
        skipped_any = False
        for route in self.spatial_topology.device_routes():
            if not route.device.ip.strip():
                continue
            colors = np.zeros((route.device.led_count, 3), dtype=np.uint8)
            colors[route.device_indices] = leds[route.global_indices]
            previous = self.last_device_leds.get(route.device.id)
            if not self._dirty_enough(colors, previous):
                skipped_any = True
                continue
            if self.config.wled_protocol == "ddp":
                try:
                    self._send_ddp(route.device.ip, colors)
                except OSError as exc:
                    self.logger.warning("WLED DDP send failed for %s: %s", route.device.id, exc)
                    continue
                self.last_device_leds[route.device.id] = colors.copy()
                sent_any = True
            elif self._send_json(route.device.ip, colors, route.device.segment_id, route.device.id):
                self.last_device_leds[route.device.id] = colors.copy()
                sent_any = True
        if sent_any:
            self.last_send = time.monotonic()
        elif skipped_any:
            self.skip_count += 1
        return sent_any

    def _send_ddp(self, ip: str, colors: np.ndarray) -> None:
        data = np.ascontiguousarray(colors, dtype=np.uint8).reshape(-1, 3).tobytes()
        port = int(getattr(self.config, "wled_udp_port", self.DDP_PORT))
        sequence = 0
        udp_socket = self._udp_socket()
        for offset in range(0, len(data), self.DDP_MAX_DATA_BYTES):
            chunk = data[offset : offset + self.DDP_MAX_DATA_BYTES]
            header = bytes(
                [
                    0x41,
                    sequence & 0xFF,
                    self.DDP_DATA_TYPE_RGB,
                    0x01,
                ]
            ) + offset.to_bytes(4, "big") + len(chunk).to_bytes(2, "big")
            udp_socket.sendto(header + chunk, (ip, port))
            sequence = (sequence + 1) & 0xFF

    def _udp_socket(self):
        if self.udp_socket is None:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return self.udp_socket

    def _dirty_enough(self, current: np.ndarray, previous: np.ndarray | None) -> bool:
        if previous is None or previous.shape != current.shape:
            return True
        threshold = int(max(0, self.config.wled_delta_threshold))
        if threshold <= 0:
            return bool(np.any(current != previous))
        delta = np.abs(current.astype(np.int16) - previous.astype(np.int16))
        return bool(np.any(delta > threshold))
