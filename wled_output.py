from __future__ import annotations

import logging
import time

import numpy as np
import requests

from config import EngineConfig
from spatial_topology import SpatialRoomTopology


class WLEDOutput:
    def __init__(self, config: EngineConfig, logger: logging.Logger, spatial_topology: SpatialRoomTopology | None = None):
        self.config = config
        self.logger = logger.getChild("wled")
        self.session = requests.Session()
        self.device_sessions: dict[str, requests.Session] = {}
        self.last_send = 0.0
        self.spatial_topology = spatial_topology
        self.enabled = bool(config.send_to_wled and (config.wled_ip or self._has_spatial_devices()))

    def send(self, leds: np.ndarray) -> bool:
        if not self.enabled:
            return False
        now = time.monotonic()
        min_interval = 1.0 / max(1, self.config.wled_fps)
        if now - self.last_send < min_interval:
            return False
        self.last_send = now

        if self.spatial_topology is not None:
            return self._send_spatial(leds)

        colors = leds.astype(int).tolist()
        segment: dict[str, object] = {"i": colors}
        if self.config.wled_segment_id is not None:
            segment["id"] = self.config.wled_segment_id
        payload = {"bri": int(max(0.0, min(1.0, self.config.brightness)) * 255), "seg": [segment]}
        url = f"http://{self.config.wled_ip}/json/state"
        try:
            response = self.session.post(url, json=payload, timeout=self.config.wled_timeout)
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            self.logger.warning("WLED send failed: %s", exc)
            self.session.close()
            self.session = requests.Session()
            return False

    def _has_spatial_devices(self) -> bool:
        if self.spatial_topology is None:
            return False
        return any(device.enabled and device.ip.strip() for device in self.spatial_topology.spatial.devices)

    def _send_spatial(self, leds: np.ndarray) -> bool:
        assert self.spatial_topology is not None
        sent_any = False
        for route in self.spatial_topology.device_routes():
            if not route.device.ip.strip():
                continue
            colors = np.zeros((route.device.led_count, 3), dtype=np.uint8)
            colors[route.device_indices] = leds[route.global_indices]
            segment: dict[str, object] = {"i": colors.astype(int).tolist()}
            if route.device.segment_id is not None:
                segment["id"] = route.device.segment_id
            payload = {"bri": int(max(0.0, min(1.0, self.config.brightness)) * 255), "seg": [segment]}
            url = f"http://{route.device.ip}/json/state"
            session = self.device_sessions.setdefault(route.device.id, requests.Session())
            try:
                response = session.post(url, json=payload, timeout=self.config.wled_timeout)
                response.raise_for_status()
                sent_any = True
            except requests.RequestException as exc:
                self.logger.warning("WLED send failed for %s: %s", route.device.id, exc)
                session.close()
                self.device_sessions[route.device.id] = requests.Session()
        return sent_any
