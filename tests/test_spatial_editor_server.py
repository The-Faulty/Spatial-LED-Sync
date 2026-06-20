from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from config import EngineConfig
from effects_engine import HeadlessEffectsEngine
from runtime import RuntimeSnapshot
from spatial_config import default_spatial_dict
from spatial_editor_server import SpatialEditorHandler


def editor_spatial_config() -> dict:
    spatial = default_spatial_dict(12)
    spatial["enabled"] = True
    spatial["room"] = {"width": 4.0, "depth": 3.0, "height": 2.4, "unit": "m"}
    spatial["devices"] = [
        {"id": "main", "name": "Main", "ip": "", "led_count": 24, "segment_id": None, "enabled": True}
    ]
    spatial["strips"] = [
        {
            "id": "front",
            "name": "Front",
            "wall": "front",
            "start_u": 0.0,
            "start_v": 1.9,
            "end_u": 4.0,
            "end_v": 1.9,
            "led_count": 6,
            "direction": "forward",
            "device_id": "main",
            "device_start": 0,
            "sync_mode": "spatial",
            "blend": 0.5,
        }
    ]
    return spatial


class EditorServerCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp.name) / "config.json"
        EngineConfig(total_leds=240, spatial=editor_spatial_config(), simulate_input=True, send_to_wled=False).save(self.config_path)
        SpatialEditorHandler.config_path = self.config_path
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), SpatialEditorHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        SpatialEditorHandler.preview_manager.stop()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def get_config(self) -> dict:
        with urllib.request.urlopen(self.base_url + "/api/config", timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))

    def post_spatial(self, spatial: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + "/api/config",
            data=json.dumps({"spatial": spatial}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))

    def post_path(self, path: str) -> dict:
        request = urllib.request.Request(self.base_url + path, data=b"{}", method="POST")
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_bytes(self, path: str) -> tuple[bytes, str]:
        with urllib.request.urlopen(self.base_url + path, timeout=3) as response:
            return response.read(), response.headers.get("Content-Type", "")

    def saved_spatial(self) -> dict:
        return json.loads(self.config_path.read_text(encoding="utf-8"))["spatial"]

    def test_device_json_changes_persist(self) -> None:
        spatial = self.get_config()["spatial"]
        spatial["devices"].append(
            {"id": "accent", "name": "Accent", "ip": "192.0.2.44", "led_count": 30, "segment_id": 1, "enabled": True}
        )
        self.assertTrue(self.post_spatial(spatial)["ok"])
        saved = self.saved_spatial()
        self.assertEqual(saved["devices"][1]["id"], "accent")
        self.assertEqual(saved["devices"][1]["ip"], "192.0.2.44")

    def test_new_strip_persists(self) -> None:
        spatial = self.get_config()["spatial"]
        spatial["strips"].append(
            {
                "id": "extension",
                "name": "Extension",
                "wall": "left",
                "start_u": 0.2,
                "start_v": 1.5,
                "end_u": 2.2,
                "end_v": 1.5,
                "led_count": 5,
                "direction": "forward",
                "device_id": "main",
                "device_start": 0,
                "sync_mode": "blend",
                "blend": 0.35,
            }
        )
        self.assertTrue(self.post_spatial(spatial)["ok"])
        saved = self.saved_spatial()
        self.assertEqual(saved["strips"][1]["id"], "extension")
        self.assertEqual(saved["strips"][1]["device_start"], 6)
        self.assertEqual(saved["strips"][1]["sync_mode"], "blend")

    def test_strip_editor_field_changes_persist(self) -> None:
        spatial = self.get_config()["spatial"]
        strip = spatial["strips"][0]
        strip["start_u"] = 1.25
        strip["end_u"] = 3.25
        strip["led_count"] = 30
        strip["device_id"] = "main"
        strip["sync_mode"] = "tv_image"
        strip["blend"] = 0.8
        self.assertTrue(self.post_spatial(spatial)["ok"])
        saved = self.saved_spatial()["strips"][0]
        self.assertEqual(saved["start_u"], 1.25)
        self.assertEqual(saved["end_u"], 3.25)
        self.assertEqual(saved["led_count"], 30)
        self.assertEqual(saved["device_id"], "main")
        self.assertEqual(saved["sync_mode"], "tv_image")
        self.assertEqual(saved["blend"], 0.8)
        self.assertGreaterEqual(self.saved_spatial()["devices"][0]["led_count"], 30)

    def test_extension_mode_fields_persist(self) -> None:
        spatial = self.get_config()["spatial"]
        spatial["strips"][0]["tv_role"] = "bottom"
        spatial["strips"].append(
            {
                "id": "edge-extension",
                "name": "Edge Extension",
                "wall": "front",
                "start_u": 0.4,
                "start_v": 0.8,
                "end_u": 0.4,
                "end_v": 0.1,
                "led_count": 4,
                "direction": "forward",
                "device_id": "main",
                "device_start": 0,
                "sync_mode": "blend",
                "blend": 0.4,
                "tv_role": "none",
                "extends_strip_id": "front",
                "extension_mode": "edge_reach",
                "extension_strength": 0.75,
                "extension_softness": 0.25,
            }
        )
        self.assertTrue(self.post_spatial(spatial)["ok"])
        saved = self.saved_spatial()["strips"][1]
        self.assertEqual(saved["extension_mode"], "edge_reach")
        self.assertEqual(saved["extension_strength"], 0.75)
        self.assertEqual(saved["extension_softness"], 0.25)

    def test_preview_runtime_endpoints_force_wled_off(self) -> None:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw["send_to_wled"] = True
        raw["spatial"]["devices"][0]["ip"] = "192.0.2.50"
        self.config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        started = self.post_path("/api/preview/start")
        self.assertTrue(started["running"], started)
        status = {}
        for _ in range(10):
            status = self.get_json("/api/preview/status")
            if status.get("leds_flat"):
                break
            time.sleep(0.1)
        self.assertTrue(status.get("running"))
        self.assertIsInstance(SpatialEditorHandler.preview_manager.runtime, HeadlessEffectsEngine)
        self.assertFalse(SpatialEditorHandler.preview_manager.runtime.config.send_to_wled)
        self.assertFalse(status.get("wled_enabled"))
        self.assertEqual(status.get("led_count"), 6)
        self.assertEqual(len(status.get("leds_flat", [])), 18)
        self.assertGreaterEqual(status.get("led_revision", 0), 1)
        stopped = self.post_path("/api/preview/stop")
        self.assertFalse(stopped["running"])

    def test_preview_frame_endpoint_returns_latest_processed_frame(self) -> None:
        frame = np.zeros((8, 12, 3), dtype=np.uint8)
        frame[:, :, 1] = 180
        manager = SpatialEditorHandler.preview_manager
        with manager.lock:
            manager.last_snapshot = RuntimeSnapshot(frame=frame, leds=np.zeros((6, 3), dtype=np.uint8))
            manager.frame_revision = 7
            manager.led_revision = 3
        data, content_type = self.get_bytes("/api/preview/frame.jpg")
        self.assertEqual(content_type, "image/jpeg")
        self.assertTrue(data.startswith(b"\xff\xd8"))
        status = self.get_json("/api/preview/status")
        self.assertEqual(status["frame_revision"], 7)
        self.assertEqual(status["led_revision"], 3)
        self.assertEqual(status["led_count"], 6)
        self.assertEqual(len(status["leds_flat"]), 18)

    def test_editor_static_refs_preview_frame_and_deselect(self) -> None:
        index, _ = self.get_bytes("/")
        app, _ = self.get_bytes("/app.js")
        self.assertIn(b'id="deselect-strip"', index)
        self.assertIn(b"/api/preview/frame.jpg", app)
        self.assertIn(b"leds_flat", app)
        self.assertIn(b"mirrorHorizontal", app)
        self.assertIn(b"drawViewHud", app)
        self.assertIn(b'id="strip-extension-mode"', index)
        self.assertIn(b"extension_strength", app)

    def get_json(self, path: str) -> dict:
        with urllib.request.urlopen(self.base_url + path, timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
