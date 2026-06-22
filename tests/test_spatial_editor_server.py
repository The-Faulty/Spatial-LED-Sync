from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from config import EngineConfig
from effects_engine import HeadlessEffectsEngine
from event_detector import LightEvent
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

    def post_json(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
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
        strip["tv_fill"] = 0.4
        strip["tv_fill_spatial"] = True
        self.assertTrue(self.post_spatial(spatial)["ok"])
        saved = self.saved_spatial()["strips"][0]
        self.assertEqual(saved["start_u"], 1.25)
        self.assertEqual(saved["end_u"], 3.25)
        self.assertEqual(saved["led_count"], 30)
        self.assertEqual(saved["device_id"], "main")
        self.assertEqual(saved["sync_mode"], "tv_image")
        self.assertEqual(saved["blend"], 0.8)
        self.assertEqual(saved["tv_fill"], 0.4)
        self.assertTrue(saved["tv_fill_spatial"])
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

    def test_effect_toggles_persist_and_update_preview_runtime(self) -> None:
        started = self.post_path("/api/preview/start")
        self.assertTrue(started["running"], started)
        payload = self.post_json(
            "/api/effects",
            {
                "enabled_effects": {"shockwave": False, "ambient_side_spill": False},
                "effect_sensitivity": {"shockwave": 0.85},
                "front_ambient_coverage": 0.35,
                "tv_image_blur": 11,
                "ambient_side_spill_base_intensity": 0.25,
                "ambient_side_spill_boost_intensity": 1.6,
            },
        )
        self.assertTrue(payload["ok"], payload)
        self.assertFalse(payload["enabled_effects"]["shockwave"])
        self.assertFalse(payload["enabled_effects"]["ambient_side_spill"])
        self.assertEqual(payload["effect_sensitivity"]["shockwave"], 0.85)
        self.assertEqual(payload["front_ambient_coverage"], 0.35)
        self.assertEqual(payload["tv_image_blur"], 11)
        self.assertEqual(payload["ambient_side_spill_base_intensity"], 0.25)
        self.assertEqual(payload["ambient_side_spill_boost_intensity"], 1.6)
        saved = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertFalse(saved["enabled_effects"]["shockwave"])
        self.assertEqual(saved["effect_sensitivity"]["shockwave"], 0.85)
        self.assertEqual(saved["front_ambient_coverage"], 0.35)
        self.assertEqual(saved["tv_image_blur"], 11)
        self.assertEqual(saved["ambient_side_spill_base_intensity"], 0.25)
        self.assertEqual(saved["ambient_side_spill_boost_intensity"], 1.6)
        runtime = SpatialEditorHandler.preview_manager.runtime
        self.assertIsNotNone(runtime)
        self.assertFalse(runtime.config.enabled_effects["shockwave"])
        self.assertEqual(runtime.config.effect_sensitivity["shockwave"], 0.85)
        self.assertEqual(runtime.config.front_ambient_coverage, 0.35)
        self.assertEqual(runtime.config.tv_image_blur, 11)
        self.assertEqual(runtime.config.ambient_side_spill_base_intensity, 0.25)
        self.assertEqual(runtime.config.ambient_side_spill_boost_intensity, 1.6)
        status = self.get_json("/api/preview/status")
        self.assertIn("enabled_effects", status)
        self.assertFalse(status["enabled_effects"]["shockwave"])
        self.assertEqual(status["effect_sensitivity"]["shockwave"], 0.85)
        self.assertEqual(status["front_ambient_coverage"], 0.35)
        self.assertEqual(status["tv_image_blur"], 11)
        self.assertEqual(status["ambient_side_spill_base_intensity"], 0.25)
        self.assertEqual(status["ambient_side_spill_boost_intensity"], 1.6)

    def test_preview_status_reports_saved_effects_when_stopped(self) -> None:
        payload = self.post_json(
            "/api/effects",
            {"enabled_effects": {"lightning": False}, "effect_sensitivity": {"lightning": 0.9}, "ambient_side_spill_base_intensity": 0.2},
        )
        self.assertTrue(payload["ok"], payload)
        status = self.get_json("/api/preview/status")
        self.assertFalse(status["running"])
        self.assertFalse(status["enabled_effects"]["lightning"])
        self.assertEqual(status["effect_sensitivity"]["lightning"], 0.9)
        self.assertEqual(status["ambient_side_spill_base_intensity"], 0.2)

    def test_config_save_effects_update_running_preview_runtime(self) -> None:
        started = self.post_path("/api/preview/start")
        self.assertTrue(started["running"], started)
        spatial = self.get_config()["spatial"]
        payload = self.post_json(
            "/api/config",
            {
                "spatial": spatial,
                "enabled_effects": {"portal_vortex": False, "front_ambient": False},
                "effect_sensitivity": {"portal_vortex": 0.8},
                "ambient_side_spill_boost_intensity": 1.4,
            },
        )
        self.assertTrue(payload["ok"], payload)
        runtime = SpatialEditorHandler.preview_manager.runtime
        self.assertIsNotNone(runtime)
        self.assertFalse(runtime.config.enabled_effects["portal_vortex"])
        self.assertFalse(runtime.config.enabled_effects["front_ambient"])
        self.assertEqual(runtime.config.effect_sensitivity["portal_vortex"], 0.8)
        self.assertEqual(runtime.config.ambient_side_spill_boost_intensity, 1.4)
        status = self.get_json("/api/preview/status")
        self.assertFalse(status["enabled_effects"]["portal_vortex"])
        self.assertEqual(status["effect_sensitivity"]["portal_vortex"], 0.8)
        self.assertEqual(status["ambient_side_spill_boost_intensity"], 1.4)

    def test_effect_toggle_removes_active_disabled_preview_waves(self) -> None:
        class FakeRuntime:
            running = True

            def __init__(self) -> None:
                self.config = EngineConfig()
                self.event_detector = SimpleNamespace(config=self.config)
                self.wave_engine = SimpleNamespace(
                    waves=[
                        SimpleNamespace(kind="lightning", intensity=1.0),
                        SimpleNamespace(kind="spill", intensity=1.0),
                    ],
                    front_ambient_intensity=1.0,
                    front_ambient_strip_colors=np.ones((2, 3), dtype=np.float32),
                    _front_ambient_led_cache=[0, 1, 2],
                    _front_ambient_led_set_cache={0, 1, 2},
                )

            def stop(self) -> None:
                self.running = False

        fake = FakeRuntime()
        with SpatialEditorHandler.preview_manager.lock:
            SpatialEditorHandler.preview_manager.runtime = fake
        payload = self.post_json(
            "/api/effects",
            {"enabled_effects": {"lightning": False, "front_ambient": False}, "front_ambient_coverage": 0.5},
        )
        self.assertTrue(payload["ok"], payload)
        self.assertFalse(fake.config.enabled_effects["lightning"])
        self.assertEqual(fake.config.front_ambient_coverage, 0.5)
        self.assertEqual([wave.kind for wave in fake.wave_engine.waves], ["spill"])
        self.assertEqual(fake.wave_engine.front_ambient_intensity, 0.0)
        self.assertIsNone(fake.wave_engine.front_ambient_strip_colors)
        self.assertIsNone(fake.wave_engine._front_ambient_led_cache)
        self.assertIsNone(fake.wave_engine._front_ambient_led_set_cache)

    def test_preview_simulation_patterns_are_listed_and_triggerable(self) -> None:
        config_payload = self.get_config()
        patterns = config_payload.get("simulation_patterns", [])
        names = {item["effect"] for item in patterns}
        self.assertIn("lightning", names)
        self.assertIn("ambient_side_spill", names)
        self.assertIn("ambient_side_spill_boost", names)
        started = self.post_path("/api/preview/start")
        self.assertTrue(started["running"], started)
        triggered = self.post_json("/api/preview/trigger", {"effect": "lightning"})
        self.assertTrue(triggered["ok"], triggered)
        self.assertEqual(triggered["effect"], "lightning")

    def test_preview_status_reports_triggered_effects(self) -> None:
        manager = SpatialEditorHandler.preview_manager
        with manager.lock:
            manager.last_snapshot = RuntimeSnapshot(
                events=[
                    LightEvent(edge="top", intensity=0.3, color=(50, 60, 70), kind="front_ambient", effect_id="front_ambient"),
                    LightEvent(edge="left", intensity=0.7, color=(255, 80, 20), kind="energy_trail", effect_id="energy_trail", primary=True),
                    LightEvent(edge="top", intensity=0.4, color=(255, 180, 80), kind="ember_particles", effect_id="ember_particles", secondary=True),
                ]
            )
        status = self.get_json("/api/preview/status")
        effects = status.get("triggered_effects", [])
        self.assertEqual(effects[0]["effect_id"], "energy_trail")
        self.assertEqual(effects[0]["edge"], "left")
        self.assertTrue(effects[0]["primary"])
        self.assertTrue(effects[1]["secondary"])

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
        self.assertIn("lit_led_count", status)
        self.assertIn("led_max", status)
        self.assertGreaterEqual(status.get("led_revision", 0), 1)
        stopped = self.post_path("/api/preview/stop")
        self.assertFalse(stopped["running"])

    def test_simulation_preview_endpoint_overrides_live_hyperhdr_setting(self) -> None:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw["simulate_input"] = False
        raw["hyperhdr_input_mode"] = "websocket"
        raw["hyperhdr_ws_port"] = 9
        self.config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        started = self.post_path("/api/preview/start-simulation")
        self.assertTrue(started["running"], started)
        self.assertEqual(started["mode"], "simulation")
        runtime = SpatialEditorHandler.preview_manager.runtime
        self.assertIsNotNone(runtime)
        self.assertTrue(runtime.config.simulate_input)

        triggered = self.post_json("/api/preview/trigger", {"effect": "lightning"})
        self.assertTrue(triggered["ok"], triggered)
        status = self.get_json("/api/preview/status")
        self.assertEqual(status["preview_mode"], "simulation")

        stopped = self.post_path("/api/preview/stop-simulation")
        self.assertFalse(stopped["running"])

    def test_preview_pattern_loop_can_start_and_stop_without_stopping_preview(self) -> None:
        started = self.post_path("/api/preview/start-simulation")
        self.assertTrue(started["running"], started)
        looped = self.post_json("/api/preview/trigger", {"effect": "loop:lightning"})
        self.assertTrue(looped["ok"], looped)
        self.assertEqual(looped["looping_effect"], "lightning")
        status = self.get_json("/api/preview/status")
        self.assertTrue(status["running"])
        self.assertEqual(status["looping_effect"], "lightning")
        stopped_pattern = self.post_json("/api/preview/trigger", {"effect": "idle"})
        self.assertTrue(stopped_pattern["ok"], stopped_pattern)
        self.assertEqual(stopped_pattern["looping_effect"], "")
        status = self.get_json("/api/preview/status")
        self.assertTrue(status["running"])
        self.assertEqual(status["looping_effect"], "")

    def test_live_preview_does_not_publish_black_led_revision_before_first_frame(self) -> None:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw["simulate_input"] = False
        raw["hyperhdr_input_mode"] = "websocket"
        raw["hyperhdr_ws_port"] = 9
        self.config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        preview = self.get_json("/api/preview-colors")
        self.assertTrue(preview.get("colors"))

        started = self.post_path("/api/preview/start")
        self.assertTrue(started["running"], started)
        time.sleep(0.3)
        status = self.get_json("/api/preview/status")
        self.assertTrue(status.get("running"))
        self.assertEqual(status.get("frame_revision"), 0)
        self.assertEqual(status.get("led_revision"), 0)
        self.assertEqual(status.get("led_count"), 6)
        self.assertEqual(len(status.get("leds_flat", [])), 18)
        self.assertEqual(max(status.get("leds_flat", [0])), 0)
        self.assertEqual(status.get("lit_led_count"), 0)
        self.assertEqual(status.get("led_max"), 0)

    def test_preview_frame_endpoint_returns_latest_processed_frame(self) -> None:
        frame = np.zeros((8, 12, 3), dtype=np.uint8)
        frame[:, :, 1] = 180
        manager = SpatialEditorHandler.preview_manager
        with manager.lock:
            manager.last_snapshot = RuntimeSnapshot(frame=frame, leds=np.zeros((6, 3), dtype=np.uint8), wled_skip_count=4)
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
        self.assertEqual(status["lit_led_count"], 0)
        self.assertEqual(status["led_max"], 0)
        self.assertEqual(status["wled_skip_count"], 4)

    def test_editor_static_refs_preview_frame_and_deselect(self) -> None:
        index, _ = self.get_bytes("/")
        app, _ = self.get_bytes("/app.js")
        styles, _ = self.get_bytes("/styles.css")
        scene3d, _ = self.get_bytes("/scene3d.js")
        vendor, _ = self.get_bytes("/vendor/three.module.js")
        orbit, _ = self.get_bytes("/vendor/OrbitControls.js")
        self.assertIn(b'id="deselect-strip"', index)
        self.assertIn(b'id="view-2d"', index)
        self.assertIn(b'id="view-3d"', index)
        self.assertIn(b'id="scene3d"', index)
        self.assertIn(b'id="start-simulation"', index)
        self.assertIn(b'id="stop-simulation"', index)
        self.assertIn(b'id="stop-pattern"', index)
        self.assertIn(b'id="effect-toggles"', index)
        self.assertIn(b'id="effect-sensitivity"', index)
        self.assertIn(b'id="ambient-side-spill-base-intensity"', index)
        self.assertIn(b'id="ambient-side-spill-boost-intensity"', index)
        self.assertIn(b'id="effect-patterns"', index)
        self.assertIn(b'id="triggered-effects"', index)
        self.assertIn(b'id="active-effects"', index)
        self.assertIn(b'type="importmap"', index)
        self.assertIn(b"/vendor/three.module.js", index)
        self.assertIn(b"/api/preview/frame.jpg", app)
        self.assertIn(b"/api/effects", app)
        self.assertIn(b"effect_sensitivity", app)
        self.assertIn(b"ambient_side_spill_base_intensity", app)
        self.assertIn(b"ambient_side_spill_boost_intensity", app)
        self.assertIn(b"tv_image_blur", app)
        self.assertIn(b'id="tv-image-blur"', index)
        self.assertIn(b"TV image blur", index)
        self.assertIn(b"TRIGGER_EFFECTS", app)
        self.assertIn(b"/api/preview/trigger", app)
        self.assertIn(b"pendingEffectOverrides", app)
        self.assertIn(b"/api/preview/start-simulation", app)
        self.assertIn(b"/api/preview/stop-simulation", app)
        self.assertIn(b"loop:", app)
        self.assertIn(b"looping_effect", app)
        self.assertIn(b"active_effect_counts", app)
        self.assertIn(b"triggered_effects", app)
        self.assertIn(b"Primary:", app)
        self.assertIn(b"Ambient side spill", app)
        self.assertIn(b"leds_flat", app)
        self.assertNotIn(b"LED rev", app)
        self.assertNotIn(b"Frame rev", app)
        self.assertIn(b"View render", app)
        self.assertIn(b"PREVIEW_MAX_FPS = 90", app)
        self.assertIn(b"Max LED", app)
        self.assertIn(b"mirrorHorizontal", app)
        self.assertIn(b"drawViewHud", app)
        self.assertIn(b".effect-range", styles)
        self.assertIn(b".effect-panel", styles)
        self.assertIn(b".effect-sensitivity-row", styles)
        self.assertIn(b'id="strip-tv-fill"', index)
        self.assertIn(b'id="strip-tv-fill-row"', index)
        self.assertIn(b'id="strip-tv-fill-spatial"', index)
        self.assertIn(b"tv_fill", app)
        self.assertIn(b"tv_fill_spatial", app)
        self.assertIn(b'id="strip-extension-mode"', index)
        self.assertIn(b"extension_strength", app)
        self.assertIn(b"createLightPreview3d", app)
        self.assertIn(b"#scene[hidden]", styles)
        self.assertIn(b"#scene3d[hidden]", styles)
        self.assertIn(b"WebGLRenderer", scene3d)
        self.assertIn(b"InstancedMesh", scene3d)
        self.assertIn(b"InstancedBufferAttribute", scene3d)
        self.assertIn(b"PointsMaterial", scene3d)
        self.assertIn(b"depthTest: false", scene3d)
        self.assertIn(b"colorIndex", scene3d)
        self.assertIn(b'count - 1 - i', scene3d)
        self.assertIn(b"makeSurfaceLightmap", scene3d)
        self.assertIn(b"updateSurfaceLightmaps", scene3d)
        self.assertIn(b"clearTvFromLightmap", scene3d)
        self.assertIn(b"TV_LIGHTMAP_CLEAR_MARGIN", scene3d)
        self.assertIn(b"updateSeeThroughWall", scene3d)
        self.assertIn(b"wallCameraIsOutsideOf", scene3d)
        self.assertIn(b'const mirror = "none"', scene3d)
        self.assertIn(b"room.depth + span * 1.55", scene3d)
        self.assertIn(b"AdditiveBlending", scene3d)
        self.assertIn(b"replaceTvTexture", scene3d)
        self.assertIn(b"OrbitControls", scene3d)
        self.assertIn(b"depthWrite: false", scene3d)
        self.assertIn(b"renderOrder = 10", scene3d)
        self.assertIn(b"WebGLRenderer", vendor)
        self.assertIn(b"InstancedMesh", vendor)
        self.assertIn(b"class OrbitControls", orbit)

    def get_json(self, path: str) -> dict:
        with urllib.request.urlopen(self.base_url + path, timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
