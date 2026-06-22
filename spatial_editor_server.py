from __future__ import annotations

import argparse
import copy
import json
import logging
import mimetypes
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from config import DEFAULT_EFFECT_SENSITIVITY, DEFAULT_ENABLED_EFFECTS, TRIGGER_EFFECTS, EngineConfig
from effects_engine import HeadlessEffectsEngine, RuntimeSnapshot
from hyperhdr_client import SIMULATION_EFFECT_PATTERNS
from spatial_config import pack_spatial_device_ranges, parse_spatial_config, spatial_config_to_dict, validate_spatial_config
from spatial_topology import SpatialRoomTopology


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web_editor"


class PreviewRuntimeManager:
    def __init__(self) -> None:
        self.runtime: HeadlessEffectsEngine | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.last_snapshot = RuntimeSnapshot()
        self.last_error = ""
        self.led_revision = 0
        self.frame_revision = 0
        self.preview_mode = "stopped"
        self.looping_effect = ""

    def start(self, config_path: Path, simulate_input: bool | None = None) -> dict[str, object]:
        with self.lock:
            self.stop_locked()
            self.last_snapshot = RuntimeSnapshot()
            self.led_revision = 0
            self.frame_revision = 0
            self.looping_effect = ""
            config = EngineConfig.load(config_path)
            if simulate_input is not None:
                config.simulate_input = simulate_input
            config.send_to_wled = False
            config.debug = False
            logger = logging.getLogger("spatial_editor.preview")
            logger.addHandler(logging.NullHandler())
            self.runtime = HeadlessEffectsEngine(copy.deepcopy(config), logger)
            try:
                self.runtime.start()
            except Exception as exc:
                self.runtime = None
                self.preview_mode = "stopped"
                self.last_error = str(exc)
                return {"running": False, "error": self.last_error}
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            self.preview_mode = "simulation" if config.simulate_input else "live"
            self.last_error = ""
            return {"running": True, "error": "", "mode": self.preview_mode}

    def stop(self) -> dict[str, object]:
        with self.lock:
            self.stop_locked()
            return {"running": False, "error": self.last_error}

    def trigger(self, effect: str) -> dict[str, object]:
        with self.lock:
            runtime = self.runtime
            if runtime is None or not runtime.running:
                return {"ok": False, "error": "preview is not running"}
            if not runtime.config.simulate_input:
                return {"ok": False, "error": "simulation input is disabled"}
            ok = runtime.trigger_simulation_effect(effect)
            if ok:
                if effect.startswith("loop:"):
                    self.looping_effect = effect.split(":", 1)[1]
                elif effect == "idle":
                    self.looping_effect = ""
            return {
                "ok": ok,
                "effect": effect,
                "looping_effect": self.looping_effect,
                "error": "" if ok else "unable to trigger simulation effect",
            }

    def stop_locked(self) -> None:
        self.stop_event.set()
        runtime = self.runtime
        thread = self.thread
        self.runtime = None
        self.thread = None
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        if runtime:
            runtime.stop()
        self.last_snapshot.running = False
        self.preview_mode = "stopped"
        self.looping_effect = ""

    def status(self, config_path: Path | None = None) -> dict[str, object]:
        with self.lock:
            snapshot = self.last_snapshot
            if snapshot.leds is not None:
                leds = snapshot.leds.astype(np.uint8, copy=False)
                leds_flat = leds.reshape(-1).astype(int).tolist()
                led_count = int(leds.shape[0])
                led_max = int(leds.max()) if leds.size else 0
                lit_led_count = int(np.count_nonzero(np.max(leds, axis=1))) if leds.size else 0
            else:
                leds_flat = []
                led_count = 0
                led_max = 0
                lit_led_count = 0
            enabled_effects = dict(DEFAULT_ENABLED_EFFECTS)
            effect_sensitivity = dict(DEFAULT_EFFECT_SENSITIVITY)
            front_ambient_coverage = 1.0
            ambient_side_spill_base_intensity = 0.45
            ambient_side_spill_boost_intensity = 1.0
            active_effect_counts: dict[str, int] = {}
            runtime = self.runtime
            if runtime is not None:
                enabled_effects.update(runtime.config.enabled_effects)
                effect_sensitivity.update(runtime.config.effect_sensitivity)
                front_ambient_coverage = runtime.config.front_ambient_coverage
                ambient_side_spill_base_intensity = runtime.config.ambient_side_spill_base_intensity
                ambient_side_spill_boost_intensity = runtime.config.ambient_side_spill_boost_intensity
                renderer = runtime.wave_engine
                waves = getattr(renderer, "waves", []) if renderer is not None else []
                for wave in waves:
                    kind = getattr(wave, "kind", "")
                    if kind:
                        active_effect_counts[kind] = active_effect_counts.get(kind, 0) + 1
            elif config_path is not None:
                config = EngineConfig.load(config_path)
                enabled_effects.update(config.enabled_effects)
                effect_sensitivity.update(config.effect_sensitivity)
                front_ambient_coverage = config.front_ambient_coverage
                ambient_side_spill_base_intensity = config.ambient_side_spill_base_intensity
                ambient_side_spill_boost_intensity = config.ambient_side_spill_boost_intensity
            triggered = [
                {
                    "kind": event.kind,
                    "effect_id": event.effect_id or event.kind,
                    "edge": event.edge,
                    "intensity": event.intensity,
                    "secondary": event.secondary,
                    "primary": event.primary,
                }
                for event in snapshot.events
                if event.kind not in {"front_ambient", "ambient_side_spill"}
            ]
            return {
                "running": bool(self.runtime and self.runtime.running),
                "preview_mode": self.preview_mode,
                "looping_effect": self.looping_effect,
                "fps": snapshot.fps,
                "active_waves": snapshot.active_waves,
                "active_effect_counts": active_effect_counts,
                "triggered_effects": triggered,
                "candidate_events": snapshot.candidate_events,
                "wled_skip_count": snapshot.wled_skip_count,
                "enabled_effects": enabled_effects,
                "effect_sensitivity": effect_sensitivity,
                "front_ambient_coverage": front_ambient_coverage,
                "ambient_side_spill_base_intensity": ambient_side_spill_base_intensity,
                "ambient_side_spill_boost_intensity": ambient_side_spill_boost_intensity,
                "buffered_frames": snapshot.buffered_frames,
                "hyperhdr_connected": snapshot.hyperhdr_connected,
                "wled_enabled": False,
                "last_error": snapshot.last_error or self.last_error,
                "led_count": led_count,
                "lit_led_count": lit_led_count,
                "led_max": led_max,
                "led_revision": self.led_revision,
                "frame_revision": self.frame_revision,
                "leds_flat": leds_flat,
            }

    def latest_frame_jpeg(self) -> tuple[bytes | None, int]:
        with self.lock:
            frame = self.last_snapshot.frame
            revision = self.frame_revision
            if frame is None:
                return None, revision
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                return None, revision
            return encoded.tobytes(), revision

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            runtime = self.runtime
            if runtime is None:
                break
            frame_interval = 1.0 / max(1, runtime.config.target_fps)
            started = time.monotonic()
            try:
                snapshot = runtime.step_once(timeout=frame_interval)
                with self.lock:
                    self.last_snapshot = snapshot
                    if snapshot.frame is not None:
                        self.frame_revision += 1
                        if snapshot.leds is not None:
                            self.led_revision += 1
            except Exception as exc:
                with self.lock:
                    self.last_error = str(exc)
                time.sleep(0.2)
            elapsed = time.monotonic() - started
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)


class SpatialEditorHandler(BaseHTTPRequestHandler):
    config_path = ROOT / "config.json"
    preview_manager = PreviewRuntimeManager()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/config":
            self._send_config()
            return
        if path == "/api/preview-colors":
            self._send_preview_colors()
            return
        if path == "/api/preview/status":
            self._send_json(self.preview_manager.status(self.config_path), compact=True)
            return
        if path == "/api/preview/frame.jpg":
            self._send_preview_frame()
            return
        self._send_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/preview/start":
            self._send_json(self.preview_manager.start(self.config_path))
            return
        if path == "/api/preview/start-simulation":
            self._send_json(self.preview_manager.start(self.config_path, simulate_input=True))
            return
        if path == "/api/preview/stop-simulation":
            self._send_json(self.preview_manager.stop())
            return
        if path == "/api/preview/stop":
            self._send_json(self.preview_manager.stop())
            return
        if path == "/api/preview/trigger":
            self._trigger_preview_effect()
            return
        if path == "/api/effects":
            self._update_effects()
            return
        if path != "/api/config":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"ok": False, "errors": ["invalid JSON"]}, status=400)
            return
        config = EngineConfig.load(self.config_path)
        if "spatial" in payload:
            config.spatial = pack_spatial_device_ranges(payload["spatial"], config.total_leds)
        if isinstance(payload.get("enabled_effects"), dict):
            config.enabled_effects.update(
                {key: bool(value) for key, value in payload["enabled_effects"].items() if key in DEFAULT_ENABLED_EFFECTS}
            )
        if isinstance(payload.get("effect_sensitivity"), dict):
            try:
                config.effect_sensitivity.update(
                    {key: float(value) for key, value in payload["effect_sensitivity"].items() if key in TRIGGER_EFFECTS}
                )
            except (TypeError, ValueError):
                self._send_json({"ok": False, "errors": ["effect_sensitivity values must be numbers"]}, status=400)
                return
        for field in ("ambient_side_spill_base_intensity", "ambient_side_spill_boost_intensity"):
            if field in payload:
                try:
                    setattr(config, field, float(payload[field]))
                except (TypeError, ValueError):
                    self._send_json({"ok": False, "errors": [f"{field} must be a number"]}, status=400)
                    return
        errors = config.validate()
        if errors:
            self._send_json({"ok": False, "errors": errors}, status=400)
            return
        config.save(self.config_path)
        self._apply_effect_settings_to_preview(config)
        self._send_json({"ok": True, "config": config.to_dict()})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_config(self) -> None:
        config = EngineConfig.load(self.config_path)
        spatial = parse_spatial_config(config.spatial, config.total_leds)
        config.spatial = spatial_config_to_dict(spatial)
        patterns = [{"effect": effect, "label": label} for effect, label in SIMULATION_EFFECT_PATTERNS]
        self._send_json({"config": config.to_dict(), "spatial": config.spatial, "simulation_patterns": patterns})

    def _trigger_preview_effect(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "invalid JSON"}, status=400)
            return
        effect = str(payload.get("effect", "")).strip()
        valid = {name for name, _label in SIMULATION_EFFECT_PATTERNS}
        legacy = {"left_exit", "right_exit", "top_spill", "top_color_exit_left", "top_color_exit_right", "pan_left", "pan_right"}
        requested = effect
        if effect.startswith("loop:"):
            requested = effect.split(":", 1)[1]
        if requested != "idle" and requested not in valid and requested not in legacy:
            self._send_json({"ok": False, "error": f"unknown simulation effect: {effect}"}, status=400)
            return
        result = self.preview_manager.trigger(effect)
        self._send_json(result, status=200 if result.get("ok") else 400)

    def _update_effects(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json({"ok": False, "errors": ["invalid JSON"]}, status=400)
            return
        requested = payload.get("enabled_effects")
        if not isinstance(requested, dict):
            self._send_json({"ok": False, "errors": ["enabled_effects must be an object"]}, status=400)
            return
        config = EngineConfig.load(self.config_path)
        config.enabled_effects.update({key: bool(value) for key, value in requested.items() if key in DEFAULT_ENABLED_EFFECTS})
        if isinstance(payload.get("effect_sensitivity"), dict):
            try:
                config.effect_sensitivity.update(
                    {key: float(value) for key, value in payload["effect_sensitivity"].items() if key in TRIGGER_EFFECTS}
                )
            except (TypeError, ValueError):
                self._send_json({"ok": False, "errors": ["effect_sensitivity values must be numbers"]}, status=400)
                return
        if "front_ambient_coverage" in payload:
            try:
                config.front_ambient_coverage = float(payload["front_ambient_coverage"])
            except (TypeError, ValueError):
                self._send_json({"ok": False, "errors": ["front_ambient_coverage must be a number"]}, status=400)
                return
        for field in ("ambient_side_spill_base_intensity", "ambient_side_spill_boost_intensity"):
            if field in payload:
                try:
                    setattr(config, field, float(payload[field]))
                except (TypeError, ValueError):
                    self._send_json({"ok": False, "errors": [f"{field} must be a number"]}, status=400)
                    return
        errors = config.validate()
        if errors:
            self._send_json({"ok": False, "errors": errors}, status=400)
            return
        config.save(self.config_path)
        self._apply_effect_settings_to_preview(config)
        self._send_json(
            {
                "ok": True,
                "enabled_effects": config.enabled_effects,
                "effect_sensitivity": config.effect_sensitivity,
                "front_ambient_coverage": config.front_ambient_coverage,
                "ambient_side_spill_base_intensity": config.ambient_side_spill_base_intensity,
                "ambient_side_spill_boost_intensity": config.ambient_side_spill_boost_intensity,
            }
        )

    def _apply_effect_settings_to_preview(self, config: EngineConfig) -> None:
        with self.preview_manager.lock:
            runtime = self.preview_manager.runtime
            if runtime is None:
                return
            runtime.config.enabled_effects.update(config.enabled_effects)
            runtime.config.effect_sensitivity.update(config.effect_sensitivity)
            runtime.config.front_ambient_coverage = config.front_ambient_coverage
            runtime.config.ambient_side_spill_base_intensity = config.ambient_side_spill_base_intensity
            runtime.config.ambient_side_spill_boost_intensity = config.ambient_side_spill_boost_intensity
            if runtime.event_detector is not None:
                runtime.event_detector.config.enabled_effects.update(config.enabled_effects)
                runtime.event_detector.config.effect_sensitivity.update(config.effect_sensitivity)
                runtime.event_detector.config.front_ambient_coverage = config.front_ambient_coverage
                runtime.event_detector.config.ambient_side_spill_base_intensity = config.ambient_side_spill_base_intensity
                runtime.event_detector.config.ambient_side_spill_boost_intensity = config.ambient_side_spill_boost_intensity
            renderer = runtime.wave_engine
            if renderer is None:
                return
            waves = getattr(renderer, "waves", None)
            if isinstance(waves, list):
                renderer.waves = [
                    wave
                    for wave in waves
                    if config.enabled_effects.get(getattr(wave, "kind", ""), True)
                ]
            if not config.enabled_effects.get("front_ambient", True):
                if hasattr(renderer, "front_ambient_intensity"):
                    renderer.front_ambient_intensity = 0.0
                if hasattr(renderer, "front_ambient_strip_colors"):
                    renderer.front_ambient_strip_colors = None
            if hasattr(renderer, "_front_ambient_led_cache"):
                renderer._front_ambient_led_cache = None
            if hasattr(renderer, "_front_ambient_led_set_cache"):
                renderer._front_ambient_led_set_cache = None
            if hasattr(renderer, "_front_ambient_reserved_mask_cache"):
                renderer._front_ambient_reserved_mask_cache = None

    def _send_preview_colors(self) -> None:
        config = EngineConfig.load(self.config_path)
        spatial = parse_spatial_config(config.spatial, config.total_leds)
        if not spatial.enabled:
            spatial.enabled = True
            config.spatial = spatial_config_to_dict(spatial)
        errors = validate_spatial_config(config.spatial, config.total_leds)
        if errors:
            self._send_json({"colors": [], "errors": errors})
            return
        topology = SpatialRoomTopology(config)
        if topology.total == 0:
            self._send_json({"colors": []})
            return
        t = np.linspace(0.0, 1.0, topology.total, dtype=np.float32)
        colors = np.stack(
            [
                (np.sin(t * np.pi * 2.0) * 0.5 + 0.5) * 255.0,
                (np.sin(t * np.pi * 2.0 + 2.0) * 0.5 + 0.5) * 255.0,
                (np.sin(t * np.pi * 2.0 + 4.0) * 0.5 + 0.5) * 255.0,
            ],
            axis=1,
        ).astype(int)
        self._send_json({"colors": colors.tolist()})

    def _send_preview_frame(self) -> None:
        data, revision = self.preview_manager.latest_frame_jpeg()
        if data is None:
            self.send_response(204)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Preview-Frame-Revision", str(revision))
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Preview-Frame-Revision", str(revision))
        self.end_headers()
        self.wfile.write(data)

    def _send_static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        target = (WEB_ROOT / path.lstrip("/")).resolve()
        if WEB_ROOT not in target.parents and target != WEB_ROOT:
            self.send_error(403)
            return
        if not target.exists() or not target.is_file():
            self.send_error(404)
            return
        content_type, _ = mimetypes.guess_type(str(target))
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: object, status: int = 200, compact: bool = False) -> None:
        if compact:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        else:
            data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Desktop 3D room visualizer/editor")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8787, help="Port to bind")
    parser.add_argument("--open", action="store_true", help="Open the editor in the default browser after the server starts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    SpatialEditorHandler.config_path = Path(args.config)
    server = ThreadingHTTPServer((args.host, args.port), SpatialEditorHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"3D room editor: {url}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
