from __future__ import annotations

import ast
import logging
import unittest
from pathlib import Path

from config import EngineConfig
from effects_engine import HeadlessEffectsEngine
from spatial_config import default_spatial_dict


ROOT = Path(__file__).resolve().parents[1]
CORE_MODULES = [
    "effects_engine.py",
    "runtime.py",
    "wave_engine.py",
    "spatial_renderer.py",
    "event_detector.py",
    "motion_detector.py",
    "frame_processor.py",
]
FORBIDDEN_IMPORTS = {
    "gui",
    "spatial_editor_server",
    "tkinter",
    "visualization",
    "web_editor",
    "webbrowser",
}


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


class EffectsEngineBoundaryTests(unittest.TestCase):
    def test_core_effects_modules_do_not_import_ui_or_preview_modules(self) -> None:
        offenders: list[str] = []
        for relative in CORE_MODULES:
            path = ROOT / relative
            forbidden = imported_roots(path) & FORBIDDEN_IMPORTS
            if forbidden:
                offenders.append(f"{relative}: {', '.join(sorted(forbidden))}")
        self.assertEqual(offenders, [])

    def test_headless_spatial_runtime_starts_without_wled(self) -> None:
        spatial = default_spatial_dict(8)
        spatial["enabled"] = True
        config = EngineConfig(
            total_leds=8,
            spatial=spatial,
            simulate_input=True,
            send_to_wled=False,
            target_fps=15,
            analysis_width=64,
            analysis_height=36,
        )
        engine = HeadlessEffectsEngine(config, logging.getLogger("test.effects_engine"))
        try:
            engine.start()
            snapshot = engine.step_once(timeout=0.05)
            self.assertTrue(snapshot.running)
            self.assertIsNotNone(snapshot.leds)
            self.assertEqual(snapshot.leds.shape, (8, 3))
            self.assertFalse(snapshot.wled_enabled)
        finally:
            engine.stop()


if __name__ == "__main__":
    unittest.main()
