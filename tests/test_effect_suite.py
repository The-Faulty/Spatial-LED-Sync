from __future__ import annotations

import unittest
import unittest.mock

import numpy as np

from config import DEFAULT_ENABLED_EFFECTS, EngineConfig
from effects_engine import HeadlessEffectsEngine
from event_mixer import EventMixer
from event_detector import EventDetector, LightEvent
from motion_detector import AnalysisRequirements, EdgeMotion, MotionAnalysis, MotionDetector
from spatial_config import default_spatial_dict
from spatial_renderer import VectorizedSpatialRenderer
from spatial_topology import SpatialRoomTopology
from topology import RoomTopology
from wave_engine import LightWave, WaveEngine


def spatial_config(total: int = 12) -> dict:
    spatial = default_spatial_dict(total)
    spatial["enabled"] = True
    return spatial


class EffectSuiteTests(unittest.TestCase):
    def event(self, kind: str, intensity: float = 0.7, edge: str = "top", **kwargs) -> LightEvent:
        return LightEvent(edge=edge, intensity=intensity, color=(255, 120, 40), kind=kind, effect_id=kind, **kwargs)

    def test_old_config_gets_all_effect_toggles(self) -> None:
        config = EngineConfig.from_dict({"enabled_effects": {"spill": False}})
        self.assertEqual(set(config.enabled_effects), set(DEFAULT_ENABLED_EFFECTS))
        self.assertFalse(config.enabled_effects["spill"])
        self.assertTrue(config.enabled_effects["ambient_side_spill"])
        self.assertTrue(config.enabled_effects["shockwave"])

    def test_analysis_requirements_skip_unused_pipeline(self) -> None:
        ambient = EngineConfig(enabled_effects={key: key == "front_ambient" for key in DEFAULT_ENABLED_EFFECTS})
        ambient_engine = HeadlessEffectsEngine(ambient, unittest.mock.Mock())
        self.assertEqual(
            ambient_engine.analysis_requirements(),
            AnalysisRequirements(color_stats=True, frame_difference=False, edge_activity=False, top_corner_activity=False, optical_flow=False, retain_flow_debug=False),
        )

        flash = EngineConfig(enabled_effects={key: key == "flash" for key in DEFAULT_ENABLED_EFFECTS})
        flash_engine = HeadlessEffectsEngine(flash, unittest.mock.Mock())
        self.assertEqual(
            flash_engine.analysis_requirements(),
            AnalysisRequirements(color_stats=True, frame_difference=True, edge_activity=False, top_corner_activity=False, optical_flow=False, retain_flow_debug=False),
        )

        disabled = EngineConfig(enabled_effects={key: False for key in DEFAULT_ENABLED_EFFECTS})
        disabled_engine = HeadlessEffectsEngine(disabled, unittest.mock.Mock())
        self.assertFalse(disabled_engine.analysis_requirements().any_analysis)

    def test_motion_detector_honors_color_only_requirements(self) -> None:
        detector = MotionDetector(EngineConfig())
        frame = np.full((20, 20, 3), (20, 80, 200), dtype=np.uint8)
        detector.analyze(frame, AnalysisRequirements(color_stats=True, frame_difference=False, edge_activity=False, top_corner_activity=False, optical_flow=False, retain_flow_debug=False))
        analysis = detector.analyze(frame, AnalysisRequirements(color_stats=True, frame_difference=False, edge_activity=False, top_corner_activity=False, optical_flow=False, retain_flow_debug=False))
        self.assertGreater(analysis.brightness, 0.0)
        self.assertEqual(analysis.edge_activity, {})
        self.assertIsNone(analysis.flow_vectors)

    def test_disabled_new_effect_does_not_emit(self) -> None:
        config = EngineConfig()
        config.enabled_effects["shockwave"] = False
        detector = EventDetector(config)
        analysis = MotionAnalysis(brightness=0.7, saturation=0.2, changed_fraction=0.8, rate_of_change=0.4, dominant_color=(240, 240, 240))
        kinds = {event.kind for event in detector.detect(analysis)}
        self.assertNotIn("shockwave", kinds)

    def test_detector_emits_representative_new_effects(self) -> None:
        detector = EventDetector(EngineConfig())
        energetic = MotionAnalysis(
            brightness=0.75,
            saturation=0.65,
            changed_fraction=0.45,
            rate_of_change=0.25,
            dominant_color=(255, 90, 30),
            color_velocity=0.3,
            flow_confidence=0.6,
            dominant_flow=(1.4, 0.5),
            edge_activity={"right": EdgeMotion(magnitude=0.5, confidence=0.5, toward_edge=0.6, coverage=0.3, color=(255, 90, 30), color_velocity=0.3, color_motion_velocity=0.5)},
            top_corner_activity={"right": EdgeMotion(magnitude=0.4, confidence=0.5, toward_edge=0.5, coverage=0.25, color=(255, 80, 30), color_velocity=0.4, color_motion_velocity=0.5)},
        )
        kinds = {event.kind for event in detector.detect(energetic)}
        self.assertTrue({"energy_trail", "shockwave", "impact_pulse", "directional_sweep", "portal_vortex"} & kinds)

    def test_mood_effects_do_not_emit_every_frame(self) -> None:
        detector = EventDetector(EngineConfig())
        saturated = MotionAnalysis(
            brightness=0.5,
            saturation=0.8,
            changed_fraction=0.04,
            dominant_color=(220, 40, 180),
        )
        first = [event.kind for event in detector.detect(saturated)]
        later = [event.kind for event in detector.detect(saturated)]
        self.assertIn("color_bloom", first)
        self.assertNotIn("color_bloom", later)

    def test_spatial_renderer_handles_new_event_kinds(self) -> None:
        config = EngineConfig(total_leds=12, spatial=spatial_config(), brightness=1.0, gamma=1.0, color_smoothing=0.0)
        topology = SpatialRoomTopology(config)
        kinds = [
            "energy_trail",
            "shockwave",
            "directional_sweep",
            "lightning",
            "impact_pulse",
            "color_bloom",
            "flame_shimmer",
            "underwater",
            "portal_vortex",
            "scene_wipe",
            "ember_particles",
        ]
        for kind in kinds:
            renderer = VectorizedSpatialRenderer(config, topology)
            renderer.add_events([LightEvent(edge="top", intensity=0.9, color=(255, 120, 40), kind=kind, direction_hint=1, width=1.0, pulse_count=4)])
            leds = renderer.step(0.2)
            self.assertEqual(leds.shape, (12, 3), kind)
            self.assertGreater(int(leds.max()), 0, kind)

    def test_mixer_keeps_lightning_primary_and_particles(self) -> None:
        mixer = EventMixer()
        events = [
            self.event("shockwave", 0.9),
            self.event("flash", 1.0),
            self.event("impact_pulse", 0.85),
            self.event("lightning", 0.7),
            self.event("ember_particles", 0.35, secondary=True),
        ]
        mixed = mixer.mix(events)
        self.assertEqual([event.kind for event in mixed], ["lightning", "ember_particles"])
        self.assertTrue(mixed[0].primary)

    def test_mixer_explosion_beats_shockwave_and_keeps_particles(self) -> None:
        mixed = EventMixer().mix([
            self.event("shockwave", 1.0),
            self.event("explosion", 0.6),
            self.event("ember_particles", 0.4, secondary=True),
        ])
        self.assertEqual([event.kind for event in mixed], ["explosion", "ember_particles"])
        self.assertTrue(mixed[0].primary)

    def test_mixer_shockwave_suppresses_edge_accents(self) -> None:
        mixed = EventMixer().mix([
            self.event("spill", 0.9, edge="left"),
            self.event("energy_trail", 0.8, edge="left"),
            self.event("shockwave", 0.6),
        ])
        self.assertEqual([event.kind for event in mixed], ["shockwave"])
        self.assertTrue(mixed[0].primary)

    def test_mixer_motion_primary_keeps_one_compatible_accent(self) -> None:
        mixed = EventMixer().mix([
            self.event("camera_pan", 0.8, direction_hint=-1),
            self.event("directional_sweep", 0.6, direction_hint=1),
            self.event("energy_trail", 0.7, edge="right", direction_hint=1),
            self.event("spill", 0.9, edge="left", direction_hint=-1),
        ])
        self.assertEqual([event.kind for event in mixed], ["directional_sweep", "energy_trail"])
        self.assertTrue(mixed[0].primary)

    def test_mixer_keeps_one_mood_effect(self) -> None:
        mixed = EventMixer().mix([
            self.event("color_bloom", 0.7),
            self.event("underwater", 0.68),
            self.event("flame_shimmer", 0.64),
        ])
        self.assertEqual([event.kind for event in mixed], ["flame_shimmer"])

    def test_wave_engine_ducks_lower_priority_waves(self) -> None:
        config = EngineConfig(total_leds=12)
        engine = WaveEngine(config, RoomTopology(config))
        engine.waves = [
            LightWave((255, 0, 0), 1.0, 0.0, 1.0, 0.95, 1.0, 0.0, 1, 6.0, "spill"),
            LightWave((255, 255, 255), 0.8, 0.0, 1.0, 0.95, 1.0, 0.0, 1, 6.0, "lightning"),
            LightWave((255, 180, 80), 0.5, 0.0, 1.0, 0.95, 1.0, 0.0, 1, 6.0, "ember_particles"),
        ]
        engine.duck_lower_priority_waves("lightning")
        self.assertAlmostEqual(engine.waves[0].intensity, 0.35)
        self.assertAlmostEqual(engine.waves[1].intensity, 0.8)
        self.assertAlmostEqual(engine.waves[2].intensity, 0.5)
        engine.duck_lower_priority_waves("shockwave")
        self.assertAlmostEqual(engine.waves[2].intensity, 0.175)

    def test_spatial_renderer_ducks_lower_priority_waves(self) -> None:
        config = EngineConfig(total_leds=12, spatial=spatial_config())
        topology = SpatialRoomTopology(config)
        renderer = VectorizedSpatialRenderer(config, topology)
        renderer.add_events([self.event("spill", 0.8), self.event("shockwave", 0.9)])
        before = [wave.intensity for wave in renderer.waves]
        renderer.duck_lower_priority_waves("shockwave")
        after_by_kind = {wave.kind: wave.intensity for wave in renderer.waves}
        self.assertLess(after_by_kind["spill"], before[0])
        self.assertGreaterEqual(after_by_kind["shockwave"], 0.89)

    def test_headless_engine_snapshot_reports_mixed_events(self) -> None:
        class Queue:
            def qsize(self) -> int:
                return 0

        class Client:
            connected = True
            frames = Queue()
            calls = 0

            def get_frame(self, timeout: float = 0.0):
                self.calls += 1
                return np.zeros((8, 8, 3), dtype=np.uint8) if self.calls == 1 else None

        class Processor:
            def prepare(self, frame):
                return frame

        class Motion:
            def analyze(self, frame, requirements):
                return MotionAnalysis(brightness=0.8, changed_fraction=0.5)

        class Detector:
            def detect(self, analysis):
                return [self_event("spill", 0.9), self_event("lightning", 0.7)]

        class Renderer:
            accepts_front_ambient_strip_sample = False
            waves: list[object] = []

            def __init__(self):
                self.events: list[LightEvent] = []
                self.ducked = ""

            def set_tv_frame(self, frame):
                return None

            def duck_lower_priority_waves(self, primary_kind: str, factor: float = 0.35):
                self.ducked = primary_kind

            def add_events(self, events):
                self.events.extend(events)

            def step(self, dt):
                return np.zeros((4, 3), dtype=np.uint8)

        class WLED:
            enabled = False

            def send(self, leds):
                return False

        def self_event(kind: str, intensity: float) -> LightEvent:
            return self.event(kind, intensity)

        engine = HeadlessEffectsEngine(EngineConfig(total_leds=4, overload_policy="fixed_quality"), unittest.mock.Mock())
        renderer = Renderer()
        engine.running = True
        engine.client = Client()
        engine.processor = Processor()
        engine.motion = Motion()
        engine.event_detector = Detector()
        engine.wave_engine = renderer
        engine.wled = WLED()
        snapshot = engine.step_once(timeout=0.0)
        self.assertEqual([event.kind for event in snapshot.events], ["lightning"])
        self.assertTrue(snapshot.events[0].primary)
        self.assertEqual(snapshot.candidate_events, 2)
        self.assertEqual(renderer.ducked, "lightning")


if __name__ == "__main__":
    unittest.main()
