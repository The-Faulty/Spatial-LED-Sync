from __future__ import annotations

import unittest
import unittest.mock
import time

import numpy as np

from config import DEFAULT_ENABLED_EFFECTS, TRIGGER_EFFECTS, EngineConfig
from effects_engine import HeadlessEffectsEngine
from event_mixer import EventMixer
from event_detector import EventDetector, LightEvent
from frame_processor import FrameProcessor
from hyperhdr_client import SIMULATION_EFFECT_PATTERNS, HyperHDRClient
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

    def bloom_frame(self, color_bgr: tuple[int, int, int], coverage: float, background: tuple[int, int, int] = (12, 10, 8)) -> np.ndarray:
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        frame[:] = background
        width = int(frame.shape[1] * coverage)
        if width > 0:
            frame[:, :width] = color_bgr
        return frame

    def test_old_config_gets_all_effect_toggles(self) -> None:
        config = EngineConfig.from_dict({"enabled_effects": {"spill": False}})
        self.assertEqual(set(config.enabled_effects), set(DEFAULT_ENABLED_EFFECTS))
        self.assertFalse(config.enabled_effects["spill"])
        self.assertTrue(config.enabled_effects["ambient_side_spill"])
        self.assertTrue(config.enabled_effects["shockwave"])
        self.assertEqual(set(config.effect_sensitivity), set(TRIGGER_EFFECTS))
        self.assertEqual(config.effect_sensitivity["shockwave"], 0.5)
        self.assertNotIn("front_ambient", config.effect_sensitivity)
        self.assertEqual(config.front_ambient_coverage, 1.0)
        self.assertEqual(config.tv_image_blur, 0)
        self.assertEqual(config.ambient_side_spill_base_intensity, 0.45)
        self.assertEqual(config.ambient_side_spill_boost_intensity, 1.0)
        self.assertEqual(config.wled_protocol, "ddp")
        self.assertEqual(config.wled_udp_port, 4048)
        self.assertEqual(config.motion_analysis_fps, 15)
        self.assertTrue(config.parallel_runtime)
        self.assertEqual(config.tv_frame_queue_size, 1)
        self.assertEqual(config.analysis_queue_size, 1)
        self.assertEqual(config.event_queue_size, 2)
        self.assertEqual(config.effect_render_skip_policy, "tv_first")
        self.assertEqual(config.spatial_priority_bands, 3)
        self.assertEqual(config.render_mode, "full_frame")
        self.assertEqual(config.edge_band_width, 192)
        self.assertEqual(config.edge_band_height, 108)
        self.assertEqual(config.edge_band_fraction, 0.12)
        self.assertEqual(config.hybrid_full_width, 64)
        self.assertEqual(config.hybrid_full_height, 36)

    def test_invalid_wled_realtime_settings_fail_validation(self) -> None:
        config = EngineConfig()
        config.wled_protocol = "http"
        config.wled_udp_port = 70000
        config.motion_analysis_fps = 0
        config.tv_frame_queue_size = 0
        config.analysis_queue_size = 0
        config.event_queue_size = 0
        config.effect_render_skip_policy = "drop_tv"
        config.spatial_priority_bands = 0
        config.spatial_far_budget_ratio = 1.4
        errors = config.validate()
        self.assertTrue(any("wled_protocol" in error for error in errors))
        self.assertTrue(any("wled_udp_port" in error for error in errors))
        self.assertTrue(any("motion_analysis_fps" in error for error in errors))
        self.assertTrue(any("tv_frame_queue_size" in error for error in errors))
        self.assertTrue(any("analysis_queue_size" in error for error in errors))
        self.assertTrue(any("event_queue_size" in error for error in errors))
        self.assertTrue(any("effect_render_skip_policy" in error for error in errors))
        self.assertTrue(any("spatial_priority_bands" in error for error in errors))
        self.assertTrue(any("spatial_far_budget_ratio" in error for error in errors))

    def test_invalid_render_mode_settings_fail_validation(self) -> None:
        config = EngineConfig()
        config.render_mode = "center_only"
        config.edge_band_width = 0
        config.edge_band_height = 0
        config.hybrid_full_width = 0
        config.hybrid_full_height = 0
        config.edge_band_fraction = 0.8
        errors = config.validate()
        self.assertTrue(any("render_mode" in error for error in errors))
        self.assertTrue(any("edge band dimensions" in error for error in errors))
        self.assertTrue(any("hybrid full-frame dimensions" in error for error in errors))
        self.assertTrue(any("edge_band_fraction" in error for error in errors))

    def test_frame_processor_full_frame_render_frames(self) -> None:
        config = EngineConfig(analysis_width=64, analysis_height=36)
        frames = FrameProcessor(config).prepare_render_frames(np.zeros((80, 120, 3), dtype=np.uint8))
        self.assertEqual(frames.tv_frame.shape, (36, 64, 3))
        self.assertEqual(frames.analysis_frame.shape, (36, 64, 3))
        self.assertEqual(frames.preview_frame.shape, (36, 64, 3))

    def test_frame_processor_edge_effects_preserves_edges_and_fills_center(self) -> None:
        config = EngineConfig(analysis_width=20, analysis_height=12, render_mode="edge_effects", edge_band_fraction=0.2)
        source = np.zeros((12, 20, 3), dtype=np.uint8)
        source[:] = (10, 20, 30)
        source[:2, :] = (80, 90, 100)
        frames = FrameProcessor(config).prepare_render_frames(source)
        self.assertEqual(frames.tv_frame.shape, (12, 20, 3))
        self.assertTrue(np.all(frames.tv_frame[:2, :] == (80, 90, 100)))
        self.assertTrue(np.any(frames.tv_frame[4:8, 6:14] != (10, 20, 30)))

    def test_frame_processor_hybrid_uses_split_frame_sizes(self) -> None:
        config = EngineConfig(
            render_mode="hybrid_edge_full",
            edge_band_width=40,
            edge_band_height=24,
            hybrid_full_width=16,
            hybrid_full_height=9,
        )
        frames = FrameProcessor(config).prepare_render_frames(np.zeros((90, 160, 3), dtype=np.uint8))
        self.assertEqual(frames.tv_frame.shape, (24, 40, 3))
        self.assertEqual(frames.analysis_frame.shape, (9, 16, 3))
        self.assertEqual(frames.preview_frame.shape, (9, 16, 3))

    def test_frame_processor_normalizes_grayscale_and_bgra(self) -> None:
        config = EngineConfig(analysis_width=16, analysis_height=9)
        processor = FrameProcessor(config)
        gray = np.zeros((20, 20), dtype=np.uint8)
        bgra = np.zeros((20, 20, 4), dtype=np.uint8)
        self.assertEqual(processor.prepare_render_frames(gray).analysis_frame.shape, (9, 16, 3))
        self.assertEqual(processor.prepare_render_frames(bgra).analysis_frame.shape, (9, 16, 3))

    def test_analysis_cadence_limits_motion_detection_rate(self) -> None:
        engine = HeadlessEffectsEngine(EngineConfig(target_fps=30, motion_analysis_fps=15), unittest.mock.Mock())
        self.assertTrue(engine._analysis_due())
        self.assertFalse(engine._analysis_due())
        engine.last_analysis_time -= 1.0
        self.assertTrue(engine._analysis_due())

    def test_parallel_runtime_starts_renders_and_stops_workers(self) -> None:
        config = EngineConfig(
            total_leds=8,
            spatial=spatial_config(8),
            simulate_input=True,
            send_to_wled=False,
            target_fps=20,
            motion_analysis_fps=10,
            parallel_runtime=True,
            analysis_width=32,
            analysis_height=18,
        )
        engine = HeadlessEffectsEngine(config, unittest.mock.Mock())
        engine.start()
        try:
            time.sleep(0.25)
            snapshot = engine.latest_snapshot()
            self.assertTrue(snapshot.running)
            self.assertIsNotNone(snapshot.leds)
            self.assertEqual(snapshot.leds.shape, (8, 3))
            self.assertGreaterEqual(snapshot.tv_frames, 1)
            self.assertGreaterEqual(snapshot.analysis_frames, 1)
        finally:
            analysis_thread = engine._analysis_thread
            render_thread = engine._render_thread
            engine.stop()
        self.assertFalse(analysis_thread and analysis_thread.is_alive())
        self.assertFalse(render_thread and render_thread.is_alive())

    def test_parallel_runtime_skips_analysis_when_no_effect_needs_it(self) -> None:
        config = EngineConfig(
            total_leds=8,
            spatial=spatial_config(8),
            simulate_input=True,
            send_to_wled=False,
            target_fps=20,
            motion_analysis_fps=10,
            parallel_runtime=True,
            enabled_effects={key: False for key in DEFAULT_ENABLED_EFFECTS},
            analysis_width=32,
            analysis_height=18,
        )
        engine = HeadlessEffectsEngine(config, unittest.mock.Mock())
        engine.start()
        try:
            time.sleep(0.2)
            snapshot = engine.latest_snapshot()
            self.assertIsNotNone(snapshot.leds)
            self.assertEqual(snapshot.analysis_frames, 0)
            self.assertGreaterEqual(snapshot.tv_frames, 1)
        finally:
            engine.stop()

    def test_invalid_effect_sensitivity_fails_validation(self) -> None:
        config = EngineConfig()
        config.effect_sensitivity["shockwave"] = 1.2
        self.assertTrue(any("effect_sensitivity.shockwave" in error for error in config.validate()))

    def test_invalid_ambient_side_spill_settings_fail_validation(self) -> None:
        config = EngineConfig()
        config.ambient_side_spill_base_intensity = 1.2
        config.ambient_side_spill_boost_intensity = 2.4
        errors = config.validate()
        self.assertTrue(any("ambient_side_spill_base_intensity" in error for error in errors))
        self.assertTrue(any("ambient_side_spill_boost_intensity" in error for error in errors))

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
        second = [event.kind for event in detector.detect(saturated)]
        later = [event.kind for event in detector.detect(saturated)]
        self.assertNotIn("color_bloom", first)
        self.assertIn("color_bloom", second)
        self.assertNotIn("color_bloom", later)

    def test_shockwave_sensitivity_suppresses_borderline_scene_change(self) -> None:
        config = EngineConfig()
        config.effect_sensitivity["shockwave"] = 1.0
        detector = EventDetector(config)
        borderline = MotionAnalysis(
            brightness=0.55,
            saturation=0.25,
            changed_fraction=0.42,
            rate_of_change=0.12,
            color_velocity=0.62,
            dominant_color=(220, 220, 230),
            edge_activity={"top": EdgeMotion(magnitude=0.1, confidence=0.1, toward_edge=0.1, coverage=0.08)},
        )
        kinds = {event.kind for event in detector.detect(borderline)}
        self.assertNotIn("shockwave", kinds)

    def test_low_sensitivity_shockwave_catches_strong_impact(self) -> None:
        config = EngineConfig()
        config.effect_sensitivity["shockwave"] = 0.0
        detector = EventDetector(config)
        impact = MotionAnalysis(
            brightness=0.78,
            saturation=0.62,
            changed_fraction=0.46,
            rate_of_change=0.24,
            dominant_color=(255, 95, 40),
            edge_activity={
                "top": EdgeMotion(magnitude=0.5, confidence=0.55, toward_edge=0.5, coverage=0.35),
                "right": EdgeMotion(magnitude=0.35, confidence=0.45, toward_edge=0.4, coverage=0.25),
            },
        )
        kinds = {event.kind for event in detector.detect(impact)}
        self.assertIn("shockwave", kinds)

    def test_high_energy_effects_are_more_separated(self) -> None:
        detector = EventDetector(EngineConfig())
        white_flash = MotionAnalysis(
            brightness=0.82,
            saturation=0.18,
            changed_fraction=0.26,
            rate_of_change=0.12,
            dominant_color=(235, 240, 245),
        )
        kinds = {event.kind for event in detector.detect(white_flash)}
        self.assertIn("lightning", kinds)
        self.assertNotIn("explosion", kinds)
        self.assertNotIn("shockwave", kinds)

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

    def test_simulation_patterns_cover_all_effect_toggles(self) -> None:
        pattern_names = {name for name, _label in SIMULATION_EFFECT_PATTERNS}
        self.assertTrue(set(DEFAULT_ENABLED_EFFECTS).issubset(pattern_names))
        self.assertIn("ambient_side_spill_boost", pattern_names)

    def test_each_simulation_pattern_produces_distinct_nonblank_frame(self) -> None:
        client = HyperHDRClient(EngineConfig(), unittest.mock.Mock())
        idle = client._simulation_frame(160, 90, None, 0.5)
        for effect, _label in SIMULATION_EFFECT_PATTERNS:
            frame = client._simulation_frame(160, 90, effect, 0.45)
            self.assertEqual(frame.shape, (90, 160, 3), effect)
            self.assertGreater(int(frame.max()), 0, effect)
            self.assertGreater(float(np.mean(np.abs(frame.astype(np.int16) - idle.astype(np.int16)))), 1.0, effect)

    def test_simulation_patterns_are_mirrored_for_preview_orientation(self) -> None:
        client = HyperHDRClient(EngineConfig(), unittest.mock.Mock())
        frame = client._simulation_frame(160, 90, "ambient_side_spill", 0.5)
        left = frame[4, 4]
        right = frame[4, -5]
        self.assertGreater(int(left[0]), int(right[0]))
        self.assertGreater(int(right[2]), int(left[2]))

    def test_ambient_side_spill_boost_pattern_brightens_uniform_scene(self) -> None:
        client = HyperHDRClient(EngineConfig(), unittest.mock.Mock())
        early = client._simulation_frame(160, 90, "ambient_side_spill_boost", 0.1)
        late = client._simulation_frame(160, 90, "ambient_side_spill_boost", 0.8)
        self.assertGreater(float(np.mean(late)), float(np.mean(early)) * 2.0)
        self.assertLess(float(np.mean(np.std(late.astype(np.float32), axis=1))), 50.0)

    def test_luminous_cyan_bloom_growth_boosts_ambient_spill(self) -> None:
        detector = MotionDetector(EngineConfig())
        req = AnalysisRequirements(color_stats=True, frame_difference=True, edge_activity=False, top_corner_activity=False, optical_flow=False, retain_flow_debug=False)
        detector.analyze(self.bloom_frame((45, 105, 135), 0.18), req)
        analysis = detector.analyze(self.bloom_frame((245, 245, 175), 0.72, background=(40, 70, 85)), req)
        self.assertGreater(analysis.luminous_color_coverage, 0.55)
        self.assertGreater(analysis.luminous_color_growth, 0.25)
        self.assertGreater(analysis.luminous_brightness_growth, 0.10)
        self.assertGreater(analysis.ambient_spill_boost_score, 0.35)
        r, g, b = analysis.luminous_bloom_color
        self.assertGreaterEqual(b, r)
        self.assertGreaterEqual(g, r)

    def test_white_flash_does_not_strongly_boost_ambient_spill(self) -> None:
        detector = MotionDetector(EngineConfig())
        req = AnalysisRequirements(color_stats=True, frame_difference=True, edge_activity=False, top_corner_activity=False, optical_flow=False, retain_flow_debug=False)
        detector.analyze(self.bloom_frame((30, 30, 30), 0.2), req)
        analysis = detector.analyze(self.bloom_frame((245, 245, 245), 0.75), req)
        self.assertLess(analysis.luminous_color_coverage, 0.05)
        self.assertLess(analysis.ambient_spill_boost_score, 0.08)

    def test_stable_cyan_bloom_is_milder_than_growing_bloom(self) -> None:
        config = EngineConfig()
        req = AnalysisRequirements(color_stats=True, frame_difference=True, edge_activity=False, top_corner_activity=False, optical_flow=False, retain_flow_debug=False)
        stable = MotionDetector(config)
        stable_frame = self.bloom_frame((245, 245, 175), 0.72, background=(40, 70, 85))
        first = stable.analyze(stable_frame, req)
        second = stable.analyze(stable_frame, req)

        growing = MotionDetector(config)
        growing.analyze(self.bloom_frame((45, 105, 135), 0.18), req)
        grown = growing.analyze(stable_frame, req)
        self.assertLess(first.ambient_spill_boost_score, grown.ambient_spill_boost_score)
        self.assertLess(second.ambient_spill_boost_score, grown.ambient_spill_boost_score)

    def test_multicolor_bright_scene_does_not_match_luminous_bloom(self) -> None:
        detector = MotionDetector(EngineConfig())
        req = AnalysisRequirements(color_stats=True, frame_difference=True, edge_activity=False, top_corner_activity=False, optical_flow=False, retain_flow_debug=False)
        start = self.bloom_frame((30, 30, 30), 0.2)
        detector.analyze(start, req)
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        frame[:, :40] = (240, 240, 40)
        frame[:, 40:80] = (40, 240, 240)
        frame[:, 80:120] = (240, 40, 240)
        frame[:, 120:] = (60, 180, 220)
        analysis = detector.analyze(frame, req)
        self.assertLess(analysis.ambient_spill_boost_score, 0.18)

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

    def test_front_ambient_coverage_controls_legacy_front_span(self) -> None:
        config = EngineConfig(total_leds=24, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient", front_ambient_coverage=1.0)
        engine = WaveEngine(config, RoomTopology(config))
        self.assertEqual(len(engine._front_ambient_leds()), len(engine.topology.range_inclusive(config.front_wall.start, config.front_wall.end)))

        config = EngineConfig(total_leds=24, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient", front_ambient_coverage=0.0)
        engine = WaveEngine(config, RoomTopology(config))
        self.assertEqual(engine._front_ambient_leds(), [])

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

    def test_high_energy_spatial_effects_fill_room(self) -> None:
        config = EngineConfig(total_leds=12, spatial=spatial_config(), brightness=1.0, gamma=1.0, color_smoothing=0.0)
        topology = SpatialRoomTopology(config)
        for kind in ("shockwave", "impact_pulse"):
            renderer = VectorizedSpatialRenderer(config, topology)
            renderer.add_events([self.event(kind, 0.95, width=1.0)])
            leds = renderer.step(0.05)
            self.assertGreater(int(np.min(np.max(leds, axis=1))), 5, kind)

    def test_lightning_room_fill_depends_on_intensity(self) -> None:
        config = EngineConfig(total_leds=12, spatial=spatial_config(), brightness=1.0, gamma=1.0, color_smoothing=0.0)
        topology = SpatialRoomTopology(config)
        high = VectorizedSpatialRenderer(config, topology)
        high.add_events([self.event("lightning", 0.9, pulse_count=1)])
        high_leds = high.step(0.01)
        low = VectorizedSpatialRenderer(config, topology)
        low.add_events([self.event("lightning", 0.55, pulse_count=1)])
        low_leds = low.step(0.01)
        self.assertGreater(int(np.min(np.max(high_leds, axis=1))), 20)
        self.assertLess(int(np.min(np.max(low_leds, axis=1))), 20)

    def test_high_energy_legacy_effects_fill_room(self) -> None:
        config = EngineConfig(total_leds=24, brightness=1.0, gamma=1.0, color_smoothing=0.0)
        for kind in ("shockwave", "impact_pulse"):
            engine = WaveEngine(config, RoomTopology(config))
            engine.add_events([self.event(kind, 0.95, width=1.0)])
            leds = engine.step(0.05)
            self.assertGreater(int(np.min(np.max(leds, axis=1))), 5, kind)

    def test_legacy_wave_render_keeps_front_ambient_reserved_leds_dark(self) -> None:
        config = EngineConfig(total_leds=240, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient")
        engine = WaveEngine(config, RoomTopology(config))
        reserved = engine._front_ambient_leds()
        engine.add_events([self.event("shockwave", 0.95, width=1.0)])
        leds = engine.step(0.05)
        self.assertTrue(reserved)
        self.assertEqual(int(leds[reserved].max()), 0)
        self.assertGreater(int(leds.max()), 0)

    def test_legacy_wave_reserved_mask_cache_matches_led_list(self) -> None:
        config = EngineConfig(total_leds=240, lighting_mode="front_ambient", front_ambient_coverage=0.5)
        engine = WaveEngine(config, RoomTopology(config))
        mask = engine._front_ambient_reserved_mask()
        self.assertEqual(set(np.nonzero(mask)[0].tolist()), set(engine._front_ambient_leds()))

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
            skip_count = 3

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
        self.assertEqual(snapshot.wled_skip_count, 3)


if __name__ == "__main__":
    unittest.main()
