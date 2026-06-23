from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from config import EngineConfig
from effects_engine import create_effect_renderer
from event_detector import EventDetector, LightEvent
from event_mixer import EventMixer
from frame_processor import FrameProcessor
from motion_detector import AnalysisRequirements, EdgeMotion, MotionAnalysis, MotionDetector
from performance import apply_runtime_profile, valid_runtime_profiles
from spatial_config import default_spatial_dict
from spatial_topology import SpatialRoomTopology
from wled_output import WLEDOutput


FRAME_BUDGET_MS_60FPS = 1000.0 / 60.0


@dataclass(frozen=True)
class BenchResult:
    name: str
    iterations: int
    avg_ms: float
    p95_ms: float
    max_ms: float
    frame_budget_percent: float
    fits_60fps_budget: bool


class BenchmarkSocket:
    def sendto(self, payload: bytes, address: tuple[str, int]) -> int:
        return len(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subsystem benchmark for the lighting engine")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--profile", choices=sorted(valid_runtime_profiles()), default="desktop_dev")
    parser.add_argument("--seconds", type=float, default=2.0, help="Seconds per benchmark case")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per case")
    parser.add_argument("--min-iterations", type=int, default=5, help="Minimum timed iterations per case")
    parser.add_argument("--led-count", type=int, default=240, help="LED count for renderer benchmarks")
    parser.add_argument("--mode", choices=["standard", "stress", "ceiling"], default="standard", help="Benchmark workload size")
    parser.add_argument("--render-mode", choices=["full_frame", "edge_effects", "hybrid_edge_full"], default=None, help="Frame preparation render mode")
    parser.add_argument("--worst-case", action="store_true", help="Alias for --mode ceiling")
    parser.add_argument("--worst-led-count", type=int, default=2000, help="LED count for stress and ceiling modes")
    parser.add_argument("--max-waves", type=int, default=None, help="Override max active waves for renderer stress cases")
    parser.add_argument("--wave-count", type=int, default=None, help="Override wave count for stress and ceiling modes")
    parser.add_argument("--worst-waves", type=int, default=None, help="Legacy alias for --wave-count")
    parser.add_argument("--spatial-renderer-backend", choices=["auto", "numpy", "numba", "gles"], default=None, help="Renderer backend for spatial benchmark cases")
    parser.add_argument("--renderer-matrix", action="store_true", help="Benchmark renderer-only cases for 240/1300/2000 LEDs and 20/50 waves")
    parser.add_argument("--backend-comparison", action="store_true", help="Run renderer matrix for numpy, numba, and gles and print a backend comparison table")
    parser.add_argument("--probe-backends", action="store_true", help="Probe spatial renderer backends and print exact initialization failures")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    return parser.parse_args()


def synthetic_frame(width: int, height: int, phase: float = 0.0) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(-1, 1)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32).reshape(1, -1)
    r = 40 + 120 * np.clip(np.sin((x + phase) * np.pi), 0.0, 1.0)
    g = 30 + 170 * np.clip(np.sin((x * 1.7 + y + phase) * np.pi), 0.0, 1.0)
    b = 25 + 210 * np.clip(np.sin((y * 1.3 + phase) * np.pi), 0.0, 1.0)
    frame = np.dstack([b + x * 20, g, r + y * 30])
    return np.clip(frame, 0, 255).astype(np.uint8)


def complex_frame(width: int, height: int, phase: float = 0.0) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(-1, 1)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32).reshape(1, -1)
    checker = ((np.floor(x * 32 + phase * 11) + np.floor(y * 18 + phase * 7)) % 2) * 65
    rings = np.sin(np.sqrt((x - 0.55) ** 2 + (y - 0.45) ** 2) * 80 - phase * 12) * 55
    streaks = np.sin((x * 23 + y * 9 + phase * 8) * np.pi) * 35
    r = 80 + checker + rings
    g = 120 + streaks + np.sin((x * 5 + phase) * np.pi) * 90
    b = 140 + checker * 0.6 + np.cos((y * 7 - phase) * np.pi) * 95
    bloom = np.exp(-(((x - 0.62) ** 2) / 0.04 + ((y - 0.35) ** 2) / 0.06)) * 150
    frame = np.dstack([b + bloom, g + bloom * 0.9, r + bloom * 0.45])
    return np.clip(frame, 0, 255).astype(np.uint8)


def spatial_config(total_leds: int) -> dict:
    spatial = default_spatial_dict(total_leds)
    spatial["enabled"] = True
    return spatial


def make_config(path: str, profile: str, led_count: int) -> EngineConfig:
    config = EngineConfig.load(path)
    config.total_leds = led_count
    config.simulate_input = True
    config.send_to_wled = False
    config.debug = False
    config.runtime_profile = profile
    config.spatial = spatial_config(led_count)
    apply_runtime_profile(config)
    return config


def renderer_backend_name(renderer: object) -> str:
    return str(getattr(renderer, "render_backend", "unknown"))


def renderer_backend_status(renderer: object) -> str:
    return str(getattr(renderer, "render_backend_status", "unknown"))


def time_case(name: str, seconds: float, warmup: int, min_iterations: int, fn: Callable[[], object]) -> BenchResult:
    for _ in range(max(0, warmup)):
        fn()
    samples: list[float] = []
    started = time.perf_counter()
    target_iterations = max(1, min_iterations)
    while time.perf_counter() - started < seconds or len(samples) < target_iterations:
        before = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - before) * 1000.0)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95)))
    avg = statistics.fmean(samples)
    return BenchResult(
        name=name,
        iterations=len(samples),
        avg_ms=round(avg, 4),
        p95_ms=round(ordered[p95_index], 4),
        max_ms=round(max(samples), 4),
        frame_budget_percent=round((avg / FRAME_BUDGET_MS_60FPS) * 100.0, 2),
        fits_60fps_budget=avg <= FRAME_BUDGET_MS_60FPS,
    )


def event_set() -> list[LightEvent]:
    return [
        LightEvent("top", 0.8, (235, 245, 255), "lightning", effect_id="lightning", pulse_count=5),
        LightEvent("top", 0.7, (255, 150, 60), "shockwave", effect_id="shockwave"),
        LightEvent("left", 0.6, (60, 220, 255), "energy_trail", effect_id="energy_trail"),
        LightEvent("right", 0.45, (255, 80, 30), "spill", effect_id="spill"),
        LightEvent("top", 0.35, (80, 190, 255), "color_bloom", effect_id="color_bloom"),
        LightEvent("top", 0.3, (255, 180, 80), "ember_particles", effect_id="ember_particles", secondary=True, pulse_count=6),
    ]


def worst_event_set() -> list[LightEvent]:
    kinds = [
        "lightning",
        "explosion",
        "shockwave",
        "impact_pulse",
        "negative_wave",
        "scene_wipe",
        "directional_sweep",
        "portal_vortex",
        "camera_pan",
        "energy_trail",
        "top_color_exit",
        "spill",
        "flame_shimmer",
        "underwater",
        "color_bloom",
        "ember_particles",
    ]
    edges = ["top", "left", "right", "bottom"]
    events: list[LightEvent] = []
    for index, kind in enumerate(kinds):
        events.append(
            LightEvent(
                edge=edges[index % len(edges)],
                intensity=0.95 - (index % 5) * 0.04,
                color=((80 + index * 41) % 256, (160 + index * 29) % 256, (240 + index * 17) % 256),
                kind=kind,
                effect_id=kind,
                direction_hint=-1 if index % 2 else 1,
                color_velocity=0.75,
                width=2.0,
                pulse_count=8 if kind in {"lightning", "ember_particles"} else 1,
                secondary=kind == "ember_particles",
            )
        )
    return events


def analysis_case() -> MotionAnalysis:
    edge = EdgeMotion(magnitude=0.6, confidence=0.5, toward_edge=0.5, parallel=0.2, coverage=0.4, color=(80, 210, 255), color_velocity=0.3)
    return MotionAnalysis(
        brightness=0.62,
        saturation=0.55,
        dominant_color=(120, 210, 245),
        color_velocity=0.22,
        luminous_color_coverage=0.55,
        luminous_color_growth=0.2,
        luminous_brightness_growth=0.15,
        luminous_bloom_color=(170, 245, 255),
        ambient_spill_boost_score=0.65,
        changed_fraction=0.35,
        rate_of_change=0.18,
        edge_activity={"top": edge, "left": edge, "right": edge, "bottom": edge},
        top_corner_activity={"left": edge, "right": edge},
        dominant_flow=(0.9, 0.2),
        flow_confidence=0.45,
    )


def renderer_step(renderer: object, dt: float = 1.0 / 60.0) -> np.ndarray:
    return renderer.step(dt)  # type: ignore[attr-defined]


def clone_waves(renderer: object) -> list[object]:
    return [replace(wave) for wave in getattr(renderer, "waves", [])]


def reset_waves(renderer: object, waves: list[object]) -> None:
    setattr(renderer, "waves", [replace(wave) for wave in waves])


def edge_only_mode(config: EngineConfig) -> bool:
    return config.render_mode == "edge_effects"


def build_cases(config: EngineConfig) -> list[tuple[str, Callable[[], object]]]:
    processor = FrameProcessor(config)
    raw = synthetic_frame(max(config.analysis_width * 2, 320), max(config.analysis_height * 2, 180))
    render_frames = processor.prepare_render_frames(raw)
    prepared = render_frames.analysis_frame
    tv_frame = render_frames.tv_frame
    prepared_next = processor.prepare_render_frames(synthetic_frame(raw.shape[1], raw.shape[0], 0.08)).analysis_frame

    color_motion = MotionDetector(config)
    color_motion.analyze(prepared, AnalysisRequirements(color_stats=True, frame_difference=False, edge_activity=False, top_corner_activity=False, optical_flow=False, retain_flow_debug=False))
    diff_motion = MotionDetector(config)
    diff_motion.analyze(prepared, AnalysisRequirements(color_stats=True, frame_difference=True, edge_activity=False, top_corner_activity=False, optical_flow=False, retain_flow_debug=False))
    edge_motion = MotionDetector(config)
    edge_motion.analyze(prepared, AnalysisRequirements(color_stats=True, frame_difference=True, edge_activity=True, top_corner_activity=True, optical_flow=False, retain_flow_debug=False))
    flow_motion: MotionDetector | None = None
    if not edge_only_mode(config):
        flow_config = EngineConfig.from_dict(config.to_dict())
        flow_config.motion_algorithm = "optical_flow"
        flow_motion = MotionDetector(flow_config)
        flow_motion.analyze(prepared, AnalysisRequirements(color_stats=True, frame_difference=True, edge_activity=True, top_corner_activity=True, optical_flow=True, retain_flow_debug=False))

    detector = EventDetector(config)
    mixer = EventMixer()
    analysis = analysis_case()
    events = event_set()

    spatial_topology = SpatialRoomTopology(config)
    spatial_idle = create_effect_renderer(config, spatial_topology)
    spatial_idle.set_tv_frame(tv_frame)
    spatial_active = create_effect_renderer(config, spatial_topology)
    spatial_active.set_tv_frame(tv_frame)
    spatial_active.add_events(events * 4)
    active_seed_waves = clone_waves(spatial_active)

    wled = WLEDOutput(config, logging.getLogger("benchmark"), spatial_topology)
    wled.udp_socket = BenchmarkSocket()  # type: ignore[assignment]
    leds = np.zeros((config.total_leds, 3), dtype=np.uint8)
    payload_leds = (np.arange(config.total_leds * 3, dtype=np.uint8).reshape(config.total_leds, 3))

    def wled_ddp_payload() -> object:
        return wled._send_ddp(config.wled_ip or "192.0.2.1", payload_leds)

    def render_active(renderer: object) -> np.ndarray:
        reset_waves(renderer, active_seed_waves)
        return renderer_step(renderer)

    hybrid60_tick = {"count": 0}
    hybrid90_tick = {"count": 0}

    def hybrid_expected_frame(flow_every: int, state: dict[str, int]) -> MotionAnalysis:
        state["count"] += 1
        if flow_motion is not None and state["count"] % flow_every == 1:
            return flow_motion.analyze(prepared_next, AnalysisRequirements(color_stats=True, frame_difference=True, edge_activity=True, top_corner_activity=True, optical_flow=True, retain_flow_debug=False))
        return diff_motion.analyze(prepared_next, AnalysisRequirements(color_stats=True, frame_difference=True, edge_activity=False, top_corner_activity=False, optical_flow=False, retain_flow_debug=False))

    cases: list[tuple[str, Callable[[], object]]] = [
        ("frame.prepare", lambda: processor.prepare(raw)),
        ("frame.prepare_render_frames", lambda: processor.prepare_render_frames(raw)),
        ("frame.top_strip_colors", lambda: processor.top_strip_colors(prepared, 120)),
        ("frame.color_stats", lambda: FrameProcessor.color_stats(prepared)),
        ("motion.color_only", lambda: color_motion.analyze(prepared_next, AnalysisRequirements(color_stats=True, frame_difference=False, edge_activity=False, top_corner_activity=False, optical_flow=False, retain_flow_debug=False))),
        ("motion.frame_difference", lambda: diff_motion.analyze(prepared_next, AnalysisRequirements(color_stats=True, frame_difference=True, edge_activity=False, top_corner_activity=False, optical_flow=False, retain_flow_debug=False))),
        ("motion.edge_activity", lambda: edge_motion.analyze(prepared_next, AnalysisRequirements(color_stats=True, frame_difference=True, edge_activity=True, top_corner_activity=True, optical_flow=False, retain_flow_debug=False))),
        ("event.detect", lambda: detector.detect(analysis)),
        ("event.mix", lambda: mixer.mix(events)),
        ("renderer.spatial_idle", lambda: renderer_step(spatial_idle)),
        (f"renderer.spatial_{len(active_seed_waves)}_waves", lambda: render_active(spatial_active)),
        ("wled.disabled_send", lambda: wled.send(leds)),
        ("wled.ddp_payload", wled_ddp_payload),
    ]
    if flow_motion is not None:
        cases.insert(7, ("motion.optical_flow", lambda: flow_motion.analyze(prepared_next, AnalysisRequirements(color_stats=True, frame_difference=True, edge_activity=True, top_corner_activity=True, optical_flow=True, retain_flow_debug=False))))
        cases.insert(8, ("motion.hybrid_expected_frame_60fps", lambda: hybrid_expected_frame(4, hybrid60_tick)))
        cases.insert(9, ("motion.hybrid_expected_frame_90fps", lambda: hybrid_expected_frame(6, hybrid90_tick)))
    return cases


def build_worst_cases(config: EngineConfig, wave_count: int, prefix: str) -> list[tuple[str, Callable[[], object]]]:
    config.max_active_waves = max(1, int(wave_count))
    processor = FrameProcessor(config)
    raw_a = complex_frame(max(config.analysis_width * 3, 640), max(config.analysis_height * 3, 360), 0.0)
    raw_b = complex_frame(raw_a.shape[1], raw_a.shape[0], 0.17)
    render_a = processor.prepare_render_frames(raw_a)
    render_b = processor.prepare_render_frames(raw_b)
    prepared_a = render_a.analysis_frame
    prepared_b = render_b.analysis_frame
    tv_b = render_b.tv_frame

    use_optical_flow = not edge_only_mode(config)
    requirements = AnalysisRequirements(
        color_stats=True,
        frame_difference=True,
        edge_activity=True,
        top_corner_activity=True,
        optical_flow=use_optical_flow,
        retain_flow_debug=use_optical_flow,
    )
    flow_config = EngineConfig.from_dict(config.to_dict())
    flow_config.motion_algorithm = "optical_flow" if use_optical_flow else "frame_difference"
    complex_motion = MotionDetector(flow_config)
    complex_motion.analyze(prepared_a, requirements)
    diff_requirements = AnalysisRequirements(
        color_stats=True,
        frame_difference=True,
        edge_activity=False,
        top_corner_activity=False,
        optical_flow=False,
        retain_flow_debug=False,
    )
    diff_motion = MotionDetector(config)
    diff_motion.analyze(prepared_a, diff_requirements)

    detector = EventDetector(config)
    mixer = EventMixer()
    heavy_analysis = analysis_case()
    heavy_analysis.changed_fraction = 0.72
    heavy_analysis.flow_confidence = 0.92
    heavy_analysis.dominant_flow = (1.8, 1.1)
    heavy_analysis.brightness = 0.82
    heavy_analysis.saturation = 0.66
    heavy_events = worst_event_set()

    spatial_topology = SpatialRoomTopology(config)
    spatial_max = create_effect_renderer(config, spatial_topology)
    spatial_max.set_tv_frame(tv_b)

    repeats = max(1, (config.max_active_waves // max(1, len(heavy_events))) + 1)
    seed_events = (heavy_events * repeats)[: config.max_active_waves]
    spatial_max.add_events(seed_events)
    max_seed_waves = clone_waves(spatial_max)
    pipeline_renderer = create_effect_renderer(config, spatial_topology)
    pipeline_renderer.set_tv_frame(tv_b)

    payload_leds = np.arange(config.total_leds * 3, dtype=np.uint8).reshape(config.total_leds, 3)
    wled = WLEDOutput(config, logging.getLogger("benchmark"), spatial_topology)
    wled.udp_socket = BenchmarkSocket()  # type: ignore[assignment]

    def refill(renderer: object) -> np.ndarray:
        reset_waves(renderer, max_seed_waves)
        return renderer_step(renderer)

    def prepare_analyze_detect_mix() -> object:
        frames = processor.prepare_render_frames(raw_b)
        analysis = complex_motion.analyze(frames.analysis_frame, requirements)
        candidates = detector.detect(analysis)
        return mixer.mix(candidates)

    def render_after_detect() -> object:
        reset_waves(pipeline_renderer, max_seed_waves)
        pipeline_renderer.set_tv_frame(tv_b)
        return pipeline_renderer.step(1.0 / 60.0)

    def full_frame_pipeline() -> object:
        frames = processor.prepare_render_frames(raw_b)
        analysis = complex_motion.analyze(frames.analysis_frame, requirements)
        candidates = detector.detect(analysis)
        mixed = mixer.mix(candidates)
        reset_waves(pipeline_renderer, max_seed_waves)
        pipeline_renderer.set_tv_frame(frames.tv_frame)
        pipeline_renderer.add_events(mixed)
        return pipeline_renderer.step(1.0 / 60.0)

    def wled_ddp_payload_large() -> object:
        return wled._send_ddp(config.wled_ip or "192.0.2.1", payload_leds)

    cases: list[tuple[str, Callable[[], object]]] = [
        (f"{prefix}.frame.prepare_complex", lambda: processor.prepare(raw_b)),
        (f"{prefix}.frame.prepare_render_frames_complex", lambda: processor.prepare_render_frames(raw_b)),
        (f"{prefix}.event.detect_heavy_analysis", lambda: detector.detect(heavy_analysis)),
        (f"{prefix}.event.mix_many_candidates", lambda: mixer.mix(heavy_events * 8)),
        (f"{prefix}.renderer.spatial_{config.max_active_waves}_waves_{config.total_leds}_leds", lambda: refill(spatial_max)),
        (f"{prefix}.wled.ddp_payload_{config.total_leds}_leds", wled_ddp_payload_large),
        (f"{prefix}.pipeline.prepare_analyze_detect_mix", prepare_analyze_detect_mix),
        (f"{prefix}.pipeline.render_after_detect", render_after_detect),
        (f"{prefix}.pipeline.total_with_render", full_frame_pipeline),
    ]
    motion_name = f"{prefix}.motion.full_optical_flow_debug" if use_optical_flow else f"{prefix}.motion.edge_frame_difference"
    cases.insert(2, (motion_name, lambda: complex_motion.analyze(prepared_b, requirements)))
    cases.insert(3, (f"{prefix}.motion.frame_difference_fill", lambda: diff_motion.analyze(prepared_b, diff_requirements)))
    return cases


def build_renderer_matrix_cases(config_path: str, config: EngineConfig, profile: str) -> list[tuple[str, Callable[[], object]]]:
    cases: list[tuple[str, Callable[[], object]]] = []
    for led_count in (240, 1300, 2000):
        for wave_count in (20, 50):
            case_config = make_config(config_path, profile, led_count)
            case_config.spatial_renderer_backend = config.spatial_renderer_backend
            case_config.max_active_waves = wave_count
            topology = SpatialRoomTopology(case_config)
            renderer = create_effect_renderer(case_config, topology)
            renderer.set_tv_frame(None)
            repeats = max(1, (wave_count // len(worst_event_set())) + 1)
            renderer.add_events((worst_event_set() * repeats)[:wave_count])
            seed = clone_waves(renderer)
            name = f"renderer.matrix.{renderer_backend_name(renderer)}_{wave_count}_waves_{led_count}_leds"

            def make_case(target: object = renderer, waves: list[object] = seed) -> Callable[[], object]:
                def run() -> object:
                    reset_waves(target, waves)
                    result = renderer_step(target)
                    return {
                        "leds": result,
                        "backend": renderer_backend_name(target),
                        "backend_status": renderer_backend_status(target),
                        "render_ms": float(getattr(target, "last_render_ms", 0.0)),
                        "readback_ms": float(getattr(target, "last_readback_ms", 0.0)),
                    }

                return run

            cases.append((name, make_case()))
    return cases


def parse_matrix_name(name: str) -> tuple[str, str]:
    prefix = "renderer.matrix."
    if not name.startswith(prefix):
        return "unknown", name
    parts = name[len(prefix) :].split("_", 1)
    if len(parts) != 2:
        return "unknown", name
    return parts[0], parts[1]


def speedup(base_ms: float, candidate_ms: float) -> float | None:
    if base_ms <= 0.0 or candidate_ms <= 0.0:
        return None
    return round(base_ms / candidate_ms, 2)


def run_backend_comparison(args: argparse.Namespace, base_config: EngineConfig) -> dict:
    requested_backends = ["numpy", "numba", "gles"]
    all_results: list[dict] = []
    by_case: dict[str, dict[str, BenchResult]] = {}
    actual_by_case: dict[str, dict[str, str]] = {}
    for requested in requested_backends:
        case_config = EngineConfig.from_dict(base_config.to_dict())
        case_config.spatial_renderer_backend = requested
        cases = build_renderer_matrix_cases(args.config, case_config, args.profile)
        results = [time_case(name, max(0.05, args.seconds), args.warmup, args.min_iterations, fn) for name, fn in cases]
        for result in results:
            actual, case_name = parse_matrix_name(result.name)
            by_case.setdefault(case_name, {})[requested] = result
            actual_by_case.setdefault(case_name, {})[requested] = actual
            row = dict(result.__dict__)
            row["requested_backend"] = requested
            row["actual_backend"] = actual
            row["case"] = case_name
            all_results.append(row)

    comparison_rows: list[dict] = []
    for case_name in sorted(by_case.keys(), key=lambda value: (int(value.split("_waves_")[1].split("_leds")[0]), int(value.split("_waves_")[0])) if "_waves_" in value else (0, 0)):
        results = by_case[case_name]
        numpy_result = results.get("numpy")
        row = {
            "case": case_name,
            "numpy_ms": numpy_result.avg_ms if numpy_result else None,
            "numba_ms": results["numba"].avg_ms if "numba" in results else None,
            "gles_ms": results["gles"].avg_ms if "gles" in results else None,
            "gles_actual_backend": actual_by_case.get(case_name, {}).get("gles", "unknown"),
            "numba_speedup": speedup(numpy_result.avg_ms, results["numba"].avg_ms) if numpy_result and "numba" in results else None,
            "gles_speedup": speedup(numpy_result.avg_ms, results["gles"].avg_ms) if numpy_result and "gles" in results else None,
        }
        comparison_rows.append(row)

    return {
        "requested_backends": requested_backends,
        "results": all_results,
        "comparison": comparison_rows,
    }


def print_backend_comparison(comparison_rows: list[dict]) -> None:
    print("Backend comparison (average ms)")
    print(f"{'Case':30} {'NumPy':>9} {'Numba':>9} {'GLES':>9} {'Numba x':>9} {'GLES x':>9} {'GLES actual':>12}")
    for row in comparison_rows:
        numpy_ms = f"{row['numpy_ms']:.4f}" if row["numpy_ms"] is not None else "-"
        numba_ms = f"{row['numba_ms']:.4f}" if row["numba_ms"] is not None else "-"
        gles_ms = f"{row['gles_ms']:.4f}" if row["gles_ms"] is not None else "-"
        numba_x = f"{row['numba_speedup']:.2f}x" if row["numba_speedup"] is not None else "-"
        gles_x = f"{row['gles_speedup']:.2f}x" if row["gles_speedup"] is not None else "-"
        print(f"{row['case']:30} {numpy_ms:>9} {numba_ms:>9} {gles_ms:>9} {numba_x:>9} {gles_x:>9} {row['gles_actual_backend']:>12}")


def probe_backends(config: EngineConfig) -> list[dict]:
    from spatial_renderer import GLESSpatialRenderer, NumbaSpatialRenderer, VectorizedSpatialRenderer

    probes: list[dict] = []
    topology = SpatialRoomTopology(config)
    for name, renderer_type in (
        ("numpy", VectorizedSpatialRenderer),
        ("numba", NumbaSpatialRenderer),
        ("gles", GLESSpatialRenderer),
    ):
        try:
            renderer = renderer_type(config, topology)
            probes.append({
                "backend": name,
                "ok": True,
                "actual_backend": getattr(renderer, "render_backend", "unknown"),
                "status": getattr(renderer, "render_backend_status", "unknown"),
                "error": "",
            })
        except Exception as exc:
            probes.append({
                "backend": name,
                "ok": False,
                "actual_backend": "",
                "status": "",
                "error": f"{type(exc).__name__}: {exc}",
            })
    return probes


def print_backend_probes(probes: list[dict]) -> None:
    print("Backend probe")
    print(f"{'Backend':10} {'OK':>5} {'Actual':>10}  Status / Error")
    for probe in probes:
        status = probe["status"] if probe["ok"] else probe["error"]
        print(f"{probe['backend']:10} {str(probe['ok']):>5} {probe['actual_backend']:>10}  {status}")


def main() -> int:
    args = parse_args()
    mode = "ceiling" if args.worst_case else args.mode
    led_count = args.worst_led_count if mode in {"stress", "ceiling"} else args.led_count
    config = make_config(args.config, args.profile, led_count)
    if args.render_mode is not None:
        config.render_mode = args.render_mode
    if args.spatial_renderer_backend is not None:
        config.spatial_renderer_backend = args.spatial_renderer_backend
    if args.max_waves is not None:
        config.max_active_waves = max(1, int(args.max_waves))
    wave_override = args.wave_count if args.wave_count is not None else args.worst_waves
    backend_comparison: dict | None = None
    backend_probes: list[dict] | None = None
    if args.probe_backends:
        backend_probes = probe_backends(config)
        cases = []
    elif args.backend_comparison:
        backend_comparison = run_backend_comparison(args, config)
        cases = []
    elif args.renderer_matrix:
        cases = build_renderer_matrix_cases(args.config, config, args.profile)
    elif mode == "stress":
        wave_count = int(wave_override if wave_override is not None else config.max_active_waves)
        cases = build_worst_cases(config, wave_count, "stress")
    elif mode == "ceiling":
        wave_count = int(wave_override if wave_override is not None else 2000)
        cases = build_worst_cases(config, wave_count, "ceiling")
    else:
        cases = build_cases(config)
    results = [] if backend_comparison is not None or backend_probes is not None else [time_case(name, max(0.05, args.seconds), args.warmup, args.min_iterations, fn) for name, fn in cases]
    output = {
        "profile": config.runtime_profile,
        "mode": mode,
        "render_mode": config.render_mode,
        "analysis_size": [config.analysis_width, config.analysis_height],
        "motion_analysis_fps": config.motion_analysis_fps,
        "optical_flow_fps": config.optical_flow_fps,
        "frame_difference_fill_enabled": config.frame_difference_fill_enabled,
        "frame_difference_fill_fps": config.frame_difference_fill_fps,
        "led_count": config.total_leds,
        "max_active_waves": config.max_active_waves,
        "spatial_renderer_backend": config.spatial_renderer_backend,
        "frame_budget": {
            "fps": 60,
            "ms": round(FRAME_BUDGET_MS_60FPS, 4),
        },
        "benchmarks": [result.__dict__ for result in results],
    }
    if backend_comparison is not None:
        output["backend_comparison"] = backend_comparison
    if backend_probes is not None:
        output["backend_probes"] = backend_probes
    if args.json:
        print(json.dumps(output, indent=2))
    elif backend_probes is not None:
        print(f"Profile: {config.runtime_profile}  Backend probe")
        print_backend_probes(backend_probes)
    elif backend_comparison is not None:
        print(f"Profile: {config.runtime_profile}  Backend comparison  60fps budget: {FRAME_BUDGET_MS_60FPS:.2f} ms")
        print_backend_comparison(backend_comparison["comparison"])
    else:
        print(f"Profile: {config.runtime_profile}  Mode: {mode}  Render: {config.render_mode}  Backend: {config.spatial_renderer_backend}  LEDs: {config.total_leds}  60fps budget: {FRAME_BUDGET_MS_60FPS:.2f} ms")
        for result in results:
            status = "OK" if result.fits_60fps_budget else "OVER"
            print(
                f"{result.name:28} avg={result.avg_ms:8.4f}ms p95={result.p95_ms:8.4f}ms "
                f"max={result.max_ms:8.4f}ms budget={result.frame_budget_percent:6.2f}% {status}"
            )
        print()
        print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
