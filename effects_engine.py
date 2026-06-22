from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Protocol

import numpy as np

from config import EngineConfig
from event_detector import EventDetector, LightEvent
from event_mixer import EventMixer, is_high_energy_primary
from frame_processor import FrameProcessor, RenderFrames
from hyperhdr_client import HyperHDRClient
from motion_detector import AnalysisRequirements, MotionAnalysis, MotionDetector
from performance import apply_runtime_profile
from spatial_config import parse_spatial_config
from spatial_renderer import SpatialRenderer, VectorizedSpatialRenderer
from spatial_topology import SpatialRoomTopology
from topology import RoomTopology
from wave_engine import WaveEngine
from wled_output import WLEDOutput


@dataclass
class RuntimeSnapshot:
    running: bool = False
    frame: np.ndarray | None = None
    frame_history: list[np.ndarray] = field(default_factory=list)
    analysis: MotionAnalysis | None = None
    leds: np.ndarray | None = None
    events: list[LightEvent] = field(default_factory=list)
    active_waves: int = 0
    fps: float = 0.0
    buffered_frames: int = 0
    hyperhdr_connected: bool = False
    wled_enabled: bool = False
    last_wled_send: bool = False
    last_error: str = ""
    missed_deadlines: int = 0
    analysis_frames: int = 0
    skipped_analysis_frames: int = 0
    wled_skip_count: int = 0
    step_ms: float = 0.0
    candidate_events: int = 0
    tv_frames: int = 0
    analysis_frame_drops: int = 0
    effect_frame_skips: int = 0
    spatial_priority_band_skips: int = 0


@dataclass
class AnalysisResult:
    frame: np.ndarray
    analysis_frame: np.ndarray | None = None
    analysis: MotionAnalysis | None = None
    events: list[LightEvent] = field(default_factory=list)
    candidate_events: int = 0
    front_ambient_strip_colors: np.ndarray | None = None
    front_ambient_intensity: float = 0.0
    ambient_spill_boost_score: float = 0.0
    ambient_spill_boost_color: tuple[int, int, int] = (0, 0, 0)


class EffectsEngine(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def step_once(self, timeout: float = 0.0) -> RuntimeSnapshot:
        ...

    def latest_snapshot(self) -> RuntimeSnapshot:
        ...

    def trigger_simulation_effect(self, effect: str) -> bool:
        ...


def create_effect_renderer(
    config: EngineConfig,
    topology: RoomTopology | SpatialRoomTopology,
) -> SpatialRenderer | WaveEngine:
    if isinstance(topology, SpatialRoomTopology):
        return VectorizedSpatialRenderer(config, topology)
    return WaveEngine(config, topology)


class HeadlessEffectsEngine:
    def __init__(self, config: EngineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.running = False
        self.topology: RoomTopology | SpatialRoomTopology | None = None
        self.processor: FrameProcessor | None = None
        self.motion: MotionDetector | None = None
        self.event_detector: EventDetector | None = None
        self.event_mixer = EventMixer()
        self.wave_engine: SpatialRenderer | WaveEngine | None = None
        self.wled: WLEDOutput | None = None
        self.client: HyperHDRClient | None = None
        self.last_step = time.monotonic()
        self.stat_time = self.last_step
        self.stat_frames = 0
        self.analysis_frames = 0
        self.skipped_analysis_frames = 0
        self.tv_frames = 0
        self.analysis_frame_drops = 0
        self.effect_frame_skips = 0
        self.spatial_priority_band_skips = 0
        self.missed_deadlines = 0
        self.last_step_duration = 0.0
        self.analysis_stride = 1
        self._analysis_tick = 0
        self.last_analysis_time = 0.0
        self.frame_history: deque[np.ndarray] = deque(maxlen=12)
        self.snapshot = RuntimeSnapshot()
        self._parallel_enabled = False
        self._stop_event = threading.Event()
        self._analysis_thread: threading.Thread | None = None
        self._render_thread: threading.Thread | None = None
        self._snapshot_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_tv_frame: np.ndarray | None = None
        self._latest_analysis_frame: np.ndarray | None = None
        self._latest_frame_revision = 0
        self._analysis_results: Queue[AnalysisResult] | None = None

    def analysis_requirements(self) -> AnalysisRequirements:
        enabled = {name for name, active in self.config.enabled_effects.items() if active}
        if not enabled:
            return AnalysisRequirements(False, False, False, False, False, False)

        color_only = {"front_ambient", "color_bloom", "underwater"}
        frame_only = {"flash", "explosion", "shockwave", "lightning", "impact_pulse", "negative_wave"}
        edge_effects = {"spill", "top_color_exit", "flame_shimmer"}
        full_flow = {"camera_pan", "portal_vortex", "directional_sweep", "scene_wipe", "energy_trail"}

        needs_full_flow = bool(enabled & full_flow)
        needs_edges = bool(enabled & edge_effects) or needs_full_flow
        needs_corners = "top_color_exit" in enabled or needs_full_flow
        needs_difference = bool(enabled & frame_only) or needs_edges or needs_full_flow
        needs_color = bool(enabled & (color_only | frame_only | edge_effects | full_flow | {"ember_particles"}))
        return AnalysisRequirements(
            color_stats=needs_color,
            frame_difference=needs_difference,
            edge_activity=needs_edges,
            top_corner_activity=needs_corners,
            optical_flow=needs_full_flow or (needs_edges and self.config.motion_algorithm != "frame_difference"),
            retain_flow_debug=needs_full_flow or needs_edges,
        )

    def start(self) -> None:
        if self.running:
            return
        apply_runtime_profile(self.config)
        spatial_enabled = parse_spatial_config(self.config.spatial, self.config.total_leds).enabled
        spatial_topology: SpatialRoomTopology | None = None
        if spatial_enabled:
            spatial_topology = SpatialRoomTopology(self.config)
            spatial_topology.apply_legacy_fields(self.config)
        errors = self.config.validate()
        if errors:
            raise ValueError("; ".join(errors))
        self.topology = spatial_topology if spatial_topology is not None else RoomTopology(self.config)
        self.processor = FrameProcessor(self.config)
        self.motion = MotionDetector(self.config)
        self.event_detector = EventDetector(self.config)
        self.wave_engine = create_effect_renderer(self.config, self.topology)
        self.wled = WLEDOutput(self.config, self.logger, spatial_topology)
        self.client = HyperHDRClient(self.config, self.logger)
        self.client.start()
        self.running = True
        self._parallel_enabled = bool(self.config.parallel_runtime)
        self._stop_event.clear()
        now = time.monotonic()
        self.last_step = now
        self.last_analysis_time = 0.0
        self.stat_time = now
        self.stat_frames = 0
        self.snapshot = RuntimeSnapshot(
            running=True,
            leds=np.zeros((self.config.total_leds, 3), dtype=np.uint8),
            frame_history=[],
            hyperhdr_connected=self.config.simulate_input,
            wled_enabled=self.wled.enabled,
        )
        if self._parallel_enabled:
            self._analysis_results = Queue(maxsize=max(1, int(self.config.event_queue_size)))
            self._analysis_thread = threading.Thread(target=self._analysis_loop, name="effects-analysis", daemon=True)
            self._render_thread = threading.Thread(target=self._render_loop, name="effects-render", daemon=True)
            self._analysis_thread.start()
            self._render_thread.start()
        self.logger.info("Cinematic spill runtime started profile=%s spatial=%s", self.config.runtime_profile, spatial_enabled)

    def stop(self) -> None:
        self._stop_event.set()
        for thread in (self._analysis_thread, self._render_thread):
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        self._analysis_thread = None
        self._render_thread = None
        self._parallel_enabled = False
        if self.client:
            self.client.stop()
        self.running = False
        with self._snapshot_lock:
            self.snapshot.running = False
        self.logger.info("Cinematic spill runtime stopped")

    def restart(self, config: EngineConfig) -> None:
        was_running = self.running
        self.stop()
        self.config = config
        if was_running:
            self.start()

    def step_once(self, timeout: float = 0.0) -> RuntimeSnapshot:
        if self._parallel_enabled:
            if timeout > 0 and self.snapshot.frame is None:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    snapshot = self.latest_snapshot()
                    if snapshot.frame is not None:
                        return snapshot
                    time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))
            return self.latest_snapshot()
        if not self.running:
            return self.snapshot
        assert self.client and self.processor and self.motion and self.event_detector and self.wave_engine and self.wled
        started = time.monotonic()
        detected: list[LightEvent] = []
        candidate_events = 0
        frame = self.snapshot.frame
        analysis = self.snapshot.analysis
        last_error = ""

        try:
            requirements = self.analysis_requirements()
            raw_frames = self._frames_for_policy(timeout)
            for raw in raw_frames:
                render_frames = self._prepare_render_frames(raw)
                frame = render_frames.preview_frame
                self.frame_history.append(frame.copy())
                self.wave_engine.set_tv_frame(render_frames.tv_frame)
                if requirements.any_analysis:
                    analysis = self.motion.analyze(render_frames.analysis_frame, requirements)
                    self.analysis_frames += 1
                    candidates = self.event_detector.detect(analysis)
                    candidate_events = len(candidates)
                    detected = self.event_mixer.mix(candidates)
                    log_events = [event for event in detected if event.kind != "front_ambient"]
                    if log_events:
                        self.logger.info(
                            "events: %s candidates=%d suppressed=%d",
                            ", ".join(f"{'*' if event.primary else ''}{event.kind}:{event.edge}:{event.intensity:.2f}" for event in log_events),
                            candidate_events,
                            max(0, candidate_events - len(detected)),
                        )
                    primary = next((event for event in detected if event.primary), None)
                    if primary and is_high_energy_primary(primary.kind):
                        self.wave_engine.duck_lower_priority_waves(primary.kind)
                    self.wave_engine.add_events(detected)
                    if (
                        self.config.lighting_mode == "front_ambient"
                        and self.config.front_ambient_source == "top_strip"
                        and self.config.enabled_effects.get("front_ambient", True)
                        and self.wave_engine.accepts_front_ambient_strip_sample
                    ):
                        colors, intensity = self.processor.top_strip_colors(render_frames.analysis_frame, self.wave_engine.front_ambient_led_count())
                        self.wave_engine.set_front_ambient_strip(colors, intensity)
                        self.wave_engine.set_ambient_spill_scene_boost(
                            analysis.ambient_spill_boost_score,
                            analysis.luminous_bloom_color,
                        )
                else:
                    analysis = None

            now = time.monotonic()
            leds = self.wave_engine.step(now - self.last_step)
            self.last_step = now
            last_wled_send = self.wled.send(leds)
            self.last_step_duration = time.monotonic() - started
            frame_interval = 1.0 / max(1, self.config.target_fps)
            if self.last_step_duration > frame_interval:
                self.missed_deadlines += 1

            self.stat_frames += 1
            if now - self.stat_time >= 1.0:
                fps = self.stat_frames / (now - self.stat_time)
                self.stat_frames = 0
                self.stat_time = now
            else:
                fps = self.snapshot.fps

            self.snapshot = RuntimeSnapshot(
                running=True,
                frame=frame,
                frame_history=list(self.frame_history),
                analysis=analysis,
                leds=leds,
                events=detected,
                active_waves=len(self.wave_engine.waves),
                fps=fps,
                buffered_frames=self.client.frames.qsize(),
                hyperhdr_connected=self.config.simulate_input or self.client.connected,
                wled_enabled=self.wled.enabled,
                last_wled_send=last_wled_send,
                last_error="",
                missed_deadlines=self.missed_deadlines,
                analysis_frames=self.analysis_frames,
                skipped_analysis_frames=self.skipped_analysis_frames,
                wled_skip_count=self.wled.skip_count,
                step_ms=self.last_step_duration * 1000.0,
                candidate_events=candidate_events,
                tv_frames=self.tv_frames,
                analysis_frame_drops=self.analysis_frame_drops,
                effect_frame_skips=self.effect_frame_skips,
                spatial_priority_band_skips=self.spatial_priority_band_skips,
            )
        except Exception as exc:
            last_error = str(exc)
            self.logger.exception("Runtime step failed")
            self.snapshot.last_error = last_error

        elapsed = time.monotonic() - started
        if elapsed < 0:
            self.snapshot.last_error = last_error
        return self.snapshot

    def latest_snapshot(self) -> RuntimeSnapshot:
        with self._snapshot_lock:
            return self.snapshot

    def trigger_simulation_effect(self, effect: str) -> bool:
        if not self.running or not self.config.simulate_input or not self.client:
            return False
        self.client.trigger_simulation_effect(effect)
        return True

    def _prepare_render_frames(self, raw: np.ndarray) -> RenderFrames:
        assert self.processor
        prepare_render_frames = getattr(self.processor, "prepare_render_frames", None)
        if callable(prepare_render_frames):
            return prepare_render_frames(raw)
        frame = self.processor.prepare(raw)
        return RenderFrames(tv_frame=frame, analysis_frame=frame, preview_frame=frame)

    def _analysis_loop(self) -> None:
        assert self.processor and self.motion and self.event_detector
        interval = 1.0 / max(1, min(int(self.config.motion_analysis_fps), int(self.config.target_fps)))
        next_due = time.monotonic()
        last_revision = 0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_due:
                time.sleep(min(0.01, next_due - now))
                continue
            next_due = now + interval
            with self._frame_lock:
                frame = self._latest_analysis_frame
                revision = self._latest_frame_revision
            if frame is None or revision <= last_revision:
                continue
            if revision > last_revision + 1 and last_revision > 0:
                self.analysis_frame_drops += revision - last_revision - 1
            last_revision = revision

            requirements = self.analysis_requirements()
            if not requirements.any_analysis:
                self.skipped_analysis_frames += 1
                continue

            result = self._analyze_frame_for_parallel(frame, requirements)
            self._put_analysis_result(result)

    def _analyze_frame_for_parallel(self, frame: np.ndarray, requirements: AnalysisRequirements) -> AnalysisResult:
        assert self.motion and self.event_detector
        analysis = self.motion.analyze(frame, requirements)
        self.analysis_frames += 1
        candidates = self.event_detector.detect(analysis)
        detected = self.event_mixer.mix(candidates)
        front_colors: np.ndarray | None = None
        front_intensity = 0.0
        if (
            self.config.lighting_mode == "front_ambient"
            and self.config.front_ambient_source == "top_strip"
            and self.config.enabled_effects.get("front_ambient", True)
        ):
            count = self._front_ambient_led_count_for_analysis()
            front_colors, front_intensity = self.processor.top_strip_colors(frame, count) if self.processor else (None, 0.0)
        return AnalysisResult(
            frame=frame,
            analysis_frame=frame,
            analysis=analysis,
            events=detected,
            candidate_events=len(candidates),
            front_ambient_strip_colors=front_colors,
            front_ambient_intensity=front_intensity,
            ambient_spill_boost_score=analysis.ambient_spill_boost_score,
            ambient_spill_boost_color=analysis.luminous_bloom_color,
        )

    def _front_ambient_led_count_for_analysis(self) -> int:
        if isinstance(self.topology, SpatialRoomTopology):
            return max(1, int(self.topology.front_ambient_indices().size))
        if self.wave_engine is not None:
            try:
                return max(1, int(self.wave_engine.front_ambient_led_count()))
            except Exception:
                return max(1, self.config.total_leds // 4)
        return max(1, self.config.total_leds // 4)

    def _put_analysis_result(self, result: AnalysisResult) -> None:
        queue = self._analysis_results
        if queue is None:
            return
        while queue.full():
            try:
                queue.get_nowait()
                self.skipped_analysis_frames += 1
            except Empty:
                break
        queue.put_nowait(result)

    def _render_loop(self) -> None:
        assert self.client and self.wave_engine and self.wled
        interval = 1.0 / max(1, int(self.config.target_fps))
        self.last_step = time.monotonic()
        self.stat_time = self.last_step
        self.stat_frames = 0
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self._render_parallel_frame(started, interval)
            except Exception as exc:
                self.logger.exception("Parallel render step failed")
                with self._snapshot_lock:
                    self.snapshot.last_error = str(exc)
            elapsed = time.monotonic() - started
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _render_parallel_frame(self, started: float, frame_interval: float) -> None:
        assert self.client and self.wave_engine and self.wled
        tv_frame = self._prepare_latest_tv_frame()
        with self._frame_lock:
            frame = self._latest_frame
        result = self._latest_analysis_result()
        if tv_frame is None and result is not None:
            tv_frame = result.analysis_frame if result.analysis_frame is not None else result.frame
        if frame is None and result is not None:
            frame = result.frame
        if tv_frame is not None:
            self.wave_engine.set_tv_frame(tv_frame)

        detected: list[LightEvent] = []
        analysis = self.snapshot.analysis
        candidate_events = 0
        if result is not None:
            analysis = result.analysis
            detected = result.events
            candidate_events = result.candidate_events
            self._apply_parallel_analysis_result(result)

        max_bands = self._spatial_priority_budget_for_frame(frame_interval)
        self._set_renderer_priority_budget(max_bands)
        now = time.monotonic()
        leds = self.wave_engine.step(now - self.last_step)
        skipped_bands = int(getattr(self.wave_engine, "last_spatial_priority_band_skips", 0))
        if skipped_bands:
            self.effect_frame_skips += 1
            self.spatial_priority_band_skips += skipped_bands
        self._set_renderer_priority_budget(None)
        self.last_step = now
        last_wled_send = self.wled.send(leds)
        self.last_step_duration = time.monotonic() - started
        if self.last_step_duration > frame_interval:
            self.missed_deadlines += 1

        self.stat_frames += 1
        if now - self.stat_time >= 1.0:
            fps = self.stat_frames / (now - self.stat_time)
            self.stat_frames = 0
            self.stat_time = now
        else:
            fps = self.snapshot.fps

        snapshot = RuntimeSnapshot(
            running=True,
            frame=frame,
            frame_history=list(self.frame_history),
            analysis=analysis,
            leds=leds,
            events=detected,
            active_waves=len(self.wave_engine.waves),
            fps=fps,
            buffered_frames=self.client.frames.qsize(),
            hyperhdr_connected=self.config.simulate_input or self.client.connected,
            wled_enabled=self.wled.enabled,
            last_wled_send=last_wled_send,
            last_error="",
            missed_deadlines=self.missed_deadlines,
            analysis_frames=self.analysis_frames,
            skipped_analysis_frames=self.skipped_analysis_frames,
            wled_skip_count=self.wled.skip_count,
            step_ms=self.last_step_duration * 1000.0,
            candidate_events=candidate_events,
            tv_frames=self.tv_frames,
            analysis_frame_drops=self.analysis_frame_drops,
            effect_frame_skips=self.effect_frame_skips,
            spatial_priority_band_skips=self.spatial_priority_band_skips,
        )
        with self._snapshot_lock:
            self.snapshot = snapshot

    def _prepare_latest_tv_frame(self) -> np.ndarray | None:
        assert self.processor
        frames = self._drain_frames(0.0)
        if not frames:
            with self._frame_lock:
                return self._latest_tv_frame
        render_frames = self._prepare_render_frames(frames[-1])
        with self._frame_lock:
            self._latest_frame = render_frames.preview_frame
            self._latest_tv_frame = render_frames.tv_frame
            self._latest_analysis_frame = render_frames.analysis_frame
            self._latest_frame_revision += 1
        self.frame_history.append(render_frames.preview_frame.copy())
        self.tv_frames += 1
        if len(frames) > 1:
            self.analysis_frame_drops += len(frames) - 1
        return render_frames.tv_frame

    def _spatial_priority_budget_for_frame(self, frame_interval: float) -> int | None:
        if self.config.effect_render_skip_policy != "tv_first":
            return None
        if not isinstance(self.wave_engine, VectorizedSpatialRenderer):
            return None
        bands = max(1, int(self.config.spatial_priority_bands))
        if self.last_step_duration > frame_interval * 1.05:
            return 1
        if self.last_step_duration > frame_interval * 0.80:
            return max(1, min(bands, 2))
        return None

    def _set_renderer_priority_budget(self, max_bands: int | None) -> None:
        setter = getattr(self.wave_engine, "set_spatial_priority_budget", None)
        if callable(setter):
            setter(max_bands)

    def _latest_analysis_result(self) -> AnalysisResult | None:
        queue = self._analysis_results
        if queue is None:
            return None
        latest: AnalysisResult | None = None
        while True:
            try:
                latest = queue.get_nowait()
            except Empty:
                return latest

    def _apply_parallel_analysis_result(self, result: AnalysisResult) -> None:
        assert self.wave_engine
        log_events = [event for event in result.events if event.kind != "front_ambient"]
        if log_events:
            self.logger.info(
                "events: %s candidates=%d suppressed=%d",
                ", ".join(f"{'*' if event.primary else ''}{event.kind}:{event.edge}:{event.intensity:.2f}" for event in log_events),
                result.candidate_events,
                max(0, result.candidate_events - len(result.events)),
            )
        primary = next((event for event in result.events if event.primary), None)
        if primary and is_high_energy_primary(primary.kind):
            self.wave_engine.duck_lower_priority_waves(primary.kind)
        self.wave_engine.add_events(result.events)
        if (
            result.front_ambient_strip_colors is not None
            and self.wave_engine.accepts_front_ambient_strip_sample
        ):
            self.wave_engine.set_front_ambient_strip(result.front_ambient_strip_colors, result.front_ambient_intensity)
            self.wave_engine.set_ambient_spill_scene_boost(
                result.ambient_spill_boost_score,
                result.ambient_spill_boost_color,
            )

    def _drain_frames(self, timeout: float) -> list[np.ndarray]:
        assert self.client
        frames: list[np.ndarray] = []
        first = self.client.get_frame(timeout=timeout)
        if first is None:
            return frames
        frames.append(first)
        while True:
            extra = self.client.get_frame(timeout=0.0)
            if extra is None:
                break
            frames.append(extra)
        return frames

    def _frames_for_policy(self, timeout: float) -> list[np.ndarray]:
        interval = 1.0 / max(1, self.config.target_fps)
        overloaded = self.last_step_duration > interval * 1.15
        if self.config.overload_policy == "output_first" and overloaded:
            self.skipped_analysis_frames += 1
            return []

        frames = self._drain_frames(timeout)
        if not frames:
            return frames
        if not self._analysis_due():
            self.skipped_analysis_frames += len(frames)
            latest = frames[-1]
            render_frames = self._prepare_render_frames(latest) if self.processor else None
            self.wave_engine.set_tv_frame(render_frames.tv_frame if render_frames else latest)
            if render_frames:
                self.frame_history.append(render_frames.preview_frame.copy())
            return []

        if self.config.overload_policy == "adaptive_quality":
            if overloaded:
                self.analysis_stride = min(4, self.analysis_stride + 1)
            elif self.analysis_stride > 1 and self.last_step_duration < interval * 0.70:
                self.analysis_stride -= 1
            self._analysis_tick = (self._analysis_tick + 1) % self.analysis_stride
            if self._analysis_tick != 0:
                self.skipped_analysis_frames += len(frames)
                latest = frames[-1]
                render_frames = self._prepare_render_frames(latest) if self.processor else None
                self.wave_engine.set_tv_frame(render_frames.tv_frame if render_frames else latest)
                if render_frames:
                    self.frame_history.append(render_frames.preview_frame.copy())
                return []
            self.last_analysis_time = time.monotonic()
            return [frames[-1]]

        self.last_analysis_time = time.monotonic()
        return frames

    def _analysis_due(self) -> bool:
        fps = min(max(1, int(self.config.motion_analysis_fps)), max(1, int(self.config.target_fps)))
        if fps >= max(1, int(self.config.target_fps)):
            return True
        now = time.monotonic()
        if self.last_analysis_time <= 0.0:
            self.last_analysis_time = now
            return True
        return now - self.last_analysis_time >= 1.0 / fps
