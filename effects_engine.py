from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from config import EngineConfig
from event_detector import EventDetector, LightEvent
from event_mixer import EventMixer, is_high_energy_primary
from frame_processor import FrameProcessor
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
    step_ms: float = 0.0
    candidate_events: int = 0


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
        self.missed_deadlines = 0
        self.last_step_duration = 0.0
        self.analysis_stride = 1
        self._analysis_tick = 0
        self.frame_history: deque[np.ndarray] = deque(maxlen=12)
        self.snapshot = RuntimeSnapshot()

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
        now = time.monotonic()
        self.last_step = now
        self.stat_time = now
        self.stat_frames = 0
        self.snapshot = RuntimeSnapshot(
            running=True,
            leds=np.zeros((self.config.total_leds, 3), dtype=np.uint8),
            frame_history=[],
            hyperhdr_connected=self.config.simulate_input,
            wled_enabled=self.wled.enabled,
        )
        self.logger.info("Cinematic spill runtime started profile=%s spatial=%s", self.config.runtime_profile, spatial_enabled)

    def stop(self) -> None:
        if self.client:
            self.client.stop()
        self.running = False
        self.snapshot.running = False
        self.logger.info("Cinematic spill runtime stopped")

    def restart(self, config: EngineConfig) -> None:
        was_running = self.running
        self.stop()
        self.config = config
        if was_running:
            self.start()

    def step_once(self, timeout: float = 0.0) -> RuntimeSnapshot:
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
                frame = self.processor.prepare(raw)
                self.frame_history.append(frame.copy())
                self.wave_engine.set_tv_frame(frame)
                if requirements.any_analysis:
                    analysis = self.motion.analyze(frame, requirements)
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
                        colors, intensity = self.processor.top_strip_colors(frame, self.wave_engine.front_ambient_led_count())
                        self.wave_engine.set_front_ambient_strip(colors, intensity)
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
                step_ms=self.last_step_duration * 1000.0,
                candidate_events=candidate_events,
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
        return self.snapshot

    def trigger_simulation_effect(self, effect: str) -> bool:
        if not self.running or not self.config.simulate_input or not self.client:
            return False
        self.client.trigger_simulation_effect(effect)
        return True

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

        if self.config.overload_policy == "adaptive_quality":
            if overloaded:
                self.analysis_stride = min(4, self.analysis_stride + 1)
            elif self.analysis_stride > 1 and self.last_step_duration < interval * 0.70:
                self.analysis_stride -= 1
            self._analysis_tick = (self._analysis_tick + 1) % self.analysis_stride
            if self._analysis_tick != 0:
                self.skipped_analysis_frames += len(frames)
                latest = frames[-1]
                prepared = self.processor.prepare(latest) if self.processor else latest
                self.wave_engine.set_tv_frame(prepared)
                return []
            return [frames[-1]]

        return frames
