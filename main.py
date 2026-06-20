from __future__ import annotations

import argparse
import signal
import time

import cv2

from config import EngineConfig, write_default_config
from logger import setup_logging
from performance import apply_runtime_profile, valid_runtime_profiles
from effects_engine import HeadlessEffectsEngine
from visualization import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HyperHDR Cinematic Light Spill Engine")
    parser.add_argument("--config", default="config.json", help="Path to configuration JSON")
    parser.add_argument("--write-default-config", action="store_true", help="Create config.json and exit")
    parser.add_argument("--debug", action="store_true", help="Force visualization on")
    parser.add_argument("--no-debug", action="store_true", help="Force visualization off")
    parser.add_argument("--simulate", action="store_true", help="Use generated frames instead of HyperHDR")
    parser.add_argument("--wled", action="store_true", help="Enable WLED output if wled_ip is configured")
    parser.add_argument("--profile", choices=sorted(valid_runtime_profiles()), help="Runtime performance profile")
    parser.add_argument(
        "--overload-policy",
        choices=["adaptive_quality", "fixed_quality", "output_first"],
        help="Behavior when the runtime misses frame deadlines",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_default_config:
        write_default_config(args.config)
        print(f"Wrote {args.config}")
        return 0

    config = EngineConfig.load(args.config)
    if args.debug:
        config.debug = True
    if args.no_debug:
        config.debug = False
    if args.simulate:
        config.simulate_input = True
    if args.wled:
        config.send_to_wled = True
    if args.profile:
        config.runtime_profile = args.profile
    if args.overload_policy:
        config.overload_policy = args.overload_policy
    apply_runtime_profile(config)

    logger = setup_logging(config.log_level, config.log_file)
    visualizer = Visualizer(config) if config.debug else None
    runtime = HeadlessEffectsEngine(config, logger)

    stopping = False

    def stop_signal(*_: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGTERM, stop_signal)

    runtime.start()
    frame_interval = 1.0 / max(1, config.target_fps)
    frames = 0
    stat_time = time.monotonic()

    try:
        while not stopping:
            started = time.monotonic()
            snapshot = runtime.step_once(timeout=frame_interval)

            if visualizer and snapshot.frame is not None and snapshot.analysis is not None and snapshot.leds is not None:
                if not visualizer.show(snapshot.frame, snapshot.analysis, snapshot.leds, snapshot.active_waves):
                    break

            frames += 1
            now = time.monotonic()
            if now - stat_time >= 5.0:
                logger.info(
                    "perf fps=%.1f active_waves=%d buffered_frames=%d",
                    frames / (now - stat_time),
                    snapshot.active_waves,
                    snapshot.buffered_frames,
                )
                frames = 0
                stat_time = now

            elapsed = time.monotonic() - started
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        runtime.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
