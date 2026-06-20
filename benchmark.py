from __future__ import annotations

import argparse
import json
import logging
import time

from config import EngineConfig
from logger import setup_logging
from performance import apply_runtime_profile, valid_runtime_profiles
from runtime import EngineRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless lighting runtime benchmark")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--profile", choices=sorted(valid_runtime_profiles()), default=None)
    parser.add_argument("--overload-policy", choices=["adaptive_quality", "fixed_quality", "output_first"], default=None)
    parser.add_argument("--seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = EngineConfig.load(args.config)
    config.simulate_input = True
    config.send_to_wled = False
    config.debug = False
    if args.profile:
        config.runtime_profile = args.profile
    if args.overload_policy:
        config.overload_policy = args.overload_policy
    apply_runtime_profile(config)

    logger = setup_logging("WARNING", config.log_file)
    logger.setLevel(logging.WARNING)
    runtime = EngineRuntime(config, logger)
    runtime.start()
    frame_interval = 1.0 / max(1, config.target_fps)
    started = time.monotonic()
    steps = 0
    try:
        while time.monotonic() - started < args.seconds:
            loop_started = time.monotonic()
            runtime.step_once(timeout=frame_interval)
            steps += 1
            elapsed = time.monotonic() - loop_started
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        snapshot = runtime.latest_snapshot()
        runtime.stop()

    duration = max(0.001, time.monotonic() - started)
    result = {
        "profile": config.runtime_profile,
        "overload_policy": config.overload_policy,
        "duration_seconds": round(duration, 3),
        "steps": steps,
        "loop_fps": round(steps / duration, 2),
        "analysis_frames": snapshot.analysis_frames,
        "analysis_fps": round(snapshot.analysis_frames / duration, 2),
        "skipped_analysis_frames": snapshot.skipped_analysis_frames,
        "missed_deadlines": snapshot.missed_deadlines,
        "last_step_ms": round(snapshot.step_ms, 3),
        "active_waves": snapshot.active_waves,
        "led_count": int(snapshot.leds.shape[0]) if snapshot.leds is not None else config.total_leds,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
