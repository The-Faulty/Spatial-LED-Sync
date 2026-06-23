#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

install_deps
echo "Running subsystem benchmarks against a 60fps frame budget..."
run_python benchmark_parts.py --profile pi_zero --seconds 0.75 "$@"
run_python benchmark_parts.py --profile pi_5 --seconds 0.75 "$@"

echo "Running stress subsystem benchmark with 2000 LEDs and the Pi 5 profile wave cap..."
run_python benchmark_parts.py --profile pi_5 --seconds 1.0 --warmup 10 --min-iterations 20 --mode stress "$@"

echo "For a 100-wave overload test, run:"
echo "scripts/linux/benchmark-parts.sh --profile pi_5 --mode stress --wave-count 100 --seconds 2 --warmup 20 --min-iterations 20"
echo "For the absolute ceiling test, run:"
echo "scripts/linux/benchmark-parts.sh --profile pi_5 --seconds 0.05 --mode ceiling"
