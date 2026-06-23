#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

PROFILE="${PROFILE:-pi_5}"
SECONDS_TO_RUN="${SECONDS_TO_RUN:-2}"

install_deps
echo "Running worst-case benchmark: profile=${PROFILE}, 2000 LEDs, heavy analysis, stress waves..."
run_python benchmark_parts.py \
  --profile "${PROFILE}" \
  --mode stress \
  --worst-led-count 2000 \
  --wave-count 100 \
  --seconds "${SECONDS_TO_RUN}" \
  --warmup 20 \
  --min-iterations 20 \
  "$@"
