#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SECONDS_TO_RUN="${SECONDS_TO_RUN:-15}"

install_deps
echo "Running Pi 5 profile benchmark for ${SECONDS_TO_RUN}s..."
run_python benchmark.py --profile pi_5 --seconds "${SECONDS_TO_RUN}" "$@"
