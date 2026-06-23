#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

install_deps
echo "Starting Pi Zero profile live engine. WLED output follows config.json unless --wled is passed."
run_python main.py --no-debug --profile pi_zero --overload-policy adaptive_quality "$@"
