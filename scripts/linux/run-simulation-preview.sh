#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

install_deps
echo "Starting simulated preview. Press q in the preview window to quit."
run_python main.py --simulate --debug "$@"
