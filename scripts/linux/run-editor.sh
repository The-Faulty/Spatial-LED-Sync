#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8787}"

install_deps
echo "Starting 3D room editor at http://${HOST}:${PORT}"
run_python spatial_editor_server.py --host "${HOST}" --port "${PORT}" "$@"
