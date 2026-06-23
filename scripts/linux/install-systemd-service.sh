#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SERVICE_NAME="${SERVICE_NAME:-ambilight-effects}"
PROFILE="${PROFILE:-pi_zero}"
OVERLOAD_POLICY="${OVERLOAD_POLICY:-adaptive_quality}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

install_deps

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl was not found. This installer is intended for systemd-based Linux systems."
  exit 1
fi

echo "Installing ${SERVICE_PATH}"
sudo tee "${SERVICE_PATH}" >/dev/null <<SERVICE
[Unit]
Description=Ambilight Effects headless engine
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_ROOT}
ExecStart=${VENV_DIR}/bin/python main.py --config config.json --no-debug --profile ${PROFILE} --overload-policy ${OVERLOAD_POLICY} ${EXTRA_ARGS}
Restart=on-failure
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"
echo "Installed and started ${SERVICE_NAME}. View logs with: journalctl -u ${SERVICE_NAME} -f"
