#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_ACCEL="${INSTALL_ACCEL:-1}"
INSTALL_CPP_RENDERER="${INSTALL_CPP_RENDERER:-1}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"

cd "${PROJECT_ROOT}"

ensure_venv() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Creating local Python environment..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
}

install_deps() {
  ensure_venv
  echo "Checking dependencies..."
  install_system_packages
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${VENV_DIR}/bin/python" -m pip install -r requirements.txt
  install_cpp_renderer
  install_acceleration_deps
}

install_system_packages() {
  if [[ "${INSTALL_SYSTEM_PACKAGES}" != "1" ]]; then
    echo "Skipping system packages because INSTALL_SYSTEM_PACKAGES=${INSTALL_SYSTEM_PACKAGES}."
    return
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get not found; skipping Raspberry Pi OS system packages."
    return
  fi

  local sudo_cmd=()
  if [[ "${EUID}" -ne 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
      echo "sudo not found; skipping system packages. Native renderer builds may need build-essential and python3-dev."
      return
    fi
    sudo_cmd=(sudo)
  fi

  echo "Installing Raspberry Pi OS packages for Python builds and OpenCV..."
  "${sudo_cmd[@]}" apt-get update

  local packages=(
    build-essential
    pkg-config
    python3-dev
    libgl1
    libgl1-mesa-dri
    libglib2.0-0
  )

  if apt_package_available libatlas-base-dev; then
    packages+=(libatlas-base-dev)
  elif apt_package_available libopenblas-dev; then
    packages+=(libopenblas-dev)
  else
    echo "Neither libatlas-base-dev nor libopenblas-dev is available; continuing without an explicit BLAS dev package."
  fi

  if ! apt_package_available libglib2.0-0 && apt_package_available libglib2.0-0t64; then
    packages=("${packages[@]/libglib2.0-0/libglib2.0-0t64}")
  fi

  "${sudo_cmd[@]}" apt-get install -y "${packages[@]}"
}

apt_package_available() {
  local package_name="$1"
  local candidate
  candidate="$(apt-cache policy "${package_name}" 2>/dev/null | awk '/Candidate:/ {print $2; exit}')"
  [[ -n "${candidate}" && "${candidate}" != "(none)" ]]
}

install_acceleration_deps() {
  if [[ "${INSTALL_ACCEL}" != "1" ]]; then
    echo "Skipping optional acceleration packages because INSTALL_ACCEL=${INSTALL_ACCEL}."
    return
  fi

  echo "Installing optional spatial renderer acceleration packages..."
  "${VENV_DIR}/bin/python" -m pip install numba
}

install_cpp_renderer() {
  if [[ "${INSTALL_CPP_RENDERER}" != "1" ]]; then
    echo "Skipping C++ renderer build because INSTALL_CPP_RENDERER=${INSTALL_CPP_RENDERER}."
    return
  fi

  echo "Building optional C++ spatial renderer..."
  "${VENV_DIR}/bin/python" setup.py build_ext --inplace
}

run_python() {
  ensure_venv
  "${VENV_DIR}/bin/python" "$@"
}
