#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "${SCRIPT_DIR}/..")
BUILD_DIR="${BASE_DIR}/build"

rm -rf "$BUILD_DIR"
