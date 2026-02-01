#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "${SCRIPT_DIR}/..")
BUILD_DIR="${BASE_DIR}/build"
CLEANUP_SCRIPT="${SCRIPT_DIR}/cleanup.sh"

OVERWRITE=false
JOBS_ARG="-j"

show_help() {
  echo "Usage: ./build.sh [OPTIONS]"
  echo
  echo "Options:"
  echo "  -o, --overwrite           Remove existing build directory before building"
  echo "  -j, --jobs <number>       Specify number of processors (default: all available)"
  echo "  -h, --help                Display this help message"
}

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -o|--overwrite)
      OVERWRITE=true
      shift
      ;;
    -j|--jobs)
      if [[ -n "${2:-}" && "$2" != -* ]]; then
        JOBS_ARG="-j$2"
        shift 2
      else
        JOBS_ARG="-j"
        shift
      fi
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "[build.sh, ERROR] Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

if [ "$OVERWRITE" = true ]; then
  echo "[build.sh] Cleaning previous build with: $CLEANUP_SCRIPT"
  if [ -x "$CLEANUP_SCRIPT" ]; then
    "$CLEANUP_SCRIPT"
  else
    rm -rf "$BUILD_DIR"
  fi
fi

mkdir -p "$BUILD_DIR"

if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
  CMAKE_HOME_DIR=$(grep -E "^CMAKE_HOME_DIRECTORY:PATH=" "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2- || true)
  if [ -n "$CMAKE_HOME_DIR" ] && [ "$CMAKE_HOME_DIR" != "$BASE_DIR" ]; then
    echo "[build.sh] CMake cache source mismatch:"
    echo "  cached: $CMAKE_HOME_DIR"
    echo "  current: $BASE_DIR"
    echo "[build.sh] Removing stale build directory."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
  fi
fi

cd "$BUILD_DIR"

CMAKE_ARGS=(-Wno-dev)

if [ -n "${PIONEERML_DATALOADERS_LIB:-}" ]; then
  CMAKE_ARGS+=("-DPIONEERML_INFERENCE_USE_EXTERNAL_DATALOADERS=ON")
  CMAKE_ARGS+=("-DPIONEERML_DATALOADERS_LIB=${PIONEERML_DATALOADERS_LIB}")
fi

echo "[build.sh] Running cmake in: $BUILD_DIR"
if [ -n "${LIBTORCH_DIR:-}" ]; then
    echo "[build.sh] Using LIBTORCH_DIR: $LIBTORCH_DIR"
    if [ -n "${CUDA_HOME:-}" ]; then
        cmake "${CMAKE_ARGS[@]}" -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" "$BASE_DIR"
    else
        cmake "${CMAKE_ARGS[@]}" -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" "$BASE_DIR"
    fi
else
    if command -v python &>/dev/null; then
        TORCH_CMAKE_PREFIX=$(python - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)
        if [ -n "$TORCH_CMAKE_PREFIX" ]; then
            echo "[build.sh] Using Torch cmake prefix: $TORCH_CMAKE_PREFIX"
            if [ -n "${CUDA_HOME:-}" ]; then
                cmake "${CMAKE_ARGS[@]}" -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" "$BASE_DIR"
            else
                cmake "${CMAKE_ARGS[@]}" -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" "$BASE_DIR"
            fi
        else
            cmake "${CMAKE_ARGS[@]}" "$BASE_DIR"
        fi
    else
        cmake "${CMAKE_ARGS[@]}" "$BASE_DIR"
    fi
fi

echo "[build.sh] Building with make $JOBS_ARG"
make $JOBS_ARG

echo "[build.sh] Build complete."
echo "[build.sh] Binaries: $BUILD_DIR/bin (if any)"
