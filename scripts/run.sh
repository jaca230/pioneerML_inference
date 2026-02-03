#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "${SCRIPT_DIR}/..")
REPO_DIR=$(realpath "${BASE_DIR}/../..")

# Prefer system libs over conda to avoid GLIBCXX mismatches at runtime.
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

MODE="group_classifier"
MODEL_PATH_DEFAULT="${REPO_DIR}/trained_models/groupclassifier/groupclassifier_20260203_055139_torchscript.pt"
INPUT_PATH_DEFAULT="${REPO_DIR}/data/ml_output_000.parquet"
OUTPUT_DIR_DEFAULT="${REPO_DIR}/data/inference_outputs/${MODE}"
OUTPUT_PATH_DEFAULT="${OUTPUT_DIR_DEFAULT}/preds.parquet"
MODEL_PATH="$MODEL_PATH_DEFAULT"
INPUT_PATHS=()
OUTPUT_PATH="$OUTPUT_PATH_DEFAULT"
DEVICE="cuda"
CONFIG_PATH=""
CHECK_ACCURACY="false"
METRICS_OUT=""
THRESHOLD=""

show_help() {
  echo "Usage: ./run.sh [OPTIONS]"
  echo
  echo "Options:"
  echo "  -m, --mode <name>          Model mode (default: group_classifier)"
  echo "  -M, --model <path>         TorchScript model path"
  echo "  -i, --input <path>         Parquet input path (repeatable)"
  echo "  -o, --output <path>        Output parquet path"
  echo "  -d, --device <cpu|cuda>    Device (default: cuda)"
  echo "  -c, --config <path>        Adapter config JSON"
  echo "  -a, --check-accuracy       Compute accuracy if targets are present"
  echo "  -t, --threshold <float>    Threshold for accuracy (default: 0.5)"
  echo "  -r, --metrics-out <path>   Write metrics JSON to path"
  echo "  -h, --help                 Show help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--mode)
      MODE="$2"; shift 2;;
    -M|--model)
      MODEL_PATH="$2"; shift 2;;
    -i|--input)
      INPUT_PATHS+=("$2"); shift 2;;
    -o|--output)
      OUTPUT_PATH="$2"; shift 2;;
    -d|--device)
      DEVICE="$2"; shift 2;;
    -c|--config)
      CONFIG_PATH="$2"; shift 2;;
    -a|--check-accuracy)
      CHECK_ACCURACY="true"; shift 1;;
    -t|--threshold)
      THRESHOLD="$2"; shift 2;;
    -r|--metrics-out)
      METRICS_OUT="$2"; shift 2;;
    -h|--help)
      show_help; exit 0;;
    *)
      echo "[run.sh] Unknown option: $1"; show_help; exit 1;;
  esac
done

if [ ${#INPUT_PATHS[@]} -eq 0 ]; then
  INPUT_PATHS+=("$INPUT_PATH_DEFAULT")
fi

if [ "$OUTPUT_PATH" = "$OUTPUT_PATH_DEFAULT" ]; then
  OUTPUT_DIR_DEFAULT="${REPO_DIR}/data/inference_outputs/${MODE}"
  OUTPUT_PATH="${OUTPUT_DIR_DEFAULT}/preds.parquet"
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

CMD=("${BASE_DIR}/build/pioneerml_inference"
  --mode "$MODE"
  --model "$MODEL_PATH"
  --output "$OUTPUT_PATH"
  --device "$DEVICE"
)

for p in "${INPUT_PATHS[@]}"; do
  CMD+=(--input "$p")
done

if [ -n "$CONFIG_PATH" ]; then
  CMD+=(--config "$CONFIG_PATH")
fi
if [ "$CHECK_ACCURACY" = "true" ]; then
  CMD+=(--check-accuracy)
fi
if [ -n "$THRESHOLD" ]; then
  CMD+=(--threshold "$THRESHOLD")
fi
if [ -n "$METRICS_OUT" ]; then
  CMD+=(--metrics-out "$METRICS_OUT")
fi

echo "[run.sh] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[run.sh] Output: $OUTPUT_PATH"
