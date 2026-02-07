#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "${SCRIPT_DIR}/..")
REPO_DIR=$(realpath "${BASE_DIR}/../..")

# Prefer system libs over conda to avoid GLIBCXX mismatches at runtime.
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

MODE="group_classifier"
INPUT_PATH_DEFAULT="${REPO_DIR}/data/ml_output_000.parquet"
OUTPUT_DIR_DEFAULT="${REPO_DIR}/data/inference_outputs/${MODE}"
OUTPUT_PATH_DEFAULT="${OUTPUT_DIR_DEFAULT}/preds.parquet"
MODEL_PATH=""
MODEL_SET="false"
INPUT_PATHS=()
INPUT_GROUPS=()
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
  echo "  -g, --input-group <csv>    Grouped parquet inputs, comma-separated (repeatable)"
  echo "  -o, --output <path>        Output parquet path"
  echo "  -d, --device <cpu|cuda>    Device (default: cuda)"
  echo "  -c, --config <path>        Adapter config JSON"
  echo "  -a, --check-accuracy       Compute accuracy if targets are present"
  echo "  -t, --threshold <float>    Threshold for accuracy (default: 0.5)"
  echo "  -r, --metrics-out <path>   Write metrics JSON to path"
  echo "  -h, --help                 Show help"
}

build_default_output_path() {
  local mode="$1"
  shift
  local inputs=("$@")
  local out_dir="${REPO_DIR}/data/inference_outputs/${mode}"
  mkdir -p "$out_dir"

  if [ ${#inputs[@]} -eq 1 ]; then
    local base
    base=$(basename "${inputs[0]}")
    base="${base%.parquet}"
    echo "${out_dir}/${base}_preds.parquet"
    return
  fi

  local first last first_base last_base
  first="${inputs[0]}"
  last="${inputs[${#inputs[@]}-1]}"
  first_base=$(basename "${first}")
  last_base=$(basename "${last}")
  first_base="${first_base%.parquet}"
  last_base="${last_base%.parquet}"
  echo "${out_dir}/${first_base}_to_${last_base}_${#inputs[@]}files_preds.parquet"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--mode)
      MODE="$2"; shift 2;;
    -M|--model)
      MODEL_PATH="$2"; MODEL_SET="true"; shift 2;;
    -i|--input)
      INPUT_PATHS+=("$2"); shift 2;;
    -g|--input-group)
      INPUT_GROUPS+=("$2"); shift 2;;
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
  if [ ${#INPUT_GROUPS[@]} -eq 0 ]; then
    INPUT_PATHS+=("$INPUT_PATH_DEFAULT")
  fi
fi

if [ "$MODEL_SET" != "true" ]; then
  case "$MODE" in
    group_classifier)
      MODEL_PATH="${REPO_DIR}/trained_models/groupclassifier/groupclassifier_20260203_055139_torchscript.pt"
      ;;
    group_classifier_event)
      MODEL_PATH="${REPO_DIR}/trained_models/groupclassifier_event/groupclassifier_event_20260203_111205_torchscript.pt"
      ;;
    group_splitter)
      LATEST_SPLITTER=$(ls -1t "${REPO_DIR}"/trained_models/groupsplitter/*_torchscript.pt 2>/dev/null | head -n1 || true)
      MODEL_PATH="${LATEST_SPLITTER}"
      ;;
    group_splitter_event)
      LATEST_SPLITTER_EVENT=$(ls -1t "${REPO_DIR}"/trained_models/groupsplitter_event/*_torchscript.pt 2>/dev/null | head -n1 || true)
      MODEL_PATH="${LATEST_SPLITTER_EVENT}"
      ;;
    *)
      MODEL_PATH="${REPO_DIR}/trained_models/groupclassifier/groupclassifier_20260203_055139_torchscript.pt"
      ;;
  esac
fi

if [ -z "$MODEL_PATH" ]; then
  echo "[run.sh] Could not infer default model path for mode '$MODE'. Please pass --model."
  exit 1
fi

if [ "$OUTPUT_PATH" = "$OUTPUT_PATH_DEFAULT" ]; then
  OUTPUT_BASIS=("${INPUT_PATHS[@]}")
  if [ ${#OUTPUT_BASIS[@]} -eq 0 ]; then
    for g in "${INPUT_GROUPS[@]}"; do
      first="${g%%,*}"
      if [ -n "$first" ]; then
        OUTPUT_BASIS+=("$first")
      fi
    done
  fi
  OUTPUT_PATH="$(build_default_output_path "$MODE" "${OUTPUT_BASIS[@]}")"
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

for g in "${INPUT_GROUPS[@]}"; do
  CMD+=(--input-group "$g")
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
