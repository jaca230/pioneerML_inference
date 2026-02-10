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
OUTPUT_PATH_DEFAULT="${OUTPUT_DIR_DEFAULT}/ml_output_000_preds_latest.parquet"
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
ALSO_TIMESTAMPED="false"

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
  echo "  -T, --also-timestamped     Also copy outputs to timestamped files"
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
    echo "${out_dir}/${base}_preds_latest.parquet"
    return
  fi

  local first last first_base last_base
  first="${inputs[0]}"
  last="${inputs[${#inputs[@]}-1]}"
  first_base=$(basename "${first}")
  last_base=$(basename "${last}")
  first_base="${first_base%.parquet}"
  last_base="${last_base%.parquet}"
  echo "${out_dir}/${first_base}_to_${last_base}_${#inputs[@]}files_preds_latest.parquet"
}

build_default_metrics_path() {
  local mode="$1"
  local out_dir="${REPO_DIR}/data/inference_outputs/${mode}"
  mkdir -p "$out_dir"
  echo "${out_dir}/metrics_latest.json"
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
    -T|--also-timestamped)
      ALSO_TIMESTAMPED="true"; shift 1;;
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
    endpoint_regressor)
      LATEST_ENDPOINT=$(ls -1t "${REPO_DIR}"/trained_models/endpoint_regressor/*_torchscript.pt 2>/dev/null | head -n1 || true)
      MODEL_PATH="${LATEST_ENDPOINT}"
      ;;
    endpoint_regressor_event)
      LATEST_ENDPOINT_EVENT=$(ls -1t "${REPO_DIR}"/trained_models/endpoints_regressor_event/*_torchscript.pt 2>/dev/null | head -n1 || true)
      MODEL_PATH="${LATEST_ENDPOINT_EVENT}"
      ;;
    event_splitter_event)
      LATEST_EVENT_SPLITTER_EVENT=$(ls -1t "${REPO_DIR}"/trained_models/event_splitter_event/*_torchscript.pt 2>/dev/null | head -n1 || true)
      MODEL_PATH="${LATEST_EVENT_SPLITTER_EVENT}"
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
if [ -z "$METRICS_OUT" ] && [ "$CHECK_ACCURACY" = "true" ]; then
  METRICS_OUT="$(build_default_metrics_path "$MODE")"
fi
if [ -n "$METRICS_OUT" ]; then
  mkdir -p "$(dirname "$METRICS_OUT")"
fi

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

if [ "$CHECK_ACCURACY" = "true" ] && [ -n "$METRICS_OUT" ] && [ -f "$METRICS_OUT" ]; then
  python - "$METRICS_OUT" "$MODE" "$MODEL_PATH" "$OUTPUT_PATH" "${INPUT_PATHS[@]}" -- "${INPUT_GROUPS[@]}" <<'PY'
import json
import sys

metrics_path = sys.argv[1]
mode = sys.argv[2]
model_path = sys.argv[3]
output_path = sys.argv[4]
argv = sys.argv[5:]

if "--" in argv:
    idx = argv.index("--")
    input_paths = argv[:idx]
    input_groups = argv[idx + 1:]
else:
    input_paths = argv
    input_groups = []

validated_files = list(input_paths)
for group in input_groups:
    for item in group.split(","):
        item = item.strip()
        if item:
            validated_files.append(item)

with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

metrics["validated_files"] = validated_files
metrics["mode"] = mode
metrics["model_path"] = model_path
metrics["output_path"] = output_path

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, sort_keys=True)
PY
fi

if [ "$ALSO_TIMESTAMPED" = "true" ]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  out_dir="$(dirname "$OUTPUT_PATH")"
  out_name="$(basename "$OUTPUT_PATH")"
  out_base="${out_name%.parquet}"
  ts_output="${out_dir}/${out_base/_latest/}_${stamp}.parquet"
  cp -f "$OUTPUT_PATH" "$ts_output"
  echo "[run.sh] Timestamped output: $ts_output"

  if [ -n "$METRICS_OUT" ] && [ -f "$METRICS_OUT" ]; then
    metrics_dir="$(dirname "$METRICS_OUT")"
    metrics_name="$(basename "$METRICS_OUT")"
    metrics_base="${metrics_name%.json}"
    ts_metrics="${metrics_dir}/${metrics_base/_latest/}_${stamp}.json"
    cp -f "$METRICS_OUT" "$ts_metrics"
    echo "[run.sh] Timestamped metrics: $ts_metrics"
  fi
fi

echo "[run.sh] Output: $OUTPUT_PATH"
if [ -n "$METRICS_OUT" ]; then
  echo "[run.sh] Metrics: $METRICS_OUT"
fi
