# pioneerML_inference

C++ inference runner for TorchScript models. Uses the C++ dataloaders input/output adapters.

## Build

```bash
./scripts/build.sh
```

## Run (group classifier)

```bash
./build/pioneerml_inference \
  --mode group_classifier \
  --model /path/to/group_classifier.ts \
  --input /path/to/ml_output_000.parquet \
  --output /path/to/predictions.parquet \
  --device cpu
```

Use `--input` multiple times to pass multiple parquet shards.

Optional config JSON:

```bash
./build/pioneerml_inference \
  --mode group_classifier \
  --model /path/to/group_classifier.ts \
  --input /path/to/ml_output_000.parquet \
  --output /path/to/predictions.parquet \
  --config /path/to/loader_config.json
```
