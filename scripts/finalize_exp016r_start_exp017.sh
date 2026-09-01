#!/bin/bash
set -euo pipefail
umask 027

ROOT=/opt/data/instance_gpu_3/fade-net-runtime
LOG_DIR="$ROOT/logs"
EXP016_PID_FILE="$LOG_DIR/launch_exp016r_input224_host_gpu3.pid"
EXP016_STATUS="$LOG_DIR/train_fade_net_exp016r_input224_host_gpu3.status"
EXP016_RESULTS="$ROOT/outputs/fade_net_exp016r_input224_host_gpu3/fold0/results.json"
EXP017_SCRIPT="$ROOT/FADE-Net/scripts/run_exp017_seed1337_host_gpu3.sh"
EXP017_LAUNCH_LOG="$LOG_DIR/launch_exp017_seed1337_host_gpu3.nohup"
EXP017_PID_FILE="$LOG_DIR/launch_exp017_seed1337_host_gpu3.pid"
FINALIZER_STATUS="$LOG_DIR/finalize_exp016r_start_exp017.status"
PYTHON=/opt/data/instance_gpu_3/fade-net-env/bin/python

EXP016_PID=$(cat "$EXP016_PID_FILE")
while kill -0 "$EXP016_PID" 2>/dev/null; do
  sleep 15
done

if [ ! -f "$EXP016_STATUS" ] || [ ! -f "$EXP016_RESULTS" ]; then
  printf 'decision\tERROR_MISSING_EXP016_ARTIFACTS\n' > "$FINALIZER_STATUS"
  exit 10
fi

EXIT_CODE=$(awk -F '\t' '$1 == "exit_code" {print $2}' "$EXP016_STATUS")
if [ "$EXIT_CODE" != "0" ]; then
  printf 'decision\tERROR_EXP016_EXIT_%s\n' "$EXIT_CODE" > "$FINALIZER_STATUS"
  exit 11
fi

BEST_VAL=$(
  "$PYTHON" - "$EXP016_RESULTS" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)

test_keys = (
    "test_mae",
    "test_base_mae",
    "evaluation_model",
    "raw_test_mae",
    "raw_test_base_mae",
    "ema_test_mae",
    "ema_test_base_mae",
)
non_null = {key: result.get(key) for key in test_keys if result.get(key) is not None}
if non_null:
    raise SystemExit(f"Test isolation violated: {non_null}")
if result.get("config", {}).get("skip_final_test") is not True:
    raise SystemExit("skip_final_test was not recorded as true")
print(float(result["best_val_mae"]))
PY
)

if "$PYTHON" - "$BEST_VAL" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) < 3.270 else 1)
PY
then
  {
    printf 'decision\tEXP016R_PROMOTED_TO_VAL_TTA\n'
    printf 'best_val_mae\t%s\n' "$BEST_VAL"
  } > "$FINALIZER_STATUS"
  exit 0
fi

if [ -e "$EXP017_LAUNCH_LOG" ] || [ -e "$EXP017_PID_FILE" ]; then
  printf 'decision\tERROR_EXP017_LAUNCH_ARTIFACT_EXISTS\n' > "$FINALIZER_STATUS"
  exit 12
fi

nohup bash "$EXP017_SCRIPT" > "$EXP017_LAUNCH_LOG" 2>&1 < /dev/null &
EXP017_PID=$!
printf '%s\n' "$EXP017_PID" > "$EXP017_PID_FILE"
sleep 3
if ! kill -0 "$EXP017_PID" 2>/dev/null; then
  printf 'decision\tERROR_EXP017_FAILED_TO_START\n' > "$FINALIZER_STATUS"
  exit 13
fi

{
  printf 'decision\tEXP016R_REJECTED_EXP017_STARTED\n'
  printf 'best_val_mae\t%s\n' "$BEST_VAL"
  printf 'exp017_wrapper_pid\t%s\n' "$EXP017_PID"
} > "$FINALIZER_STATUS"
