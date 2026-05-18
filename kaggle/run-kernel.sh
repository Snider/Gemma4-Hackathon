#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
KAGGLE_DIR="$ROOT_DIR/kaggle"
METADATA="$KAGGLE_DIR/kernel-metadata.json"
EXAMPLE="$KAGGLE_DIR/kernel-metadata.example.json"
EVIDENCE_DIR="${KAGGLE_EVIDENCE_DIR:-$KAGGLE_DIR/run-evidence}"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}"
TIMEOUT="${KAGGLE_TIMEOUT:-1800}"
POLL_INTERVAL="${KAGGLE_POLL_INTERVAL:-30}"
POLL_LIMIT="${KAGGLE_POLL_LIMIT:-80}"

if [ ! -f "$METADATA" ]; then
  echo "Missing $METADATA"
  echo "Copy $EXAMPLE to $METADATA and replace YOUR_KAGGLE_USERNAME."
  exit 2
fi

KERNEL_ID=$(
  python3 - "$METADATA" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    metadata = json.load(f)
print(metadata.get("id", ""))
PY
)

case "$KERNEL_ID" in
  ""|*YOUR_KAGGLE_USERNAME*)
    echo "Invalid kernel id in $METADATA: $KERNEL_ID"
    echo "Replace YOUR_KAGGLE_USERNAME before running."
    exit 2
    ;;
esac

mkdir -p "$EVIDENCE_DIR"

echo "Pushing $KERNEL_ID with accelerator=$ACCELERATOR timeout=$TIMEOUT"
"$KAGGLE_BIN" kernels push -p "$KAGGLE_DIR" --accelerator "$ACCELERATOR" --timeout "$TIMEOUT" \
  | tee "$EVIDENCE_DIR/push.log"

attempt=1
while [ "$attempt" -le "$POLL_LIMIT" ]; do
  echo "Polling status attempt $attempt/$POLL_LIMIT"
  "$KAGGLE_BIN" kernels status "$KERNEL_ID" | tee "$EVIDENCE_DIR/status.log"
  STATUS_TEXT=$(cat "$EVIDENCE_DIR/status.log")

  case "$STATUS_TEXT" in
    *complete*|*Complete*|*COMPLETE*|*succeeded*|*Succeeded*|*SUCCEEDED*)
      echo "Kernel completed."
      break
      ;;
    *error*|*Error*|*ERROR*|*failed*|*Failed*|*FAILED*)
      echo "Kernel failed; fetching logs."
      "$KAGGLE_BIN" kernels logs "$KERNEL_ID" > "$EVIDENCE_DIR/logs.txt" || true
      exit 1
      ;;
  esac

  attempt=$((attempt + 1))
  sleep "$POLL_INTERVAL"
done

if [ "$attempt" -gt "$POLL_LIMIT" ]; then
  echo "Timed out waiting for $KERNEL_ID after $POLL_LIMIT polls."
  "$KAGGLE_BIN" kernels logs "$KERNEL_ID" > "$EVIDENCE_DIR/logs.txt" || true
  exit 1
fi

"$KAGGLE_BIN" kernels logs "$KERNEL_ID" > "$EVIDENCE_DIR/logs.txt"
"$KAGGLE_BIN" kernels output "$KERNEL_ID" -p "$EVIDENCE_DIR/output"

cat > "$EVIDENCE_DIR/README.md" <<EOF
# Kaggle Run Evidence

- Kernel: https://www.kaggle.com/code/$KERNEL_ID
- Commit: $(git -C "$ROOT_DIR" rev-parse HEAD)
- Accelerator: $ACCELERATOR
- Timeout: $TIMEOUT seconds
- Logs: logs.txt
- Output directory: output/

Copy the final Cell 11 toxic and benign responses from logs.txt into the submission notes.
EOF

echo "Evidence written to $EVIDENCE_DIR"
