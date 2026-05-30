#!/usr/bin/env bash
# matting_bench.sh — GPU-vs-CPU + resolution + thread-count sweep for BiRefNet
# background removal (vision.cpp). Two layers:
#   1. CLI compute timing (pure inference + VRAM, no HTTP)  -> vision-cli birefnet
#   2. End-to-end server timing (resident/warm) + res sweep -> matting-server /remove
#
# Requires a built tree (cmake -DVISP_SERVER=ON ...) and a model + test image.
# Nothing here builds; run it once the device is free.
#
# Usage:
#   MODEL=/path/RMBG-2.0-F16.gguf IMG=/path/photo.jpg ./bench/matting_bench.sh
#
# Env knobs:
#   BUILD     build dir (default: build)
#   MODEL     path to *.gguf BiRefNet model               (required)
#   IMG       path to a test image                         (required)
#   REPS      timed repetitions per config (default 5)
#   RES_LIST  process_res values for server sweep (default "0 512 1024")  [0 = native]
#   THREADS   CPU thread counts to try (default "auto $(nproc) $(($(nproc)/2))")
#   PORT      server port (default 8898)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="${BUILD:-build}"
CLI="$BUILD/bin/vision-cli"; [ -x "$CLI" ] || CLI="$BUILD/src/cli/vision-cli"
SRV="$BUILD/bin/matting-server"; [ -x "$SRV" ] || SRV="$BUILD/src/server/matting-server"
: "${MODEL:?set MODEL=/path/to/model.gguf}"
# Default to the in-repo fur/edge sample so the bench is turnkey.
IMG="${IMG:-$SCRIPT_DIR/../tests/input/cat-and-hat.jpg}"
[ -f "$IMG" ] || { echo "IMG not found: $IMG  (set IMG=/path/to/image)"; exit 1; }
REPS="${REPS:-5}"
PORT="${PORT:-8898}"
NPROC="$(nproc)"
THREADS="${THREADS:-auto $NPROC $((NPROC/2))}"
RES_LIST="${RES_LIST:-512 768 1024 1536 2048}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"; kill "${SRV_PID:-0}" 2>/dev/null || true' EXIT

has_gpu() { command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; }

# VRAM attribution is by PROCESS NAME, not PID — robust when the binary runs
# inside a container (PID namespace differs from the host nvidia-smi view).
VRAM_NAME="${VRAM_NAME:-matting-server}"

# peak VRAM (MiB) for the named process while $1 (a pid) runs
vram_peak_during() { # $1: pid to watch
  local pid="$1" peak=0 used
  while kill -0 "$pid" 2>/dev/null; do
    used="$(vram_now)"
    [ -n "${used:-}" ] && [ "$used" -gt "$peak" ] 2>/dev/null && peak="$used"
    sleep 0.05
  done
  echo "$peak"
}
vram_now() { # instantaneous MiB summed over processes matching VRAM_NAME
  has_gpu || { echo 0; return; }
  nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v n="$VRAM_NAME" 'index($1,n){s+=$2} END{print s+0}'
}

# ----- Layer 1: CLI compute timing (pure inference + VRAM at native res) ------
echo "=== CLI compute (model=$(basename "$MODEL") img=$(basename "$IMG"), native res) ==="
printf '%-8s %-6s %-12s %-10s\n' backend rep infer_ms vram_MiB
cli_run() { # $1 backend
  local be="$1"
  for r in $(seq 1 "$REPS"); do
    "$CLI" birefnet -b "$be" -m "$MODEL" -i "$IMG" -o "$TMP/m.png" >"$TMP/log" 2>&1 &
    local pid=$!; local vram; vram="$(vram_peak_during "$pid")"; wait "$pid" || { cat "$TMP/log"; continue; }
    local ms; ms="$(grep -oiE 'complete \([0-9.]+ ms\)' "$TMP/log" | grep -oE '[0-9.]+' | head -1)"
    printf '%-8s %-6s %-12s %-10s\n' "$be" "$r" "${ms:-?}" "${vram:-0}"
  done
}
cli_run cpu
has_gpu && cli_run gpu

# ----- Layer 2: end-to-end server (warm) + threads + process_res sweep --------
echo; echo "=== Server /remove (warm, REPS=$REPS) ==="
printf '%-8s %-8s %-8s %-12s %-12s %-10s\n' backend threads res first_ms warm_ms vram_MiB

post_warm() { # $1 res -> echoes "first warm"
  local res="$1" q="" first sum=0 e
  [ "$res" != 0 ] && q="?process_res=$res"
  first="$(curl -s -o "$TMP/o.png" -D - --data-binary @"$IMG" \
            "http://127.0.0.1:$PORT/remove$q" | awk -F': ' '/X-BG-Elapsed-Seconds/{print $2*1000}' | tr -d '\r')"
  for _ in $(seq 1 "$REPS"); do
    e="$(curl -s -o /dev/null -D - --data-binary @"$IMG" \
          "http://127.0.0.1:$PORT/remove$q" | awk -F': ' '/X-BG-Elapsed-Seconds/{print $2*1000}' | tr -d '\r')"
    sum="$(awk -v a="$sum" -v b="${e:-0}" 'BEGIN{print a+b}')"
  done
  echo "$first $(awk -v s="$sum" -v n="$REPS" 'BEGIN{printf "%.1f", s/n}')"
}

server_run() { # $1 backend  $2 threads
  local be="$1" thr="$2" thr_arg=()
  [ "$thr" != auto ] && thr_arg=(--threads "$thr")
  MATTING_BACKEND="$be" "$SRV" --model "$MODEL" --port "$PORT" --backend "$be" "${thr_arg[@]}" \
      >"$TMP/srv.log" 2>&1 & SRV_PID=$!
  for _ in $(seq 1 150); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 0.2; done
  # background peak-VRAM sampler for this server's lifetime (cumulative max;
  # captures the inference peak, not just the idle-loaded baseline).
  : > "$TMP/peak"; echo 0 > "$TMP/peak"
  ( pk=0; while kill -0 "$SRV_PID" 2>/dev/null; do
      u="$(vram_now "$SRV_PID")"
      [ -n "${u:-}" ] && [ "$u" -gt "$pk" ] 2>/dev/null && { pk="$u"; echo "$pk" > "$TMP/peak"; }
      sleep 0.1
    done ) & local SAMP=$!
  for res in $RES_LIST; do
    read -r first warm < <(post_warm "$res")
    local vpk; vpk="$(cat "$TMP/peak" 2>/dev/null || echo 0)"
    printf '%-8s %-8s %-8s %-12s %-12s %-10s\n' "$be" "$thr" "$res" "${first:-?}" "$warm" "${vpk:-0}"
  done
  kill "$SAMP" 2>/dev/null || true
  kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; SRV_PID=
}

for thr in $THREADS; do server_run cpu "$thr"; done
has_gpu && server_run gpu auto

echo; echo "done."
