#!/usr/bin/env bash
# Measure peak GPU VRAM (nvidia-smi per-process) for a single birefnet inference.
# Usage: measure_vram.sh <label> [extra_env...]
#   extra_env passed as "VAR=val VAR2=val2" before the binary.
# Prints: <label> peak_MiB=<N> wall_ms=<M> YAVG=<Y>
set -u
LABEL="${1:-run}"; shift || true
EXTRA_ENV="${*:-}"
CNAME="vrammeas_$$"
OUT="/src/bench/vm_${LABEL}.png"
HOST_OUT="/home/dbrain/dev/vision.cpp/bench/vm_${LABEL}.png"
LOG="/home/dbrain/dev/vision.cpp/bench/vm_${LABEL}.log"

# Launch inference detached in a named container.
docker run -d --name "$CNAME" --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$HOME/dev/vision.cpp:/src" flux2-dev:builder bash -lc \
  "VISP_FLASH_ATTENTION=0 ${EXTRA_ENV} /src/build/bin/vision-cli birefnet -b gpu \
     -m /src/models/RMBG-2.0-F16.gguf -i /src/tests/input/cat-and-hat.jpg -o ${OUT}" \
  > "$LOG" 2>&1

PEAK=0
# Sample peak until the container exits.
while docker ps --format '{{.Names}}' | grep -q "^${CNAME}\$"; do
  # nvidia-smi reports host PIDs; match by used_memory of any compute app (one GPU job at a time).
  M=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  if [ -n "$M" ] && [ "$M" -gt "$PEAK" ] 2>/dev/null; then PEAK=$M; fi
done

docker logs "$CNAME" >> "$LOG" 2>&1
WALL=$(grep -oE 'detected objects in [0-9]+ ms' "$LOG" | grep -oE '[0-9]+' | tail -1)
docker rm -f "$CNAME" >/dev/null 2>&1

# YAVG via host ffmpeg.
YAVG=""
if [ -f "$HOST_OUT" ]; then
  # NOTE: the matte is a single-channel gray PNG; signalstats needs a YUV input, so
  # convert via yuv444p (format=gray,signalstats yields EMPTY on a gray PNG).
  YAVG=$(ffmpeg -i "$HOST_OUT" -vf "format=yuv444p,signalstats" -f null - 2>&1 | grep -oE 'YAVG:[0-9.]+' | head -1 | cut -d: -f2)
fi
echo "${LABEL} peak_MiB=${PEAK} wall_ms=${WALL} YAVG=${YAVG}"
