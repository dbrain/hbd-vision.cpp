# BiRefNet / RMBG-2.0 matting — PROFILE (Phase 1)

Model: `models/RMBG-2.0-F16.gguf` (Swin-L encoder + ASPP/deformable decoder).
Input: `tests/input/cat-and-hat.jpg` @ 1024x1024. FA off (`VISP_FLASH_ATTENTION=0`).
Box: RTX 3060 (12 GB), 12-core CPU, builder image `flux2-dev:builder`.
dbrain/ggml `depend/ggml` @ 9ae8f517 (ships a CUDA CONV_2D_DEFORM kernel).

## Method

Op-level profiler added to `src/visp/ml.cpp` `compute()`, gated by env
`VISP_PROFILE_OPS=<csv>`. Uses `ggml_backend_sched_set_eval_callback`; the
backend is synchronized before and after every node so each op is timed in
isolation. For the CPU (non-sched) path a temporary single-backend
`ggml_backend_sched` is built so the same callback applies. Zero cost when the
env var is unset (no callback installed).

Per-node sync inflates absolute wall slightly (GPU 4.06s real -> 4.01s summed,
CPU 9.9s -> 9.97s summed) — here the summed node time tracks real wall closely
because the dominant ops are large GPU kernels, not launch-bound. Shares (%)
are the trustworthy signal.

Stage boundaries from the CSV: exactly 2 `IM2COL` nodes (the two Swin
`patch_embed` convs) at idx 9 (enc1, 256x256) and idx 177 (enc2, 128x128).
First `CONV_2D_DEFORM` at idx 3403 marks the decoder. Stage cuts: enc1
[0,177), enc2 [177,~3403), decoder [3403,end].

Baseline (un-profiled, correctness-gated):
- GPU @1024: 4061.8 ms, matte YAVG 181.014 (`bench/baseline_gpu.png`).
- CPU @1024: ~9.9 s, matte YAVG 181.014.

## Headline: GPU and CPU bottlenecks are DIFFERENT

| Stage                 | GPU %  | CPU %  |
|-----------------------|--------|--------|
| enc1 (full-res Swin)  | 10.7   | 57.3   |
| enc2 (half-res Swin)  | 2.6    | 15.8   |
| decoder (ASPP+deform) | 86.7   | 26.9   |

**This OVERTURNS the handoff.** The handoff claimed the Swin encoder is the
bottleneck and "deform is NOT the bottleneck" (measured 3.98 vs 4.06s,
deform-on-CPU vs deform-on-CUDA). That A/B is stale: it predates the real
per-op profile. On **GPU the decoder (deformable + ASPP convs) is 86.7%** of
the time; the Swin encoder is only ~13%. On **CPU the Swin encoder is 73%** and
the decoder 27% — the opposite. The earlier "no diff" A/B happened because the
deform op cost ~the same on CPU and on the early CUDA kernel; what it actually
showed is that the deform op is the cost on BOTH, not that it's negligible.

## Top ops — GPU (% of 4.01 s)

| op             | us       | %     | count |
|----------------|----------|-------|-------|
| CONV_2D_DEFORM | 1842546  | 45.9  | 20    |
| CONV_2D        | 1580693  | 39.4  | 80    |
| MUL_MAT        | 223227   | 5.6   | 296   |
| REPEAT         | 96562    | 2.4   | 48    |
| ADD            | 71081    | 1.8   | 553   |
| CONT           | 56627    | 1.4   | 310   |
| SOFT_MAX       | 47635    | 1.2   | 48    |
| CONCAT         | 19555    | 0.5   | 50    |
| (UNARY/UPSCALE/MUL/NORM/ROLL/CPY/GET_ROWS/PAD… each <0.5%)   |

**85.3% of GPU time = CONV_2D_DEFORM (45.9%) + CONV_2D (39.4%), all in the
decoder.** Swin's MUL_MAT (the encoder attention) is only 5.6%.

### Deform conv detail (the single biggest sink)

20 deform nodes; grouped by output resolution (4 modules per decoder block —
aspp1 k=1 + 3 aspp_deforms k=1/3/7):

| resolution (HxW)        | total us | n | per-node |
|-------------------------|----------|---|----------|
| 128x128 (256 ch)        | 604033   | 4 | ~151 ms  |
| 64x64  (128 ch)         | 599210   | 4 | ~150 ms  |
| 256x256 (512 ch)        |  43095   | 4 | ~11 ms   |
| 32x32  (64 ch)          |  84926   | 4 | ~21 ms   |
| 16x16  (64 ch)          |   ~12 ms each            |

Key finding: within a block, the k=1, k=3 and k=7 deform modules ALL cost the
same (~150 ms) at a given resolution. **The kernel is NOT tap/FLOP-bound — it
is memory-latency-bound on the per-input-channel gather.** The naive CUDA
kernel (`depend/ggml/src/ggml-cuda/conv2d-deform.cu`) assigns one thread per
output element and loops over all IC input channels doing uncoalesced bilinear
gathers from global memory, re-reading the weight every iteration. At 256
channels that's a long serial dependent-load loop per thread with no reuse.

## Top ops — CPU (% of 9.97 s)

| op             | us       | %     | count |
|----------------|----------|-------|-------|
| MUL_MAT        | 5278358  | 52.9  | 331   |
| CONV_2D        | 1238310  | 12.4  | 45    |
| CONT           | 768166   | 7.7   | 333   |
| ADD            | 722913   | 7.2   | 553   |
| CONV_2D_DEFORM | 510737   | 5.1   | 20    |
| SOFT_MAX       | 331245   | 3.3   | 48    |
| UNARY          | 267641   | 2.7   | 110   |
| CONCAT         | 239755   | 2.4   | 21    |
| UPSCALE        | 181345   | 1.8   | 17    |
| (REPEAT/NORM/PAD/MUL/ROLL… each ~1% or less)               |

CPU is MUL_MAT-bound (53%, the Swin attention/linears). Notably CONV_2D_DEFORM
is only 5.1% on CPU — the CPU deform kernel (multithreaded across 12 cores) is
RELATIVELY cheaper than the single naive CUDA kernel. The GPU deform kernel is
the under-optimized outlier.

## The single biggest time sink

**GPU: the decoder's deformable convolutions** — CONV_2D_DEFORM = 45.9%
(1.84 s), dominated by 8 nodes at 128x128/256ch and 64x64/128ch (~150 ms each,
~1.2 s combined). Memory-latency-bound, not FLOP-bound (see above).

**CPU: the Swin-L encoder attention** — MUL_MAT = 52.9% (5.28 s).

## Phase-2 candidate levers (ranked by the profile)

1. **GPU deform-conv kernel memory access** (targets 45.9% of GPU). The kernel
   is latency-bound on the IC gather. Coalescing the input read / caching the
   tiny weight in registers/shared should give a real win. Lives in SHARED
   dbrain/ggml — must stay bit-exact and not regress other consumers (no other
   prod model uses CONV_2D_DEFORM, so blast radius is BiRefNet-only).
2. **GPU decoder CONV_2D** (39.4%) — large 256-ch 1x1/3x3 convs at 128x128.
   Already cuBLAS/MMQ-backed; less headroom.
3. **CPU thread count / MUL_MAT** — encoder attention is 53% of CPU; scales

---

# Phase 2 — NO optimization landed (honest status)

NOTHING was optimized. An earlier draft of this file claimed a "-25% deform
kernel win" — that was WRONG and has been deleted. The edit targeted a naive
per-thread kernel that does not exist; the real `conv2d-deform.cu` is already a
cooperative shared-memory im2col kernel (one block/output-pixel, smem column,
threads split output channels). The edit silently failed to apply; the rebuild
produced identical numbers (deform still 45.9%, GPU still 4.06s). depend/ggml
is UNCHANGED, clean at 9ae8f517.

Real remaining lever (NOT attempted): the deform kernel uses one block per
output pixel and only `CUDA_CONV2D_DEFORM_BLOCK_SIZE=256` threads, each looping
the full im2col column (up to c_in*7*7=12544 elems) per output channel
serially. The k=7/256-ch node at 256x256 alone is ~1.15s. The dot-product loop
(step 2) has no tiling/warp-reduction and re-reads the smem column per oc with
no register blocking — that is the genuine hot spot to attack, but it is a real
CUDA kernel rewrite in SHARED ggml and was not done this session.

# Verified numbers (profiler OFF)
- GPU @1024: 4061.8 ms, YAVG 181.014.
- CPU @1024: ~9.9 s, YAVG 181.014.

# Only code change kept
- src/visp/ml.cpp: VISP_PROFILE_OPS op-level profiler (zero cost when unset).
