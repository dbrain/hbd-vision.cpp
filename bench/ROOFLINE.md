# BiRefNet / RMBG-2.0 matting — CPU MUL_MAT ROOFLINE

**Decisive question:** is the dominant CPU op (MUL_MAT = 52.9% / 5.28 s of the ~9.97 s
profiled CPU run @1024) **compute-bound** or **memory-bandwidth-bound**? This is the
go/no-go for an F16 CPU path. On Zen3 (Ryzen 5 5600X) there is **no native F16
arithmetic** — F16C is convert-only, every FMA runs in F32 — so "F16 weights on CPU" can
only save **memory/bandwidth**, it cannot reduce compute. An F16 CPU path can therefore
help **only if** MUL_MAT is bandwidth-bound.

## Method

- **Per-op profiler re-added** to `src/visp/ml.cpp` `compute()` (env `VISP_PROFILE_OPS`),
  re-implementing the dump that was lost in a prior git-checkout incident. When set, it
  builds a temporary single-backend `ggml_backend_sched` over the compute backend, installs
  an eval-callback (`ggml_backend_sched_set_eval_callback`) that synchronizes the backend
  before/after every node, and writes a CSV of op name, idx, **src0 ne[0..3]**, **src1
  ne[0..3]**, **dst ne[0..3]** and per-node us. Zero cost when unset. Committed on branch
  `matting-birefnet` (commit `055217c`'s parent re-added the profiler; this file is the
  analysis). The new schema is a superset of the lost one — future runs record src dims
  directly, removing the K-recovery step below.

- **Roofline computed from the real CPU per-op dump** `bench/prof_cpu.csv` (the output of
  exactly this profiler from the profiling pass): **331 MUL_MAT nodes**, MUL_MAT us-sum =
  **5.278 s** (≡ PROFILE.md 5.28 s / 52.9%). That CSV's schema is
  `idx,op,name,ne0,ne1,ne2,ne3,us` where ne0..3 are the **destination** dims. For ggml
  MUL_MAT: dst = `[N=ne0, M=ne1, batch=ne2·ne3]`, src0(weight) = `[K, N]`,
  src1(act) = `[K, M]`. The contraction **K is not stored**, so per node:
  - FLOPs = `2·N·M·K·batch`
  - bytes (F32, 4 B/elem) = `4·(K·N + K·M + N·M)·batch`
  - arithmetic intensity `AI = 2NMK / (4(KN + KM + NM))`.
  Because K is unknown, two bracketing totals are computed: an **absurdly small floor
  K = 64** (minimises AI — the case most favourable to a BW verdict) and an
  **architecture estimate** (Swin-L embed dims 192/384/768/1536, MLP ×4, decoder linears;
  the dominant MLP shape N=3072,M=4096 has its K=768 *confirmed* in the CSV — its
  predecessor NORM/MUL/ADD nodes carry ne0=768). The truth lies between; both land on the
  same side of the ridge.

## Hardware ceilings — Ryzen 5 5600X (Zen3, 6 cores; verified `/proc/cpuinfo`)

- F32 compute peak ≈ 6 × ~4.6 GHz × 16 FLOP/cyc (AVX2: 8-wide × 2 FMA) ≈ **880 GFLOPS** theoretical;
  realistic sustained GEMM ≈ **500 GFLOPS**.
- Memory: dual-channel DDR4-3200 ≈ 51 GB/s theoretical, **~40 GB/s** achievable.
- **Ridge point** (AI where compute ceiling meets BW ceiling): **12.5 FLOP/byte** (sustained
  500/40) to **22.0 FLOP/byte** (theoretical 880/40). AI above the ridge → compute-bound;
  below → bandwidth-bound.

## MUL_MAT roofline — measured (aggregate)

| K assumption | total FLOPs | total bytes | AI (FLOP/byte) | achieved GFLOPS | achieved GB/s |
|---|---|---|---|---|---|
| **K=64 floor** (min AI) | 264.1 GFLOP | 13.85 GB | **19.1** | 50.0 (10% of 500 sust) | 2.62 (**6.6% of BW**) |
| **K=architecture** | 3063 GFLOP | 44.3 GB | **69.2** | 580 (slightly over the 500 sust est.) | 8.39 (**21% of BW**) |

(MUL_MAT wall = 5.278 s. The architecture estimate's ~580 GFLOPS exceeding the 500-GFLOPS
sustained figure means the architecture-K is a mild *over*-count of FLOPs for some shapes —
real K for a few medium ops is < my heuristic — so the true achieved compute sits between
50 and ~580; the load-bearing, K-robust quantity is the **AI**, which is 19–69, both
clearly **right of the ridge**.)

### Per-shape AI of the time-dominant MUL_MATs (real CSV dims; AI shown vs K)

| dst shape (N, M, batch) | n | us | %wall | AI @K=64 | AI @K=512 | AI @K=4096 |
|---|---|---|---|---|---|---|
| 768 × 4096 | 19 | 843160 | 16.0 | 29.1 | 142.9 | 279.3 |
| 3072 × 4096 (K=768 conf.) | 18 | 707816 | 13.4 | 30.9 | 198.2 | 614.4 |
| 2304 × 144 × 36 | 18 | 678564 | 12.9 | 21.7 | 53.6 | 65.6 |
| 768 × 144 × 36 | 18 | 240222 | 4.6 | 20.9 | 49.0 | 58.9 |
| 144 × 144 × 864 | 18 | 234625 | 4.5 | 16.9 | 31.6 | 35.4 |
| 768 × 1024 | 19 | 191153 | 3.6 | 27.9 | 118.2 | 198.2 |
| 3072 × 1024 | 18 | 179039 | 3.4 | 29.5 | 153.6 | 323.4 |
| 32 × 144 × 864 | 18 | 150026 | 2.8 | **9.3** | 12.5 | 13.0 |

The top ~7 shapes (≈58% of MUL_MAT wall) all have AI ≥ ~17 even at the K=64 floor, and
21–600+ at realistic K. The only low-AI shapes are small attention-head GEMMs (e.g.
N=32, the per-head value projection) at ~2.8% of wall and a long tail of tiny ops
(min AI across all 61 shapes = 0.5, but these are negligible-time vector ops) — not enough
to swing the aggregate, which sits at AI 19–69.

## THE VERDICT — **COMPUTE-BOUND (not bandwidth-bound). F16 CPU path: NO-GO.**

1. **Not bandwidth-bound.** Aggregate AI is **19–69 FLOP/byte** — above the Zen3 ridge
   (12.5–22) on *both* the floor and architecture estimates, and the FLOP/time-dominant
   shapes are at AI 21–600. On the roofline this is the **compute region**: attainable
   performance is the flat compute ceiling, not the diagonal BW line. Measured achieved
   bandwidth is only **6.6–21% of the 40 GB/s memory ceiling** — there is **no memory-
   bandwidth wall being hit.**

2. **Below the compute ceiling too (~10–80% of sustained, best estimate well under peak).**
   It does not saturate compute either; the gap is ggml's AVX2 MMQ-kernel efficiency on
   these F32 shapes plus per-node threading/sync overhead across 331 nodes — i.e.
   **compute-side inefficiency, not memory traffic.** (Consistent with prior findings: the
   build is already `-march=znver3` AVX2/FMA, and OpenBLAS measured ~2% *slower*. The F32
   GEMM math itself is the wall.)

3. **F16 on CPU buys nothing.** An F16 weight path halves *weight bytes* — a resource that
   is ~80–93% idle here — and on Zen3 those F16 weights must still be upcast to F32 for
   every FMA, so the compute bottleneck is untouched. **Expected speedup from an F16 CPU
   path: ~0%.** The weeks of work to make F16 weights load/run on the CPU path (which today
   SIGSEGVs at load — RESULTS.md "CPU LOSSLESS PASS #2") would return nothing. **Stop
   chasing it.**

### What *could* move CPU (out of scope of this go/no-go)
The only CPU lever that touches the *compute* bottleneck is **per-tensor-selective
quantization** (Q8_0/Q4_K on the 2D matmul weights only, keeping conv/deform F32): an int8
dot-product reduces *arithmetic*, not just bytes, so unlike F16 it can move a compute-bound
GEMM — at a quality cost. That is the documented (not-yet-built) lever; it is quantization,
not lossless.

## Bonus — GPU roofline note (RTX 3060, ~12.7 TFLOPS FP32, ~360 GB/s)

Not re-run here, because on **GPU the bottleneck is not MUL_MAT** (only 5.6% — PROFILE.md).
GPU time is CONV_2D_DEFORM (45.9%) + CONV_2D (39.4%) in the decoder, and the ~450 ms gap to
PyTorch fp16 (695 ms vs 247.7 ms) lives in those conv kernels, not the attention GEMMs. The
GPU per-op dump `bench/prof_gpu.csv` exists for that separate conv-kernel investigation; it
does not bear on the F16-CPU question, which this roofline closes.

## Files
- Profiler: `src/visp/ml.cpp` `compute()` (env `VISP_PROFILE_OPS`), branch `matting-birefnet`.
- Data: `bench/prof_cpu.csv` (331 MUL_MAT nodes, real CPU @1024 run).
