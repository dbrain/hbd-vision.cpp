# Matting (BiRefNet/RMBG-2.0) — VERIFIED results only

Host: RTX 3060 (12GB) + Ryzen 5 5600X. Model RMBG-2.0-F16 (fixed 1024px native).
Build: flux2-dev:builder, static (-DBUILD_SHARED_LIBS=OFF -DVISP_STATIC_GGML=ON
-DVISP_SERVER=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86). Run --gpus all.

## Verified (artifact-on-disk + visual/luma checked)
- CPU @1024: **10.8 s**, mask correct (YAVG 181.5, full cat+hat silhouette). bench/VERIFY_cpu.png
- GPU @1024: runs 4.6 s but **mask WRONG** — near-empty (YAVG 5.8). bench/VERIFY_gpu.png
- cast_like bias fix (nn.cpp): CPU correct => fix is good, did not break CPU.
- Scheduler CPU-fallback (ml.cpp): fixes the CONV_2D_DEFORM crash; GPU now runs but
  output is garbage (see hypothesis below).

## Crash fixes (sequence on GPU/CUDA, all real)
1. Mixed-precision bias add (binbcast nb10%sizeof) -> nn.cpp cast_like. FIXED.
2. Flash attention fattn.cu:492 on head_dim=32 -> VISP_FLASH_ATTENTION=0 workaround.
3. CONV_2D_DEFORM "op not supported" on CUDA -> ggml_backend_sched {GPU,CPU} fallback
   in compute_graph_allocate/compute. Crash FIXED, but correctness BROKEN (next).

## GPU output near-empty — FIXED (verified YAVG 181.5, visually correct)
Root cause (confirmed via per-op eval trace): NOT layout/scheduler. The CPU
deformable-conv kernel (ggml-cpu/ops.cpp conv_2d_deform_whcn) hardcodes its GEMM to
GGML_TYPE_F32 with no F16 path. On CPU the model is F32 so it works; on GPU the model
keeps native F16, and when the deform op offloads to CPU those F16 weight bytes get
reinterpreted as F32 -> garbage GEMM -> ASPP decoder explodes to inf -> mask collapses.
Fix: src/visp/nn.cpp conv_2d_deform() casts weight to F32 before the op (no-op on CPU,
lossless F16->F32). GPU mask now YAVG 181.5 == CPU 181.1. bench/VERIFY_gpu.png.

## VERIFIED headline (current binary)
- CPU @1024: ~10.0 s, correct mask.
- GPU @1024: ~4.0-4.6 s, correct mask. ~2.2-2.5x faster than CPU.
  NOTE: modest speedup because CONV_2D_DEFORM runs on CPU (no CUDA kernel) with
  GPU<->CPU copies at the boundary each call. The deform kernel is the obvious next
  optimization target (CUDA kernel or keep whole deform subgraph resident).
- Both fixes (cast_like bias, deform F32 cast, sched CPU-fallback) confirmed correct.

## dbrain/ggml PORT + CUDA deform kernel — DONE (2026-05-30)
- vision.cpp builds against dbrain/ggml (submodule depend/ggml @ 9ae8f517,
  branch consolidated-v0.13 = remote dbrain-ssh/master 048cba4d + matting commits,
  clean linear FF). conv_2d_deform CPU op + ggml_concat_n ported; CUDA kernel added.
- Ported-build gate PASSED: CPU YAVG 181.087 (~10-13.5s), GPU YAVG 181.501 (~4.0-4.8s).

## KEY FINDING — see bench/PROFILE.md (real per-op profile, supersedes the A/B)
The earlier A/B ("deform CPU 3.98 vs CUDA 4.06 → deform not the bottleneck") was a
MISREAD. The real per-op profile (VISP_PROFILE_OPS) shows the OPPOSITE:
- **GPU: decoder = 86.7%** — CONV_2D_DEFORM 45.9% (1.84s) + CONV_2D 39.4% (1.58s).
  Swin encoder only ~13%. Deform IS ~half of GPU time — the A/B just showed our CUDA
  deform kernel is no faster than CPU deform (both ~1.8s), NOT that deform is cheap.
- **CPU: encoder = 73%** — MUL_MAT (Swin attention) 52.9% (5.28s). Decoder only 27%.
GPU and CPU bottlenecks are DIFFERENT. The GPU deform kernel is memory-latency-bound
(per-input-channel gather, no tiling/warp-reduction) — a real CUDA rewrite is the lever
(~46% of GPU), but it's a substantial change in SHARED ggml, NOT attempted this session.
(History of wrong calls on this: '#1 target=deform' then 'NOT deform' then '53%
depthwise' [fabricated] — the PROFILE.md numbers are the trustworthy answer. Measure.)

## ggml-builder tooling gotcha
flux2-dev:builder has NO ffmpeg/ffprobe/python → the handoff's `ffmpeg ... YAVG`
luma check silently returns empty inside the container. Use a tiny stb_image luma
checker, or run ffmpeg on the HOST (host has ffmpeg). Visual Read of the PNG is the
reliable cross-check.

## OPTIMIZATION LANDED — GPU 4061ms → 1639ms (2.48×), matte correct (2026-05-30)
Independently re-verified (my own fresh build + 3 runs: 1642.9/1639.0/1633.8 ms;
visual matte == reference). Two kernels in dbrain/ggml @ af69870c (consolidated-v0.13):
1. **CONV_2D_DEFORM rewrite** (ggml-cuda/conv2d-deform.cu): one-block-per-pixel serial
   dot-product → tiled register-blocked GEMM w/ fused deformable im2col (BM=128 ch ×
   BN=64 px, BK=16, 8×4 reg tile). Bilinear gather done once per (pixel,k) + reused
   across 128 ch; gather coalesced across pixels. **1.84s → 57ms (~32×)**. Bit-faithful
   to CPU ref (YAVG 181.5).
2. **1×1-conv-as-mul_mat fast path** (vision.cpp src/visp/nn.cpp conv_2d whcn branch):
   decoder's 256-ch 1×1 convs were on ggml_conv_2d_direct (1-thread/output, no reuse) —
   NOT cuBLAS as PROFILE.md guessed. Reroute 1×1 → permute C-inner → mul_mat (cuBLAS/MMQ)
   → permute back. **CONV_2D 1.58s → 895ms (−44%)**. BiRefNet-only call site, zero blast
   radius on shared CONV_2D kernel.
Op shares now: CONV_2D 895ms (largest), MUL_MAT 256ms, deform 57ms. CPU unaffected (9.9s).
Ceiling: deform near floor; remaining headroom = 3×3 decoder convs still on conv2d_direct
+ the permute CONTs from the 1×1 path. Diminishing — cheap safe wins banked.
⚠️ Numbers self-verified on disk (not agent-reported) per session anti-fabrication rule.

## TWO FINAL LEVERS LANDED — GPU 1625ms → 860ms (−47%), CPU default tuned (2026-05-30)
Self-verified (fresh incremental build, 3 GPU runs 861.4/860.6/859.5 ms; host-ffmpeg
YAVG 181.557 == baseline 181.6; visual matte == reference). ggml UNCHANGED at af69870c
(both ~/dev/ggml and depend/ggml in sync, 0 dirty) — both levers are vision.cpp-side only.

### LEVER 1 (GPU) — k>1 decoder convs: ggml_conv_2d_direct → im2col+mul_mat
src/visp/nn.cpp conv_2d() whcn branch, the k>1 `else`: was `ggml_conv_2d_direct`
(1-thread/output, loops IC*KH*KW with no reuse). Rerouted to `ggml_conv_2d`
(im2col + ggml_mul_mat → cuBLAS/MMQ on CUDA) — same idea as the 1×1 fast path but
with a real im2col GEMM. BiRefNet-only call site; shared CONV_2D CUDA kernel untouched.
- GPU baseline (current binary, pre-lever): ~1625 ms (1627/1624/1625), YAVG 181.6.
- GPU after lever 1: **~860 ms** (3 runs 864.8/858.0/858.9), YAVG 181.557. **−765 ms / −47%.**
- Op shares (VISP_PROFILE_OPS, per-node-synced; absolute us include sync overhead but
  shares are sound): the 45 k>1 decoder convs left the CONV_2D op entirely — CONV_2D
  (895ms/80 pre-lever) no longer appears in the post-lever profile; that work is now
  **MUL_MAT 275ms/376** (was 256ms; +the conv GEMMs) + new **IM2COL 104ms/45** +
  **REPEAT 102ms/48**. The slow 1-thread/output direct kernel is gone.
- CONT 72ms/427, ADD 71ms, CONV_2D_DEFORM 57ms/20 (unchanged — the prior win).

### LEVER 2 (CPU) — thread default = physical cores (was hw-2)
src/visp/ml.cpp backend_init(): default was `hw_concurrency - 2` (=10 on a 5600X).
Changed to `hw/2` (=6 physical cores) + new `VISP_CPU_THREADS` env override. The
matting-server already had MATTING_THREADS/--threads (it calls backend_set_n_threads
after backend_init when >0), so the new env mainly governs the CLI + the server's
default-0 case. Sweep (RMBG-2.0-F32, cat-and-hat @1024, 2 runs each; all YAVG 181.557):

| threads | run1 (ms) | run2 (ms) |
|---------|-----------|-----------|
| 4       | 12737.4   | 12752.1   |
| **6**   | **9910.4**| **9935.1**| ← knee (physical cores)
| 8       | 10175.4   | 10188.0   |
| 12      | 10153.2   | 10137.0   | (all logical / SMT — slower than 6)

Knee = 6 = physical cores, as predicted (GEMM is BW/latency-bound, SMT hurts).
6 wins clearly; 8 and 12 are ~2.5% slower, 4 is ~28% slower. New default 6 is optimal.

### FINAL VERIFIED NUMBERS (binary with BOTH levers)
- GPU @1024: **~860 ms** (was ~1625), YAVG 181.557 (correct). bench/FINAL_gpu.png
- CPU @1024: **~9.9 s** @ 6 threads default (~9910 ms; was ~9.9s @ 10 threads — 6
  ties/edges the old default and avoids the SMT-thrash of the old hw-2=10), YAVG 181.557.
  bench/FINAL_cpu.png
- ggml af69870c unchanged; depend/ggml in sync. matting-server target rebuilds clean.

### Honest ceiling
- LEVER 1: real, banked, bigger than expected (−47%). The k>1 convs left CONV_2D
  entirely; that work is now MUL_MAT (275ms/376) + IM2COL (104ms/45) — the cuBLAS/im2col
  path, near the floor for these decoder convs without a fused-im2col custom kernel
  (a shared-ggml rewrite, out of scope). CONT (72ms) is structural: the 1×1 reshape
  needs a contiguous operand and birefnet.cpp's residual conts are load-bearing
  (binbcast assert). Reducing CONT needs
  a layout-tracking refactor (defer permute-back when the next op re-permutes) — invasive
  and risky on shared correctness, NOT done. Diminishing returns from here.
- LEVER 2: at the knee. 6 threads optimal; nothing more from thread count alone.

## GPU VRAM TUNING — PLANNED, GATED on CPU pass (skip if CPU fast enough)
If the deep-CPU pass makes CPU fast enough, CPU-ungated wins (no VRAM pressure, always
available) and this is moot. Otherwise, TWO ORDERED STEPS:
1. FREE SAVINGS FIRST — VRAM-tune the current 866ms config WITHOUT losing speed. Hunt
   waste in the ~3GB transient scratch: ggml compute-buffer over-reserve, im2col scratch
   not reused across the 2 encoder passes, gallocr/sched reserve slop, alignment. Pure win.
2. ONLY IF still too high — the speed↓/VRAM↓ ladder: env-gate nn.cpp conv_2d so one binary
   measures all-direct(~4061ms) / +1×1(~1639ms) / +k>1(~866ms) peak VRAM (intermediate
   VRAM currently UNMEASURED — only final 524 resident / 3580 peak is known). Selectively
   revert the biggest-im2col convs to direct → min-VRAM at acceptable speed, target
   ~1-1.5GB peak so it sits resident alongside the LLM. Hypothesis: high-res k>1 im2col is
   VRAM-dominant; 1×1 mul_mat cheap (prove, don't assume).

## GPU 866ms → ~700ms — scheduler removal (verified 2026-05-30)
The deep-CPU agent's `git checkout` accidentally removed the ml.cpp ggml_backend_sched
{GPU,CPU} CPU-fallback (compute()/compute_graph_allocate back to single-backend gallocr).
This turned out to be a NET WIN, not damage: the sched-fallback was OBSOLETE once dbrain/ggml
gained the CUDA CONV_2D_DEFORM kernel (af69870c) — the deform op now runs natively on GPU,
so the fallback was only adding GPU↔CPU copies. Single-backend GPU: **~700 ms** (701/698/699/
700, self-verified fresh build) vs 866ms with the sched. Matte correct (visual == reference).
bench/POSTREVERT_gpu.png. ml.h still declares cpu_fallback/sched members (now unused, harmless).
LOST in the same checkout: VISP_PROFILE_OPS profiler (dev tool, low stakes — re-add if needed).
Session GPU total: 4061ms → **700ms = 5.8×**.

## NOT yet measured (next)
- VRAM (GPU), process_res sweep, server /remove e2e, ncu profile.

## PyTorch reference (briaai/RMBG-2.0) — MEASURED 2026-05-30
Ran the original PyTorch RMBG-2.0 (bg_server.py's `process_rmbg` math) on
10.0.0.151 (`pleader`). NOTE: that box is an **AMD Radeon 890M iGPU / ROCm-HIP**
stack, NOT the RTX 3060 — the 3060 is the porting *target* in HANDOVER_RTX3060.md,
not this host. Reproduced pre-proc torchvision-free (PIL bilinear Resize→1024,
ImageNet normalize, sigmoid). Reference mask pulled to `bench/pytorch_ref_mask.png`.

| device | process_res | sec/img (median fwd) | matte YAVG |
|---|---|---|---|
| CPU (AMD) | 1024 | 10.20 (min 9.68, warmup excluded, 3 iters) | 181.06 |
| HIP GPU | 1024 | not measured (ROCm torch wheel exceeded /tmp tmpfs quota) | — |

### C++ vs PyTorch parity (cat-and-hat, 512×512 grayscale matte)
| C++ matte | mean-abs-diff | max | p99 | within 5 levels | Pearson r |
|---|---|---|---|---|---|
| VERIFY_gpu.png | 1.13 | 254 | 17.0 | 98.4% | 0.9934 |
| VERIFY_cpu.png | 1.14 | 254 | 16.0 | 98.4% | 0.9934 |
| cpu_ref.png    | 1.14 | 254 | 16.0 | 98.4% | 0.9934 |

PASS — PyTorch ref YAVG 181.06 == C++ ~181; 98.4% of pixels within 5/255,
r=0.993. The C++ port faithfully reproduces the original model's matte. (max=254
is a few edge/halo pixels: F16-vs-F32 + bilinear resize-back, not structural.)

## MEMORY — MEASURED 2026-05-30 (was never measured before; one earlier "730MiB" was fabricated/retracted)
GPU F16 model @1024, RTX 3060 (12GB), measured by nvidia-smi (VRAM, by process) +
/proc VmHWM (host RSS), peak over a multi-image burst (3 independent samplers agreed):
- **Peak VRAM: ~3580 MiB** (steady ~3218 MiB mid-run; ~29% of the 12GB card — fits
  comfortably alongside the rest of the stack, but NOT tiny).
- **Peak host RSS: ~1593 MiB** (VmHWM). The static CUDA binary is itself ~846 MB
  (statically-linked ggml-cuda), so roughly half of host RSS is the binary image, not data.
- process_res scales peak (bigger res = bigger activation/im2col buffers); 1024 is native/default.
- ⚠️ CORRECTION: a first draft said "1605 MiB" — typo of a guess. Measured peak ~3580 MiB.

### VRAM PHASE DECOMPOSITION — MEASURED (lazy-load server, by process_name)
| phase | VRAM | component |
|-------|------|-----------|
| server up, model NOT loaded | **0 MiB** | CUDA context is lazy (nothing until first use) |
| model loaded, idle | **524 MiB** | F16 weights (~440MB) + CUDA/cuBLAS context |
| peak during inference | **~3580 MiB** | + activation / im2col / GEMM scratch |
=> **~3GB of the 3.5GB peak is TRANSIENT inference scratch, NOT weights.** Resident
footprint is only 524 MiB. The scratch is largely self-inflicted by the im2col+GEMM
speed opts (1×1 + k>1 convs materialize big im2col matrices the old direct kernels
didn't). KEY: the scratch is the lever — selectively reverting the biggest-im2col convs
to direct, or tiling the GEMM, could cut peak toward ~1-1.5GB while keeping most of the
866ms speed → would make GPU-resident-alongside-the-LLM viable (vs CPU-ungated fallback).
(Supersedes the earlier guessed "440+300+700" breakdown, which was wrong + didn't sum.)
- Self-verified GPU @1024 (independent rebuild, 6 runs 863-872ms): ~866ms. bench/FINAL_verified_866ms.png

## CPU DEEP-PERF PASS — 2026-05-30 (deploy is CPU-only, GPU peak VRAM ~3.5GB too big for shared card)
Host: Ryzen 5 5600X (Zen3, 6c/12t), builder flux2-dev:builder. Model RMBG-2.0-F16.
Baseline (verified, this pass): CPU @1024 F32 = ~9.88 s, YAVG 181.087, clean cat+hat silhouette.

### Lever 1 — ISA flags: ALREADY OPTIMAL (no win)
GGML_NATIVE=ON on x86 takes the `-march=native` path; the per-feature GGML_AVX2/FMA/F16C
cache vars stay OFF but are INERT (native path doesn't set them). Verified the ggml-cpu .o
flags (flags.make + compile_commands.json) = `-O3 -march=native`, and inside the builder
`-march=native` resolves to `-march=znver3 -mavx -mavx2 -mfma -mf16c`. So the 9.88 s baseline
is already a full Zen3 AVX2+FMA+F16C build. No 2-4× hiding here; the earlier hope is dead.

### Lever 2 — quantization (Q8_0 / Q4_K): DEAD END as a blanket lever
Tried `preferred_float_type()` returning Q8_0/Q4_K so decoder linears use ggml's AVX2
quant dot-product. CRASHES at model load: `ggml-cpu/ops.cpp:321 not implemented` — the
blanket cast also tries to quantize the conv2d / deformable-conv weights, and ggml's CPU
cast to a k-quant/Q8_0 of those shapes is unimplemented -> abort. A working quant lever
needs per-tensor selectivity (quantize only 2D matmul weights, keep conv/deform F32) — a
larger change using the model's conv2d_weights index. Left as a documented follow-up; the
crashing knob was removed (no degraded matte ever shipped).

### Lever 3 — BLAS (OpenBLAS): not pursued to a number
Builder image has no OpenBLAS; a throwaway image + GGML_BLAS=ON CPU build was scaffolded but
not landed. Given Lever-1 already gives the full AVX2 MMQ kernel (which on Zen3 typically
matches/beats OpenBLAS for these sizes — cf. flux2 lap-5/6 finding), deprioritized vs Lever 4.

### Lever 4 — process_res sweep: THE CPU WIN (opt-in, quality varies — visual-checked)
New env `VISP_BIREFNET_RES` (birefnet.cpp birefnet_detect_params) overrides inference res;
server already exposes the same knob as `process_res`. REAL measured CPU sec/img (2-3 runs),
matte YAVG, and SSIM of the mask vs the 1024 F32 reference. All three were Read visually and
are clean cat+hat silhouettes (bench/FINAL_cpu_default.png, R768.png, R512.png):

| res  | CPU time (runs)        | YAVG    | SSIM vs 1024 | note                              |
|------|------------------------|---------|--------------|-----------------------------------|
| 1024 | 9.89 / 9.89 / 9.91 s   | 181.087 | 1.000        | native default (KEPT)             |
| 768  | 5.09 / 5.23 s          | 167.848 | 0.908        | -47%, clean silhouette; cat lower-body inclusion differs from 1024 |
| 512  | 2.48 / 2.46 / 2.45 s   | 184.018 | 0.978        | -75%, clean silhouette, slightly softer edges |

IMPORTANT (anti-fabrication): the SSIM/YAVG do NOT monotonically degrade — 512 has HIGHER
SSIM-vs-1024 (0.978) than 768 (0.908). That is because lowering res shifts genuine
segmentation *decisions* (at 768 the cat's lower body is cut differently than at 1024), not
just edge sharpness — so "SSIM vs 1024" is not a pure quality metric here. The trustworthy
check is the visual one: all three are clean, usable mattes. Treat res as a real
speed/quality *tradeoff* the user opts into per request, NOT free speed. Default stays 1024.

### Lever 5 — encoder 2x pass / CONT: not separately optimized (encoder is 73% of CPU, structural).

### FINAL
- Best CPU at FULL (1024-native) quality: **9.89 s, YAVG 181.087** — UNCHANGED. The model is
  already on the Zen3 AVX2/FMA floor (Lever 1 proved the build is already `-march=znver3`
  with AVX2+FMA+F16C; Lever 2 blanket-quant crashes at load). No safe full-quality speedup found.
- Biggest CPU lever = process_res, but it is a real tradeoff (segmentation shifts), not free:
  **768 ≈ 5.1 s (-47%)**, **512 ≈ 2.46 s (-75%)** — both visually clean. Expose as opt-in
  `process_res`; recommend 768 for a good balance, 512 for speed-first, 1024 when matte
  fidelity matters most. Do NOT change the default away from 1024.
- Honest ceiling at 1024/full-quality: 9.89 s is the floor without a real per-tensor-selective
  quant kernel (quantize only the 2D matmul weights, keep conv/deform F32) — the one untried
  real lever, deferred as it needs the conv2d_weights-index plumbing.
- ggml UNTOUCHED: ~/dev/ggml and depend/ggml both @ af69870c, 0 dirty (all changes vision.cpp-side:
  birefnet.cpp VISP_BIREFNET_RES + ml.cpp thread default = physical cores + VISP_CPU_THREADS).

## CPU LOSSLESS PERF PASS #2 — 2026-05-30 (no win found; F16-weights crashes, OpenBLAS slower)
Host: Ryzen 5 5600X (Zen3, 6c/12t), builder flux2-dev:builder, 6 threads default.
Baseline reproduced THIS pass (2 runs): F32-CPU @1024 = 9893 / 9951 ms, YAVG 180.974,
clean cat+hat silhouette. Matches PROFILE.md's 9.89s. ggml af69870c (both trees in sync).
Goal: faster CPU at ZERO quality loss (no quant). Both candidate lossless levers FAILED.

### DEAD — F16 weights on CPU (VISP_CPU_F16=1): SEGFAULTS at load, reverted
Tried preferred_float_type() returning GGML_TYPE_COUNT for CPU when VISP_CPU_F16=1, so the
RMBG-2.0-F16.gguf keeps native F16 weights (idea: ggml CPU MUL_MAT upcasts F16->F32 per op,
lossless, but halves weight BW on the BW-bound Swin GEMMs). Built clean (ml.cpp.o recompiled,
binary relinked, grep-confirmed). Result: **SIGSEGV (exit 139)** during model load — the run
dies right after "found 1 CUDA devices", before "Initializing backend... done" / weight load
finishes. Root cause: the CPU path uses preferred_layout()==cwhn, and the whcn->cwhn weight
conversion + several CPU ops are written assuming F32 weights (same class of failure the handoff
documented for the F32-keep rationale: "not all operations support F16"). Making it work would
need a deep per-op F16 audit in the shared correctness path — out of scope / high risk. Edit
REVERTED (ml.cpp == committed HEAD, grep VISP_CPU_F16 == 0). No degraded matte shipped, no
fabricated number kept. (NOTE: an earlier draft of this section claimed "8688 ms, lossless" —
that was WRONG, carried over from a cancelled command batch; the lever actually crashes.)

### DEAD — OpenBLAS (GGML_BLAS=ON / OpenBLAS): measured ~2% SLOWER, do not deploy
Built throwaway flux2-dev:builder-blas (apt libopenblas-dev) + separate build-blas dir with
-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS (CMake: "Found BLAS: .../libopenblas.so",
"Including BLAS backend"). Prod build dir untouched.
  BLAS F32-CPU (3 runs): 10128 / 10116 / 10110 ms, YAVG 180.974 (correct).
  vs baseline 9893 / 9951 ms  =>  ~+2% SLOWER.
Confirms the flux2 lap-5/6 / memory finding: on Zen3 ggml's own AVX2 MMQ kernel matches/beats
OpenBLAS for these GEMM shapes (BiRefNet's Swin attention/MLP). DEPLOY IMPLICATION: do NOT add
libopenblas to the deploy image — it only slows CPU. Throwaway image + build dir removed after.

### Levers 3/4 (2x encoder, CONT/ADD) — inspected, no lossless redundancy to remove
- Swin encoder runs twice on DIFFERENT inputs (full-res x AND downscale_by(x,2)); BiRefNet's
  multi-scale design. No weight/activation sharing is possible without changing the output.
  downscale_by is a cheap cont->interpolate->cont; nothing recomputed redundantly.
- The 3 decoder `ggml_cont(x1/x2/x3)` before the lateral adds (birefnet.cpp) are CUDA binbcast
  stride-assert guards. On CPU the add has no such assert, so they are pure CPU overhead — but
  they are 3 of 333 CONT nodes (~7ms of the ~0.77s total CONT, <0.1% of wall). Not worth the
  layout-fragility risk on the shared GPU correctness path. Skipped.

### FINAL (this pass) — honest ceiling
Best LOSSLESS CPU @1024/full-quality stays **~9.9 s, YAVG 180.974** — UNCHANGED. No safe
full-quality CPU speedup found:
  - ISA flags already -march=znver3 (AVX2+FMA+F16C) — floor (PROFILE.md / pass #1).
  - Threads at the knee (6 = physical cores) — floor (pass #1).
  - Quant (Q8_0/Q4_K) — off the table (quality) AND crashes at load (pass #1).
  - F16 weights on CPU — crashes (this pass).
  - OpenBLAS — measured slower (this pass).
The model is MUL_MAT-bound (53%, Swin attention) on ggml's AVX2 MMQ kernel, which is already the
fast path for these F32 GEMM shapes. The remaining real lever is the documented per-tensor-
selective quant (quantize only 2D matmul weights, keep conv/deform F32) — but that is NOT
lossless (it's quantization) so it is out of scope for this zero-quality-loss task.
ggml UNTOUCHED (af69870c, both ~/dev/ggml and depend/ggml). src/visp/ml.cpp == committed HEAD.
