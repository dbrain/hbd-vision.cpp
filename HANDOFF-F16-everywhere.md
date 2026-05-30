# ═══════════════════════════════════════════════════════════════════════════
# OUTCOME (2026-05-31) — VRAM TARGET MET LOSSLESSLY; F16 NOT NEEDED
# ═══════════════════════════════════════════════════════════════════════════
# The mission's GOAL was VRAM (peak 3072 -> ~1.6-2.0 GB so the matting worker can be
# co-resident on the shared 12GB card). F16 was the PROPOSED means. It turned out F16
# is NOT the right means here, and is no longer needed:
#
# WHY F16 was the wrong lever (verified, see "F16 op audit" below / git log):
#   - On CUDA, CONV_2D and MUL_MAT are F32-in/F32-out and im2col ASSERTS F32 input.
#     The decoder head is conv-dominated, so F16-ing a buffer just forces a cast back
#     to F32 at the next conv -> no net peak win without rewriting the shared CUDA
#     conv/im2col/upscale kernels to F16-accumulate-F32 (multi-week, and risky in the
#     shared dbrain/ggml that the prod LLM/avatar/TTS/siglip all link).
#
# WHAT ACTUALLY WORKED — two lossless ALGEBRAIC identities on the 1x1 output head
# (conv_out1.0 is 1x1, 240->1), in src/visp/arch/birefnet.cpp decode():
#   (1) conv1x1(concat[A,B]) == conv_a(A) + conv_b(B)      -> kills [1024,1024,240] concat (960 MiB)
#   (2) conv1x1(upscale(x))  == upscale(conv1x1(x))        -> kills [1024,1024,192] upscale (768 MiB)
#       (1x1 conv is channel-linear, bilinear upscale is spatial-linear -> commute)
#   => GPU PEAK VRAM 3072 -> 1480 MiB (-52%), full 1024 quality, NO precision loss
#      (YAVG 181.416 == baseline; PSNR 78-81 dB vs pre-change = sub-LSB FP order only).
#      Below PyTorch fp16's 2840 MiB peak and the ~1.6 GB active target. Latency ~unchanged.
#   Commits: a7d0ceb (split) + 23a27ad (upscale-commute). Verified on cli AND server.
#
# DEPLOY — worker-isolated service SHIPPED & VERIFIED (commit b469f1c,
#   src/server/birefnet_server.cpp). Parent holds 0 CUDA ctx; model runs in a forked
#   worker over an AF_UNIX socketpair (12-byte length-prefixed frames); SIGKILL on evict
#   = true-0 VRAM. Per-request backend (?backend=cpu|gpu, GPU default, CPU uses 0 GPU);
#   IDLE_UNLOAD_SECONDS=0 default (evict the instant idle) / >0 warm-then-evict /
#   MATTING_KEEP_RESIDENT; serialized 1-at-a-time with an in_flight gate so overlapping
#   requests reuse ONE warm worker (no load/unload churn); batch = multipart N images ->
#   one load, N infers, JSON of base64 PNGs. Timings (RTX 3060): cold load 0.52s, unload
#   0.17s, warm infer ~1.2s e2e (internal birefnet_compute ~1.3s; the docs' "700ms"
#   internal does NOT reproduce on the current binary — flagged, not load-bearing).
#
# DEEPER LEVERS (2026-05-31, all MEASURED on the RTX 3060):
#   - ASPP conv1 split (same identity, 1280->64 1x1 over the 5-way [256,256,1280] concat):
#     peak 1480 -> 1440 MiB (-40), YAVG 181.43, PSNR 60dB. Lossless, committed (c0afb60).
#     conv_1x1_split_sum helper added to birefnet.cpp (reusable for any 1x1-over-concat).
#   - swin rel-pos-mask REPEAT removal (encoder): peak UNCHANGED 1480 -> DISPROVEN, reverted.
#     The encoder is NOT the binding peak (encoder/decoder passes are sequential).
#   - ipt_blk1 full-res path: probe (VISP_NOIPT, skip the path) left peak at 1480 -> DISPROVEN.
#     Costs ~200ms compute but its VRAM is not binding.
#   => After the head fusion the peak is a BROAD PLATEAU of medium (64-256MiB) buffers
#      (deform working sets + persistent 2-pass Swin features + decoder intermediates), not a
#      single giant. Cheap concat-splits net only ~40 MiB each. The cheap lossless bucket is
#      effectively bottomed at ~1440 MiB.
#   - F16 ceiling (measured by op-type, no off-the-shelf conv2d F16 exists — llama.cpp #19505
#     only does transformer activation ops): ~62% of big-buffer traffic is F16-capable
#     (ADD/CONT/CONCAT/UNARY), ~38% F32-locked (MUL_MAT/SOFT_MAX/CONV/DEFORM). Optimistic
#     ceiling ~1000-1100 MiB, i.e. only ~350-450 MiB below 1440, for multi-week custom
#     conv2d/im2col F16 work in the shared fork. Bounded; NOT clearly sub-1GB. (The old
#     "F16 = key to 1.7GB" claim is moot — algebra already beat it.)
#   - PERF breakdown: e2e warm ~1231ms = ~1224ms (IPC+compute, compute dominates, IPC <5ms)
#     + ~7ms (decode+post-proc+PNG+HTTP combined). There is NO fat overhead to trim — it's
#     ~98% GPU graph compute. Big perf wins require better conv/deform kernels (the cuDNN gap,
#     PROFILE.md: deform 46% + conv 39%) = multi-week, exceeds a medium-effort cap.
#   PEAK ANATOMY (graph-truncation probes, MEASURED 2026-05-31 — the decisive map):
#     full-res Swin pass only ......... 1290 MiB   <- THE FLOOR (encoder stage-1 attention)
#     + half-res pass + encode_concat . 1296 MiB   (+6)
#     + squeeze block ................. 1432 MiB   (+136  <- 2nd target, basic_decoder_block)
#     + all 4 decoder blocks + head ... 1440 MiB   (+8    decoder is NOT the peak)
#     => The binding peak is the Swin ENCODER (1290 floor), not the decoder. Tiling the
#        decoder (the original plan B) is FUTILE. Sub-1GB is gated by the Swin encoder.
#   FA-FOR-SWIN is BLOCKED by more than head_dim: fattn.cu:488 returns NONE when mask->ne[2]
#     != 1, and Swin's mask is PER-HEAD ([n,n,n_heads,b]) because the relative-position bias
#     is per-head. head_dim=32 padding to 64 works (lossless, tried) but the per-head mask
#     still disqualifies every FA kernel. => enabling FA needs PER-HEAD additive-mask support
#     added to fattn.cu (load/index mask by head) — a real shared-ggml CUDA change, but a
#     FLEET lever (any per-head-bias attention benefits) and it removes the encoder's [n,n]
#     scores buffers (the 1290 floor) AND speeds the encoder. THIS is the sub-1GB lever.
#   FA NOW WORKS (2026-05-31, SHIPPED opt-in): per-head-mask support added to fattn MMA F16
#     (dbrain/ggml 13f25957) + head_dim 32->64 zero-pad in nn.cpp attention (vision 8f56ad3).
#     VISP_FLASH_ATTENTION=1 -> GPU peak 1440->1294 MiB (-146). Visually-clean matte (YAVG
#     181.2) but F16 K/V -> near-, not bit-lossless (PSNR 24-67 dB by image), so OPT-IN: the
#     worker defaults FA off (lossless 1440); container sets =1 for the 1294 low-VRAM mode.
#   CORRECTED ROADMAP after FA: with FA on, the peak (1294) IS the Swin-encoder full-res pass
#     floor — a TRANSFORMER (layernorm + QKV/MLP mul_mat + softmax), NOT conv. Therefore:
#       * F16 CONV (old item ③) targets DECODER convs, which are BELOW the peak -> it will
#         NOT lower matting's peak (same trap as decoder tiling). Still a FLEET win for
#         conv-heavy models (flux2 etc.), just not for matting VRAM.
#       * The matting sub-1GB lever is F16 TRANSFORMER ACTIVATIONS in the Swin encoder:
#         additive F16 in/out for NORM, SOFT_MAX, SCALE (the F32-locked ops in the encoder),
#         then thread F16 activations through swin.cpp. This is the ggml-org #19505 direction
#         and a FLEET win (the LLM benefits most). Multi-week; ggml-org may land it upstream.
#   REAL-WORLD OOM MATH (user's box): 6.5-7.5GB resident + up to one 4.5GB gated service.
#     Worst case 7.5+4.5 ~= 12GB -> NOTHING fits (even sub-1GB), needs gating/evict-timing.
#     Common case (resident only, ungated slot ~4.5GB free): matting 1294/1440 ALREADY fits.
#     => VRAM returns are diminishing; 1294-1440 is good-enough for the realistic ungated slot.
#   F16 ENCODER PROGRESS (2026-05-31): additive F16 in/out landed in dbrain/ggml for the
#     transformer F32-locked islands — NORM/RMS/L2_NORM, SOFT_MAX, SCALE (4b0d597d) — float
#     internal math, F32 path BYTE-IDENTICAL (matting FA-off PSNR=inf vs pre-change). Fleet
#     win (LLM can use them). BUT threading F16 through Swin is still BLOCKED: the encoder's
#     big remaining buffer is the MLP hidden [768,65536] which is a MUL_MAT *output*, and
#     ggml hardcodes mul_mat dst = F32. Casting that to F16 churns (F32+F16 both live ->
#     no peak win). => the gating prereq is **F16 mul_mat OUTPUT** (a core ggml change).
#     ggml-org is building exactly this (discussion #19505: configurable F16 intermediate
#     precision for the activation path) — TRACK/BORROW from upstream rather than fork-hack
#     the most-used op. Once mul_mat can emit F16, thread F16 from patch_embed (cast conv
#     output F16) through the blocks (norm/softmax/scale/gelu/add all F16-ready now) and
#     measure the encoder floor drop. THIS is the remaining matting sub-1GB lever.
#   SUB-1GB PATHS (all deliberate, multi-day, NOT done; pick in this order):
#     (0) Swin-encoder FA via per-head-mask fattn.cu + head_dim<64 zero-pad (NEW, best:
#         attacks the 1290 floor, fleet-wide VRAM+perf, additive to ggml). Est: removes the
#         per-window [144,144] scores -> encoder floor could drop a few hundred MiB.
#     (B') squeeze-block +136 trim (cheaper, but floor stays 1290 -> not sub-1GB alone).
#  (A) F16 custom conv kernels -> ~1000 ceiling,
#     multi-week, shared-fleet regression risk; (B) decoder spatial tiling -> lossless,
#     birefnet-local, multi-hour, could go lower but real restructure with halo handling.
#
# REMAINING (task 5, better with user awake): docker image for matting-server; wire into
#   kobbler/koblem; push dbrain/ggml consolidated-v0.13 (af69870c, UNPUSHED) + pin submodule;
#   fork -> dbrain/hbd-vision.cpp; remove any temp debug header (current server has none).
# FURTHER VRAM (optional, diminishing): new peak 1480 is distributed 256^2 decoder + 2x Swin
#   encoder buffers (no single giant left). F16 is now only a small marginal play on those;
#   the cheap lossless head wins are banked.
# ═══════════════════════════════════════════════════════════════════════════

# MISSION: F16 activations/compute everywhere — vision.cpp BiRefNet/RMBG-2.0 matting

You are a fresh orchestrator picking up a well-scoped, high-value, multi-week effort.
READ THESE FIRST (all committed on branch `matting-birefnet`):
- `HANDOFF-matting-server.md` — full project state, build/run mechanics, all prior fixes.
- `bench/RESULTS.md` — every verified number (perf, VRAM, PyTorch baseline, roofline).
- `bench/PROFILE.md` — per-op profile (GPU decoder 86.7%: deform 46% + CONV_2D 39%).
- `bench/ROOFLINE.md` — CPU is compute-bound (F16-CPU is dead); GPU gap is conv kernels.
- `bench/VRAM.md` — peak decomposition: 3072 MiB floor = REAL F32 decoder-head activations.

## The one-paragraph why
The matting service is correct (matte == real RMBG-2.0, Pearson r=0.993) and works on GPU
(~700ms @1024) and CPU (~9.9s). But PyTorch fp16 does the SAME model in **248ms / 2840 MiB
peak**, vs our **700ms / 3072 MiB**. The entire gap has ONE root cause: **vision.cpp computes
in F32** (weights load F16 but activations + matmul/conv compute are F32; "not all ops support
F16"). PyTorch uses true fp16 (F16 weights AND F16 tensor-core compute). Closing this — F16
activations/compute throughout the GPU graph — is the single lever that:
- **~2.8× speed** (F16 tensor-core matmul + halved data movement on the memory-latency-bound conv kernels) → toward 248ms
- **~halves the 3GB transient activation peak** → ~1.7-2.0 GB (VRAM.md estimate), since the 3072 MiB floor is REAL F32 activation tensors (CONCAT [1024,1024,240]=960MiB, UPSCALE [1024,1024,192]=768MiB, full-res output convs)
It DOMINATES the alternatives (direct-conv VRAM revert is VRAM↓/perf↓; F16 is VRAM↓/perf↑)
and subsumes the "GPU conv kernel rewrite" task. This is THE endgame.

## Deploy stakes (why VRAM matters as much as speed)
Single 12GB RTX 3060 shared with the prod LLM (Qwen3.5-VL-9B), avatar, TTS, etc.
- At today's 3.5GB peak: ANY concurrent service risks OOM → must evict aggressively (harsh
  worker-isolation, evict-per-request). 3GB peak = "guarded service" territory.
- At F16's ~1.7GB peak + 524MB resident: plausibly **keep resident**, reserve ~1.7GB headroom,
  brief (<1s) spike during processing — much more comfortable, may not need harsh evict at all.
So how low F16 gets the peak DIRECTLY drives the deploy architecture. Lower is better, always.
NOTE: weights are ALREADY F16 (~440MB resident, the lossless floor — quantization is OFF the
table for quality). F16-everywhere shrinks the ACTIVATION PEAK, not the resident weights.

## Scope of the work (the hard part — why it's weeks, not days)
vision.cpp deliberately runs F32 because "not all operations support F16." The job:
1. **Audit every op** in the BiRefNet graph (Swin encoder + ASPP/deformable decoder) for F16
   support in dbrain/ggml's CUDA backend. Many ggml CUDA ops support F16; some assert F32.
2. **Make the graph run F16 activations** — likely a build/runtime flag (mirror how the GPU
   path already picks F16 weights via preferred_float_type, but extend to activations/compute).
   Today an attempt to force F16 SIGSEGVs at load + the decoder explodes to inf on type
   mismatch — so this needs careful per-op type threading, not a blanket flip.
3. **The deform CUDA kernel** (dbrain/ggml src/ggml-cuda/conv2d-deform.cu) currently requires
   F32 weights (nn.cpp casts them). Teach it to accept F16 in/out, or keep a tight F32 island
   around just the deform with casts at the boundary (measure which is faster).
4. **Ops that genuinely can't do F16** → keep F32 with explicit cast boundaries (like the
   current deform F32-cast), minimizing cast churn.
5. **Flash attention**: currently OFF (fattn.cu:492 crashes on head_dim=32). F16 may interact;
   re-evaluate whether FA-on becomes viable + beneficial under F16.

This is in SHARED dbrain/ggml (other prod models depend on it @ af69870c / consolidated-v0.13)
— any ggml change MUST stay correct + fast for them. Prefer adding F16 paths over changing
F32 ones. Keep ~/dev/ggml and depend/ggml in sync.

## Success criteria (hard gates — VERIFY on disk, never claim)
- Matte stays correct: YAVG ~181 on cat-and-hat, visually clean on the hard furry cases
  (bench/hardcases/), parity vs bench/rmbg2_pytorch_matte_1024.png. F16 is near-lossless here
  (PyTorch fp16 already matched at YAVG 181) — but PROVE it per-image, don't assume.
- GPU latency: target toward PyTorch's 248ms (any real drop from 700ms is progress).
- GPU peak VRAM: target ~1.7-2.0 GB (from 3072). This is the deploy-critical number.
- Report the new resident/peak/latency triple + matte parity for each milestone.

## Build/run/verify mechanics (CRITICAL — host has NO cuda/cmake)
All in HANDOFF-matting-server.md. Summary: build in `flux2-dev:builder` docker (static,
GGML_CUDA=ON, sm_86), run with `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility`,
model models/RMBG-2.0-F16.gguf. Correctness = matte YAVG via HOST ffmpeg
`format=yuv444p,signalstats` (gray PNG → NOT format=gray, that returns empty) or visual Read.
VRAM = nvidia-smi --query-compute-apps by process during a burst. Per-op profiler:
VISP_PROFILE_OPS=1 (committed, src/visp/ml.cpp compute()).

## HARD PROCESS RULES (this session had fabrication + file-destruction incidents)
- ONE bash command per step for git/file ops; NEVER batch many interdependent shell calls
  (cascades cancel the whole batch).
- NEVER `git checkout/restore/reset` tracked source — a prior agent DESTROYED uncommitted
  work that way. COMMIT early/often on branch `matting-birefnet`. Verify `git branch --show-current`.
- NEVER run >1 agent writing the same files concurrently. Serialize GPU agents (one card).
- VERIFY every artifact on disk (PNG exists + YAVG + visual) before reporting ANY number.
  Multiple agents fabricated numbers this session ("157ms", "−25%", "8688ms", "bit-exact") —
  all caught by re-measuring. Re-measure, don't trust.
- nvidia-smi before every GPU run; never two GPU jobs at once.

## Current verified baseline to beat (all on disk, branch matting-birefnet)
| | latency @1024 | peak VRAM | resident | matte |
|--|--|--|--|--|
| Our C++ (current) | ~700ms | 3072 MiB | 524 MiB | YAVG 181 ✓ |
| PyTorch fp16 (target) | 248ms | 2840 MiB | 1022 MiB | YAVG 181 |
| F16-everywhere (goal) | →248ms | →~1.7GB | 524MiB | must stay 181 |

## Deferred sibling tasks (NOT this mission, but downstream of it)
- Worker-isolation evict deploy (blueprint extracted from qwen3-tts.cpp — see below). Its
  harshness depends on F16's VRAM result: low peak → maybe resident, no harsh evict needed.
- Wire matting-server into kobbler/koblem; remove temp X-BG-Debug header; push dbrain/ggml
  (consolidated-v0.13 NOT pushed yet — af69870c + your F16 commits); fork → dbrain/hbd-vision.cpp.

## Worker-isolation deploy shape (LOCKED by user, for the deferred deploy task)
Pattern = copy from qwen3-tts.cpp (worker_ipc.{h,cpp} + worker_session.{h,cpp}: socketpair+
fork+execv `--worker <fd>`, 12-byte FrameHeader length-prefixed frames, SIGKILL on unload =
true-0 VRAM). USER'S REQUIRED SHAPE:
- **Strict evict-per-request** (worker dies after serving → true-0 VRAM between requests),
  BECAUSE 3GB peak is unguarded-OOM territory. (F16 may relax this to resident — re-decide
  after F16's VRAM number.)
- **Serialize execution**: only ONE inference at a time. Concurrent requests QUEUE/chain.
- **Don't evict while a request is waiting** in the queue (evict only when truly idle).
- **Batch API**: "rmbg on these N images" = one load, N images, then evict (amortize load).
MATTING_WORKER_ISOLATION env gate. See the qwen3-tts blueprint (recon done this session).
