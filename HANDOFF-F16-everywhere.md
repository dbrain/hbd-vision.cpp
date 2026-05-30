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
