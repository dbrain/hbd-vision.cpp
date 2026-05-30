# BiRefNet / RMBG-2.0 matting — GPU PEAK VRAM decomposition & reduction

Goal: cut GPU **peak** VRAM. Deploy = worker-isolated lazy-evict GPU service that must fit the
free-VRAM window when the heavy GPU services are concurrently resident. Latency is NOT a
concern (CPU alternative 9.9 s; GPU has huge headroom), so trading latency for VRAM is a win.

Box: RTX 3060 (12 GB) + Ryzen 5 5600X. Model `RMBG-2.0-F16.gguf` (1024 native).
Build: flux2-dev:builder, static, GGML_CUDA, sm86. ggml submodule @ af69870.
Measurement: `bench/measure_vram.sh` — runs one inference in a named container, polls
`nvidia-smi --query-compute-apps=used_memory` for the peak, host-ffmpeg YAVG of the output
(NOTE the matte is a gray PNG → signalstats needs `format=yuv444p`, not `format=gray`).
Peaks reproduced across reps (3606/3606, 3072/3072) — stable, not jitter.

Baseline reconfirmed THIS pass @1024: **peak 3606 MiB, 695 ms, YAVG 181.090** (matches prior
~3580/695). Resident-after-load 524 MiB (prior sessions; deploy is lazy-load). => ~3.08 GB of
the 3.6 GB peak is transient compute scratch + activations.

---

## PHASE 1 — decomposition (real per-op dst sizes, `bench/prof_gpu.csv`)

Method: the `VISP_PROFILE_OPS` profiler dumps each node's **dst** ne dims; bytes = Πne·4 (F32).
ggml's gallocr reuses buffers, so **peak ≈ the largest concurrent live set**, not the sum.

### Top transient buffers — they are the DECODER OUTPUT HEAD, not im2col

| idx | op | dst ne | MiB | stage |
|-----|----|--------|-----|-------|
| **4015** | CONCAT  | [1024,1024,240] | **960** | final full-res feature concat |
| **4006** | UPSCALE | [1024,1024,192] | **768** | upsample features to 1024² |
| 3996 | CONCAT  | [256,256,1280]  | 320 | decoder block concat |
| 4007 | CONV_2D | [1024,1024,64]  | 256 | full-res output conv |
| 4010 | ADD     | [1024,1024,64]  | 256 | |
| 3986 | CONCAT  | [256,256,1024]  | 256 | |
| 4011 | CONV_2D | [1024,1024,48]  | 192 | full-res output conv |
| 3966 | CONCAT  | [256,256,768]   | 192 | |
| (enc) 55/121 | MUL_MAT | [144,144,6,484] | 230 | Swin attention scores |

**These are genuine activation tensors at the 1024×1024 output resolution — NOT im2col
scratch.** The peak is dominated by BiRefNet's decoder head, which concatenates the
multi-scale feature pyramid and upscales it to full 1024² (up to 240 channels). The 960 MiB
CONCAT + 768 MiB UPSCALE being live near-simultaneously sets the floor.

### im2col is a SECONDARY contributor, not the peak driver

Only **2 IM2COL nodes exist** in the entire graph (idx 9 = 12 MiB patch_embed, idx 177 =
3 MiB) — the Swin encoder convs are strided/tiny-kernel and do not blow up im2col. The
decoder's k>1 convs DO route through `ggml_conv_2d` (im2col+mul_mat) via the `whcn` branch;
their transient im2col matrices are the ~534 MiB the lever below removes — but gallocr reuses
them against the larger decoder-head activations, so removing them lowers the floor by exactly
that 534 MiB, not to ~1 GB.

⚠️ Correction to an earlier draft of this file: it claimed a 1152 MiB "idx-179 encoder
IM2COL" peak driver and a 2524 MiB floor — BOTH WRONG (idx 179 is a 48-elem RESHAPE; there is
no 1152 MiB IM2COL node; the measured floor is 3072). All numbers below are re-measured.

### Peak-vs-resolution curve (MEASURED, nvidia-smi per-process peak, baseline build th=0)

| process_res | peak MiB | note |
|-------------|----------|------|
| 512  | 1308 | quality shift (segmentation differs; not lossless) |
| 768  | 2288 | quality shift |
| **1024** | **3606** | native default — the deploy target |
| 1536 | 7530 | |
| 2048 | 11542 | nearly fills the 12 GB card |

Peak scales ~quadratically with res (activation area). Dropping to 768 hits ~2.3 GB but shifts
segmentation decisions (YAVG 768 ≈ 167.8, RESULTS.md) — a real quality tradeoff, not lossless.
Deploy needs full-quality 1024, so res is not the lossless lever.

---

## PHASE 2 — free / near-lossless savings

The im2col→direct reroute (Phase 3) is the main win for −534 MiB. Quality (default th=128 vs
disabled th=0, measured by ffmpeg PSNR + md5, all 3 test images):
- **wardrobe:      PSNR 67.4 dB**
- **cat-and-hat:   PSNR 54.5 dB**, YAVG 181.090→181.083, visually identical clean silhouette
  (both Read — `q0_*` vs `q128_*`)
- **vase-and-bowl: PSNR 44.1 dB** (the noisiest, still well above any perceptual threshold)
- md5 differs on all 3 → **near-lossless, NOT strictly bit-exact**.

`ggml_conv_2d_direct` and the im2col+`ggml_mul_mat` path differ in F16-weight handling / FP
accumulation order; this surfaces as sub-perceptual edge noise (PSNR 44–67 dB). It passes the
correctness gate (YAVG ~181 + clean silhouette) on all 3, but do NOT claim bit-identity.

No gallocr over-reservation was found beyond this: with the im2col matrices gone the floor is
the decoder-head activation set (Phase 1), which is real output data, not reclaimable scratch.
The two Swin encoder passes (full + half res) are BiRefNet's multi-scale design — no activation
sharing possible without changing the output (RESULTS.md). No further zero-cost VRAM available.

## PHASE 3 — im2col threshold sweep (`VISP_IM2COL_MAX`, MiB of the F32 im2col matrix)

conv_2d() routes a k>1 conv to the streaming `ggml_conv_2d_direct` kernel (no im2col buffer)
when its im2col matrix [IC·KH·KW, OH·OW] would exceed VISP_IM2COL_MAX; else im2col+mul_mat.
All @1024, GPU, FA off, cat-and-hat. peak = nvidia-smi per-process (2+ reps, stable).

| VISP_IM2COL_MAX | peak MiB | wall ms | YAVG (cat) | quality vs th=0 |
|-----------------|----------|---------|------------|-----------------|
| 0 (disabled / always im2col) | **3606** | 695 | 181.090 | baseline |
| 128 (DEFAULT)   | **3072** | 717 | 181.083 | PSNR 44–67 dB (per image), visually identical |
| 64              | 3072 | 730 | 181.083 | |
| 16              | 3072 | 764 | 181.083 | |
| 4               | 3072 | 801 | 181.083 | |
| 1               | 3072 | 809 | 181.083 | |

**Floor is 3072 MiB, reached at threshold ≤128 and FLAT all the way down to 1** (more convs go
direct, latency keeps rising, VRAM does not drop further — proves the remaining peak is the
decoder-head activations, not im2col). **Chosen default = 128 MiB**: past the knee for the full
−534 MiB with maximal latency headroom; catches the decoder's high-res k>1 convs, leaves the
cheap small ones on the faster im2col path. `VISP_IM2COL_MAX=0` disables it.

### FINAL VERIFIED NUMBERS (default build, no env)
- GPU @1024: **peak 3072 MiB** (was 3606), **~717 ms** (was 695), YAVG **181.083** (cat) —
  passes gate (YAVG ~181 + clean silhouette); near-lossless (PSNR 44–67 dB), NOT bit-identical.
- `VISP_IM2COL_MAX=0` restores 3606 MiB / 695 ms / YAVG 181.090 (knob verified both directions,
  repeatable).

---

## FINAL — honest ceiling

- **Banked: GPU peak 3606 → 3072 MiB (−534 MiB / −15%)** at full 1024 resolution, near-lossless
  matte (YAVG 181.083 cat; PSNR 44–67 dB; sub-perceptual edge noise; passes YAVG~181 +
  clean-silhouette gate), ~22 ms latency cost. Shipped as the compiled default, env-tunable.
- **3072 MiB is the floor for any im2col/scratch lever.** The peak is NOT im2col scratch — it is
  the BiRefNet **decoder output head**: a 960 MiB CONCAT [1024,1024,240] + 768 MiB UPSCALE
  [1024,1024,192] + several 256/192 MiB full-res convs, all genuine F32 activations at the 1024²
  output. gallocr is not over-reserving; these are real intermediate results. (PyTorch fp16 peak
  is 2840; we are now within ~230 MiB of it and our resident 524 is half PyTorch's 1022.)
- **~1–1.5 GB is NOT reachable losslessly at 1024.** Honest options below 3072:
  1. **Lower process_res** — measured 768→2288 MiB, 512→1308 MiB, but shifts segmentation
     (quality tradeoff, not lossless). Opt-in `process_res` already exists. The hard ~1.5 GB
     target is met only at ≤768 res with reduced fidelity.
  2. **F16 activations (Phase 4 — NOT implemented, flagged):** the entire decoder-head working
     set is F32. Halving activation precision to F16 would ~halve the dominant buffers
     (960→480, 768→384, …), estimated GPU peak ≈ **~1.7–2.0 GB** at 1024. This is the only
     lossless-at-1024 path under ~2.5 GB. BUT it is a large, risky **shared-ggml** change: many
     ops (CONCAT/UPSCALE/CONV_2D/ADD + the ASPP/deform decoder, which already explodes to inf on
     a type mismatch — RESULTS.md) would need a vetted F16-activation path; the F16 path SIGSEGVs
     today and several ops assume F32. Real numerical-quality risk in the F16-accumulating
     decoder. Estimate only — do NOT attempt without explicit sign-off, a per-op F16 audit, and a
     bit/MAD gate. Even then it lands ~1.7 GB, not necessarily ≤1.5 GB.
  3. **Tile the decoder head** (compute the 1024² output in spatial strips) — would cut the
     960/768 MiB buffers proportionally, lossless, but is a real birefnet.cpp restructure of the
     output stage. Not attempted; flagged as the highest-value next lever if <2.5 GB is required.

### Deploy implication
At **3072 MiB peak** (worker-isolated, true-0 when evicted) the matting worker has a −534 MiB
safety margin vs the old 3606 against OOM when the heavy GPU services are resident, at near-zero
quality and ~zero latency cost. Default baked in; `VISP_IM2COL_MAX` is there if a future tighter
window wants more (peak stays 3072 — no further gain past the knee). If the free-VRAM window is
ever < ~3 GB, the only routes are reduced process_res (lossy) or the F16-activations /
decoder-tiling work above (large, flagged, not done).
