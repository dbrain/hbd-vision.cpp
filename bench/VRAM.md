# BiRefNet / RMBG-2.0 matting — GPU PEAK VRAM decomposition & reduction

Goal: cut GPU **peak** VRAM (deploy = worker-isolated lazy-evict GPU service; it must fit in
the free-VRAM window when the heavy GPU services are resident). Latency is NOT a concern
(CPU alternative is 9.9 s; current GPU 695 ms has huge headroom).

Box: RTX 3060 (12 GB) + Ryzen 5 5600X. Model `RMBG-2.0-F16.gguf` (1024 native).
Build: flux2-dev:builder, static, GGML_CUDA, sm86. ggml submodule @ af69870.

Verified starting point (prior sessions): GPU @1024 ~695 ms, matte YAVG ~181,
resident-after-load 524 MiB, **peak VRAM ~3580 MiB (nvidia-smi)**. The ~3 GB delta over the
524 MiB resident is TRANSIENT inference scratch.

---

## PHASE 1 — decomposition of the transient scratch (from `bench/prof_gpu.csv`)

Method: the per-op profiler (`VISP_PROFILE_OPS`) dumps each node's **dst** ne dims. For
im2col-based convs the IM2COL node's *dst* IS the materialised `[IC·KH·KW, OH·OW]` scratch
matrix — exactly the transient buffer we care about. Bytes = product(ne)·4 (F32).
ggml's gallocr reuses buffers, so **peak ≈ the largest single concurrent buffer + live
activations**, NOT the sum. The single biggest buffer therefore sets the floor.

Stage boundary (PROFILE.md): encoder = idx < 3403, decoder (ASPP+deform) = idx ≥ 3403.

### Top transient buffers (all are IM2COL, ALL in the ENCODER)

| idx | op | dst ne | MiB | what |
|-----|----|--------|-----|------|
| **179** | IM2COL | [4608, 65536] | **1152.0** | enc conv IC=512 k=3 @256×256 → out 384ch. **THE peak driver.** |
| 176 | IM2COL | [1152, 65536] | 288.0 | enc conv IC=128 k=3 @256×256 → out 192ch |
| 9   | IM2COL | [48, 1048576] | 192.0 | patch_embed k=4 IC=3 @1024×1024 (1048576=1024²) |
| (decoder per-block IM2COL, idx≥1188) | IM2COL | [4608,16384] etc | 288.0 each (×4) | — see note below |

IM2COL by size class (count × MiB-each):
- 1152 MiB ×1  (idx 179)            = 1152 MiB
- 288  MiB ×4                        = 1152 MiB
- 144  MiB ×4                        =  576 MiB
- 128  MiB ×4                        =  512 MiB
- 72   MiB ×4                        =  288 MiB
- 36   MiB ×6, 9 MiB ×2, ≤2 MiB tail
- **IM2COL dst total across 47 nodes = 5072 MiB** (NOT concurrent — gallocr reuses).

Decoder (idx ≥ 3403): the **largest single buffer is only ~16 MiB** (CONT/CONV_2D/CONCAT at
256×256×64ch). The decoder convs run on the deform CUDA kernel + the 1×1 mul_mat path; they
do **not** materialise any large im2col matrix. **The decoder is NOT a VRAM problem.**

### KEY FINDING — overturns the task's stated hypothesis

The task brief hypothesised the VRAM hogs are the **decoder** high-res/high-channel k>1 convs
(the `whcn` branch in `nn.cpp conv_2d`). **That is wrong.** Measured:

- The peak is dominated by **ONE buffer: idx 179 = 1152 MiB**, an **encoder** conv im2col at
  full 256×256 spatial × IC=512 × k=3, plus idx 176 (288 MiB) and idx 9 (192 MiB), also encoder.
- These encoder convs go through the **`cwhn`** branch of `conv_2d` (line 104–108:
  `ggml_cont(permute) → ggml_conv_2d → ggml_cont(permute)`), because `patch_embed` sets the
  `cwhn` flag for the whole Swin encoder.
- The **decoder** (the `whcn` branch the task pointed at) tops out at ~16 MiB/buffer — reverting
  it to direct would save ~nothing.

So the lever is in the **`cwhn`** branch (encoder), specifically the few full-resolution k>1
convs whose im2col matrix is huge. Routing those through `ggml_conv_2d_direct` (no im2col,
streams the conv) removes the 1152/288/192 MiB buffers at a small latency cost.

### Peak-vs-resolution curve (MEASURED, nvidia-smi per-process peak)

| process_res | peak MiB | wall ms | YAVG | note |
|-------------|----------|---------|------|------|
| 512  | 1592 | 358  | 183.96 | quality shift (segmentation differs) |
| 768  | 1592 | 400  | 167.80 | same gallocr bucket as 512; quality shift |
| **1024** | **3596** | **689** | **181.09** | native default — the deploy target |
| 1536 | 4666 | 1207 | 180.62 | — |
| 2048 | 8862 | ~8800 | 180.62 | — |

Peak scales steeply with resolution (≈ activation area). Lowering res WOULD cut peak
(3596→1592 at ≤768) but it is a real quality tradeoff (768 YAVG 167.8 = the segmentation
decisions shift, not just edges — see RESULTS.md CPU pass). The deploy needs full-quality
**1024**, so resolution is NOT the lever we want; the im2col-buffer lever (Phase 3) keeps 1024.

Baseline @1024 reconfirmed THIS pass: **peak 3596 MiB, 691 ms, YAVG 181.090** (matches prior
3580/695). Resident-after-load 524 MiB (prior). => ~3.07 GB of the 3.6 GB peak is transient.

---

## PHASE 2 — free savings & the structural floor

The im2col lever (Phase 3) turned out to ALSO be the "free" win: routing the big-im2col
convs to the streaming `ggml_conv_2d_direct` kernel is **bit-identical** (MAD=0.0000, max=0
on all 3 test images, def-path vs im2col-path — see `bench/diff_mattes.py`) and costs only
~20 ms (~691→~712 ms). So it is effectively free for this latency-insensitive deploy.

Why peak is NOT just "the largest single buffer": the threshold sweep proves gallocr reuse.
- Pushing the 1152 MiB buffer (idx 179) to direct alone (threshold 256–1000) left peak at
  **3596** — that buffer was already being reused, it did not set the peak.
- Peak only dropped (to **2524**) once the threshold fell to **≤160 MiB**, i.e. once the
  192 MiB patch_embed im2col (idx 9, the 1024×1024 conv) was also pushed to direct. That
  192 MiB buffer is the marginal one that sets the high-water mark of the concurrent set.
- Below 160 MiB the peak is **flat at 2524** all the way down to 4 MiB (latency keeps rising).

=> The remaining **2524 MiB floor is STRUCTURAL**, not im2col. With the big im2col buffers
gone, the largest single remaining buffer is only ~192 MiB (and ≤16 MiB in the decoder), yet
peak is 2524 — so peak is the **sum of many concurrent live activations**, dominated by the
Swin-L encoder feature pyramid kept resident for the decoder's lateral skip connections, plus
the two encoder passes (full-res + half-res, BiRefNet's multi-scale design — no scratch
sharing possible without changing the output, confirmed in RESULTS.md). Compute-buffer
working set ≈ 2524 − ~524 resident ≈ ~2000 MiB; PyTorch fp16's compute working set is
comparable (~1818 MiB). gallocr is NOT meaningfully over-reserving — no free slop to reclaim.

## PHASE 3 — im2col threshold sweep (VISP_IM2COL_MAX, MiB of the F32 im2col matrix)

All @1024, GPU, FA off, cat-and-hat. peak = nvidia-smi per-process. (`bench/measure_vram.sh`.)

| VISP_IM2COL_MAX | peak MiB | wall ms | YAVG | note |
|-----------------|----------|---------|------|------|
| 0 (disabled / always im2col) | **3596** | 701 | 181.090 | legacy baseline |
| 1000 | 3596 | 708 | 181.090 | only 1152 buf → direct; no peak change (reused) |
| 256  | 3596 | 719 | 181.090 | knee not yet crossed |
| 200  | 3596 | 706 | 181.090 | |
| **160**  | **2524** | 710 | 181.090 | **knee** — patch_embed (192 MiB) now direct |
| **128 (DEFAULT)** | **2524** | 713 | 181.090 | chosen default (comfortably past knee) |
| 64   | 2524 | 730 | 181.090 | |
| 32   | 2524 | 735 | 181.090 | |
| 16   | 2524 | 760 | 181.090 | latency rising, no further VRAM gain |
| 8    | 2524 | 792 | 181.090 | |
| 4    | 2524 | ~800 | 181.090 | floor; everything on direct |

**Chosen knee / default = 128 MiB.** Past the knee (≤160) for a solid 2524, with maximal
latency headroom. Catches patch_embed (192), the 1152 encoder conv, and decoder 288/144
convs; leaves the cheap small convs on the faster im2col path.

Bit-exactness gate (default 128 vs disabled 0), all 3 test images:
`cat-and-hat MAD=0.0000 max=0 | vase-and-bowl MAD=0.0000 max=0 | wardrobe MAD=0.0000 max=0`.
YAVG 181.090 == baseline. Direct-conv and im2col+mul_mat compute the identical convolution.

### FINAL VERIFIED NUMBERS (default build, no env)
- GPU @1024: **peak 2524 MiB** (was 3596), **713 ms** (was 691), YAVG **181.090**, bit-exact.
- `VISP_IM2COL_MAX=0` restores 3596 MiB / 701 ms (knob verified both directions).
- Resident-after-load unchanged at 524 MiB (weights + CUDA ctx; the lever only touches scratch).

## FINAL — honest ceiling

- **Banked: GPU peak 3596 → 2524 MiB (−1072 MiB / −30%)** at full 1024 quality, bit-identical
  matte, ~free latency (+22 ms). Now **below PyTorch fp16's 2840 MiB peak**, and our resident
  (524) is already half of PyTorch's (1022). Shipped as the compiled default (env-tunable).
- **2524 MiB is the floor for this class of lever.** It is structural: the concurrent Swin-L
  encoder activation pyramid (multi-scale skip connections + the two encoder passes). The
  largest remaining single buffer is ~192 MiB; the rest is the live activation set. gallocr is
  not over-reserving (compute working set ~2000 MiB ≈ PyTorch's). No more free/lossless VRAM.
- **~1–1.5 GB is NOT reachable losslessly at 1024.** The only routes below ~2.5 GB:
  1. **Lower process_res** — 512/768 give peak **1592 MiB** (measured) but shift the
     segmentation (quality tradeoff, not lossless; YAVG 768=167.8). Opt-in `process_res`
     already exists; viable if a request accepts lower fidelity. The hard ~1.5 GB target is
     met ONLY at reduced resolution.
  2. **F16 activations (Phase 4, NOT implemented — flagged):** the ~2000 MiB compute working
     set is F32. Halving activation precision to F16 would roughly halve it → estimated GPU
     peak ≈ **524 + ~1000 ≈ ~1.5 GB** at 1024. This is the one path to ~1.5 GB at full
     resolution. BUT it is a big, risky **shared-ggml** change (many ggml ops would need an
     F16-activation path; per RESULTS.md the F16 path SIGSEGVs today and several ops assume
     F32), with real quality risk (F16 accumulation in the decoder ASPP/deform, which already
     "explodes to inf" on type mismatch). Estimate only — do NOT attempt without explicit
     sign-off and a proper per-op F16 audit + bit/MAD gate.

### Deploy implication
At 2524 MiB peak (worker-isolated, true-0 when evicted), the matting worker fits the
free-VRAM window far more comfortably than the old 3596 — a −1072 MiB safety margin against
OOM when the heavy GPU services are concurrently resident, with zero quality or meaningful
latency cost. Default is baked in; `VISP_IM2COL_MAX` lets ops trade latency for even more VRAM
headroom if a future tighter window demands it (peak stays 2524 but the knob is there).

