# Brief: verify vision-server on CUDA before deployment

You are on the GPU box (3060 + 5060). Everything below was developed on an AMD gfx1150 iGPU
over **Vulkan and CPU only**. No CUDA kernel in this changeset has ever been compiled. Your job
is to establish a baseline on what is running today, then verify the replacement against it.

**Read `HANDOFF-gpu-verification.md` in this repo first** — 11 sections, written as the work
happened. Sections 7 and 8 are **retracted in place**: they measured flat six-colour placeholder
art, not real renders, and their conclusions do not stand. Do not act on them.

**Work in this order. Do not skip Phase A** — once the new container is up, the old numbers are
unrecoverable, and "it seems about the same" is not a result.

---

## What changed

One binary, `vision-server`, replaces `matting-server`. It serves three endpoint groups from one
worker process with one VRAM budget:

| endpoint | model | consumer |
|---|---|---|
| `POST /remove` | BiRefNet / RMBG-2.0 | koblem `MATTING_SERVICE_URL` |
| `POST /depth` | **Depth-Anything V2** (DA3 via `?model=depth-anything-3`) | koblem `DEPTH_SERVICE_URL` |
| `POST /v1/*` | SigLIP2 | **not wired** — see below |

- `dbrain/hbd-vision.cpp` `main` @ `2847b6b`
- `depend/ggml` → branch **`vision-conv2d-deform`** @ `a1bf8b90` (v0.18.0 + one re-ported op).
  `dbrain/ggml` `master` is untouched, so the other `hbd-*` repos are unaffected.

---

## Phase A — baseline what is running now (BEFORE changing anything)

Measure the **currently deployed** `matting` and `siglip2` services as they stand. This is the
control; everything in Phase D is compared against it.

1. **Record the running state**: image tags/digests, `VISION_REF`, container env, which GPU each
   sits on, and the current `gpu_engine_config` rows (`matting` should be 4000 MiB, home
   `GPU-bd93e020…` (5060), candidates 5060+3060, priority 12).
2. **VRAM, idle and under load.** Per-process, not just `nvidia-smi` total — the parent holds no
   CUDA context, the worker child does. Capture: idle after `IDLE_UNLOAD_SECONDS`, matting at
   `process_res` 1024 and 2048, siglip2 embedding a batch, and both together.
3. **Latency.** Matting at 1024 and 2048; siglip2 `/v1/embeddings` warm and cold; `/v1/classify`.
   Several runs, report spread not just best.
4. **Functional output.** Keep the actual matting PNGs — Phase C compares against these
   byte-wise/RMSE, not against a fresh subjective look.
5. **Note anything already broken.** If matting is slower or worse than `bench/PROFILE.md`
   claims today, that predates this work and must not be attributed to it.

> Also grep the deployment for **`VISP_F16_ENCODER`** and **`SIGLIP2_DISABLE_FA`**. If either is
> set to a falsy value (`0`, `false`, empty), it has been doing the **opposite** of what it says —
> those flags tested presence, not value, so `=0` turned the feature **on**. That is fixed in this
> changeset, so behaviour will change on deploy. For `VISP_F16_ENCODER` that silently changes the
> `bench/VRAM.md` numbers. Record what they are set to before you change anything.

---

## Phase B — build the new codebase

1. `git clone --recurse-submodules` fresh, or update in place. **Verify the submodule resolves** —
   `depend/ggml` must land on `a1bf8b90` from branch `vision-conv2d-deform`. `.gitmodules` uses
   the **HTTPS** URL deliberately: Docker builders clone with no SSH agent.
2. **Build `-DGGML_CUDA=ON`. This is the highest-risk step in the whole brief.**
   `src/ggml-cuda/conv2d-deform.{cu,cuh}` is carried forward byte-identical from a pre-rewrite
   commit and **has never been through a compiler**. Static audit says it only touches
   `ctx.stream()`, `GGML_ASSERT`, `ggml_is_contiguous` and raw CUDA, and `ggml-cuda/CMakeLists.txt`
   globs `*.cu` so no CMake edit is needed — but that is an audit, not a build.
   **If it fails to compile, the CPU op is complete and correct; the fix is confined to the `.cu`.**
3. Run `test-models` on CUDA. Baseline on Vulkan/CPU here was **11 passed / 1 skipped / 1 failed**,
   where the only failure is `test_birefnet[gpu]` at rmse ~1.15. **That failure should NOT
   reproduce on CUDA** — its cause is that there is no scheduler on the inference path
   (`ml.cpp` calls `ggml_backend_graph_compute` directly; `compute_graph::cpu_fallback` is declared
   and never read), so on Vulkan the `CONV_2D_DEFORM` op never executes and its buffer is never
   written. CUDA has the kernel. **If it DOES fail on CUDA, the kernel is not being selected** —
   that is the single most important negative result you can return.

---

## Phase C — functional parity

Against Phase A's captured outputs:

1. **Matting**: same images, compare RMSE against Phase A's PNGs and against `bench/` baselines.
   Watch for a numeric shift from upstream `efee796c` ("uniformize im2col dst_type"), which made
   `ggml_conv_2d` build its im2col as **F16** where it previously used `a->type`. On CPU that was
   measured harmless (rms 2.7e-4, 37x headroom) but **that does not clear CUDA**, where the weights
   are F16 and the change lands differently.
2. **SigLIP2**: the four `/v1/*` routes must be byte-compatible with what koblem sends —
   `/v1/embeddings`, `/v1/classify`, `/v1/text_embeddings`, `/v1/classify_from_embeddings`.
   Note the new binary was validated against HF at image cosine 0.9994 / text 1.0000, but with
   **`base-patch16-naflex`**, not the so400m your deployed service runs.
3. **Depth** is new — no parity to check, but sanity-check the maps: `/depth` returns a 16-bit
   greyscale PNG, min/max-normalised, at the **input** resolution, with `depth_min`/`depth_max`
   in the JSON (they are the only way to denormalise). **V2 emits disparity: larger = NEARER.**
   DA3 emits distance: larger = farther. They correlate at Pearson −0.947. If a depth map looks
   inverted, that is the first thing to check.

---

## Phase D — performance and VRAM on CUDA

Measure the same axes as Phase A, then compare.

**Reference numbers, all from the gfx1150 iGPU over Vulkan with unified memory. Treat as shape,
not as targets** — UMA does not OOM the way a discrete card does:

| | VRAM delta | inference |
|---|---|---|
| DA2 @native (518²) | **+254 MiB** | ~0.30 s |
| DA3 @504 | +297 MiB | ~0.30 s |

Staged co-residency, cumulative: 84 MiB (matting weights) → 385 (+depth@504) → 1081
(+depth@1008) → 1802 (+siglip) → 2423 (+`/remove`@1024).

**The number that decides the gate config: combined CUDA peak with RMBG + DA2 + SigLIP2 resident,
measured at the resolution production actually uses.**

⚠️ **The existing 4000 MiB `matting` reservation is almost certainly wrong — too high.** That
figure traces to RMBG-2.0 at `process_res` **2048** (migration `0020`'s own comment says
"RMBG2, ~4 GB peak"), but production mattes **1024×1024**. Peak scales with activation area, so
the real 1024 figure should be far below 4000 MiB.

So the likely outcome is that `peak_mb` comes **DOWN**, not up — and that matters: the `matting`
row is homed on the 5060 at priority 12, so an over-reservation holds budget hostage from
flux/songgen on the same card for VRAM that is never touched.

**Measure, then set it from the measurement:**
1. Peak at `process_res` **1024** (production), not 2048.
2. Peak with DA2 resident alongside — depth deliberately shares this line (`depth.rs:237` calls
   `acquire_for_matting()`) rather than taking its own, because a separate line would let the
   placer put them on different cards, and a changed `gpu` field respawns the worker, so
   alternating requests would thrash one process between two GPUs. **Do not add a `depth` row.**
3. Peak with SigLIP2 also resident, if you ever intend `/v1/*` to be served from this binary.
4. Set `peak_mb` on the existing `matting` row to the measured figure plus whatever headroom you
   normally allow, and say what you picked.

Also confirm the server-side `process_res` clamp actually pins 1024 in production — if a caller
can ask for 2048, the reservation has to cover it regardless of what the common case costs.

One knob worth knowing: `worker.cpp` forces `VISP_FLASH_ATTENTION=0` for BiRefNet's head_dim=32
Swin, and that is **process-wide** — it costs DA2/DA3 ~20% (113 ms → 135 ms measured on Vulkan).
Overridable by container env if depth latency ever matters more than matting's.

**Also check `CONV_2D_DEFORM` is actually selected on CUDA, not silently falling back.** BiRefNet's
ASPP decoder is ~46% of matting GPU time per `bench/PROFILE.md`; a CPU fallback shows up as roughly
a 2x wall-clock regression, not as a failure.

---

## kobbler / docker-compose.yml

**These changes were deliberately NOT pushed.** They were written on a base 11 commits behind
origin, and deployment belongs to whoever owns the box. `kobbler-vision-server.patch` in this repo
is the working diff — apply it, cherry-pick from it, or ignore it and redo the change; it exists so
the findings below are not rediscovered the hard way, not because it should be applied blind.

**One part is not optional.** `docker/matting/Dockerfile` builds the target **`matting-server`,
which no longer exists** — the binary is `vision-server` now. That container build is broken
against current `main` whether or not you take the rest.

What the patch does:

- `docker/matting/` → **`docker/vision-server/`**; compose service `matting` → **`vision-server`**,
  with **`matting` kept as a network alias** so koblem's `MATTING_SERVICE_URL=http://matting:8898`
  keeps resolving with no koblem change.
- Fixed in passing: the Dockerfile still built target `matting-server`, which stopped existing at
  the rename — **that build was already broken.**
- `IDLE_UNLOAD_SECONDS` **0 → 300**. Necessary: with three model families sharing one worker, 0
  means every request reloads every model it touches.
- The `matting-models` volume name is deliberately unchanged, so the cached RMBG download survives.

**Model provisioning — no manual copy needed.** The entrypoint already fetches from HF:
```
DEPTH_MODEL_PATH="$(fetch "${DEPTH_HF_REPO:-}" "${DEPTH_MODEL_FILE:-Depth-Anything-V2-Small-F16.gguf}")"
```
Set **`DEPTH_HF_REPO=Acly/Depth-Anything-V2-GGUF`** and it pulls the 49 MB file into the volume on
first boot. With it empty, `/depth` answers **503 naming the missing env var** and the other
endpoints are unaffected — graceful degradation is intentional, verify it still works.

**DA3 is optional and has no published GGUF.** Leave `DEPTH3_HF_REPO` empty unless you want
`?model=depth-anything-3` reachable; if you do, copy `models/Depth-Anything-3-Small-F16.gguf`
(67 MB, produced by `scripts/convert.py`) into the volume and set `DEPTH3_MODEL_PATH`.

**Do NOT repoint `VISION_SERVICE_URL` at vision-server.** The `/v1/*` routes are built in and
wire-compatible, but the deployed `siglip2` service runs **so400m from a different repo**
(`dbrain/hbd-siglip2.cpp`) and **koblem holds persisted image embeddings that must stay
comparable**. Migrating is a one-line URL change plus `SIGLIP_HF_REPO`/`SIGLIP_MODEL_FILE` — a
deliberate decision, not a tidy-up. Related: koblem's `VISION_SERVICE_URL` default was
`http://vision:8890`, which **has never resolved** (no such alias); it is now `http://siglip2:8890`.

---

## Also worth checking while you are in there

- **`scripts/scan-gguf-type42.py`** over the whole model store. The fork's `GGML_TYPE_F8_E4M3` was
  id **42**, and upstream v0.18 reassigns **42 to `Q2_0`**. Any GGUF written with the old FP8 type
  now reads as Q2_0 — **wrong output, no error, no warning**. Clean across all 59 GGUFs on the dev
  laptop, but the FP8 imports (`import_ltx_fp8.py`, `import_wan_nvfp4.py`) write on the server.
- **Do NOT repoint `hbd-qwen3-tts.cpp` at ggml master.** The same history rewrite dropped
  `ggml_snake`, `ggml_conv_1d_direct{,_to}` and `ggml_conv_transpose_1d_to`, which it uses. Note
  upstream landed snake as a graph **autofusion** (`mul→sin→sqr→mul→add`) with CUDA and Metal
  paths, so expressing it with stock ops may fuse back to one kernel — worth trying before a
  re-port. The other two have no upstream equivalent.
- The `2028dcb8` sched/alloc hardening was dropped by the rewrite for **all** consumers. The
  `hash_set` sizing has a real failure mode (the `size >= n_nodes + n_leafs` assert it prevented),
  and large graphs are likelier to trip it — the server's big models more than anything here.

---

## Report back

1. Phase A baseline table (VRAM, latency, outputs).
2. **Did `conv2d-deform.cu` compile? Is the CUDA kernel selected at runtime?**
3. `test-models` on CUDA — and specifically whether `test_birefnet[gpu]` passes there.
4. Matting parity vs Phase A, with the im2col-F16 question answered.
5. Phase D table vs Phase A, and the combined-peak number for the gate.
6. Whether `peak_mb` on the `matting` row needs raising, and to what.
7. What `VISP_F16_ENCODER` / `SIGLIP2_DISABLE_FA` were set to before, and what changed.
8. Anything that regressed. A clear negative is more useful than a reassuring summary — several
   conclusions in this project were overturned by measuring rather than assuming, twice because
   the input was wrong rather than the code.

---

## 12. SAM 3 `/parts` — the CUDA work, and why it wants doing on device

`POST /parts` (commit `f6829d2`) serves SAM 3 concept segmentation: image plus a list of noun
phrases in, instance masks out. It exists because paperworld's silhouette rig approximates each
body part with an **ellipse** — measured across 34 parts on 14 real renders, those ellipses are
**37.7% background**, SAM's masks are **0.0%**. It also finds limbs no silhouette method can:
a bear's forelimbs merged into its body, an owl's folded wings, a wizard's arms inside sleeves.

**Status: the four gaps are closed** (`7e696c7`; `depend/sam3.cpp` now points at
`dbrain/sam3.cpp` branch `gpu-backend-support` @ `09b80ad`). It runs on Vulkan here. What is left
is CUDA itself.

Measured on the dev box (Radeon 890M, RADV), 1008 square encode:

| | encode | decode/prompt |
|---|---|---|
| CPU | 12.9 s cold | 1.8 s |
| **Vulkan iGPU** | **3.6-7.3 s** (mean 5.3) | **0.9-1.4 s** (mean 1.07) |
| Vulkan, FA forced off (worst-case CUDA shape) | 7.1-8.7 s | 1.2-1.8 s |

End-to-end through `POST /parts`, owl with 3 prompts: **7.3 s**. CUDA should sit nearer the Vulkan
row than the FA-off row, because only the head_dim-32 sites take the fallback and the ViT — which
dominates — keeps its native head_dim-64 kernel.

What was closed, for context when reading the diff:

| gap | fix |
|---|---|
| no GPU backend init at all (Metal only) | `ggml_backend_init_best()` — note `init_by_type(GPU)` returns null on a UMA part, which registers as `IGPU` |
| `GGML_OP_WIN_PART` / `WIN_UNPART` | one strided `ggml_view_4d` + `ggml_cont`; `ggml_pad` only when not a window multiple |
| `GGML_OP_POOL_1D` | `ggml_sum_rows` + `ggml_scale(1/T)` |
| `FLASH_ATTN_EXT` @ head_dim 32 | **conditional** — builds the FA node, asks `ggml_backend_supports_op`, materialises the scores matrix only when the device says no. This is why CPU output is byte-identical. |
| `worker.cpp` hardcoded `p.use_gpu = false` | follows the worker's real device; `/parts` no longer reports `"backend": "cpu"` unconditionally |

`SAM3_NO_FLASH_ATTN=1` forces the fallback path — the only way to exercise CUDA's graph shape on
hardware that supports head_dim 32.

**WHAT IS LEFT FOR CUDA, and only CUDA can answer it:**

1. **Does it run.** Vulkan is strong evidence — it rejects `WIN_PART`/`POOL_1D` identically and now
   executes the whole graph — but Vulkan *accepts* head_dim 32, so the fallback only ran there
   under `SAM3_NO_FLASH_ATTN=1`. On CUDA it must trigger by itself via `supports_op`.
2. **Are CUDA's `supports_op` answers what we read them to be.**
   `ggml_cuda_flash_attn_ext_supported` → `get_best_fattn_kernel` was read to return `NONE` rather
   than abort, so the probe should work — never executed.
3. **VRAM: the activation arena is still unmeasured on any backend.** The fenc self-attention
   materialises a scores matrix on the fallback path, so the CUDA figure may exceed the Vulkan one.
   Weight buffer is 1748 MB F16 / 1.0 GB q8_0 / 707 MB q4_0. If `/parts` shares the worker with
   RMBG and DA2, this is a real addition to the `matting` gate row's `peak_mb`.
4. **SAM2 / EdgeTAM** went through the same helper and no model exists here to test. Behaviour-
   preserving on CPU by construction; only a GPU run exercises it.

**NOT a gap:** the F16 `SOFT_MAX` you re-ported for matting is not needed here. sam3 keeps its one
`soft_max_ext` mask in F32 deliberately (in-source: the F16 mask path loses precision on the
box-relative positional bias), and its ViT is head_dim 64 with F32 K/V.

**Why this half belongs on the GPU box.** The dev laptop is AMD/Vulkan, so it can verify the
rewrites are *correct* — the masks must come out byte-identical to the archived CPU results — but
it cannot verify they *run* on CUDA, and it cannot measure anything. That is the whole question.
Expected ~2–5 s once the ops land, i.e. 20–50×, but nobody should trust a number extrapolated from
a path that currently aborts.

**Model:** not published. Convert with sam3.cpp's own script; point `SAM3_MODEL_PATH` at the
result. `*.ggml` is gitignored. Weight buffer **1748 MB F16 / 1.0 GB q8_0 / 707 MB q4_0**;
**activation arena unmeasured** — budget ~2.5–3 GB F16 or ~1.7–2 GB q8_0 and verify. If it shares
the worker with RMBG and DA2 it is a real addition to the `matting` gate row's `peak_mb`.

**SAM 3.1 is not worth chasing:** sam3.cpp hardcodes SAM 3's hparams, and 3.1's changes are all
video — Object Multiplex is shared-memory joint multi-object *tracking*, explicitly "without
changing the model architecture" and "without sacrificing accuracy". Nothing in it touches
single-image PCS. Its checkpoints are also HF access-gated.
