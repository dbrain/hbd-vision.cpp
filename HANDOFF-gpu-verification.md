# HANDOFF: GPU-server verification (3060 / 5060)

All of this was developed on the AMD HX 370 laptop (gfx1150 iGPU, unified memory) against the
**Vulkan and CPU** backends. Nothing here has run on CUDA. This document is the checklist for
vetting it on the server before going live.

Living document — appended to as each task lands. Status column is honest: `unverified` means
nobody has run it on CUDA, not that it is expected to fail.

---

## 0. Why the divergence exists

`hbd-vision.cpp/CMakeLists.txt:125` only wires `GGML_VULKAN`. There is no HIP/ROCm path, so the
laptop exercises Vulkan + CPU exclusively. Three classes of thing therefore go unexercised here:

1. every CUDA kernel, including ones that only exist in the fork;
2. anything where the scheduler falls back to CPU on Vulkan but would stay on GPU under CUDA;
3. all real VRAM behaviour — the laptop has unified memory, so nothing OOMs the way a 12 GB or
   8 GB discrete card does.

---

## 1. The ggml bump — dropped fork ops

`depend/ggml` moved from `8e6e481c` (2026-07-09) to `a7478f58` (2026-08-02,
"Merge upstream ggml 78de6069 (v0.18.0 + llama.cpp sync)"). That merge was scoped to
hbd-longcat-avatar.cpp's needs and rewrote fork history, dropping ops other repos rely on.

| Op | Consumer | Status |
|---|---|---|
| `ggml_mul_mat_ext` | hbd-vision.cpp `nn.cpp:64,375,377` | **survived byte-identical** |
| ggml-alloc `RMS_NORM_CHANNELS` in-place | all | survived |
| ggml-cpu `ROPE_PE` unsupported | all | survived, improved |
| `GGML_OP_COL2IM_1D` | acestep | survived as *upstream's* independent implementation, identical semantics |
| `ggml_conv_2d_deform` | hbd-vision.cpp `nn.cpp:258` | dropped → **re-ported**, see §1.1 |
| `ggml_concat_n` | hbd-vision.cpp `ml.cpp:869` | dropped → **not re-ported**, consumer changed, see §1.1a |
| `ggml_snake`, `ggml_conv_1d_direct{,_to}`, `ggml_conv_transpose_1d_to` | hbd-qwen3-tts.cpp | dropped — **do not repoint that repo**, but see §1.5 |
| `2028dcb8` sched/alloc hardening | all | dropped, see §1.2 |

### 1.1 CONV_2D_DEFORM is the one that matters on CUDA

Dropped commit `af69870c` added, together: `GGML_OP_CONV_2D_DEFORM` (enum + CPU op in
`ggml-cpu/ops.cpp` + **a CUDA kernel**), `ggml_conv_2d_deform()`, `ggml_concat_n()`, and a
channels-last branch in `ggml_conv_2d`.

The laptop build only strictly needs the **CPU** half, and the Vulkan build goes green either way.
**That masks the CUDA requirement entirely.** BiRefNet's ASPP decoder is ~46% of GPU time per
`bench/PROFILE.md`, so on the 3060/5060 the CUDA kernel is not optional.

> **Correction — an earlier draft of this document said Vulkan "falls back to CPU" for this op.
> That is wrong, and the truth is worse.** There is no scheduler on the inference path at all
> (§5): the op is never executed and its output buffer is never written, so Vulkan BiRefNet
> produces noise rather than a slow-but-correct result. Anything this repo reports about matting
> on Vulkan is therefore meaningless. Only the CPU and CUDA paths mean anything.

> **Verify on server:** confirm the CUDA `CONV_2D_DEFORM` kernel is present and actually selected.
> Compare matting wall-clock against the `bench/` baselines.

**Status: re-ported.** `depend/ggml` branch `vision-conv2d-deform`, commit `a1bf8b90` on top of
`a7478f58`. Local only — not pushed. Searched upstream v0.18 first: no deformable conv, no
`grid_sample`, no gather-with-bilinear primitive, so there is no stock expression for this and
cherry-picking was the only option. Applied as a hand-filtered subset of `af69870c`, not a raw
cherry-pick — the full commit conflicts hard in `ggml-cpu/ops.cpp` (354 commits of drift), and most
of what it carried is either unwanted or superseded. What landed:

| File | What | Δ |
|---|---|---|
| `include/ggml.h` | `GGML_OP_CONV_2D_DEFORM` enum + `ggml_conv_2d_deform()` decl | +12 |
| `src/ggml.c` | op name/symbol tables, `GGML_OP_COUNT` 104→105, builder, file-local `ggml_set_permuted_strides` | +65/−2 |
| `src/ggml-cpu/ops.cpp` | `ggml_compute_forward_conv_2d_deform{,_cwhn,_whcn}` + bilinear helpers | +310 |
| `src/ggml-cpu/{ggml-cpu.c,ops.h}` | dispatch, `n_tasks`, `GGML_IM2COL_WORK_SIZE` in `graph_plan` | +7 |
| `src/ggml-cuda/conv2d-deform.{cu,cuh}` | **the CUDA kernel, carried forward byte-identical** | new |
| `src/ggml-cuda/ggml-cuda.cu` | include, `compute_forward` case, `supports_op` gate | +9 |

Deliberately **not** carried from `af69870c`: `ggml_concat_n` (§1.1a), its channels-last branches in
`ggml_conv_2d` / `ggml_conv_2d_direct` (§1.1b), the `ggml_is_contiguous_channels` `>` → `>=`
relaxation, a `dup_bytes` size-4 memcpy specialization, and an MSVC `__debugbreak` in `ggml_abort`.
Result is +401/−2 against upstream — the two deletions are the `GGML_OP_COUNT` static_asserts.

One deviation from `af69870c`, and it is load-bearing: BiRefNet's `aspp1` is a **1x1** deformable
conv, and a 1x1 kernel is channel-packed but has `nb[1] == nb[0]`, which upstream's strict-`>`
`ggml_is_contiguous_channels` rejects. `af69870c` handled this by relaxing that predicate globally
to `>=`, which also changes `conv_2d_dw_direct` and the Vulkan dispatch. Instead the builder's
assert now allows the 1x1 case explicitly, matching what the CPU op's own assert already did. No
global predicate change.

> **CUDA kernel is UNCOMPILED.** There is no CUDA toolchain on the laptop (`nvcc` absent, no
> `/usr/local/cuda`), so `conv2d-deform.cu` has never been through a compiler at `a7478f58`. Static
> audit of its API surface: it uses only `ctx.stream()`, `GGML_ASSERT`, `ggml_is_contiguous` and raw
> CUDA — `ctx.stream()` is unchanged at `common.cuh:1500` and `conv2d-dw.cu:145` calls it
> identically, so drift risk is low but not zero. `ggml-cuda/CMakeLists.txt:105` globs `*.cu`, so
> the new file is picked up with no CMake edit.
>
> **First thing to do on the server:** build `-DGGML_CUDA=ON` and confirm `conv2d-deform.cu`
> compiles. If it does not, the CPU op is complete and correct — the fix is confined to the `.cu`.

#### 1.1a `ggml_concat_n` — not re-ported, consumer changed instead

Upstream has no variadic concat. It was only ever a node-count optimization behind
`model_build_flag::concat_n`, set for the **CPU backend only**, and `ml.cpp` already had a working
`else` branch folding pairwise `ggml_concat`. Re-porting would have meant rewriting the CPU concat
forward against heavy drift *and* teaching CUDA/Vulkan concat to take >2 srcs, for zero benefit on
CUDA where the flag was never set. Dropped the flag instead: `model_build_flag::concat_n` is gone
from `include/visp/ml.h`, the branch is gone from `ml.cpp:868`, and `backend_default_flags(cpu)` no
longer sets it. **No CUDA impact** — pure CPU-path change.

#### 1.1b `conv_2d_direct_cwhn` — removed, and it broke the CPU gate first

Also CPU-only, also removed, but this one is worth reading because it *failed loudly* rather than
silently. `af69870c` carried a channels-last branch in `ggml_conv_2d_direct` that gave the result
permuted (CWHN-in-memory) strides. Without it the result is WHCN-contiguous, so `nn.cpp`'s
`permute_whcn_to_cwhn` on it yields `nb[0] = W*4` — not contiguous rows. Upstream v0.18 *also*
tightened `ggml_unary` to `GGML_ASSERT(ggml_is_contiguous_rows(a))` (the fork had relaxed it for
non-inplace unary). The two together abort Depth-Anything V2 on CPU at
`depth-anything.cpp residual_conv`'s `ggml_relu`.

Fixed stock, without touching ggml: deleted the `conv_2d_direct_cwhn` branch in `nn.cpp:143` so the
cwhn path always takes the `else` branch, which already `ggml_cont`s to WHCN and back. That branch
keeps the same `VISP_IM2COL_MAX` VRAM guard (default 2048 MiB routes big convs to
`ggml_conv_2d_direct`), so the memory protection is unchanged — the cost is extra `cont` copies on
CPU only. **The flag was never set for gpu/vulkan, so CUDA is unaffected**, but note the tightened
`ggml_unary` assert is global and will bite any other code path handing a permuted view to a unary.

### 1.2 Sched hardening regression

`2028dcb8` had three parts, all gone: `hash_set` sized `2x graph_size`
(`ggml-backend.cpp:1789`, FIXME comment restored), `ggml_gallocr_reserve_n`'s return value
discarded again (`:1563`), and `ggml_backend_tensor_copy`'s slow path back to unchecked `malloc`
(`:493`). The first has a real failure mode: the `size >= n_nodes + n_leafs` assert it was added
to prevent. Large graphs are likelier to trip it, so this is more likely to bite the server's big
models than anything here.

### 1.3 Type id 42 — silent misread hazard

The fork had `GGML_TYPE_F8_E4M3 = 42`. Upstream v0.18 assigns `GGML_TYPE_Q2_0 = 42`. Both verified
by reading the two `include/ggml.h` revisions directly. **Any GGUF written with the fork's id-42
FP8 will now be read as Q2_0 — wrong output, no error, no warning.**

Swept all 59 GGUFs on the laptop: zero id-42 tensors, clean. The exposure is on the server, where
`import_ltx_fp8.py` / `import_wan_nvfp4.py` write.

> **Verify on server:** run the header scanner over the whole model store. It only reads GGUF
> headers, so it is fast even over a large collection. Script: `scripts/scan-gguf-type42.py`.

### 1.4 im2col dst_type — a numeric change, not a compile break

Upstream `efee796c` "uniformize im2col dst_type for all conv ops" changed `ggml_conv_2d` to build
its im2col as **F16** (F32 only when the kernel is BF16), where it previously used `a->type`. An
F32 BiRefNet silently gains an F16 intermediate. Nothing errors.

> **Verify on server:** matting output against `bench/` baselines on CUDA specifically. F16
> intermediates behave differently across backends, so a clean Vulkan result does not clear CUDA.

**Measured on CPU: live, and harmless here.** The change only bites when the kernel is F32, because
an F16 kernel already produced an F16 im2col under the old `a->type`. CPU is exactly that case —
`backend_device::preferred_float_type()` returns `GGML_TYPE_F32` for CPU — so the CPU path now runs
the conv GEMM in F16 where it used to run F32. Against `tests/reference/`:

| test | rms vs reference | tolerance | headroom |
|---|---|---|---|
| `birefnet-cpu` | 0.000267 | 0.010 | 37x |
| `depth-anything-cpu` | 0.002774 | 0.010 | 4x |
| `migan-cpu` | 0.000005 | 0.010 | 1846x |
| `esrgan-cpu` | 0.000581 | 0.010 | 17x |

Non-zero, so the change is definitely active, but far below tolerance. **This does not clear CUDA**
— on GPU the BiRefNet weights are F16 and the im2col was already F16, so CUDA sees no change *from
this commit*; the thing to actually check on CUDA is whether the F32 casts `nn.cpp` does for the
deform weight interact with it.

The `bench/` baselines themselves were **not** reproducible here: they are all RMBG-2.0
(`YAVG 181.x`, cat-and-hat @1024) and `RMBG-2.0-F16.gguf` is not on this box — the only BiRefNet
model present is `BiRefNet-lite-F16.gguf`. `bench/matting_bench.sh` also needs `nvidia-smi` for its
VRAM layer. So `tests/reference/birefnet-cpu.png` is the substitute numeric gate used above.

### 1.5 Upstream v0.18 vs the qwen3-tts dropped ops — partial good news

Checked while looking for stock replacements. **Not for action here; hbd-qwen3-tts.cpp untouched.**

- **`ggml_snake` — probably no re-port needed.** Upstream has no `ggml_snake` op and no
  `GGML_UNARY_OP_SNAKE`, but it landed snake as a **graph autofusion**: `ggml-metal-ops.cpp:3132`
  "Snake activation autofuse: mul -> sin -> sqr -> mul -> add", with a matching CUDA path
  (`ggml-cuda.cu:3653` `ggml_cuda_op_snake_fused`, `ggml-cuda/snake.cu`). So expressing snake with
  stock ops gets fused back into one kernel on CUDA and Metal. Worth trying before re-porting the op
  — verify the fusion actually triggers for that repo's tensor shapes/types.
- **`ggml_conv_1d_direct{,_to}` and `ggml_conv_transpose_1d_to` — no upstream equivalent.** Upstream
  has `ggml_conv_1d`, `ggml_conv_1d_dw`, `ggml_conv_1d_dw_ph`, `ggml_conv_transpose_1d`, but no
  direct 1D conv and no `_to` (explicit-dst-type) variants at all. Those need a real re-port.
- **`GGML_OP_COL2IM_1D` is confirmed present** at `include/ggml.h:541` / `ggml_col2im_1d` — the §1
  table's claim holds.

---

## 2. Depth-Anything 3

New model, new arch, `models/Depth-Anything-3-Small-F16.gguf` (507 tensors, 39 KV, 66.5 MB,
`general.architecture=depthanything3`).

**The HF model card is wrong about two things** and the checkpoint is the authority: it is
**34.3M params, not 0.08B**, and it is **not a vanilla DINO**. From block 4 onward it has per-head
QK LayerNorm, 2D RoPE (frequency 100), alternating local/global attention, and a learned camera
token overwriting token 0. The head is a DualDPT — two independent DPT chains (depth + ray).

Monocular use is much cheaper than the architecture suggests: at S=1 the global blocks are
shape-identical to local ones and receive RoPE-neutral positions, so only even blocks ≥4 carry
real 2D RoPE.

Outputs we actually consume: `depth = exp(output_conv2[...,0])` and
`depth_conf = 1 + exp(output_conv2[...,1])`. The ray/camera branch is stubbed — a matted sprite
has no meaningful camera. **Keep `depth_conf`**: the game's inflation mesh blends on it
(`disp = mix(edt_inflation, da3_depth, confidence)`), so a broken confidence map degrades
silhouette quality rather than erroring.

Preprocessing differs from Depth-Anything V2: ImageNet mean/std as before, but **longest side
≤ 504 rounded to a multiple of 14** (an upper bound), where `depthany_image_extent` bounds the
*shortest* side.

> **Verify on server:** DA3 output vs the PyTorch reference on CUDA, not just Vulkan/CPU.
> Measure peak VRAM — the koblem gate needs a real budget number, and 34.3M params means the
> weights are trivial but activations at 504² are not.

### 2.1 Implemented — status and what is new on the op surface

`src/visp/arch/depth-anything-3.{h,cpp}`, wired through `model_family::depth_anything_3`,
`vision-cli depthany3` and the `depthany3_*` API. The DINOv2 block code is **shared, not forked**:
`dino::layer`/`self_attention` gained a defaulted `dino::block_params` (QK norm, RoPE positions),
so the V2 gate is byte-for-byte unaffected — `test_depth_anything` still passes on both backends.

**Verified against the PyTorch reference** (upstream `depth_anything_3` at
`~/models/da3/upstream`, F32, fed the *identical* preprocessed tensor so no resize noise):

| image | backend | depth rel-RMS | conf rel-RMS |
|---|---|---|---|
| wardrobe (364×504) | CPU | 0.00028 | 0.0024 |
| wardrobe (364×504) | Vulkan | 0.00101 | 0.0031 |
| cat-and-hat (504×504) | CPU | 0.00113 | 0.00090 |
| vase-and-bowl (462×504) | CPU | 0.00083 | 0.00319 |

That is F16-weight noise (the GGUF is F16, the reference F32). Correlation ≥ 0.99987 throughout.
**This clears CPU and Vulkan only.** Three things are new on the op surface and none has run on CUDA:

| Op | Where | CUDA note |
|---|---|---|
| `GGML_OP_ROPE` (NEOX, `freq_base` 100, `n_dims` 32) | `dino.cpp rope_2d`, 4 calls per block × 8 blocks | first RoPE use in this repo. Stock upstream kernel, but check it is *selected* — a CPU offload here would be on the critical path |
| `GGML_UNARY_OP_EXP` | `depth-anything-3.cpp head`, both outputs | fed a `ggml_cont` deliberately: upstream v0.18 tightened `ggml_unary` to `GGML_ASSERT(ggml_is_contiguous_rows(a))` (§1.1b). Do not "optimize" the `cont` away |
| `ggml_scale_bias` | confidence `1 + exp(x)` | trivial, but new here |

2D RoPE is applied by slicing each head's 64 features in half, `ggml_cont`-ing each half and running
two NEOX ropes (y then x), then concatenating. ggml's `GGML_ROPE_TYPE_VISION` looks like a fit and
**is not** — it pairs dim `i` with `i+32` *across* the two halves, where DA3 rotates within each
half. If someone "simplifies" this to one VISION rope it will be silently wrong.

`hbd-longcat-avatar.cpp/tools/check_rope.py` remains the right oracle if CUDA RoPE disagrees.

### 2.2 The position-embedding kludge is resampled on the host, at graph-build time

DINOv2 interpolates its 37×37 position grid by passing `scale_factor = (target + 0.1) / 37` rather
than a target size. ggml's `interpolate` is size-based only, and ignoring the offset is **not**
cosmetic — measured against torch, 6–10% RMS on the embedding itself (DA3 always runs off-grid:
504 → 36×36 vs M=37). So `da3::create_pos_encoding` reads the weight back with
`transfer_from_backend` and does a PyTorch-exact bicubic (Keys A = −0.75, replicated border) on the
CPU, once per resolution.

Two CUDA-relevant consequences:

1. It performs a **device→host read of a model weight during graph construction**. Fine on CUDA, but
   it means graph build is no longer pure — a lazily-loaded or offloaded weight buffer would break it.
2. It asserts the weight is F32. `preferred_float_type()` returns `GGML_TYPE_COUNT` for GPU, so the
   converter's explicit-F32 `position_embeddings` / `cls_token` / `camera_token` survive as F32 while
   everything else is F16. Any future device that forces F16 weights trips that assert, and would
   also make the camera-token `ggml_concat` (F32 token into F32 activations) type-mismatch.

### 2.3 Persistent constant buffers — a real, if small, VRAM line item

`depthany3_precompute` pins 9 buffers into the graph buffer as input+output (never reused for
scratch), sized by resolution. At 504²:

| Buffer | Size |
|---|---|
| `da3.uv_out` (32ch UV embedding at full res) | 32.5 MB |
| `da3.uv_0..3` (48/96/192/384ch at 36×36) | 3.7 MB |
| `da3.pos_embed` | 2.0 MB |
| `da3.rope_{y,x,c}` | 15 KB |

≈ 38 MB, F32, dominated by the full-resolution UV embedding. It is separable in principle
(`sin/cos(u)` on the first half of the channels, `sin/cos(v)` on the second) but expressing that in
ggml needs two `ggml_repeat`s that materialise the same bytes as transient graph memory, so storing
it won is the cheaper option. If the 5060 turns out tight, this is the knob: store F16 and
`cast_like` at the add.

Laptop peak RSS at 504×504 (whole process, weights + graph): **499 MiB CPU / 170 MiB host with
Vulkan**. Neither is a VRAM number — the iGPU is UMA. **The gate budget still needs a real
`nvidia-smi` figure on the target card.**

### 2.4 The ray/camera branch is loaded but not built

`refinenet*_aux`, `output_conv{1,2}_aux.*`, `cam_dec.*`, `cam_enc.*` are all still in the GGUF and
still transferred to the device by `model_transfer` (which loads every tensor unconditionally), so
they cost VRAM but contribute no graph nodes. Roughly a third of the 66.5 MB file. If VRAM matters
more than round-tripping the converter, re-running `convert.py` without them is the cheap win —
nothing in the C++ references them.

---

## 3. Test cases

| # | What | How | Status |
|---|---|---|---|
| 1 | ggml builds at `a7478f58` | cmake, CPU+Vulkan | **passed** — library, `vision-cli`, `matting-server`, tests, zero warnings |
| 2 | Depth-Anything **V2** still works | `tests/test-models.cpp::test_depth_anything` vs `Depth-Anything-V2-Small-F16.gguf`, both backends | **PASSED both backends** — the DA3 gate is green |
| 2b | Rest of `test-models` | `./build-vk/bin/test-models -v` | **11 passed, 1 skipped, 1 failed** — only `test_birefnet[gpu]`, see §5 |
| 2c | DA3 regression gate | `test_depth_anything_3[cpu,gpu]`, depth + confidence | **PASSED both backends**; references generated locally, **not on the CDN** — the test self-skips without them, run `scripts/upload_references.py` to fix |
| 3 | CONV_2D_DEFORM on CUDA | present, selected, not falling back | **unverified — CUDA only**, and the `.cu` has never been compiled (§1.1) |
| 4 | Matting numeric parity | output vs `bench/` baselines | **unverified on CUDA**; `bench/` not reproducible on laptop, substitute gate in §1.4 |
| 5 | Matting wall-clock | vs `bench/PROFILE.md` | **unverified — CUDA only** (§1.1) |
| 6 | GGUF type-42 sweep | `scripts/scan-gguf-type42.py` over model store | clean on laptop, **unverified on server** |
| 7 | DA3 vs PyTorch reference | out-of-tree harness against upstream `depth_anything_3` | **PASSED on CPU + Vulkan**, rel-RMS ≤ 0.003 (§2.1). **unverified on CUDA** |
| 8 | DA3 peak VRAM | for the koblem gate budget | **unverified — CUDA only**; laptop host-RSS numbers and the 38 MB constant-buffer breakdown in §2.3 |
| 9 | Unified server co-residency | matting + depth + siglip2 resident together | **unverified — see §4** |
| 10 | DA3 RoPE / EXP / scale_bias on CUDA | present, selected, not CPU-offloaded | **unverified — CUDA only** (§2.1) |

`tests/workbench.{cpp,py}` is the harness of choice for #7 — a ctypes bridge that invokes a single
ggml module against torch tensors in-process, already wired for `dino.h` and `depth-anything.h`.
Preferable to the dump-file oracle pattern in hbd-longcat-avatar.cpp/tools/, though that repo's
`check_rope.py` is still worth borrowing: DA3's 2D RoPE (split-half rotation, per-axis y/x,
frequency 100) is exactly the kind of thing that mismatches silently.

---

## 5. BiRefNet is broken on Vulkan, and it is not a fallback — there is no fallback

Requested: record where the Vulkan scheduler falls back to CPU for an op CUDA would keep on GPU.
The finding is worse than that. **There is no scheduler on the inference path at all.**

`src/visp/ml.cpp` `compute()` calls `ggml_backend_graph_compute(b, g.graph)` — a single-backend
compute. `compute_graph::cpu_fallback` and `compute_graph::sched` (`include/visp/ml.h:160-161`) are
**declared and never read**, and the comment above them describing scheduler routing for
`CONV_2D_DEFORM` describes machinery that does not exist. This is pre-existing, not fallout from the
bump: `git show 14d9787:src/visp/ml.cpp` — the commit that added those fields — already has the
one-line `compute()`. `af69870c`'s own message says why ("Keeps the op on the GPU, no scheduler CPU
fallback / copies"): the CUDA kernel replaced the fallback and the dead fields were left behind.

Consequence on Vulkan: `CONV_2D_DEFORM` has no Vulkan kernel, the graph is handed straight to the
Vulkan backend, and the op is simply **not executed**. Its destination buffer is never written, so
BiRefNet returns uninitialised memory. Not a tolerance failure — `tests/results/birefnet-gpu.png` is
pure noise (rms 1.15 against a 0.015 tolerance; mean 70.3 vs the reference's 130.8). Confirmed
directly: the `VISP_PROFILE_OPS` path is the *only* place that builds a `ggml_backend_sched` with a
CPU backend appended, and forcing it produces the honest error the default path swallows —

```
VISP_PROFILE_OPS=CONV_2D_DEFORM ./build-vk/bin/test-models -v test_birefnet
ggml-backend.cpp:898: pre-allocated tensor (squeeze_module.0.dec_att.aspp1.conv)
  in a buffer (Vulkan0) that cannot run the operation (CONV_2D_DEFORM)
```

(That path fails too, because `compute_graph_allocate` has already pinned the tensors into the
Vulkan buffer via `gallocr` before any scheduler sees them.)

**This should not affect CUDA**, which has the kernel — but it is exactly the case the divergence
warning in §0 is about, in the direction that flatters CUDA rather than the laptop. Two things
follow:

1. Vulkan BiRefNet was already broken before this ggml bump and is out of scope here; it needs
   either a Vulkan `CONV_2D_DEFORM` kernel or `compute()` actually wired to a scheduler (which means
   giving up `gallocr` pre-allocation, so it has real VRAM consequences on the 3060 — do not do it
   casually).
2. `test_birefnet[gpu]` failing on the laptop is **not** a signal about CUDA. The meaningful laptop
   gate for the deform op is `test_birefnet[cpu]`, which passes at rms 2.7e-4. Do not "fix" the
   Vulkan failure by loosening the tolerance.

The stale comment at `include/visp/ml.h:157-161` says `CONV_2D_DEFORM` "has no CUDA kernel". That
was wrong before this task and is still wrong — it does, and it is back. Left in place rather than
edited to avoid colliding with concurrent DA3 work in the same header.

---

## 4. Deployment notes for the server

- **The unified container** (matting + depth + siglip2 in one binary) changes the co-residency
  picture. BiRefNet/RMBG-2.0 alone peaks ~3.5 GB at `process_res` 2048 and must be clamped
  server-side. Adding DA3 and SigLIP2 to the same process means one gate budget covering all
  three. Measure the real combined peak on the target card before setting it.
- **Every gated submit must carry the `gpu` field.** Disabled-card enforcement depends on it;
  an engine that omits it silently ignores operator card toggles. Matting's `/remove` passes it
  as a multipart field — the new `/depth` endpoint must do the same.
- **Composite RGBA onto mid-grey before depth inference**, never black or white. Depth models read
  a black field as a hole and a white one as sky, and either drags the subject's relief with it.
  Those pixels are discarded via alpha anyway.
- **`.gitmodules` uses the HTTPS fork URL**, not SSH — Docker builders clone with no SSH agent.
  hbd-longcat-avatar.cpp learned this the hard way and carries a comment saying so.
- **Do not repoint hbd-qwen3-tts.cpp** at ggml master until its dropped ops are re-ported (§1).

### Open — fill in on the server

| Question | Why it matters |
|---|---|
| 3060 / 5060 VRAM, and which card hosts the vision container | sets the gate budget and the matting `process_res` clamp |
| Combined peak VRAM: matting + DA3 + siglip2 resident | §4 co-residency |
| ~~Does upstream v0.18 cover any qwen3-tts dropped ops~~ | answered in §1.5 — snake yes (as autofusion), conv_1d_direct no |
| Does `conv2d-deform.cu` still compile at `a7478f58` | §1.1 — never compiled, no CUDA toolchain on the laptop |
| Does the deform kernel still get *selected* (not CPU-offloaded) on CUDA | `supports_op` gates on contiguous whcn F32; confirm BiRefNet hits that, not the cwhn CPU path |
| Push `depend/ggml` branch `vision-conv2d-deform` (`a1bf8b90`) | local-only right now; the superproject submodule pointer is uncommitted |

---

## 5. SigLIP2 fold-in (task 6) — what runs on this laptop, what needs CUDA

`hbd-siglip2.cpp` was CUDA-only in practice. Its model code now lives here under
`src/siglip2/` and is served by `vision-server` (renamed from `matting-server`;
`src/server/` split into `server_ipc` / `server_state` / `worker` / `*_endpoints` / `main`).
Verified on this box: **AMD Radeon 890M (RADV STRIX1, gfx1150) via Vulkan, and CPU.**

### 5.1 SigLIP2 needs nothing CUDA-only

Every op in both towers runs on Vulkan and on CPU as-is:
`mul_mat` (F16 weight x F32 activation), `norm`, `gelu`, `soft_max_ext`,
`flash_attn_ext`, `interpolate` (BILINEAR|ANTIALIAS), `get_rows`, `pad`, `cast`, `mean`.
The two CUDA-shaped hacks that came with the source were carried but are inert here:

- **FA d-head padding to a multiple of 16** (`SIGLIP2_DISABLE_FA_PAD=1` opts out).
  It exists because CUDA's MMA/WMMA flash-attn tiles reject so400m's `d_head=72`.
  The `-base-` checkpoint has `d_head=64`, so the pad is a no-op there — **the
  d_head=72 path is unexercised on this laptop and still needs a so400m run**.
- **`SIGLIP2_DISABLE_FA=1`** falls back to explicit `mul_mat`+`soft_max_ext` if a
  backend's `flash_attn_ext` misbehaves. Not needed on RADV; left in as the escape hatch.

Dropped on the way in, all CUDA-only and none load-bearing for correctness:
the whole `src/cuda/` megakernel (fused LayerNorm / QKV-prep / custom MMQ, plus its
QKV scratch slabs and CUDA-graph priming), per-encoder private CUDA streams with
`SIGLIP2_VISION_STREAM_PRIORITY`, and the `siglip2_megakernel::profile_*` hooks.
On CUDA these were worth ~30% on the text encode; re-add them only against a
measured regression, and only behind `GGML_CUDA`.

### 5.2 Numeric parity, measured here

vs HuggingFace `google/siglip2-base-patch16-naflex` in F32 (`transformers` 5.14.1),
`tests/input/cat-and-hat.jpg`, `max_num_patches=256`, F16 GGUF:

| | image emb cosine | text emb cosine |
|---|---|---|
| CPU | 0.999474 | 1.000000 |
| Vulkan | 0.999439 | 0.999962 |

Above the source repo's documented 0.999 parity floor on both backends. `/v1/classify`
and `/v1/classify_from_embeddings` return bit-identical scores for the same image, which
is the contract koblem relies on when it classifies persisted embeddings.

Warm timings on the 890M via Vulkan: ~28 ms per `/v1/embeddings` (base, 256 patches),
~28 ms per 2-prompt `/v1/text_embeddings`, ~65 ms for a 2x2 `/v1/classify`.
CPU is ~0.45 s per image embed. so400m will be roughly 3x the base model.

### 5.3 `VISP_F16_ENCODER` is CUDA-only and was aborting everything else

`src/visp/arch/swin.cpp:258` reads the flag with `getenv(...) != nullptr`, i.e. it tests
**presence**, so `VISP_F16_ENCODER=0` *enables* it. The old `birefnet_server.cpp` hard-set
it to `1` for every backend, which means matting has never worked on this laptop:

- CPU: `ggml-cpu/ops.cpp` `ggml_compute_forward_norm` has no F16 case -> `GGML_ABORT`.
- Vulkan: `ggml-vulkan.cpp:11659`, no F16 unary pipeline -> `GGML_ABORT`.

Both reproduce with the plain CLI (`VISP_F16_ENCODER=1 vision-cli birefnet -b cpu|gpu`),
so this is pre-existing, not a fold-in regression. `worker.cpp` sets the flag only
when the initialized device is `backend_type::gpu` (CUDA/HIP) and `unsetenv`s it otherwise
(Vulkan is `backend_type::vulkan`, so it takes the force-off branch); with that, matting is
green on CPU (~0.95 s @512) and Vulkan (~0.76 s @1024) here.

#### 5.3a Fixed — and the CUDA path to F16 changed as a result

`swin.cpp` now reads the flag through `visp::env_flag()` (`src/util/env.h`): unset falls back
to the caller's default, `0`/empty/`false`/`no`/`off` are off, `1`/`true`/`yes`/`on` are on
(case-insensitive), unrecognised values fall back to the default. It follows the allow-list
convention already used by `env_truthy()` in `src/server/main.cpp`. `worker.cpp` needed no
functional change — "unset" still means off — only the now-stale comment sentence was replaced.

Verified on the laptop (CLI + server, both backends): unset -> off, `0` -> off, `1` -> on.
`unset` and `0` produce byte-identical masks; `1` still aborts at `ops.cpp:3780` (CPU) and
`ggml-vulkan.cpp:11659` (Vulkan), which is what proves the flag is genuinely being read.
The pre-fix binary in `build-vk/` aborts on `VISP_F16_ENCODER=0`; the fixed one does not.
`test-models` baseline unchanged: 11 passed / 1 skipped / 1 failed (`test_birefnet[gpu]`, §5).

**Check on the 3060/5060.** The F16 encoder path is now reached by a different code path than
before, and the change is silent in both directions:

- Anything that set `VISP_F16_ENCODER` to a falsy value — `0`, `false`, a stray empty
  assignment in a compose file or unit — used to get F16 **on** and now gets it **off**. That
  loses the `bench/VRAM.md` VRAM/latency win without any error. Grep the deployment for the
  variable before assuming the numbers still hold.
- Conversely a truthy value now behaves as documented for the first time. On CUDA the flag
  should normally be left unset and delegated to `worker.cpp`, which sets it to `1` with
  `overwrite=0` when the device comes up as `backend_type::gpu` — so an explicit operator
  value still wins. Confirm the CUDA worker actually reports `backend_type::gpu` (not
  `vulkan`) and that F16 activations still deliver the measured win.

Two more flags were audited at the same time. `VISP_FLASH_ATTENTION` was already tri-state
but inspected only the first character, so `true`/`yes`/`on` silently fell through to the
per-model default; it now goes through `env_flag(name, default_enabled)` and accepts the full
spelling set. `SIGLIP2_DISABLE_FA` had the identical presence bug as `VISP_F16_ENCODER`:
`SIGLIP2_DISABLE_FA=0` **disabled** flash-attention on CUDA, the opposite of the intent, and
`=1` and `=0` were indistinguishable. Both `vision.cpp` and `text.cpp` are fixed.
`SIGLIP2_DISABLE_FA_PAD` was already parsed by value and is unchanged in behaviour — but note
that the §5.5 so400m `d_head=72` pad experiment on CUDA must now use `=1` to disable the pad;
under the old code any value, including `0`, would have done it for `_FA` but not for `_FA_PAD`.

### 5.4 sentencepiece needs two build patches

`v0.2.0` does not build with a current toolchain: `cmake_minimum_required(VERSION 3.1)`
is rejected by CMake >= 4, and `sentencepiece_processor.h` uses `uint32_t` without
`<cstdint>` (GCC >= 13). `src/siglip2/CMakeLists.txt` sets `CMAKE_POLICY_VERSION_MINIMUM`
around the fetch and force-includes `<cstdint>`; the train/CLI targets are
`EXCLUDE_FROM_ALL` since only `sentencepiece-static` is linked.

### 5.5 Still unverified — do on the server

| Question | Why |
|---|---|
| so400m (`d_head=72`) on CUDA, with and without `SIGLIP2_DISABLE_FA_PAD` | the pad exists only for that shape; untestable on the base checkpoint |
| Q4_K / Q8_0 GGUFs through the K-padding path (`pad_x_to_w`) | only F16 was exercised here; `siglip2-quantize` was **not** ported |
| Combined resident VRAM: BiRefNet + DA3 + SigLIP2-so400m in one worker | the whole point of the fold-in; sets the gate budget |
| Re-add the megakernel behind `GGML_CUDA`? | ~30% of the CUDA text encode; measure first |

---

## 6. `/depth` — Depth-Anything 3 folded into `vision-server` (2026-08-02, Vulkan/gfx1150)

> **Superseded in part by §10.** `/depth`'s DEFAULT model is now Depth-Anything V2; DA3 is
> `?model=depth-anything-3`, and `process_res` no longer exists. The plumbing, the mid-grey
> composite (§6.4) and the DA3 parity and VRAM figures below all still stand.

Third endpoint on the same worker. Shape is exactly the one the restructure predicted:
`src/server/depth_endpoints.cpp` + one frame pair (`DEPTH_REQ 0x50` / `DEPTH_RESP 0x51`)
+ one slot in `WorkerModels`. Nothing under `src/visp/arch/` was touched.

### 6.1 Verified here (Vulkan, AMD 890M, RADV)

All three families served by ONE worker pid, one device, one budget — `/remove`,
`/depth`, `/v1/embeddings` and `/v1/classify` all answered 200 against a single warm
process. `gpu` is honoured on `/depth` from both the multipart field and the query
string, and it respawns the worker (`respawn worker (backend vulkan->gpu, gpu ''->'0')`)
exactly like `/remove`. Idle-unload evicts and the next `/depth` transparently reloads.

Parity vs `vision-cli depthany3` on the same image/backend: depth **corr 0.99999**,
mean |diff| 0.0015 (the CLI reference is 8-bit, which alone accounts for ~0.002);
confidence corr 0.99999. Warm `/depth` at native 504: **~0.17–0.38 s** end-to-end
including PNG encode.

### 6.2 Per-process GPU allocation, staged (fdinfo `drm-total-vram`, cumulative)

| after | per-process |
|---|---|
| worker spawn, BiRefNet-lite F16 weights loaded, no inference | 84 MiB |
| + one `/depth` @504 (DA3-Small F16 weights + graph) | 385 MiB |
| + one `/depth` @1008 (`process_res=1008`) | 1081 MiB |
| + one `/v1/embeddings` (SigLIP2-base F16) | 1802 MiB |
| + one `/remove` @1024 | 2423 MiB |

Monotonic high-water of the Vulkan allocator, not a concurrent peak — but for a
single worker that is the number that matters. **The transferable figure is the DA3
delta: ~300 MiB at its native 504, ~700 MiB more if you force 1008.** Everything else
is Vulkan/UMA-specific and will not match CUDA (§"GPU PEAK VRAM REDUCTION" measured
3072 MiB for RMBG-2.0 @1024 on CUDA alone).

### 6.3 CUDA-relevant, unverified here

| Question | Why it matters |
|---|---|
| Combined CUDA peak: RMBG-2.0 @1024 + DA3-Small @504 + so400m in one worker | sets the gate budget. Vulkan says +300 MiB for DA3; CUDA's F32 activation shape may differ |
| `VISP_F16_ENCODER` with DA3 resident | `worker.cpp` sets it only when `device->type() == backend_type::gpu` (CUDA/HIP — Vulkan is a distinct enum value, so it stays OFF here). Only `swin.cpp` reads it, so DA3 should be unaffected, but that is untested on CUDA |
| `VISP_FLASH_ATTENTION=0` costs DA3 ~20% | `worker.cpp` forces it off (with `overwrite=0`) for BiRefNet's head_dim=32 Swin. Measured here: DA3 CLI 113 ms with FA, 135 ms without. Harmless but it is a matting-motivated default now applied process-wide; set `VISP_FLASH_ATTENTION=1` in the container env if depth latency matters more than matting's |
| 16-bit PNG encode cost at 1024² on the server CPU | the endpoint hand-rolls a PNG16 writer (stb only does 8-bit) using `stbi_zlib_compress` + Sub filter. Two maps per request; ~340–430 KB each at 512² here |

### 6.4 Mid-grey compositing — measured, not assumed

`/depth` composites RGBA over **mid-grey** before inference. Same cut-out, four ways,
comparing depth over the subject's own (alpha>0) pixels against the RGBA input:

| background | mean abs diff vs RGBA input | subject depth range |
|---|---|---|
| grey (128) | **0.00000** | 0.822 .. 1.123 |
| black | 0.29121 | 1.096 .. 1.418 |
| white | 0.06721 | 0.937 .. 1.021 |

Grey matching the RGBA path exactly confirms the composite fires. Black shoves the whole
subject 0.29 further from the camera; white collapses its relief from 0.301 to 0.084 — it
reads the field as sky and flattens the silhouette. Neither is recoverable downstream,
which is why this is not a knob.

---

## 7. DA3 on flat-lit cut-out sprites — paperworld's watershed premise (NEGATIVE)

> **RETRACTED — §9 overturns this section. Its corpus was invalid.** Every number below was
> measured on `paperworld/server/stub_images/*.png`, which are the `PAPERWORLD_STUBS=1` test-seam
> placeholders — **six flat colours per image**, no shading, no gradients, and RGB `(0,0,0)` under
> the alpha, i.e. no underlying render at all. They are not krea2 output and never were. "A depth
> model keys on albedo when handed an image made entirely of albedo" is a tautology, and the
> 10–18x luminance-edge lift below is an artifact of an 8% ink base rate that only flat vector art
> can produce. On real krea2 renders the same lift is 2.2x, against 1.7x for a photograph of a real
> 3D object through the identical pipeline. **Read §9 instead; keep this only as a record of a
> degenerate input class.**

CPU/Vulkan-valid: this is about what the model *says*, not how fast a kernel runs. Nothing
here needs re-measuring on CUDA.

**The claim under test.** `kob/paperworld/server/src/watershed.rs` is built on one assertion:
that DA3 depth splits parts that share a silhouette — an arm crossing a torso, or a head
sitting on shoulders with no neck to pinch at, has no alpha boundary but a clear depth step.
Real depth had never been obtainable when it was written (koblem's route is Bearer-gated,
`DEPTH_SERVICE_URL` is an internal docker name), so its published numbers used the distance
field's own ridges as a synthetic stand-in. The premise had never been tested.

**Rig.** `build-depth/bin/vision-server`, `Depth-Anything-3-Small-F16.gguf`, `backend=cpu`,
`process_res=504` (native). Corpus: the ten `paperworld/server/stub_images/*.png`. Input built
exactly as `Cutout::for_depth` does — crop to subject, composite on mid-grey 128, CatmullRom to
504 longest side. Response parsed exactly as `paperworld/server/src/vision.rs::relief` does,
including the `FLAT_CONFIDENCE` fallback (which never fired: no sprite had a flat confidence map).
~0.6 s/sprite on CPU.

### 7.1 The model is not broken and the depth is not flat

| sprite | img depth range | subject range | subject/img |
|---|---|---|---|
| thing_00 | 0.845 .. 1.575 (0.731) | 0.845 .. 1.528 (0.683) | 93.5% |
| thing_01 | 0.917 .. 1.312 (0.395) | 0.922 .. 1.253 (0.331) | 83.7% |
| thing_02 | 0.864 .. 1.414 (0.549) | 0.864 .. 1.282 (0.417) | 76.0% |
| thing_03 | 0.920 .. 1.585 (0.666) | 0.920 .. 1.466 (0.546) | 82.0% |
| thing_04 | 0.909 .. 1.301 (0.391) | 0.909 .. 1.273 (0.363) | 92.9% |
| thing_05 | 0.867 .. 1.303 (0.436) | 0.867 .. 1.249 (0.383) | 87.7% |
| thing_06 | 0.897 .. 1.560 (0.664) | 0.897 .. 1.441 (0.545) | 82.1% |
| thing_07 | 0.862 .. 1.457 (0.595) | 0.862 .. 1.343 (0.481) | 80.9% |
| thing_08 | 0.919 .. 1.161 (0.242) | 0.919 .. 1.136 (0.217) | 89.8% |
| thing_09 | 0.889 .. 1.131 (0.242) | 0.899 .. 1.126 (0.226) | 93.6% |

Confidence (raw, pre-normalisation) spans 0.97..1.00 low to 1.77..5.15 high; nothing degenerate.
The same server on `tests/input/cat-and-hat.jpg` and `vase-and-bowl.jpg` returns textbook depth —
the hat separates from the cat's head, the table from the wall, the floor recedes. **The model
and this build are healthy. The negative below is about the input class, not the port.**

But 37–85% of each subject's apparent depth range lives within 2 mask texels of the alpha edge —
it is matte-edge halo, not relief:

| sprite | subject span | span excl. d<=2 | excl. d<=4 | interior/all |
|---|---|---|---|---|
| thing_00 | 0.7248 | 0.3494 | 0.2188 | 48% |
| thing_01 | 0.3829 | 0.1409 | 0.1285 | 37% |
| thing_02 | 0.5130 | 0.2553 | 0.2500 | 50% |
| thing_03 | 0.6549 | 0.3340 | 0.3255 | 51% |
| thing_04 | 0.3709 | 0.1649 | 0.1413 | 44% |
| thing_05 | 0.4308 | 0.2386 | 0.1329 | 55% |
| thing_06 | 0.6559 | 0.4027 | 0.2225 | 61% |
| thing_07 | 0.5301 | 0.2206 | 0.2138 | 42% |
| thing_08 | 0.2353 | 0.1736 | 0.1626 | 74% |
| thing_09 | 0.2256 | 0.1920 | 0.1829 | 85% |

### 7.2 There is no step at the head/torso junction

Median interior depth in the 5 mask texels above vs below the join the alpha-only rig already
found. This is the exact place a step would have to be for the watershed to earn its keep.

| sprite | above | below | jump / subject range | jump / interior range |
|---|---|---|---|---|
| thing_00 | 0.8829 | 0.8904 | 1.0% | 2.2% |
| thing_01 | 0.9739 | 0.9716 | 0.6% | 1.6% |
| thing_02 | 0.9220 | 0.9171 | 0.9% | 1.9% |
| thing_03 | 0.9315 | 0.9320 | 0.1% | 0.2% |
| thing_04 | 0.9280 | 0.9315 | 0.9% | 2.1% |
| thing_05 | 0.9278 | 0.9266 | 0.3% | 0.5% |
| thing_06 | 0.9401 | 0.9410 | 0.1% | 0.2% |
| thing_07 | 0.9161 | 0.9160 | 0.0% | 0.1% |
| thing_08 | 0.9870 | 0.9830 | 1.7% | 2.3% |
| thing_09 | 1.0183 | 1.0071 | 5.0% | 5.8% |

Nine of ten are under 2.3% of the subject's own interior range, and the **sign alternates** —
on five the head reads *further* than the torso. A real head-in-front-of-shoulders step would be
consistently signed. Down the vertical centreline the subject is a smooth monotonic ramp with a
single spike at the top row (the halo), never a plateau-step-plateau.

### 7.3 What DA3 *does* respond to: paint, not shape

Interior texels only (>3 from the alpha edge), depth-gradient magnitude vs input-luminance-
gradient magnitude on the mask grid. "Ink" = luminance gradient > 8/255.

| sprite | Pearson | top 1% depth gradients on ink | base rate | lift |
|---|---|---|---|---|
| thing_00 | 0.113 | 57.7% | 5.6% | 10.3x |
| thing_01 | 0.268 | 96.9% | 5.4% | 18.1x |
| thing_02 | 0.236 | 78.1% | 5.4% | 14.6x |
| thing_03 | 0.104 | 61.9% | 5.6% | 11.0x |
| thing_04 | 0.280 | 78.9% | 5.1% | 15.5x |
| thing_05 | 0.198 | 86.2% | 6.7% | 12.9x |
| thing_06 | 0.285 | 73.8% | 4.9% | 15.1x |
| thing_07 | 0.255 | 84.2% | 5.8% | 14.4x |
| thing_08 | 0.218 | 82.6% | 5.4% | 15.4x |
| thing_09 | 0.414 | 94.3% | 5.9% | 15.9x |

Rendering the cut overlays makes it unmissable: **every cut the watershed produces traces the
outline of the sprite's lighter belly patch.** DA3 reads a flat albedo boundary in vector art as
a depth discontinuity. On a photograph, shading and albedo are entangled and this is the right
prior; on flat-lit generated sprites there is no shading at all, so albedo is the only thing
left for it to key on.

### 7.4 Head width, real depth vs both baselines

`head.radius[0] * 2.0` as a fraction of sprite width, same measurement as the original task.

| sprite | alpha-only | synthetic ridges | **real DA3** |
|---|---|---|---|
| thing_00 | 0.610 | 0.610 | 0.610 |
| thing_01 | 0.563 | **0.120** | 0.563 |
| thing_02 | 0.374 | 0.374 | 0.374 |
| thing_03 | 0.663 | 0.663 | 0.663 |
| thing_04 | 0.765 | **0.212** | 0.765 |
| thing_05 | 0.555 | 0.555 | 0.555 |
| thing_06 | 0.485 | 0.485 | 0.485 |
| thing_07 | 0.640 | **0.138** | 0.640 |
| thing_08 | 0.513 | 0.513 | 0.513 |
| thing_09 | 0.345 | 0.345 | 0.343 |

**Real depth changes nothing on 9 of 10, and thing_09's 0.345 -> 0.343 is a 1.3% cut, i.e.
noise.** The synthetic stand-in split 3 of 10; real depth splits 0 of 10. Part counts are
unchanged (3 everywhere), so nothing is lost either — `derive`'s keep-or-drop rule is doing its
job and the feature is inert rather than harmful.

### 7.5 Are the thresholds wrong, or is the signal absent? Both — but fixing the thresholds does not help

`STEP = 0.04` is the binding constraint (`RIDGE` passes on 2–6 boundaries per sprite; `TRUSTED`
passes on 5–14). Median boundary steps are 0.005–0.026 against a 0.04 threshold, so almost
nothing clears it.

Two genuine bugs found, both real and both insufficient:

1. **Confidence damping is uncalibrated.** `gradient()` multiplies every gradient by the
   min-neighbourhood confidence, and `TRUSTED`/`STEP` were tuned against the synthetic relief's
   uniform `confidence = 1.0`. Real DA3 confidence over the subject averages 0.48–0.67, so every
   gradient is scaled by roughly half before meeting a threshold calibrated for undamped values.
   Measured: hardest boundary step drops 3–5x with damping on (thing_05 0.2822 -> 0.0679,
   thing_09 0.1504 -> 0.0437, thing_00 0.0698 -> 0.0145).
2. **The confidence map's dip is in the wrong place for this use.** It *does* behave as
   documented — mean confidence rises monotonically with distance from the alpha edge
   (0.13–0.54 at d<=1, 0.56–0.71 at d>8, and lowest of all on the background) — so the
   "unreliable at matte edges and on thin limbs" claim holds. But a head sitting on shoulders
   joins at a *silhouette pinch*, i.e. near the outline, which is exactly where confidence is
   lowest. The backstop damps hardest precisely where the watershed most wants to cut.

Neither matters, because the recovered signal is not the neck. Sweeping `STEP` down with damping
disabled does move head widths (5 of 10 change by 0.02), but the cut centroid lands **44–115
mask texels below** the alpha rig's head/torso join on every single sprite, and the overlays show
it tracing the belly patch. Head width falls because the lower body fragments and the finders
reshuffle, not because a head came off. Lowering `STEP` to 0.01–0.02 also produces obvious
garbage (thing_02 head -> 0.032, thing_08 -> 0.030 — 3% of sprite width is a speck).

**No thresholds were changed.** There is no setting of `STEP`/`RIDGE`/`TRUSTED` that finds necks
here, because there is nothing at the neck to find.

### 7.6 Verdict

**The watershed premise does not hold for flat-lit generated cut-out sprites.** DA3 is working
correctly and returns a coherent, wide-range relief — it is simply keying on albedo, because a
cut-out with no scene, no ground plane, no perspective cues and no shading gives a model trained
on photographs nothing else to key on. The head/torso junction carries 0.0–2.3% of the subject's
interior depth range on nine of ten sprites, with inconsistent sign.

Consequences:

- The alpha-only rig in `paperworld_shared::rig` is the real product. `watershed.rs` and the
  `/depth` round trip in `refine.rs` are dead weight on this class of image, and paperworld
  should not pay a GPU call per summon for them.
- The failure is input-class-specific, not a port defect. If paperworld ever feeds photographs
  or shaded 3D renders, re-test before concluding anything — the cat/hat separation shows the
  mechanism works when the input has real shading.
- Any future attempt should either (a) light the sprite generator so albedo and geometry stop
  coinciding, or (b) drop depth and reach for a part-segmentation model, which is what the
  problem actually is.

Reproduction: the probe was a temporary `cfg(test)` module in paperworld and has been removed.
It hit `POST /depth?backend=cpu&process_res=504` with `Cutout::for_depth` bytes and re-ran
`watershed::split` on the result; four measurements (subject depth range, junction step,
albedo correlation, cut centroid vs join) are all that is needed to reproduce the tables above.

---

## 8. DA3 depth as inflation relief — the second consumer (NEGATIVE)

> **RETRACTED — §9 overturns this section.** Same invalid corpus as §7 (six-colour placeholder art,
> not krea2 renders), and the same consequence: the "pale belly stands proud by up to 16.7% of the
> crown" result is measuring the only interior feature those images have. On real renders the same
> pale-minus-dark residual is **+0.004–0.005 crowns**, ~13x smaller. The `distance <= 0` rim-gate
> finding (§8.4) and the two latent bugs it names are corpus-independent and still stand; the
> verdict does not. **Read §9.**

CPU/Vulkan-valid, same as §7: this is about what the model says, not how fast it says it.

**The claim under test.** `paperworld/client/src/inflate.rs::displacement` computes
`disp = mix(edt_inflation, da3_depth, confidence)` and still passes `None`. §7 killed the
watershed; this blend was the only remaining justification for a GPU call per summon. The
specific worry, following from §7.3: if DA3 keys on albedo then the blend puffs a sprite
according to its lighter patches rather than its form.

**Rig.** Identical to §7 — `build-depth/bin/vision-server`, `Depth-Anything-3-Small-F16.gguf`,
`backend=cpu`, `process_res=504`, input built by `Cutout::for_depth`, response parsed by
`vision.rs::relief` including the `FLAT_CONFIDENCE` fallback (again never fired: raw confidence
spans 0.97..1.00 low to 1.77..5.15 high). The returned `depth_min`/`depth_max` match §7.1 to
three decimals, so the two evaluations are looking at the same maps. Measured in an external
scratch harness with `displacement` and `inflate_rigged` copied verbatim and
`paperworld_shared::mask::DistanceField` used as-is. **Nothing in paperworld was changed.**

**Normalisation into `DepthMap`'s documented units** (half-thickness toward the viewer as a
fraction of sprite width). Lower DA3 value = nearer, verified on `tests/input/vase-and-bowl.jpg`
(foreground floorboards 0.84, back wall 1.10). The subject's own depth range is then rescaled so
its nearest texel gets exactly the height the pure EDT gives the medial axis
(`ROUNDNESS * peak / width`) and its furthest gets zero — DA3's scale is affine-free, so the
inflation's own crown is the only anchor available, and this is not tuned per sprite. Every
measurement below was also run at half that gain (which is roughly where mean-matching would put
it: DA3's mean over the subject is 0.67–0.95 crowns against the inflation's 0.52–0.59). Halving
changes nothing qualitative and makes §8.3 worse; it is reported where it differs.

### 8.1 What comes back is a slab, and DA3 is right about that

Mean height by distance from the alpha edge, in crowns (1.0 = the pure inflation's medial-axis
height). Bins are mask texels: 0–2, 2–5, 5–10, 10–20, 20+.

| sprite | relief 0–2 | 2–5 | 5–10 | 10–20 | 20+ | EDT 0–2 | 2–5 | 5–10 | 10–20 | 20+ |
|---|---|---|---|---|---|---|---|---|---|---|
| thing_00 | 0.73 | 0.83 | 0.86 | 0.90 | 0.92 | 0.05 | 0.13 | 0.27 | 0.48 | 0.82 |
| thing_01 | 0.78 | 0.82 | 0.84 | 0.86 | 0.91 | 0.06 | 0.15 | 0.29 | 0.52 | 0.83 |
| thing_02 | 0.68 | 0.76 | 0.82 | 0.87 | 0.94 | 0.04 | 0.10 | 0.21 | 0.38 | 0.74 |
| thing_03 | 0.74 | 0.85 | 0.91 | 0.96 | 0.98 | 0.05 | 0.14 | 0.28 | 0.50 | 0.83 |
| thing_04 | 0.71 | 0.82 | 0.87 | 0.92 | 0.95 | 0.05 | 0.14 | 0.28 | 0.51 | 0.84 |
| thing_05 | 0.69 | 0.77 | 0.80 | 0.86 | 0.90 | 0.05 | 0.14 | 0.28 | 0.51 | 0.83 |
| thing_06 | 0.71 | 0.82 | 0.85 | 0.90 | 0.94 | 0.04 | 0.10 | 0.21 | 0.38 | 0.75 |
| thing_07 | 0.68 | 0.77 | 0.82 | 0.88 | 0.92 | 0.05 | 0.14 | 0.28 | 0.50 | 0.82 |
| thing_08 | 0.52 | 0.58 | 0.64 | 0.70 | 0.80 | 0.04 | 0.11 | 0.22 | 0.41 | 0.75 |
| thing_09 | 0.57 | 0.61 | 0.62 | 0.64 | 0.73 | 0.05 | 0.12 | 0.25 | 0.45 | 0.78 |

The relief is already at 0.52–0.78 crowns **two texels inside the alpha edge** and climbs only
0.16–0.26 crowns from there to the medial axis. That is not a failure of the model: the true
depth of a flat cut-out standing on a grey field *is* a flat card with a step at its outline, and
that is exactly what DA3 returns. It is the wrong quantity for this consumer. The inflation
exists to invent the roundness the picture does not have; at confidence ~0.55 the mix trades away
half of that dome for half of a slab.

### 8.2 M1 — luminance goes straight into the geometry

§7.3's albedo test, moved off the raw depth and onto the displacement. Interior texels only
(>3 from the alpha edge); "ink" = luminance gradient > 8/255; the figure is the share of the top
1% steepest displacement gradients that land on ink.

| sprite | ink base rate | pure EDT | **mixed** | the injected term alone | relief~luma given distance | pale-minus-dark, silhouette removed |
|---|---|---|---|---|---|---|
| thing_00 | 7.5% | 0.0% | 34.6% | 18.1% | **-0.399** | **-5.4%** of crown |
| thing_01 | 7.2% | 0.0% | 25.5% | 6.2% | +0.525 | +6.0% |
| thing_02 | 7.2% | 0.0% | 90.5% | 83.6% | +0.356 | +3.9% |
| thing_03 | 7.6% | 0.0% | 50.6% | 44.3% | +0.003 | +0.2% |
| thing_04 | 6.9% | 0.0% | 53.0% | 25.9% | -0.023 | -1.0% |
| thing_05 | 8.9% | 0.0% | 69.7% | 49.0% | +0.463 | +5.3% |
| thing_06 | 6.7% | 0.0% | 71.3% | 71.7% | +0.043 | +0.5% |
| thing_07 | 7.9% | 0.7% | 52.0% | 32.9% | +0.456 | +5.3% |
| thing_08 | 7.3% | 0.0% | 87.5% | 71.2% | +0.402 | **+10.6%** |
| thing_09 | 8.4% | 0.7% | 92.9% | 58.9% | +0.451 | **+16.7%** |

The pure inflation scores *below* the base rate — its only steep slopes are at the rim, which is
not ink. The mixed surface scores 25–93%, a 3–12x lift: **on the blended mesh the paint boundary
is the crease.** The last two columns take the silhouette out of it, since these sprites have dark
outlines and luminance is itself correlated with distance-from-edge at 0.32–0.78. Partial
correlation of relief with luminance controlling for the distance field is 0.36–0.53 on six of
ten; the last column is the plain-language version — mean relief of the pale half of the subject
minus the dark half, each texel measured against the mean relief of every texel at the same
distance from the edge. **The pale belly sits proud of the dark body by up to 16.7% of the crown
height on thing_09 and 10.6% on thing_08**, which at a mix weight of ~0.55 delivers roughly 9% of
the crown as a step whose only cause is paint. On thing_00 the sign is reversed and the pale
belly sinks. That inconsistency is the point: this is not even a stable stylisation.

Raw correlation of the final displacement with luminance is a wash (mean 0.344 mixed vs 0.354
EDT) purely because of the dark-outline confound, which is why it is not the measurement.

Rendering the height fields confirms it by eye: the belly patch appears in the mixed surface as a
disc-shaped plateau with a hard scalloped step exactly on its albedo boundary, the painted eyes
appear as raised rings, and the smooth torso/head dome of the EDT surface is gone.

### 8.3 M2 — the blend dents and grows false ridges

The pure inflation is monotone in the distance field by construction, and the harness confirms it:
**0** interior local minima and **0.0%** of interior texels sloping downhill toward the medial
axis, on all ten sprites. The blend, at gain 1.0 / at half gain:

| sprite | interior local minima (dents) | interior local maxima (EDT: medial-axis ridge) | texels sloping wrong way | rim→axis profiles that dip | worst dip |
|---|---|---|---|---|---|
| thing_00 | 40 / 53 | 61 / 62 (8) | 2.9% / 5.8% | 30% / 44% | 0.12 / 0.11 crowns |
| thing_01 | 31 / 32 | 43 / 67 (13) | 2.2% / 3.9% | 26% / 57% | 0.10 / 0.08 |
| thing_02 | 50 / 66 | 60 / 90 (5) | 2.7% / 6.0% | 31% / 40% | 0.17 / 0.08 |
| thing_03 | 14 / 72 | 39 / 82 (5) | 0.8% / 5.5% | 7% / 17% | 0.12 / 0.05 |
| thing_04 | 64 / 91 | 74 / 92 (8) | 7.0% / 7.4% | 73% / 70% | 0.32 / 0.12 |
| thing_05 | 19 / 29 | 33 / 49 (7) | 1.3% / 4.2% | 16% / 62% | 0.15 / 0.07 |
| thing_06 | 12 / 115 | 37 / 106 (13) | 0.5% / 6.7% | 5% / 29% | 0.07 / 0.09 |
| thing_07 | 71 / 116 | 66 / 116 (8) | 6.1% / 8.1% | 77% / 84% | 0.37 / 0.13 |
| thing_08 | 53 / 90 | 46 / 105 (8) | 4.7% / 5.8% | 52% / 48% | 0.31 / 0.12 |
| thing_09 | 17 / 52 | 21 / 55 (9) | 3.6% / 4.9% | 25% / 29% | 0.07 / 0.10 |

The EDT control for the dip test reads 1.9–24.7% of profiles at a fixed 0.03–0.04 crowns, which is
the ray-walker's own discretisation floor; the blend's worst dips are 2–10x that. Lowering the
gain does not rescue this — the surface gets flatter, so the noise dominates the ordering and the
dent and extrema counts roughly double.

### 8.4 M3 — the rim gate holds, but the rim becomes a cliff

Task #15's ordering (the `distance <= 0` early return *before* the mix) survives contact with real
DA3, on all ten sprites: **zero** grid vertices outside the mask were lifted, and **zero** edges
into the bulge are used an odd number of times — the caps are still welded and the mesh is still
closed. It is load-bearing, not decorative: the relief that the gate is holding off those rim
vertices is **0.13–0.51 crowns** (2–9% of sprite width). Without the gate the caps separate there
and the mesh cracks open exactly as §15 predicted.

What is *not* protected is the slope just inside it. The tallest lift on a vertex adjacent to a
rim vertex, as a fraction of sprite width:

| sprite | pure EDT | mixed | ratio (half gain) |
|---|---|---|---|
| thing_00 | 0.024 | 0.114 | 4.7x (2.4x) |
| thing_01 | 0.024 | 0.090 | 3.8x (2.0x) |
| thing_02 | 0.021 | 0.125 | 6.0x (3.0x) |
| thing_03 | 0.028 | 0.129 | 4.6x (2.4x) |
| thing_04 | 0.027 | 0.140 | 5.2x (2.6x) |
| thing_05 | 0.021 | 0.112 | 5.3x (2.7x) |
| thing_06 | 0.020 | 0.104 | 5.1x (2.7x) |
| thing_07 | 0.026 | 0.133 | 5.2x (2.6x) |
| thing_08 | 0.020 | 0.105 | 5.4x (2.7x) |
| thing_09 | 0.018 | 0.089 | 4.9x (2.6x) |

The silhouette stops being a rounded lip 2% of the sprite wide and becomes a near-vertical wall
9–14% of the sprite tall — the §7.1 matte halo, delivered as geometry. Because `shade()` keys on
`|n.z|`, that wall also bakes a `SHADE_FLOOR` dark band right around the outline of every thing.

Two notes for whoever revisits this:

- **Latent, not triggered: `displacement` can return exactly 0 inside the mask.** The `.max(0.0)`
  at the end of the mix has no floor, and `inflate_rigged` treats `lift[k] <= 0.0` as "this is the
  rim" and welds the back cap onto the front vertex there — an interior pinch through the mesh.
  Real DA3 leaves 2.9–4.1% of a crown of headroom at mask resolution and produces zero occurrences
  at the 65² mesh grid, so nothing is broken today, but the contract permits relief 0 at
  confidence 1 and measured confidence maxima are 0.988–1.000. If depth is ever wired up, the weld
  test needs to be `distance <= 0.0` rather than `lift <= 0.0`. **Left unfixed — this was an
  evaluation.**
- **`confidence` is not calibrated.** `vision.rs::relief` decodes the confidence PNG with bounds
  `(0.0, 1.0)`, i.e. the picture's own min/max normalisation, so it is a per-image rank rather than
  a probability. It averages 0.50–0.68 over the body and 0.17–0.55 in the two texels nearest the
  edge no matter how sure DA3 actually was. Same complaint as §7.5, now against the second
  consumer: the mix leans ~half on DA3 by construction.

### 8.5 Verdict

**Drop DA3 from paperworld.** The blend is not a wash, it is a regression: it replaces a
monotone, silhouette-honest dome with a flat-topped slab that has a 4–6x taller wall at the rim,
12–71 dents and 21–74 false ridges per sprite where the parabola can have none, and creases that
land on the artwork's paint boundaries 25–93% of the time against a ~7% base rate. The one
interior structure it reliably adds is the sprite's own lighter patch standing proud (or, on
thing_00, sunk) by up to 17% of the crown — the failure condition named in the task, measured, and
not a nose. Nothing in it is real relief: the part of the depth that does track form (partial
correlation with the distance field 0.12–0.73) is information the EDT already has for free, and
the residual is albedo.

No mix weight or gain rescues it, because the problem is in the signal rather than in the blend.
Halving the gain trades the rim cliff (down to 2–3x) for more dents. A high-pass "add detail on
top of the inflation" formulation would fail for the same reason §8.2 gives — the detail DA3
carries on this input class *is* the paint.

Consequences, taken with §7.6:

- **`inflate.rs` should keep passing `None`**, and the `DepthMap` parameter is now dead weight on
  both consumers. paperworld's product is the EDT inflation plus the alpha-only rig; there is no
  remaining justification for a `/depth` call per summon.
- The `distance <= 0` gate stays regardless — it is cheap and it is the only reason a blend this
  aggressive did not crack the mesh open.
- Input-class-specific, as in §7.6. Shaded renders or photographs would need re-measuring; the
  slab in §8.1 is a property of flat cut-outs on a flat backdrop, not of the model.

Reproduction: an out-of-tree scratch crate (deleted) that `path`-depends on `paperworld-shared`,
copies `displacement`/`inflate_rigged` verbatim, POSTs `Cutout::for_depth` bytes to
`POST /depth?backend=cpu&process_res=504`, and normalises the reply as §8's preamble describes.
Five measurements reproduce every table: height-by-distance bins, top-1%-gradient-on-ink,
pale-minus-dark residual per distance bin, interior local extrema plus uphill-derivative sign, and
the grid's rim gate/weld/edge-parity counts.

---

## 9. Re-test on real renders — §§7–8 are overturned, and the answer is Depth-Anything **V2**

**§§7 and 8 did not test Depth-Anything 3. They tested a corpus.** Both ran on
`paperworld/server/stub_images/*.png`, the `PAPERWORLD_STUBS=1` placeholders that
`imagegen.rs::StubBackend::load` serves when no image backend is configured. Every one of the ten
is 512×512 RGBA with **exactly six distinct colours** and RGB `(0,0,0)` beneath the alpha — there
is no render under them, they are flat vector art. Two flat body tones, white eye discs, black
pupils, a pale belly blob. There is no geometric information in those images for any depth model to
recover, so "DA3 reads albedo rather than shape" was true by construction and said nothing about
the model. The user caught this; we did not.

Everything below is on **12 real krea2 renders**, generated through koblem with paperworld's own
`prompting.rs::build()` prompt shape (`REQUIRED` verbatim, both framing variants), spread across
subject types: `knight goblin owl dwarf robot bear` (bipedal, head on shoulders) plus `spider`
(limbs), `reacher` (an ogre punching *at the camera* — an unambiguous near/far ground truth),
`face` (portrait bust), `truck` (a rigid object with a known front/back axis) and `dragon`, `lamp`.

### 9.0 The pre-matte render needs no new endpoint — it is already in the cutout

`/api/v1/krea2/generate` is Bearer-gated and paperworld only holds the `x-api-key` doorstop, so the
raw render looked unreachable. It is not. `vision-server`'s `/remove` in its default `refine=false`
path (`matting_endpoints.cpp`) copies the **source RGB verbatim into every pixel**, including the
ones it sets `alpha=0`, and koblem's `RemoveOptions::default()` never sends `refine`. So the RGBA
PNG `/api/v1/oneshot` returns carries the complete pre-matte render: **drop the alpha channel and
you have it, byte-exact.** Verified — the alpha-zero region of all 12 comes back as the flat
light-grey/beige backdrop krea2 actually painted (mean 196–228, per-channel σ ≈ 1 on nine of them),
not black and not a constant. This is worth knowing generally: *any* consumer that wants the render
and the matte from one `/oneshot` call already has both.

### 9.1 §7.3's diagnostic was measuring the corpus, not the model

The single number §7 leaned on — what share of the top 1% steepest interior depth gradients land on
a luminance edge, against the base rate of luminance edges. Reproduced exactly, plus the control
§7 never had: **real photographs of real 3D objects pushed through paperworld's identical pipeline**
(BiRefNet matte → crop → composite on mid-grey 128 → CatmullRom to 504).

| corpus | model | ink base rate | top 1% on ink | **lift** |
|---|---|---|---|---|
| stub placeholder art (what §7 measured) | DA3 | **8%** | 91% | **11.2x** |
| stub placeholder art | DA2 | 8% | 43% | 5.3x |
| real krea2 renders, matted on grey | DA3 | 37% | 75% | **2.2x** |
| real krea2 renders, matted on grey | DA2 | 37% | 81% | 2.4x |
| **real photographs**, matted on grey | DA3 | 54% | 91% | **1.7x** |
| **real photographs**, matted on grey | DA2 | 54% | 82% | 1.5x |

The lift is almost entirely the reciprocal of the base rate, and the base rate is a property of how
much texture the picture has. Six-colour art has an 8% base rate; that is where 10–18x comes from.
A photograph of a genuinely 3D subject scores 1.7x. **2.2x on real renders is a photograph-like
number, not a paint-reading one**, and §7.3 does not survive the control.

Same for §8.2's plain-language version — mean relief of the pale half of the subject minus the dark
half, each texel measured against every texel at its own distance from the edge:

| corpus | DA3 | DA2 |
|---|---|---|
| stub art (§8 reported +0.039..+0.167, mean +0.066) | +0.066 crowns | −0.102 |
| real krea2 renders | **+0.004** | **+0.005** |

### 9.2 Baseline on real renders — raw vs matted-on-grey, DA3 at native 504

Same units as §§7–8 throughout: mask grid at `MASK_MAX_SIDE = 256` (paperworld's own), distance in
mask texels, "crowns" = subject range normalised so its nearest texel is 1 and its furthest 0, so
0.16 crowns of climb here means exactly what it meant in §8.1. `raw` = the pre-matte render on the
same crop; `matte` = `Cutout::for_depth` verbatim.

| | int/all (span excl. d≤2 ÷ span) | crowns at d 0–2 | at d 20+ | **climb** |
|---|---|---|---|---|
| DA3 raw | 0.527 | 0.67 | 0.86 | 0.188 |
| DA3 matte (today) | 0.406 | 0.75 | 0.90 | 0.153 |
| **DA2 raw** | 0.835 | 0.46 | 0.69 | 0.233 |
| **DA2 matte** | **0.849** | **0.41** | **0.66** | **0.253** |
| *pure EDT inflation, for scale (§8.1)* | — | *0.05* | *0.82* | *0.77* |

**H1 (matte halo) is real but second-order, and it is a DA3 problem only.** Removing the manufactured
alpha edge lifts DA3's interior fraction 0.406 → 0.527 and its climb 0.153 → 0.188. It does nothing
for DA2, which never had the halo (0.85 either way). §7's "37–85% of the span is halo" becomes
15–68% (median ~35%) for DA3 on real renders and 5–27% for DA2.

**§8.1's slab is a DA3 property, not an input-class property.** On identical pixels DA2 starts at
0.41 crowns two texels inside the rim and climbs to 0.66; DA3 starts at 0.75 and climbs to 0.90.

### 9.3 Ground truth: does the depth get near-vs-far right?

Three tests with an answer known from the picture. "Head-minus-torso" is the §7.2 measurement with
the join row **hand-annotated** per sprite rather than found by heuristic (7 sprites with a real
head/torso join); positive = head nearer, which is the correct sign for all seven.

| config | fist-vs-torso (reacher) | truck front-vs-bed | head/torso signs | median \|step\| |
|---|---|---|---|---|
| DA3 raw @504 | +0.077 | +0.127 | `+++-+++` 6/7 | 0.010 |
| DA3 matte @504 (today) | +0.046 | +0.079 | `+++-+++` 6/7 | 0.004 |
| DA3 feather @504 | +0.070 | +0.136 | `+++-+++` 6/7 | 0.018 |
| DA2 raw @native | +0.267 | +0.317 | `+++-+++` 6/7 | 0.024 |
| **DA2 matte @native** | **+0.302** | **+0.326** | **`+++++++` 7/7** | **0.029** |
| DA2 feather @native | +0.307 | +0.359 | `+++++++` 7/7 | 0.023 |

All figures are fractions of the subject's own depth span. **Both models get every sign right; DA3's
magnitudes are 4–10x smaller.** A fist punched straight at the camera moves 4.6% of the subject's
range under DA3-as-shipped and 30% under DA2. §7.2's alternating head/torso sign does not reproduce
on real renders: 6 of 7 for DA3, **7 of 7 for DA2 in the shipping configuration**. The lone DA3
dissenter is `robot`, a flat-shaded cartoon — i.e. the one sprite closest to the stub corpus.

Look at the maps and it is not subtle: DA2 separates the punching fist as a distinct near blob,
resolves the truck's cab from its bed and its running board from its body, and gives the portrait
bust a nose. DA3 returns a near-uniform slab for every bipedal figure.

### 9.4 H4 — resolution is not the ceiling (NEGATIVE)

The token-grid argument is sound in principle (ViT-S/14 at 504 is a 36×36 grid, and a neck is one
to three tokens) but it does not survive measurement. Swept 252 → 1260 on all 12, both variants:

| DA3 res | fist-torso | truck | head/torso | int/all (raw) | climb (raw) | lift |
|---|---|---|---|---|---|---|
| 252 | −0.013 | +0.136 | 5/7 | 0.595 | 0.160 | 2.0x |
| 378 | +0.050 | +0.141 | 6/7 | 0.567 | 0.179 | 2.2x |
| **504 (native)** | +0.077 | +0.127 | 6/7 | 0.527 | 0.188 | 2.3x |
| 728 | +0.083 | +0.192 | 6/7 | 0.523 | 0.187 | 2.3x |
| 756 | **+0.084** | +0.163 | 6/7 | 0.525 | **0.192** | 2.2x |
| 1008 | +0.032 | +0.196 | 5/7 | 0.450 | 0.164 | 2.1x |
| 1260 | **−0.034** | +0.144 | 5/7 | 0.473 | 0.141 | 2.0x |
| 1512 | *Vulkan OOM on this box* | | | | | |

**The signal peaks at 504–756 and then degrades — at 1260 the punching fist reads as the*furthest*
part of the ogre.** Mean per-pixel total variation does keep rising with resolution (3.3 → 4.4 at
504 → 1512), so the output gets *busier* without getting *righter*: exactly the "bigger, not better"
the port notes predicted from the `scale_factor = (w0 + 0.1)/37` position-embedding kludge running
further and further off-grid. 728 is a free ~10% on the two ground-truth tests over 504 and costs
~2x the activation memory (§6.2: 385 → 1081 MiB from 504 → 1008); it is not the answer to anything.

### 9.5 H3 — background and edge treatment (marginal)

DA3 at 504, all 12, everything else identical:

| variant | int/all | climb | fist | truck | head/torso | lift |
|---|---|---|---|---|---|---|
| matte, grey 128 (today) | 0.406 | 0.153 | +0.046 | +0.079 | 6/7, 0.004 | 2.2x |
| white 240 | 0.377 | 0.146 | — | — | 6/7 | 2.2x |
| the render's own backdrop colour | 0.406 | 0.161 | — | — | 6/7 | 2.3x |
| **feathered alpha** (12 px Gaussian, no hard cut) | **0.674** | 0.169 | +0.070 | +0.136 | 6/7, **0.018** | 2.0x |
| nearest-colour dilation (no chroma step at all) | 0.487 | 0.182 | −0.016 | +0.020 | 6/7, 0.004 | 2.0x |
| raw pre-matte render | 0.527 | 0.188 | +0.077 | +0.127 | 6/7, 0.010 | 2.3x |

Mid-grey is not specially good but nothing beats it by enough to matter — §6.4's "not a knob" holds.
Softening the alpha instead of cutting it is the one edge treatment that helps DA3 (halo roughly
halved, head/torso step 4.5x) and it is free, client-side, and needs no ordering change. Dilation
looks like it should be the fix and is actively the worst: it invents a background that *is* the
subject and the model reads the whole frame as one surface. Feeding alpha itself is not an option —
`/depth` takes RGB and the model's stem is 3-channel; feathering is the nearest thing there is.

### 9.6 H5 — the ray / point-map branch carries no shape (NEGATIVE, by construction)

Resolved by reading the reference at `~/models/da3/upstream` and the checkpoint, without building
anything. Three independent reasons it cannot help:

1. **It is 7 channels at patch scale, not a point map.** `output_conv2_aux.*.5` is `[7, 32, 1, 1]` —
   6 ray channels (origin + direction) plus one confidence. And in `dualdpt.py` only `fused_main`
   gets `custom_interpolate` to `h_out × w_out`; `last_aux` is read straight off `fused_aux_pyr[-1]`
   at the fusion pivot scale. There is no full-resolution aux output to recover.
2. **Its only consumer collapses it to seven global numbers.** `da3.py::_process_ray_pose_estimation`
   feeds it to `get_extrinsic_from_camray` → `camray_to_caminfo`, a RANSAC homography fit that
   returns one 4×4 extrinsic, one focal length and one principal point — then `del output.ray`.
   Upstream's default is `use_ray_pose=False`; pose normally comes from `cam_dec` and the ray tensor
   is deleted unread either way.
3. **The point map is the depth scalar reprojected.** `glb`/`ply`/`gs_ply` export
   `unproject_depth(depth, intrinsics)` — pixel rays through a pinhole times the depth. It contains
   strictly nothing the depth map does not.

For a single matted sprite with an invented camera this is a focal-length regressor on an image with
no perspective cues. **Do not build the ray branch.** §2.4's note stands: the weights are dead
VRAM and re-running `convert.py` without them is the only thing to do with them.

### 9.7 Verdict

**Was the original negative a fair test of the model? No — it was a test of six-colour placeholder
art, and §§7–8 do not stand.** But re-testing on valid input does not simply reverse them either,
and the honest answer is more useful than that:

- **DA3-Small as shipped is genuinely weak on this input class, and that part of §§7–8 survives.**
  On real, shaded, photographic krea2 renders it still returns a slab (0.75 → 0.90 crowns from rim
  to medial axis, against the EDT's 0.05 → 0.82), and a fist thrown at the camera buys 4.6% of the
  subject's depth span. It is not reading paint — its signs are right, and it beats a photograph's
  own luminance-edge lift — it is reading form *and compressing it into nothing*.
- **Depth-Anything V2 does not have that problem.** Same pixels, same pipeline, same
  `paperworld_shared::mask` grid: 0.41 → 0.66 crowns, +30% of span for the punching fist, +33% for
  the truck's front, and a **consistently signed head/torso step on 7 of 7** in the exact
  configuration paperworld ships (matted on mid-grey, native res). DA2 is a plain relative-depth DPT
  with no camera head and no unified depth-ray representation, and on a single flat image with no
  real camera that is an advantage, not a limitation.
- **The fix is the model, not the stage order.** Depth-before-matte buys DA3 1.6–2.5x on every form
  metric — real, and still not enough to be usable — and buys DA2 nothing at all (matte 0.302 vs
  raw 0.267 on the fist; matte is if anything better). **Do not reorder `summon.rs`.** The
  ordering change is real work in a streaming endpoint, and it is the wrong lever.

Recommended, in order of cost:

1. **Serve DA2 from `/depth`.** *(Done — §10.)* `depthany_*` is already ported, `test_depth_anything` passes on both
   backends, and DA2-Small-F16 is already in `models/`. This is a model slot and a query param on an
   endpoint that already has `backend` and `process_res`. **Note DA2 bounds the *shortest* side
   (`depthany_image_extent`, 518) where DA3 bounds the longest** — a caller sending `process_res`
   through unchanged will silently get a different effective resolution.
2. **Feather the alpha before compositing** if DA3 is kept for any reason. Client-side, free,
   halves the halo.
3. Leave `process_res` at 504. 728 is a ~10% improvement for ~2x the activations; above 1008 the
   output degrades.

Two things to settle before wiring DA2 into paperworld, neither of which this evaluation resolves.
*§10.3 settles the first — a missing confidence map is an absent field, and the blend weight comes
from the consumer's own distance field. The second is task 33's.*

- **DA2 has no confidence map**, and both consumers blend on one — `inflate.rs::displacement` mixes
  `edt` and `depth` at `confidence`, and `watershed::gradient` damps every gradient by it. §7.5 and
  §8.4 already showed that confidence is uncalibrated (it is the PNG's own min/max rank), so this is
  a chance to replace it with an explicit constant rather than a loss.
- **The head/torso steps, though now consistently signed, are still small** — median 2.9% of the
  subject's span under DA2, against `watershed.rs`'s `STEP = 0.04` threshold. So DA2 plausibly
  rescues the **inflation blend** (§8's consumer); whether it rescues the **watershed** (§7's) needs
  its own test against real rig output, which this evaluation did not run.

Scope: 12 renders, one seed each, one prompt shape, `krea2` only. Vulkan throughout; spot-checked
against CPU at corr **0.999997**, mean |diff| 0.00055 on a span of 1.31, so the backend is not in
play.

**Reproduction.** No repo was modified. (a) `POST /api/v1/oneshot` with `x-api-key:
paperworld-shoddy-key` and `prompting::build()`'s positive/negative, 1024², one call per subject;
(b) `raw` = the returned RGBA with the alpha channel dropped, `matte` = `Cutout::for_depth` verbatim
(crop to the alpha bbox, composite on 128, CatmullRom to 504 longest side); (c) a throwaway
single-file tool linking `libvisioncpp` that sets `model.params.image_size` and dumps
`depthany{,3}_compute`'s raw f32 rather than the CLI's normalised 8-bit; (d) six measurements
reproduce every table — span excluding a distance-field ring, crowns by distance bin,
top-1%-gradient-on-ink against its own base rate, pale-minus-dark per distance bin, the two
hand-boxed ground-truth region pairs, and the hand-annotated head/torso step. The photograph control
is `tests/input/{cat-and-hat,wardrobe,vase-and-bowl,bench-image}.jpg` matted with
`vision-cli birefnet` and pushed through (b) unchanged.

---

## 10. `/depth` now serves Depth-Anything **V2** by default (task 32)

§9's recommendation, implemented. `/depth` answers from DA2-Small unless asked otherwise; DA3 is
still there, still built, still validated to F16 noise against PyTorch, and reachable in one query
param. Nothing under `src/visp/arch/` changed — this is a model slot, a selector and a response
shape. **`summon.rs`'s stage order was NOT touched**, per §9.7: depth-before-matte buys DA3
1.6–2.5x from an unusable base and buys DA2 nothing (matte 0.302 vs raw 0.267 on the fist).

### 10.1 Selecting a model, and two models on disk

| | Depth-Anything V2 | Depth-Anything 3 |
|---|---|---|
| `?model=` | `depth-anything-v2` / `v2` / *omitted* | `depth-anything-3` / `v3` / `3` |
| env / flag | `DEPTH_MODEL_PATH` / `--depth-model` | `DEPTH3_MODEL_PATH` / `--depth3-model` |
| GGUF | `Depth-Anything-V2-Small-F16.gguf` (48 MiB) | `Depth-Anything-3-Small-F16.gguf` (66 MiB) |
| resolution knob | `process_res_short` | `process_res_long` |
| confidence map | **none** | per-pixel |
| `depth_polarity` | `disparity` — larger is **nearer** | `distance` — larger is **farther** |

The two GGUFs load into two independent slots on the same worker, lazily and separately: a
deployment that only ever calls the default never loads DA3 and never pays its VRAM. An unknown
`model` string is a 400, and a model whose path is unset is a **503 naming the env var that is
missing** — the pre-existing graceful degradation, now per-family. `DEPTH_HF_REPO` and the new
`DEPTH3_HF_REPO` both stay empty by default: neither GGUF is published, both are converted locally
and dropped into the volume.

> **Deploy note.** `DEPTH_MODEL_FILE`'s default changed from the DA3 filename to the V2 one, and
> `DEPTH3_MODEL_FILE` defaults to the DA3 name the volume already holds. So an existing container
> keeps serving DA3 under `?model=depth-anything-3` from cache with no action, and `/depth`'s
> default answers 503 until `Depth-Anything-V2-Small-F16.gguf` is copied in
> (`scripts/convert.py depth-anything <DA-V2-Small checkpoint> -q f16` — `arch` is positional;
> the pre-existing `--arch` spelling in `entrypoint.sh` was wrong and is fixed).

### 10.2 `process_res` is gone, and that is deliberate

The two families do not measure inference resolution on the same axis, and §9.7 flagged that a
caller forwarding one number would silently get a different picture:

- **V2** — `depthany_image_extent` is `max(image_size, next_multiple(min_side, 14))` scaled onto
  the input's aspect. `image_size` (native 518) is a **floor on the SHORTEST side**, and it never
  downscales: hand it a 1024² image at `image_size=504` and it runs at **1024**, not 504.
- **DA3** — `depthany3_image_extent` scales by `image_size / longest`, a genuine **bound on the
  LONGEST side** (native 504, a multiple of 14).

Normalising the two was rejected: forcing V2 onto a longest-side bound means bypassing its own
preprocessing, which is what the port is validated against and what every number in §9 was measured
in. So each axis is **named after itself** and the endpoint refuses to guess:

```
process_res_short   depth-anything-v2 only — floor on the shortest side
process_res_long    depth-anything-3 only  — bound on the longest side
process_res         400. "ambiguous across depth models"
```

Sending the *other* model's axis is also a 400 rather than a silent drop. The same three names
propagate through `koblem/api/src/depth.rs` (`DepthOptions`), `koblem/api/src/routes/depth.rs`
(`DepthQuery`) and out to paperworld, which now sends **no resolution override at all**
(`server/src/vision.rs`: `"/api/v1/depth?inline=true"`). Native V2 is exactly the configuration
§9.3's `DA2 matte @native` row was measured in, so the shipping path and the evaluation agree.

### 10.3 A missing confidence map is an absent field

DA2 has no confidence head. Nothing fabricates one — not a constant, not zeros, not ones. When the
model has no confidence map, `confidence_png_base64`, `confidence_min` and `confidence_max` are
**not present** in the result object, and koblem's stored shape files **one** output per result
instead of two (the urls are consecutive, not `2i`/`2i+1` — a fixed stride would have handed
result 1 result 2's depth).

This is not a loss. §§7.5/8.4 measured DA3's confidence being decoded with `(0.0, 1.0)` bounds —
it is the PNG's own min/max rank, a **per-image ordering and not a probability** — and it rose
monotonically with distance from the alpha edge, which is precisely what the consumers' own EDT
distance field already gives them for free.

> **For task 33.** Two consumers, and they are not in the same state.
> `server/src/watershed.rs::gradient` damps every gradient by `confidence` and runs on every
> refined summon. `client/src/inflate.rs::displacement` mixes the EDT inflation and the depth at
> `confidence` — but every shipping call site passes `depth: None` (`net.rs`, `rig.rs`, `hands.rs`;
> only its own tests build a `DepthMap`), so that consumer is **dormant** and task 33 is wiring it
> up as much as re-testing it. Both should derive the weight from the distance field rather than
> from the model. `server/src/vision.rs::relief` currently fills `1.0` when the field is absent —
> the same branch `FLAT_CONFIDENCE` already took for a map with no spread — which leaves the depth
> untouched and is a holding position, not the design.

### 10.4 Polarity — the one way to get this badly wrong

DA2's DPT head ends in a ReLU and `depthany_process_output` min/max-normalises it, so V2 emits
**relative disparity in [0,1] where larger is NEARER**. DA3 emits `exp(logit)`, an unnormalised
**distance where larger is FARTHER**. Swapping the default therefore inverts near and far for any
consumer that does not look. The response now carries `model` and `depth_polarity` at the top
level (and `X-Depth-Model`), and both travel through koblem's `DepthRouteResponse`. Note also that
the two models' bounds are on completely different scales — measured on the same picture,
V2 came back `-0.0056 .. 0.9994` (the model pipeline already min/max-normalised; the small
overshoot is the CatmullRom rescale to the input extent) against DA3's `0.7560 .. 2.7107`. Neither
carries an absolute scale; only the relative ordering means anything, so a consumer must not read
V2's numbers as metres or DA3's as a fraction.

A side observation that corroborates §10.3: DA3's confidence on that same image came back
`1.0000 .. 4.6135` — `1 + exp(logit)`, so its floor is 1 by construction — while
`paperworld/server/src/vision.rs` decodes it with `(0.0, 1.0)` bounds. §§7.5/8.4 said that made it
a per-image rank rather than a probability; this is that in two numbers.

### 10.5 Measured here (Vulkan, AMD 890M, RADV)

One warm worker, one process, one 504×504 PNG posted to `/depth` six times per model after a
throwaway warm-up. Staged fdinfo `drm-total-vram`, cumulative, exactly as §6.2:

| after | per-process | delta |
|---|---|---|
| worker spawn, BiRefNet-lite F16 loaded, no inference | 84 MiB | — |
| + one `/depth` at V2 native (DA2-Small F16 weights + graph) | 338 MiB | **+254 MiB** |
| + five more warm V2 requests | 338 MiB | +0 |
| + one `/depth?model=depth-anything-3` @504 (DA3-Small F16) | 635 MiB | **+297 MiB** |
| + five more warm DA3 requests | 635 MiB | +0 |

**DA3's delta reproduces §6.2's ~300 MiB, and DA2's is ~15% smaller at 254 MiB** — so the default
swap does not raise the gate budget, it lowers it slightly. The two slots are independent and
additive: a worker that has served both holds 551 MiB of depth, which only a caller that alternates
`?model=` can reach.

| | wall (s, incl. PNG encode) | `elapsed_seconds` (inference only) |
|---|---|---|
| **DA2 @native (518²)** | 0.398 – 0.446 | **0.296 – 0.322** |
| DA3 @504 | 0.425 – 0.525 | 0.294 – 0.306 |

**Inference cost is a wash** — ~0.30 s either way, and DA2 is doing 518² to DA3's 504² while
carrying a third of the parameters. The whole wall-clock difference is DA3's *second* 16-bit PNG:
one map to encode instead of two. Cold-start on an already-warm worker was 0.502 s (V2) / 0.837 s
(DA3), the extra being weight load.

Polarity was checked rather than assumed. Same picture, same pipeline, the two depth PNGs correlate
at **Pearson −0.947** — they agree about the scene and disagree about which end of the scale is
near, which is §10.4 in one number.

Gate: `test-models` holds its baseline, **11 passed / 1 skipped / 1 failed**, the only failure
being the pre-existing `test_birefnet[gpu]` of §5. `test_depth_anything[cpu]` and `[gpu]` both pass,
as does `test_depth_anything_3` on both — the DA3 references are unchanged, because nothing under
`src/visp/arch/` was touched.

The 400s were exercised end-to-end, not just written:

```
?process_res=504                              -> 400 "process_res is ambiguous across depth models…"
?model=depth-anything-v2&process_res_long=504 -> 400 "…not a depth-anything-v2 parameter — it takes process_res_short"
?model=depth-anything-3&process_res_short=518 -> 400 "…not a depth-anything-3 parameter — it takes process_res_long"
?model=bogus                                  -> 400 "unknown model — use depth-anything-v2 or depth-anything-3"
```

and the response shapes are what §10.3 claims:

```
v2 result keys: depth_max depth_min depth_png_base64 height width
v3 result keys: confidence_max confidence_min confidence_png_base64 depth_max depth_min depth_png_base64 height width
```

### 10.6 Still unverified on CUDA

| Question | Why it matters |
|---|---|
| Combined CUDA peak with RMBG-2.0 @1024 + DA2-Small resident | §6.3's question, re-asked for the smaller model. Vulkan's DA2 delta is below DA3's, so the gate budget does not need raising, but CUDA's F32 activation shape may differ |
| `VISP_F16_ENCODER=1` with DA2 resident | `worker.cpp` sets it on CUDA/HIP only. Only `swin.cpp` reads it, so DA2's DINOv2 backbone should be unaffected — untested on CUDA, same as DA3 |
| Both depth families resident at once | Only reachable by a caller that alternates `?model=`. Two independent slots, so the deltas add; nothing evicts one for the other |

---

## 11. MoGe-2 instead of DA2 — assessment only, no port (task 35)

The question: paperworld wants characters shaped like characters, `/depth` now serves DA2 (§10), and
MoGe-2 is **already ported to ggml** on `hbd-longcat-avatar.cpp` branch `origin/spike/sparse-conv-3d`
(`tools/m1_ref/cpp_port/moge_{neck,cam,fov}.hpp` + three tests, plus `SCOPE-moge2-camera-port.md`).
MoGe-2 predicts a dense **point map**, not a depth scalar, which is a better shape of answer than
anything DA2 or DA3 returns. This section costs the move and recommends against making it now.

**Verdict up front. Do not port MoGe-2 yet, and if it is ever ported, port `moge-2-vits-normal`, not
the ViT-L the spike branch targets.** The engineering is cheaper than it looks — the encoder is
almost free in `visp` and the hard numeric details are already solved twice over. But the token-grid
premise does not survive measurement (11.7), and a first run of the PyTorch reference on one real
matted render puts MoGe *behind* DA2 on §9's own metric (11.8). The experiment that would settle it
costs half a day of Python and no C++ at all. Finish that first.

### 11.0 What was and was not done here

Read-only. Nothing in `hbd-vision.cpp` or `hbd-longcat-avatar.cpp` was modified; the spike branch was
read via `git show`, never checked out. The spike branch is **pre-rebuild** (last commit 2026-07-19,
pinned to ggml `05a32239`, which is *not* an ancestor of `a7478f58`), so its test binaries were
deliberately not built — a latency number from a resurrected dead ggml would measure nothing anyone
uses. Its headers were read as an **architecture spec**, which is what they are worth.

Measured here instead: DA2-Small on this box at three token grids, the MoGe-2 checkpoint itself
(downloaded, parsed, config and parameter census extracted), rocBLAS GEMM throughput on the same
iGPU as a scaling anchor, and one run of the **PyTorch MoGe-2 reference** on a real matted character
render (11.8). MoGe latency and VRAM *on ggml/Vulkan* are **estimates**, labelled as such.

### 11.1 The point map is real, complete and validated — not a stub the FOV fit tolerates

`moge_cam.hpp` returns only `camera_angle_x`, but that is an API choice, not the limit of the port.
The graph builds `points` (3ch) and `mask` (1ch) at 960², marks both `ggml_set_output`, and
`moge_cam.hpp` already reads both back to host buffers (`pbuf`, `mbuf`) before throwing them away.
**Exposing the point map is deleting the discard, ~10 lines.**

`moge_test.cpp` compares all three of `neck_out` [32,960,960], `points_head_out` [3,960,960] and
`mask_head_out` [1,960,960] against banked PyTorch goldens and gates on
`neck.meanabs < 1e-3 && points.meanabs < 1e-3 && mask.maxabs < 0.2` (mask is a pre-sigmoid logit).
So the heads are validated elementwise at full resolution, not to whatever a global focal fit
tolerates. One caveat: the points gate is **mean**-absolute, with no max-error bound — a localized
defect would pass. The goldens live at `/mnt/hdd/3d/avatar-shootout/moge_goldens` on the GPU server
and were not re-verifiable here.

`moge_fov.hpp` is a faithful host port: nearest-downsample to 64², the `exp` remap
(`x·e^z, y·e^z, e^z` — MoGe's `remap_output: "exp"`, confirmed against the checkpoint config), and a
1-D Levenberg-Marquardt matching `scipy least_squares(method='lm', ftol=1e-3)`. The FOV solver is
**not optional if you want the point map**: `infer()` adds the recovered `shift` to `points[...,2]`
before returning, so the geometry depends on the fit.

### 11.2 The checkpoint — fetched and parsed, not guessed

`Ruicheng/moge-2-vitl`, MIT, one 1.305 GB `model.pt` (the `weights_npy` dir in the headers is a
derived numpy dump that lives on the GPU server and is irrelevant to a `visp` port — go from
`model.pt` to GGUF via `scripts/convert.py`). Config read out of the checkpoint's pickle header
without downloading the tensors:

| | `moge-2-vits-normal` | `moge-2-vitb-normal` | `moge-2-vitl` (the spike's target) |
|---|---|---|---|
| backbone | `dinov2_vits14` | `dinov2_vitb14` | `dinov2_vitl14` |
| intermediate layers | `[5,11]` | `[5,11]` | `[5,11,17,23]` |
| neck widths | 384/256/128/64/32 | 768/256/128/64/32 | 1024/256/128/64/32 |
| neck res-blocks | `[0,1,1,1,0]` | `[0,1,1,1,0]` | `[0,2,2,2,0]` |
| heads | points, **normal**, mask, scale | points, **normal**, mask, scale | points, mask, scale |
| **total params** | **35.1 M** | 104.7 M | 326.2 M |
| encoder / heads | 22.4 / 12.7 M | 87.8 / 16.9 M | 308.6 / 17.6 M |
| F16 GGUF (est.) | **~70 MB** | ~209 MB | ~652 MB |

`num_tokens_range` is `[1200, 3600]`; the spike branch's fixed 60×60 grid at 840² **is the top of the
range**, not a natural resting point. `res_block_*_norm` is `"none"` everywhere, so there is no
GroupNorm to port. Two findings that contradict the notes:

* **`SCOPE-moge2-camera-port.md`'s one open backbone question is settled: 0 register tokens.** The
  checkpoint has `cls_token`, `pos_embed [1,1370,1024]` (= 1 + 37²) and `mask_token`, and no
  register tensor. `dinov2_graph.hpp` already hard-codes `N_PREFIX = 1` for the same reason.
  **`dino.cpp` needs no register-token support.**
* **`moge-2-vitl` has no `normal_head`** — normals are a *separate checkpoint family*
  (`-vits/-vitb/-vitl-normal`), and those also exist at **ViT-S and ViT-B**. The 35 M ViT-S variant is
  the same size class as DA2-Small (25 M) and DA3-Small (34.3 M) and it ships points **and** normals
  **and** mask. This is the single most useful thing found in this assessment.

### 11.3 The encoder is nearly free in `visp` — this is the crux, and it lands the right way

MoGe-2's backbone is stock DINOv2 ViT-L/14: pre-norm, LayerScale, learned interpolated pos-embed,
MLP FFN, LN eps 1e-6, no RoPE, no QK-norm, no registers. `src/visp/arch/dino.cpp` **is** a DINOv2
implementation — it was written for DA2 and gained QK-norm/2D-RoPE as *defaulted-off* `block_params`
for DA3 (§2.1). Item by item:

| MoGe needs | `visp` today | new code |
|---|---|---|
| dim 1024, 24 blocks, 16 heads, patch 14 | `dino_detect_params` reads all four from GGUF KV | **none** — metadata |
| LayerScale, LN 1e-6, GELU MLP | `dino::layer` verbatim | none |
| no RoPE / no QK-norm | `block_params` defaults | none |
| register tokens | not needed (11.2) | none |
| taps at blocks {5,11,17,23}, each final-`layernorm`ed | `dino::get_intermediate_layers` does exactly this | none |
| pos-embed resampled with DINOv2's `(target+0.1)/37` kludge | `da3::create_pos_encoding` — host bicubic, Keys A=−0.75, offset 0.1 (§2.2) | ~10 lines to lift its signature off `depthany3_params` |
| fused `attn.qkv` → split q/k/v; `blocks.N` → `encoder.layer.N`; `ls1.gamma` → `layer_scale1.lambda1`; `patch_embed.proj` → `embeddings.patch_embeddings.projection` | `da3_encoder_renames` + `da3_split_qkv` in `convert.py`. MoGe's `encoder.backbone.*` keys match DA3's DINOv2 convention **one for one** — checked key by key against the checkpoint | reuse verbatim |

The only genuine numeric risk in the encoder — the position-embedding kludge, worth 6–10 % RMS if
skipped — was already found, implemented and validated for DA3. **The encoder is a converter change
and a ~20-line block loop; the real work is entirely in the neck and heads.**

One parity nit: `dino::mlp` uses `ggml_gelu` (tanh approximation, F16 LUT); DINOv2 is exact erf.
DA2/DA3 both live with this at F16 noise, but `dinov2_graph.hpp` used `gelu_erf_` explicitly, so if
stage-by-stage parity is tight, switch to `ggml_gelu_erf`.

### 11.4 What is actually new: one file, and `visp` already has every op

The neck and heads are five-level `ConvStack`s: 1×1 input blocks, residual blocks that are
`relu → conv3×3 → relu → conv3×3 → +skip` with **no norms**, ConvTranspose2d k2s2 ×3 then a bilinear
×2, and a 1×1 output conv per head. `visp/nn.h` has `conv_2d` (1×1 routed to `mul_mat`, k>1 to
im2col with a `VISP_IM2COL_MAX` cap that falls back to `ggml_conv_2d_direct`), `conv_transpose_2d`,
and `interpolate`. The one gap is **replicate padding** — MoGe's convs are `padding_mode="replicate"`
and ggml only zero-pads — and `moge_neck.hpp::replicate_pad2d` already solves it with edge-column
`ggml_concat`s.

Estimated new code for a `model_family::moge2`, following `docs/model-implementation-guide.md`:

| piece | lines | risk |
|---|---|---|
| `scripts/convert.py::convert_moge2` | ~120 py | low — reuses the DA3 encoder renames |
| `src/visp/arch/moge2.{h,cpp}` graph (convstack, resblock, resamplers, replicate pad, UV pyramid precompute, predict) | ~300 | low — spec is captured |
| host post: `exp` remap, mask sigmoid, LM `recover_focal_shift`, shift/reprojection | ~200 | low — lift `moge_fov.hpp` near-verbatim |
| `vision.h` API block + `vision.cpp` load/compute + `model_detect_family` | ~100 | low |
| `c-api.cpp` family arm | ~20 | **awkward** — `model_funcs::compute` returns one `image_data`; a 3-channel point map does not fit. DA3 already collapses to normalized depth here |
| `cli.cpp` command | ~40 | low |
| `tests/workbench.cpp` + python parity tests + `test-models` entry | ~150 | medium — the bulk of the real time |
| `src/server/depth_endpoints.cpp` | ~100 | **response shape** — see below |

On the response shape: `/depth` returns one 16-bit greyscale PNG plus `depth_min`/`depth_max`
(§10.3), and collapsing a point map to that throws away the exact property that motivated the move.
It does **not** need three PNGs though — MoGe's point map is `depth` unprojected through the
intrinsics it recovers, so the existing PNG plus **two extra JSON numbers (`fx`, `fy`)** is a
complete, losslessly-invertible encoding. That is a small extension of an established contract, but
it is still a `paperworld` client change as well as a server one: `DepthMap` in `inflate.rs` holds
`depth` + `confidence` and has no field for intrinsics.

≈ 900–1000 new lines touching 8 existing files — the size of `birefnet.cpp` (426) plus its
scaffolding. With the spec already captured, **3–5 focused days including stage-by-stage validation
against the PyTorch reference.** Reviving the spike branch in place is strictly worse and is not
costed here: it needs a dead ggml resurrected, carries `m1_ggml.hpp`'s hand-rolled harness (641
lines) and `.npy` weight loading instead of GGUF, has no F16 path, no backend abstraction and no
`test-models` gate.

### 11.5 `normal_head` is nearly free — and mostly unnecessary

Architecturally `normal_head` is **byte-for-byte the same ConvStack config as `points_head`**, 3
output channels, `[0,1,1,1,0]` res-blocks. In the port that is *one more call to `convstack()`* —
call it 5 lines of C++ and a converter passthrough. Runtime cost is one extra 960² head: ~+5 % of
total FLOPs on ViT-L, ~+9.4 MB of F16 weights. Cheap, as suspected.

But it buys less than it looks:

1. **`moge-2-vitl` does not have one.** Normals mean switching to the `-normal` checkpoint family,
   which invalidates the spike branch's goldens (captured against plain `vitl`).
2. **A dense point map already gives normals** — cross products of neighbouring points, which is
   what MoGe's own `points_to_normals` does. `normal_head` is a better-at-edges refinement.
3. **The consumer cannot spend them.** `inflate.rs` tessellates at `GRID = 64`: 65² vertices, and
   `shade()` bakes vertex colours from `normals()`, which are computed from the *inflated positions*.
   Feeding 960² predicted normals into a 65²-vertex mesh throws away 99.5 % of them. Exploiting them
   means baking a normal/AO texture instead of vertex colours — a **consumer-side** change strictly
   larger than the model-side one.

`scale_head` (2.1 M params, an MLP to one metric scale) is correctly skipped: paperworld wants
relative geometry and MoGe's point map is already internally consistent without it.

### 11.6 Cost — measured anchors, estimated totals

Measured here (Vulkan, AMD 890M, RADV), DA2-Small F16 on one matted 1024² character render,
`vision-cli depthany`, inference only:

| DA2-Small input | token grid | inference |
|---|---|---|
| 518² (the shipping config) | 37×37 = 1369 | **225–239 ms** |
| **840² (MoGe's exact grid)** | **60×60 = 3600** | **781–799 ms** |
| 1022² (native for a 1024² crop) | 73×73 = 5329 | 1524–1545 ms |

Also measured, as a scaling ceiling: rocBLAS F16 GEMM on this iGPU sustains **3.3 TFLOP/s** at
MoGe-shaped sizes (3600×1024×4096) and 3.2 at DA2-shaped ones. DA2 @840 is ≈0.46 TFLOP of analytic
work in 0.79 s, so **ggml/Vulkan achieves ~0.6 TFLOP/s** here; MoGe's GEMMs are 2.7× wider so call
it 0.6–1.2.

Analytic FLOP counts, from the checkpoint's real widths (encoder / neck / per head):

| | tokens | output | total | vs DA2 @840 |
|---|---|---|---|---|
| `moge-2-vitl`, 2 heads | 3600 | 960² | 3.45 / 0.31 / 0.21 = **≈ 4.2 TFLOP** | **9.1×** |
| `moge-2-vits-normal`, 3 heads | 3600 | 960² | 0.39 / 0.20 / 0.20 = **≈ 1.2 TFLOP** | **2.6×** |
| `moge-2-vits-normal` | 1225 | 560² | 0.08 / 0.07 / 0.07 = **≈ 0.35 TFLOP** | 0.8× |
| *DA2-Small @840², measured 0.79 s* | *3600* | *840²* | *≈ 0.46 TFLOP* | *1×* |

One real MoGe datapoint, from 11.8: the PyTorch reference, ViT-L, 3600 tokens, **CPU fp32, 49.7 s**
on 24 cores — 0.084 TFLOP/s against the 4.2 TFLOP above, where a bare 4096³ CPU GEMM on this box
manages 0.177. The FLOP model is therefore not fantasy; the ggml/Vulkan rows below are still
**estimates** (the ROCm reference run hard-hung the iGPU, so there is no GPU reference number):

| | this box, Vulkan | 3060, CUDA F16 | VRAM delta |
|---|---|---|---|
| `moge-2-vitl` @3600 | **~3.5–7 s** | ~0.4–0.9 s | **~1.2–1.8 GiB** |
| `moge-2-vits-normal` @3600 | ~1–2 s | ~0.15–0.3 s | ~0.6–1.0 GiB |
| `moge-2-vits-normal` @1225 | ~0.3–0.6 s | ~0.1 s | ~0.3–0.5 GiB |
| *DA2 @518, measured (§10.5)* | *0.30 s* | — | *+254 MiB* |

The VRAM figures are dominated not by weights but by **960²-resolution convolution activations** —
one 960²×32ch F32 tensor is 113 MiB and several are live at once, and the 3×3 conv at 960²×32ch has a
~1.0 GiB F32 im2col matrix. That is precisely the BiRefNet decoder-head profile of `bench/VRAM.md`,
and `visp` already has the lever for it (`VISP_IM2COL_MAX` → `ggml_conv_2d_direct`). Flash attention
is on by default on Vulkan and CUDA at head_dim 64, so the 3600² attention matrix (~790 MiB F32 if
materialized) is never built — worth knowing, because on a backend without FA it is the peak.

**MoGe's cost floor is the 960² output, not the encoder.** The 35 M ViT-S variant is still 2.6× DA2
at the same token grid, because three ConvStack heads run to 16× the token resolution. The consumer
samples the result on a 65×65 grid.

### 11.7 The premise, checked — and the token-grid argument does not survive

Two claims motivate the request. One is sound, one is not.

**Sound: MoGe predicts geometry, so the units problem disappears.** `inflate.rs::displacement`
consumes `relief` as "half-thickness toward the viewer **as a fraction of the sprite's width**, under
the same planar projection as the mesh". DA2 returns min/max-normalized disparity: no scale, wrong
sign convention (§10.4), and no way to get to that unit without an invented focal length. MoGe's
point map is affine-invariant with **x, y and z sharing one unknown global scale**, so `z_span /
x_span` is scale-free and is *exactly* the quantity the consumer wants. Polarity stops being a
question too. This is a real advantage and DA2 cannot be made to have it.

With one string attached: what the heads emit is *affine*, and it only becomes a true perspective
point cloud after the LM fit's `shift` is added to z. So the focal/shift recovery — the part
`SCOPE-moge2-camera-port.md` treats as a means to an end — is **on the critical path for the
geometry being right**, and it is a least-squares fit over 64² samples of a subject that may be out
of distribution. If MoGe misjudges the shift on a matted cut-out, `z_span / x_span` is wrong by
exactly that much. 11.8's experiment measures this too; nothing else does.

Note this does **not** inherit §9.6's negative. That was DA3's *ray* branch, where the point map is
"the depth scalar reprojected" and contains strictly nothing the depth does not. MoGe is the reverse:
the point map is the primary prediction and depth is `points[...,2]` derived from it.

**Not sound: "60×60 tokens vs 37×37 is the detail ceiling."** §9.4 already swept DA3 252→1260 and
found the signal peaks at 504–756 and then *degrades*. Measured here on real renders, DA2, same two
subjects, same metric as §9.2 (mask grid 256, crowns normalized to the subject, climb = crowns at
d≥20 minus crowns at d≤2):

| subject | DA2 @518 (1369 tok) | DA2 @840 (3600 tok) | DA2 @1022 (5329 tok) |
|---|---|---|---|
| donkey | rim 0.27 → med 0.62, **climb 0.351** | 0.27 → 0.60, climb 0.332 | 0.27 → 0.59, climb 0.319 |
| child | rim 0.46 → med 0.73, **climb 0.269** | 0.45 → 0.71, climb 0.259 | 0.45 → 0.70, climb 0.253 |

Both subjects, monotonically **down** — 2.6× the tokens buys 3.5× the latency and slightly *less*
form. (The @518 climbs of 0.35/0.27 also bracket §9.2's corpus figure of 0.253, so this pipeline
reproduces the existing DA2 baseline.) **The 60×60 grid can be had from DA2 today for 0.79 s and
zero engineering, and it is not an improvement.** n=2, but the direction agrees with §9.4's n=12.

### 11.8 First look — one real subject, PyTorch reference, and it is not encouraging

Nothing about architecture establishes that MoGe-2 returns character-shaped geometry on a flat-lit
matted sprite; MoGe is trained on photographs with real perspective, and a cut-out composited on
mid-grey is out of distribution in exactly the way §§7–8 got burned by. So the reference was run.

**Setup.** `Ruicheng/moge-2-vitl` (MIT, 1.305 GB `model.pt`) at
`/home/dbrain/.cache/moge-assess/`, venv = system ROCm torch 2.12.1 + `utils3d`, reference repo
cloned. Input: a real 1216×1600 RGBA character render (`/home/dbrain/output/*_rgba.png`, April
generations — *not* the retracted six-colour placeholders in
`paperworld/docs/shape-comparison/`, which must never be used again), matted exactly as
`Cutout::for_depth` does it — crop to the alpha bbox, square-pad, composite on 128, resize.
`infer(num_tokens=3600)`. **ROCm hard-hung the iGPU** (`HW Exception by GPU node-1 … GPU Hang`,
gfx1150 + experimental AOTriton SDPA) so this is the CPU fp32 path; also note the reference defaults
to `use_fp16=True`, which on CPU is emulated and pathologically slow — pass `use_fp16=False`.

| measured | |
|---|---|
| ViT-L, 3600 tokens, CPU fp32, 24 cores | **49.7 s** (≡ 0.084 TFLOP/s achieved on 11.6's 4.2 TFLOP — the FLOP model checks out against a real run) |
| recovered `fov_x` | 22.4° — a plausible telephoto read of a centred subject, not a degenerate fit |
| MoGe mask vs the alpha matte | 0.448 vs 0.438 of frame — **the predicted mask tracks the matte**, a free validity map that DA2 has no analogue for |
| **`z_span / x_span` (p1–p99)** | **1.279** — the subject is read as 1.28× as deep as it is wide. Not a slab. This is the number DA2 structurally cannot produce |
| crowns, §9.2 units | rim **0.49** → med **0.77**, **climb 0.285**, int/all 0.984 |
| DA2 @840 on the identical crop | rim 0.27 → med 0.60, **climb 0.332**, int/all 0.977 |
| Pearson, MoGe relief vs DA2 disparity, over the subject | **0.911** |

**On the metric §9 chose, MoGe loses to DA2** — climb 0.285 against 0.332 at the same token grid
(0.351 at DA2's shipping 518²) — and the two models agree with each other at r = 0.91, so MoGe is
not seeing a different subject, it is seeing the same one slightly flatter. Its distinctive
contribution is the *absolute* number: a metrically-consistent 1.28 depth-to-width ratio, which is
what §11.7 says is genuinely new and which the normalized crown metric is blind to by construction.

**n = 1.** One subject, one seed, CPU, ViT-L. This is a first look, not a verdict, and it is the
reason 11.9 asks for the full comparison rather than treating this as settled. But it is a first
look that points away from the port, not toward it.

The full experiment is still half a day of Python and no C++: regenerate §9's 12 krea2 renders,
matte them with `Cutout::for_depth` verbatim, run `MoGeModel.infer`, and report §9.2/§9.3's exact
metrics — crowns at d≤2 and d≥20, climb, int/all, fist-vs-torso, truck front-vs-bed, the seven
head/torso signs — beside the DA2 row, plus `z_span / x_span` per subject.

**The environment is left ready** at `/home/dbrain/.cache/moge-assess/` (1.6 GB, outside every repo):
`model.pt`, the cloned reference, a venv, `run_moge.py`, `crops.py` (the `Cutout::for_depth`
mirror) and `crowns.py` (the §9.2 metric, validated by reproducing DA2's published climb). Delete it
if the answer is "stay on DA2". Note `crowns.py` must `nan_to_num` the point map first — `infer`'s
`apply_mask=True` writes NaN outside the predicted mask.

### 11.9 Recommendation

**Do not port MoGe-2 now.** In order:

1. **Finish 11.8's Python comparison on all 12 renders.** Half a day, no C++, no repo changes. The
   n=1 first look already has MoGe *behind* DA2 on climb (0.285 vs 0.332) at r = 0.91 agreement, so
   the bar is: does MoGe's climb and its ground-truth signs beat DA2's 0.253 / `+0.302` / 7-of-7 by a
   wide margin, and is `z_span / x_span` per subject plausible rather than merely well-formed? If
   not, the question is closed and the answer is DA2, already shipped.
2. **If it does win, port `moge-2-vits-normal`, not `moge-2-vitl`.** 35 M params, ~70 MB F16 GGUF,
   the same size class as the two depth models already in `models/`, and it carries the normal head
   for free. The architecture is identical to ViT-L — in `visp` the difference is GGUF metadata — so
   nothing in the spike branch's captured spec is wasted. Take the ViT-L only if the small one
   measurably loses, and expect ~1.5 GiB of VRAM and multi-second latency on this box if you do.
3. **Run at `num_tokens = 1200`, not 3600.** The bill is the 960² conv heads, the consumer samples a
   65×65 grid, and §9.4 plus 11.7 both say the fine grid is not where the form is.
4. **Skip `scale_head`. Take `normal_head` only when the consumer can spend it** — that means baking
   a normal texture instead of `inflate.rs`'s vertex colours, which is the larger half of the job.
5. **Do not revive `spike/sparse-conv-3d` in place.** Read it as the spec it is. The expensive
   knowledge — ConvStack layouts, replicate padding, the UV pyramid, the `exp` remap, the LM fit, and
   the confirmation that there are no register tokens — is already captured and transfers unchanged.

**Still unverified on CUDA:** everything in 11.6's estimate table. If MoGe is ever built, the
`ggml_conv_transpose_2d_p0` at 1024→256 and the `VISP_IM2COL_MAX` fallback path at 960² are the two
things to watch, and the gate budget needs a real `nvidia-smi` figure exactly as §2 asks for DA3.
