# SigLIP2 fold-in — state, decisions, and what a native visp port would cost

Folds `hbd-siglip2.cpp` (separate repo, last touched `06cbf31`) into this tree so matting
and SigLIP2 run from one container against one visp/ggml, one backend device and one VRAM
budget. GPU/parity measurements live in `HANDOFF-gpu-verification.md` §5.

## Depth of port: (a), not (b)

Two options were on the table: (a) move the self-contained model code across, or (b) rewrite
SigLIP2 as a first-class visp arch on `visp/ml.h` + `nn.cpp` + the `model_family` registry.
**(a) was taken, with the loader and the process topology genuinely unified** (see below).
Reasons, in order of weight:

1. **Persisted embeddings.** koblem stores image embeddings and later scores them against
   text via `/v1/classify_from_embeddings`. A rewrite onto visp's F16-everywhere conventions
   would have to re-derive which activations are safe in F16 and then prove the result still
   matches stored vectors — and the only card that matters for that (CUDA) is not this box.
   The moved code carries its numerics with it; measured cosine vs HF is 0.9994 image /
   1.0000 text, above the source repo's 0.999 floor.
2. **`src/visp/arch/` is occupied.** (b) lives exactly there and in `include/visp/vision.h`,
   where the Depth-Anything-3 port is active. (b) cannot be done concurrently.
3. **Size.** See the estimate below — it is not a small job.

### What (a) still consolidated

- **`gguf_loader.{cpp,h}` is gone.** `src/siglip2/model.cpp` loads through
  `visp::model_load` + `model_init` + `model_transfer` onto a `visp::backend_device`.
  `float_type = GGML_TYPE_COUNT` and `layout = unknown` means the transfer is a verbatim
  byte copy, so weights are identical to what the old loader produced.
- **One weight buffer for both towers.** The source gave the vision and text encoders a
  backend and a copy of the weights each (for overlapping CUDA streams). They now share one
  `siglip2::Model`, halving resident weights.
- **`worker_ipc.cpp` / `worker_session.cpp` did not come across** — see below.
- **`src/cuda/` (megakernel, custom MMQ, ~2000 lines) did not come across** — CUDA-only, and
  its call sites were deleted rather than stubbed.
- **`VISION_ONLY` / `TEXT_ONLY` dropped.** They saved ~1 GiB by skipping a tower at load;
  with a shared buffer that no longer factors, nothing in the kobbler deployment sets them,
  and the part-rig pipeline needs both towers.

## worker_ipc / worker_session / IDLE_UNLOAD_SECONDS

**Deleted, because this repo already had the identical machinery.** The premise that the
matting server is a resident single-process service is wrong: `birefnet_server.cpp` was
already worker-isolated with the same fork + `execv --worker <fd>` + AF_UNIX socketpair +
12-byte length-prefixed frames shape (both were borrowed from `hbd-qwen3-tts.cpp`), and
already had `IDLE_UNLOAD_SECONDS` plus an in-flight-gated idle watchdog.

Keeping siglip2's copy would have meant two IPC layers and two worker lifecycles in one
binary — the opposite of the point. Instead `src/server/server_ipc.{h,cpp}` is the matting
layer generalised: `spawn_worker` takes arbitrary extra argv, and `Frame` gained
`SIGLIP_REQ` / `SIGLIP_RESP` alongside `MATTE_REQ` / `MATTE_RESP` (numeric values unchanged).
One worker process owns the device and every model; models load lazily per family.

`IDLE_UNLOAD_SECONDS` and `MATTING_KEEP_RESIDENT` keep their matting semantics — 0 means
evict the instant the server goes idle. That is right for matting (small model, fast reload)
and wrong for SigLIP2, so the server logs a warning at boot if siglip2 is configured with
`IDLE_UNLOAD_SECONDS=0` and no keep-resident. The old siglip2 container ran 300.

## Layout

```
src/siglip2/         model.cpp (visp-backed GGUF load) vision.cpp text.cpp
                     tokenizer.cpp preproc.cpp score.cpp siglip2.h
src/server/          server_ipc.*    frames, socketpair IPC, spawn, base64
                     server_state.*  worker lifecycle, request round-trip, in-flight guard
                     worker.cpp      the GPU-owning child; one device, one slot per family
                     matting_endpoints.cpp   /remove
                     siglip_endpoints.cpp    /v1/embeddings /classify /text_embeddings
                                             /classify_from_embeddings
                     main.cpp        config, /health, /v1/gpu/status, admin, idle watchdog
scripts/convert_siglip2_to_gguf.py   copied verbatim from hbd-siglip2.cpp
```

A third family (depth) slots in as `depth_endpoints.cpp` + a `DEPTH_REQ` frame + a lazily
loaded slot in `WorkerModels`; nothing else needs to move.

## Wire API

`/v1/embeddings`, `/v1/classify`, `/v1/text_embeddings`, `/v1/classify_from_embeddings`
match `koblem/api/src/vision.rs` field for field. Two things are deliberate and easy to
"fix" by mistake:

- **Scalar knobs are read from the query string / urlencoded body only, never from multipart
  parts.** cpp-httplib files multipart fields under `req.files`, never `req.params`, so the
  siglip2-server this replaces has always ignored `max_num_patches`, `pooling`,
  `return_last_hidden` and `return_logits` when koblem sends them as multipart text — which
  it does, on every call that sets them. Honouring them now would change the embeddings
  koblem has already persisted. `prompts` is the exception; it was always collected from both
  sources. Any change here is a data migration, not a bug fix.
- `return_last_hidden=true` still answers 501.

Two intentional differences:

- **The `gpu` field is now honoured.** The old siglip2-server parsed it nowhere; `/remove`
  already used it for gate placement, and siglip requests now go through the same path.
  Consequence: if the gate targets a different card than the live worker, the worker
  respawns and reloads. Disabled-card enforcement depends on this.
- A wrong-dimension `image_embeddings` row now answers 500 rather than 400 (the check moved
  into the worker, which only reports errors as one frame type).

## Reproducing the test model

The base checkpoint's `config.json` omits every default, and the converter's fallbacks are
so400m's, so the config has to be resolved first:

```
uv venv -p /usr/bin/python3 --system-site-packages .venv-siglip
uv pip install --python .venv-siglip/bin/python safetensors huggingface_hub transformers
# stage: symlink model.safetensors + preprocessor_config.json, write a resolved config.json
#   from transformers.Siglip2Config.from_pretrained(<snapshot>).to_dict()
.venv-siglip/bin/python scripts/convert_siglip2_to_gguf.py -i <stage> \
  -o models/SigLIP2-base-patch16-naflex-F16.gguf -t f16
cp <snapshot>/tokenizer.model models/siglip2-tokenizer.model
```

Run:

```
SIGLIP_MODEL_PATH=models/SigLIP2-base-patch16-naflex-F16.gguf \
SIGLIP_TOKENIZER_PATH=models/siglip2-tokenizer.model \
MODEL_PATH=models/BiRefNet-lite-F16.gguf \
MATTING_BACKEND=gpu MATTING_KEEP_RESIDENT=1 PORT=8898 build-siglip/bin/vision-server
```

## What (b) — a native visp arch — would actually take

Roughly 1200–1600 lines of new visp code plus a parity campaign that has to run on CUDA.

| Piece | Reuse from visp | New work |
|---|---|---|
| Vision ViT | `arch/dino.cpp` has `layer`, `self_attention`, `mlp`, `layer_norm`, `attention` | ~60% new: no CLS token, NaFlex bilinear+antialias pos-embed resize on a non-square grid (dino does bicubic on square), **fused QKV** weights (dino expects split q/k/v), `gelu_pytorch_tanh`, and the `Siglip2MultiheadAttentionPoolingHead` probe cross-attention. ~350 lines |
| Text tower | nothing — visp has no text model at all | token embedding + learned position embedding + 12–27 non-causal blocks + last-position pooling + linear head. ~300 lines |
| Tokenizer | nothing | sentencepiece stays a dependency either way; the wrapper is ~120 lines and would have to live outside `visioncpp` or drag spm into the shared library |
| NaFlex preproc | `visp/image.h` has resize, but not the aspect binary-search or the patchify | mostly mechanical, ~250 lines |
| Score head | nothing | trivial, ~50 lines |
| Registry | `c-api.cpp`, `model_family`, `vision.h` | params detection, load/compute entry points, C API. ~150 lines, and it collides with the DA3 work |
| **F16-everywhere** | `model_build_flag::f16_activations`, `nn.cpp` `cast_like` | the real cost: decide per-tensor what is safe in F16 for this arch and prove the embeddings still match production. Cannot be validated on this laptop |
| Converter | — | either keep `v.blk.*` / `t.blk.*` naming (visp's `model_ref::find` is name-agnostic, so this is free) or re-map, which invalidates every existing GGUF |

The graph code itself is the tractable half. The expensive half is proving parity against
persisted embeddings on the card that serves production. Sensible sequencing: do (b) after
DA3 lands and `src/visp/arch/` is quiet, on the server, with the so400m checkpoint and a
fixture set of stored embeddings to diff against — not before.
