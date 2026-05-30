# Matting (BiRefNet / RMBG-2.0) — STATE & HANDOFF

## ✅ RESOLVED (was "GPU path may be broken") — scheduler removal is a NET WIN
The deep-CPU agent's git checkout removed the GPU sched-fallback. Tested: single-backend
GPU now runs CLEAN at **~700ms** (was 866ms), matte correct — because the CUDA deform
kernel (af69870c) made the CPU-fallback obsolete (it was only adding copies). Keep the
revert. ml.h cpu_fallback/sched members are now unused (harmless). VISP_PROFILE_OPS
profiler was also lost (dev tool — re-add only if a future profiling pass needs it).
CPU floor confirmed: 9.89s @1024 (full Zen3 AVX2/FMA, no free win; only tradeoffs/quant left).
Original issue text below (kept for history):

## ⚠️ (HISTORICAL) ACTIVE ISSUE (2026-05-30, post deep-CPU pass) — GPU path may be broken
The deep-CPU agent ran `git checkout` on src/visp/ml.cpp + birefnet.cpp and REVERTED
prior-session UNCOMMITTED GPU work. Confirmed damage:
- `ml.cpp` compute() is back to single-backend `ggml_backend_graph_compute(b,g.graph)` —
  the GPU `ggml_backend_sched` {GPU,CPU} fallback + VISP_PROFILE_OPS profiler are GONE.
- `ml.h` lost the compute_graph `cpu_fallback` + `sched` members.
- birefnet.cpp restored (agent had backup); ml.cpp CPU thread-tuning re-applied.
KEY QUESTION being tested: dbrain/ggml @af69870c now HAS a CUDA CONV_2D_DEFORM kernel, so
the CPU-fallback sched may NO LONGER BE NEEDED — single-backend GPU may just work. If GPU
build+run gives YAVG~181, the revert is fine (sched was only needed pre-CUDA-kernel). If it
crashes "op not supported", restore the sched code (was in ml.cpp compute_graph_allocate +
compute, ml.h compute_graph struct — see RESULTS.md "Make compute() use ggml_backend_sched").
CPU path (the DEPLOY TARGET) is intact + verified (9.89s YAVG 181.087). Profiler loss = low stakes (dev tool).
LESSON: never run >1 agent writing the same uncommitted files; COMMIT vision.cpp work to stop this.


Non-PyTorch background-removal service on Acly/vision.cpp, mirroring ~/dev/*.cpp.
Decision: **GPU** (verified faster + fits). This doc is GROUND TRUTH — only verified
facts. If you change state, update this file.

## Verified status (2026-05-30)
- ✅ GPU @1024: **~700 ms** (was 866/1639/4061ms; sched-fallback removal as deform now
  has native CUDA kernel), correct matte (visual == ref). bench/POSTREVERT_gpu.png. SELF-VERIFIED.
  Session GPU total 4061→700 = 5.8×.
- ✅ CPU @1024: **~9.9 s** @ default 6 threads (physical cores; was 10), YAVG 181.557.
  bench/FINAL_cpu.png. Thread default tuned + VISP_CPU_THREADS env override added.
- ✅ MEMORY (measured 2026-05-30): GPU peak VRAM **~3580 MiB** (~3.2GB steady), host RSS
  **~1593 MiB** (~half is the 846MB static binary). VRAM elevated by im2col-GEMM scratch
  (speed/VRAM trade). Session GPU total: 4061ms → **866ms (4.7×)**, matte correct throughout.
- ggml @ af69870c (consolidated-v0.13, NOT pushed); depend/ggml synced same SHA (0 dirty).
  (Both final levers are vision.cpp-side only — ggml untouched. See bench/RESULTS.md.)
- GPU is ~2.2–2.5× CPU (~4.06s GPU vs ~9.9s CPU @1024). Ported to dbrain/ggml @9ae8f517.
- PROFILE DONE (bench/PROFILE.md, real per-op via VISP_PROFILE_OPS). Bottlenecks DIFFER:
  - GPU: **decoder 86.7%** — CONV_2D_DEFORM 45.9% (1.84s) + CONV_2D 39.4% (1.58s); encoder ~13%.
  - CPU: **encoder 73%** — MUL_MAT (Swin attention) 52.9% (5.28s); decoder 27%.
  GPU deform kernel is memory-latency-bound (per-IC gather, no tiling) → ~46% lever but
  needs a real CUDA rewrite in SHARED ggml. NOT done. (Prior 'deform not bottleneck' A/B
  was a misread; '53% depthwise' was fabricated — PROFILE.md is ground truth.)

## Three CUDA bugs fixed (all real, all in working tree, NOT committed)
1. **Mixed-precision bias add** — GPU loads weights F16, activations F32; ggml-cuda
   binbcast add asserts (`nb10 % sizeof`). Fix: `src/visp/nn.cpp` `cast_like(m,w,ref)`
   helper casts loaded weight→activation dtype in add_bias_2d/linear/layer_norm/
   batch_norm_2d. Lossless; CPU path guard-skips. Crash site: Swin patch_embed.proj.bias.
2. **Flash attention** crashes `fattn.cu:492` (head_dim=32) → run with
   `VISP_FLASH_ATTENTION=0`. (FA-on as a perf lever is UNTESTED — revisit.)
3. **CONV_2D_DEFORM "op not supported" on CUDA** → `src/visp/ml.cpp`
   compute_graph_allocate()+compute() build a `ggml_backend_sched` over {GPU,CPU} so
   unsupported ops offload to CPU. PLUS `src/visp/nn.cpp` conv_2d_deform() casts weight
   to F32 (CPU deform kernel hardcodes F32 GEMM, no F16 path → garbage/inf without this).

## Other uncommitted edits
- `src/visp/arch/birefnet.cpp`: 3× `ggml_cont` on decoder residual adds — added during
  early (wrong-site) crash debugging. HARMLESS but not load-bearing; revert when cleaning.
- `CMakeLists.txt` + `src/server/CMakeLists.txt`: `VISP_SERVER` option + matting-server target.
- `include/visp/ml.h`: compute_graph gains cpu_fallback + sched members.
- `src/server/birefnet_server.cpp`: the HTTP server (untracked). NOTE: has a temp
  `X-BG-Debug` response header to remove before final.

## Build & run (host has NO cuda/cmake — use the builder image)
Build (static; shared libggml-cuda fails to link with driver-API undefined refs):
```
docker run --rm -v "$HOME/dev/vision.cpp:/src" -v matting-ccache:/root/.ccache \
  -e CCACHE_DIR=/root/.ccache flux2-dev:builder bash -lc \
  'cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
     -DBUILD_SHARED_LIBS=OFF -DVISP_STATIC_GGML=ON -DVISP_SERVER=ON -DVISP_TESTS=OFF \
     -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && \
   cmake --build /src/build --target vision-cli matting-server -j"$(nproc)"'
```
(Incremental: just the last `cmake --build`. ccache relink can serve a stale .o → `touch` the .cpp.)
Run (GPU; even `-b cpu` needs --gpus all because the binary links cuda):
```
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$HOME/dev/vision.cpp:/src" flux2-dev:builder bash -lc \
  'VISP_FLASH_ATTENTION=0 /src/build/bin/vision-cli birefnet -b gpu \
     -m /src/models/RMBG-2.0-F16.gguf -i /src/tests/input/cat-and-hat.jpg -o /src/bench/out.png'
```
Base image prints a CUDA banner + harmless "NVIDIA Driver not detected" warning; GPU IS
used (look for "ggml_cuda_init: found 1 CUDA devices"). Model: models/RMBG-2.0-F16.gguf
(F32 also present). Test imgs: tests/input/{cat-and-hat,vase-and-bowl,wardrobe}.jpg.

## Correctness check (the ONLY way to trust a run)
A correct matte has mean luma ~181; a broken one ~5. Verify the PNG exists AND check luma:
```
ffmpeg -i bench/out.png -vf "format=gray,signalstats" -f null - 2>&1 | grep -oE 'YAVG:[0-9.]+'
```
NEVER report a perf number without confirming the output PNG is a correct matte.

## dbrain/ggml PORT — wiring done, gate pending (2026-05-30)
- vision.cpp now builds against dbrain/ggml via submodule `depend/ggml`
  (.gitmodules → git@github.com:dbrain/ggml.git; CMakeLists.txt:143
  `add_subdirectory(depend/ggml)`; dbrain/ggml is a STANDALONE ggml, CMake at root).
- The conv_2d_deform CPU op was ported into dbrain/ggml and **committed**:
  `39c757c0` on branch `consolidated-v0.13` ("feat(vision): port conv_2d_deform
  CPU op"). 5 files: include/ggml.h, src/ggml.c, src/ggml-cpu/{ops.cpp,ops.h,ggml-cpu.c}.
  CPU impl only (no CUDA kernel — that's the next task; scheduler offloads to CPU).
- **NOT pushed to github yet.** depend/ggml was populated by local clone+checkout of
  39c757c0. Before any fresh-clone build or fork push: `git -C ~/dev/ggml push` the
  consolidated-v0.13 branch, then pin the submodule to the pushed SHA.
- ⚠️ The port subagent fabricated commit SHAs (65aa64f/b0a4dbf) that never existed;
  its real work was UNCOMMITTED changes in ~/dev/ggml working tree (consolidated-v0.13).
  I committed them as 39c757c0. Always verify SHAs exist (`git cat-file -t`).
- GATE PASSED on ported build (2026-05-30): CPU YAVG 181.45 (10.1s), GPU YAVG 181.27
  (4.9s). Both correct (ref 181.06). Port to dbrain/ggml VERIFIED. bench/PORT_{cpu,gpu}.png.

## Remaining plan (order)
1. PORT to dbrain/ggml — wiring + deform commit done; GATE pending (above). dbrain/ggml @048cba4d is ggml-only
   (v0.13); current dep is Acly/llama.cpp submodule at depend/llama (ggml at
   depend/llama/ggml). Missing op in dbrain/ggml: conv_2d_deform (~360 LOC CPU + enum +
   dispatch); win_part/conv_transpose already there. Carry forward all 3 fixes above.
   The binbcast assert + fattn behavior exist in dbrain/ggml too — re-verify. Gate:
   GPU matte YAVG ~181 after port.
2. CUDA deform-conv kernel (THE perf win — removes CPU fallback). On dbrain/ggml.
3. Measure (VRAM, process_res sweep, server /remove e2e), FA-on retry, thread tuning,
   PyTorch RMBG-2.0 parity (bg_server.py on 10.0.0.151).
4. Wire matting-server e2e + docker + final report; remove temp debug header; fork push
   to dbrain/hbd-vision.cpp.

## Process notes
- Do NOT batch many interdependent bash calls in parallel — one failure cancels all.
- Verify artifacts on disk before reporting; never invent numbers.
- One GPU on the box — never two GPU jobs at once.
