// sam3-bench — single-process SAM 3 PCS driver for profiling and VRAM accounting.
//
// vision-server fork+execv's a `--worker` child that owns the CUDA context, so
// nsys/ncu pointed at the server trace a process that never touches the GPU
// (--trace-fork-before-exec was not sufficient — the report came back with no
// CUDA kernel data at all). This runs the identical sam3 calls the worker makes,
// in one process, so the profilers have something to attach to.
//
//   sam3-bench <model.ggml> <image> <prompt> [reps] [threshold]
//
// Reports device free/total around each stage, so the weight buffer and the
// compute arena are attributed separately rather than read off nvidia-smi as one
// number.

#include "ggml-backend.h"
#include "ggml.h"
#include "sam3.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// Device memory in MiB. Returns used, or -1 where there is no CUDA device.
double device_used_mib() {
#ifdef GGML_USE_CUDA
    size_t free_b = 0;
    size_t total_b = 0;
    ggml_backend_cuda_get_device_memory(0, &free_b, &total_b);
    return double(total_b - free_b) / (1024.0 * 1024.0);
#else
    return -1.0;
#endif
}

void mark(const char* label, double baseline) {
    const double used = device_used_mib();
    if (used < 0) {
        printf("  %-28s (no cuda device)\n", label);
        return;
    }
    printf("  %-28s used %8.1f MiB   delta %+8.1f MiB\n", label, used, used - baseline);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: sam3-bench <model.ggml> <image> <prompt> [reps] [threshold]\n");
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string prompt = argv[3];
    const int reps = argc > 4 ? atoi(argv[4]) : 3;
    const float threshold = argc > 5 ? (float)atof(argv[5]) : 0.35f;

    const double base = device_used_mib();
    printf("device used at start: %.1f MiB\n", base);

    sam3_params p;
    p.model_path = model_path;
    p.use_gpu = getenv("SAM3_FORCE_CPU") == nullptr;
    p.n_threads = getenv("SAM3_THREADS") ? atoi(getenv("SAM3_THREADS")) : 12;
    if (const char* e = getenv("SAM3_ENCODE_IMG_SIZE")) {
        p.encode_img_size = atoi(e);
        printf("encode_img_size override: %d (grid %d)\n", p.encode_img_size,
               p.encode_img_size / 14);
    }

    auto t0 = clk::now();
    auto model = sam3_load_model(p);
    if (!model) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    printf("\nload: %.0f ms\n", ms_since(t0));
    mark("after weights", base);

    t0 = clk::now();
    auto state = sam3_create_state(*model, p);
    if (!state) {
        fprintf(stderr, "state alloc failed\n");
        return 1;
    }
    printf("state: %.0f ms\n", ms_since(t0));
    mark("after state", base);

    sam3_image img = sam3_load_image(image_path);
    if (img.data.empty()) {
        fprintf(stderr, "image load failed: %s\n", image_path.c_str());
        return 1;
    }
    printf("image: %dx%d\n\n", img.width, img.height);

    for (int i = 0; i < reps; ++i) {
        t0 = clk::now();
        if (!sam3_encode_image(*state, *model, img)) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }
        const double enc = ms_since(t0);
        mark("after encode", base);

        sam3_pcs_params pcs;
        pcs.text_prompt = prompt;
        pcs.score_threshold = threshold;
        t0 = clk::now();
        sam3_result r = sam3_segment_pcs(*state, *model, pcs);
        const double dec = ms_since(t0);
        mark("after pcs", base);

        printf("rep %d: encode %.1f ms  pcs %.1f ms  detections %zu\n\n",
               i, enc, dec, r.detections.size());
    }

    printf("peak device used: %.1f MiB (baseline %.1f)\n", device_used_mib(), base);
    return 0;
}
