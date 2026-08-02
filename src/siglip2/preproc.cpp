#include "siglip2/siglip2.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace siglip2 {

namespace {

// Snap a scaled dimension up to a multiple of patch_size, floor of one patch.
int scaled_size(double scale, int size, int patch_size) {
    double v = (double)size * scale;
    int rounded = (int)std::ceil(v / (double)patch_size) * patch_size;
    return std::max(patch_size, rounded);
}

// Mirrors HF get_image_size_for_max_num_patches.
void compute_target_size(
    int height, int width, int patch_size, int max_num_patches, int& out_h, int& out_w) {
    double const eps = 1e-5;
    double lo = eps / 10.0;
    double hi = 100.0;
    while ((hi - lo) >= eps) {
        double mid = 0.5 * (lo + hi);
        int th = scaled_size(mid, height, patch_size);
        int tw = scaled_size(mid, width, patch_size);
        long long n = ((long long)th / patch_size) * ((long long)tw / patch_size);
        if (n <= (long long)max_num_patches) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    out_h = scaled_size(lo, height, patch_size);
    out_w = scaled_size(lo, width, patch_size);
}

// Separable triangular low-pass, matching torchvision's
// F.interpolate(mode='bilinear', antialias=True) which HF Siglip2ImageProcessor
// uses for both up- and down-sampling. The support radius widens with the
// downscale factor, which is what makes it antialiased.
struct AAFilter1D {
    int max_taps = 0;
    std::vector<int> first;
    std::vector<int> taps;
    std::vector<float> weights; // [out * max_taps + tap]
};

void build_aa_filter(int src_size, int dst_size, AAFilter1D& f) {
    double const scale = (double)src_size / (double)dst_size;
    double const r = std::max(scale, 1.0);
    f.max_taps = (int)std::ceil(2.0 * r) + 2;
    f.first.assign(dst_size, 0);
    f.taps.assign(dst_size, 0);
    f.weights.assign((size_t)dst_size * f.max_taps, 0.0f);

    for (int i = 0; i < dst_size; ++i) {
        double const center = (i + 0.5) * scale - 0.5;
        int k0 = (int)std::ceil(center - r);
        int k1 = (int)std::floor(center + r);
        f.first[i] = k0;
        int const n = k1 - k0 + 1;

        double sum = 0.0;
        for (int t = 0; t < n; ++t) {
            double const d = ((k0 + t) - center) / r;
            double const w = std::max(0.0, 1.0 - std::abs(d));
            f.weights[(size_t)i * f.max_taps + t] = (float)w;
            sum += w;
        }
        // r >= 1 guarantees a centre tap with w=1, so sum > 0.
        float const inv = (float)(1.0 / sum);
        for (int t = 0; t < n; ++t) {
            f.weights[(size_t)i * f.max_taps + t] *= inv;
        }
        f.taps[i] = n;
    }
}

void resize_bilinear_u8(
    uint8_t const* src, int src_h, int src_w, int channels, uint8_t* dst, int dst_h, int dst_w) {
    AAFilter1D fy;
    AAFilter1D fx;
    build_aa_filter(src_h, dst_h, fy);
    build_aa_filter(src_w, dst_w, fx);

    std::vector<float> tmp((size_t)dst_h * src_w * channels);
    for (int y = 0; y < dst_h; ++y) {
        int const k0 = fy.first[y];
        int const n = fy.taps[y];
        float const* wrow = fy.weights.data() + (size_t)y * fy.max_taps;
        for (int x = 0; x < src_w; ++x) {
            float* out = tmp.data() + ((size_t)y * src_w + x) * channels;
            for (int c = 0; c < channels; ++c) {
                out[c] = 0.0f;
            }
            for (int t = 0; t < n; ++t) {
                int sy = std::clamp(k0 + t, 0, src_h - 1);
                float const w = wrow[t];
                uint8_t const* sp = src + ((size_t)sy * src_w + x) * channels;
                for (int c = 0; c < channels; ++c) {
                    out[c] += w * (float)sp[c];
                }
            }
        }
    }

    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            int const k0 = fx.first[x];
            int const n = fx.taps[x];
            float const* wrow = fx.weights.data() + (size_t)x * fx.max_taps;
            uint8_t* out = dst + ((size_t)y * dst_w + x) * channels;
            float acc[8] = {0};
            for (int t = 0; t < n; ++t) {
                int sx = std::clamp(k0 + t, 0, src_w - 1);
                float const w = wrow[t];
                float const* sp = tmp.data() + ((size_t)y * src_w + sx) * channels;
                for (int c = 0; c < channels; ++c) {
                    acc[c] += w * sp[c];
                }
            }
            for (int c = 0; c < channels; ++c) {
                out[c] = (uint8_t)std::clamp((int)std::lrintf(acc[c]), 0, 255);
            }
        }
    }
}

// HF convert_image_to_patches works on CHW: reshape (3, n_h, p, n_w, p) then
// permute (1,3,2,4,0), which puts channels innermost — the same order as the
// HWC input here, so the copy is a straight gather.
void rescale_normalize_patchify(
    uint8_t const* rgb,
    int height,
    int width,
    int patch_size,
    float rescale_factor,
    float const mean[3],
    float const std_v[3],
    std::vector<float>& out) {
    int const n_h = height / patch_size;
    int const n_w = width / patch_size;
    int const feat = 3 * patch_size * patch_size;
    out.assign((size_t)n_h * n_w * feat, 0.0f);

    for (int ph = 0; ph < n_h; ++ph) {
        for (int pw = 0; pw < n_w; ++pw) {
            float* dst = out.data() + ((size_t)ph * n_w + pw) * feat;
            for (int py = 0; py < patch_size; ++py) {
                int const y = ph * patch_size + py;
                for (int px = 0; px < patch_size; ++px) {
                    int const x = pw * patch_size + px;
                    uint8_t const* src_pix = rgb + ((size_t)y * width + x) * 3;
                    size_t const off = ((size_t)py * patch_size + px) * 3;
                    for (int c = 0; c < 3; ++c) {
                        dst[off + c] = ((float)src_pix[c] * rescale_factor - mean[c]) / std_v[c];
                    }
                }
            }
        }
    }
}

} // namespace

bool preprocess_image_rgb(
    uint8_t const* rgb,
    int height,
    int width,
    int max_num_patches,
    int patch_size,
    float rescale_factor,
    float const image_mean[3],
    float const image_std[3],
    PreprocResult& out,
    std::string& error) {
    if (!rgb || height <= 0 || width <= 0) {
        error = "empty input";
        return false;
    }
    if (patch_size <= 0 || max_num_patches <= 0) {
        error = "invalid patch_size or max_num_patches";
        return false;
    }

    int target_h = 0;
    int target_w = 0;
    compute_target_size(height, width, patch_size, max_num_patches, target_h, target_w);

    std::vector<uint8_t> resized((size_t)target_h * target_w * 3);
    resize_bilinear_u8(rgb, height, width, 3, resized.data(), target_h, target_w);

    rescale_normalize_patchify(
        resized.data(), target_h, target_w, patch_size, rescale_factor, image_mean, image_std,
        out.pixel_values);

    out.n_patches_h = target_h / patch_size;
    out.n_patches_w = target_w / patch_size;
    return true;
}

} // namespace siglip2
