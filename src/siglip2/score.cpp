#include "siglip2/siglip2.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace siglip2 {

namespace {

void l2_normalize_rows(float* x, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* row = x + (size_t)r * cols;
        double sumsq = 0.0;
        for (int c = 0; c < cols; ++c) {
            sumsq += (double)row[c] * (double)row[c];
        }
        float const inv = (float)(1.0 / (std::sqrt(sumsq) + 1e-12));
        for (int c = 0; c < cols; ++c) {
            row[c] *= inv;
        }
    }
}

} // namespace

void score_image_text(
    float const* image_embeds,
    int n_image,
    float const* text_embeds,
    int n_text,
    int hidden,
    ScoreParams const& params,
    float* logits_per_image,
    float* probs_per_image) {
    std::vector<float> img((size_t)n_image * hidden);
    std::vector<float> txt((size_t)n_text * hidden);
    std::memcpy(img.data(), image_embeds, sizeof(float) * img.size());
    std::memcpy(txt.data(), text_embeds, sizeof(float) * txt.size());
    l2_normalize_rows(img.data(), n_image, hidden);
    l2_normalize_rows(txt.data(), n_text, hidden);

    float const scale = std::exp(params.logit_scale);
    float const bias = params.logit_bias;

    for (int i = 0; i < n_image; ++i) {
        float const* imrow = img.data() + (size_t)i * hidden;
        for (int j = 0; j < n_text; ++j) {
            float const* txrow = txt.data() + (size_t)j * hidden;
            double dot = 0.0;
            for (int c = 0; c < hidden; ++c) {
                dot += (double)imrow[c] * (double)txrow[c];
            }
            float const logit = (float)(dot * (double)scale + (double)bias);
            if (logits_per_image) {
                logits_per_image[(size_t)i * n_text + j] = logit;
            }
            if (probs_per_image) {
                float p;
                if (logit >= 0.0f) {
                    p = 1.0f / (1.0f + std::exp(-logit));
                } else {
                    float const e = std::exp(logit);
                    p = e / (1.0f + e);
                }
                probs_per_image[(size_t)i * n_text + j] = p;
            }
        }
    }
}

} // namespace siglip2
