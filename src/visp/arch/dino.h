#pragma once

#include "util/math.h"
#include "visp/ml.h"
#include "visp/vision.h"

#include <vector>

namespace visp::dino {

// Per-block extensions used by Depth-Anything-3. Defaults reproduce plain DINOv2.
struct block_params {
    bool qk_norm = false;
    tensor rope_pos_y = nullptr;
    tensor rope_pos_x = nullptr;
    float rope_frequency = 100.f;
};

tensor interpolate_pos_encoding(model_ref m, tensor x, int64_t w, int64_t h, int patch_size);
tensor prepare_tokens(model_ref m, tensor x, int patch_size, tensor pos_encoding = nullptr);
tensor layer_scale(model_ref m, tensor x);
tensor mlp(model_ref m, tensor x);
tensor self_attention(model_ref m, tensor x, int n_heads, block_params const& = {});
tensor layer(model_ref m, tensor x, dino_params const& p, block_params const& = {});

std::vector<tensor> get_intermediate_layers(
    model_ref m, tensor x, std::span<int const> layers, dino_params const& p);

} // namespace visp::dino
