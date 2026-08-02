#pragma once

#include "visp/ml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace siglip2 {

enum class Pooling {
    MEAN,  // mean over patches (HF last_hidden_state.mean(dim=1))
    PROBE, // Siglip2MultiheadAttentionPoolingHead
};

struct VisionConfig {
    int hidden_size = 0;
    int num_attention_heads = 0;
    int num_hidden_layers = 0;
    int patch_size = 16;
    int num_patches = 256; // native grid, must be a perfect square
    int num_channels = 3;
    float layer_norm_eps = 1e-6f;
};

struct TextConfig {
    int hidden_size = 0;
    int num_attention_heads = 0;
    int num_hidden_layers = 0;
    int max_position_embeddings = 64;
    int projection_size = 0;
    float layer_norm_eps = 1e-6f;
};

struct ScoreParams {
    float logit_scale = 0.0f; // gets exponentiated
    float logit_bias = 0.0f;
};

struct PreprocParams {
    int default_max_num_patches = 256;
    float rescale_factor = 1.0f / 255.0f;
    float image_mean[3] = {0.5f, 0.5f, 0.5f};
    float image_std[3] = {0.5f, 0.5f, 0.5f};
};

// Weights of one SigLIP2 GGUF, uploaded once onto `device`. Both towers share
// the single backend buffer, so a dual-tower deployment pays for the weights
// once instead of per-encoder.
struct Model {
    visp::model_weights weights;
    ggml_backend_t backend = nullptr;

    VisionConfig vision;
    TextConfig text;
    ScoreParams score;
    PreprocParams preproc;
    bool has_vision = false;
    bool has_text = false;

    ggml_tensor* get(char const* name) const; // throws if missing
};

// Throws visp::except on failure.
Model load_model(std::string const& gguf_path, visp::backend_device const& device);

// Result of preprocessing an image for the vision encoder. pixel_values is
// row-major fp32 with n_patches_h*n_patches_w rows of num_channels*patch_size^2.
struct PreprocResult {
    std::vector<float> pixel_values;
    int n_patches_h = 0;
    int n_patches_w = 0;
};

// Mirrors HF Siglip2ImageProcessor: binary-search the largest scale whose
// patch grid fits max_num_patches, antialiased-bilinear resize, rescale,
// normalize, patchify. `rgb` is channels-last uint8 (y*width*3 + x*3 + c).
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
    std::string& error);

// Vision tower. NaFlex: the caller supplies pre-patchified pixel_values plus
// the spatial grid; the native position embedding is interpolated in-graph.
class VisionEncoder {
  public:
    VisionEncoder();
    ~VisionEncoder();

    bool load(Model const& model);
    void close();

    std::string const& last_error() const { return error_msg_; }

    bool encode(
        float const* pixel_values,
        int n_patches_h,
        int n_patches_w,
        Pooling pooling,
        std::vector<float>& out_embedding);

  private:
    std::string error_msg_;
    struct State;
    State* state_ = nullptr;
};

// Text tower. Bidirectional self-attention, pooled at the last position then
// projected through the linear head (HF Siglip2TextModel).
class TextEncoder {
  public:
    TextEncoder();
    ~TextEncoder();

    bool load(Model const& model);
    void close();

    std::string const& last_error() const { return error_msg_; }

    // token_ids is n_tokens*n_batch I32, batch-major, every prompt padded to
    // the same n_tokens. attention_mask must be null for n_batch > 1 (HF's
    // path passes no mask). out_embeddings is projection_size*n_batch.
    bool encode_batch(
        int32_t const* token_ids,
        int n_tokens,
        int n_batch,
        int32_t const* attention_mask,
        std::vector<float>& out_embeddings);

  private:
    std::string error_msg_;
    struct State;
    State* state_ = nullptr;
};

// Sentencepiece tokenizer (Gemma-style 256K vocab). Padding mirrors HF:
// padding="max_length", truncation, EOS appended.
class Tokenizer {
  public:
    Tokenizer();
    ~Tokenizer();

    bool load(std::string const& spm_model_path);
    void close();

    std::string const& last_error() const { return error_msg_; }

    bool encode(
        std::string const& text,
        int max_length,
        std::vector<int32_t>& out_token_ids,
        std::vector<int32_t>& out_attention_mask);

  private:
    struct State;
    State* state_ = nullptr;
    std::string error_msg_;
    int pad_id_ = 0;
    int eos_id_ = -1;
};

// logits_per_image[i,j] = (l2(img[i]) . l2(txt[j])) * exp(logit_scale) + logit_bias,
// probs = sigmoid(logits). Both outputs are (n_image, n_text) row-major and
// either may be null.
void score_image_text(
    float const* image_embeds,
    int n_image,
    float const* text_embeds,
    int n_text,
    int hidden,
    ScoreParams const& params,
    float* logits_per_image,
    float* probs_per_image);

} // namespace siglip2
