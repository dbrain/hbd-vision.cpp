#include "siglip2/siglip2.h"
#include "util/string.h"

#include <gguf.h>

namespace siglip2 {

using visp::except;

namespace {

int32_t kv_u32(gguf_context* g, char const* key, int32_t fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) {
        return fallback;
    }
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return (int32_t)gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32: return gguf_get_val_i32(g, id);
        default: return fallback;
    }
}

float kv_f32(gguf_context* g, char const* key, float fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) {
        return fallback;
    }
    return gguf_get_val_f32(g, id);
}

bool kv_bool(gguf_context* g, char const* key, bool fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_BOOL) {
        return fallback;
    }
    return gguf_get_val_bool(g, id);
}

void kv_f32_array(gguf_context* g, char const* key, float* out, int n) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_arr_type(g, id) != GGUF_TYPE_FLOAT32 ||
        (int)gguf_get_arr_n(g, id) != n) {
        return;
    }
    auto const* data = (float const*)gguf_get_arr_data(g, id);
    std::copy(data, data + n, out);
}

// mm.logit_scale / mm.logit_bias are [1]-shaped F32 tensors. model_load reads
// tensor data into host memory, so they can be dereferenced directly.
float scalar_tensor(ggml_context* data, char const* name) {
    ggml_tensor* t = ggml_get_tensor(data, name);
    if (!t || t->type != GGML_TYPE_F32 || ggml_nelements(t) < 1) {
        throw except("SigLIP2 model is missing the F32 scalar tensor '{}'", name);
    }
    return *(float const*)t->data;
}

} // namespace

ggml_tensor* Model::get(char const* name) const {
    ggml_tensor* t = ggml_get_tensor(weights.context.get(), name);
    if (!t) {
        throw except("Missing tensor in SigLIP2 model: {}", name);
    }
    return t;
}

Model load_model(std::string const& gguf_path, visp::backend_device const& device) {
    visp::model_file file = visp::model_load(gguf_path.c_str());
    if (file.arch() != "siglip2") {
        throw except("Expected a siglip2 model, got arch '{}'", file.arch());
    }
    gguf_context* g = file.gguf.get();

    Model m;
    m.has_vision = kv_bool(g, "siglip2.has_vision", true);
    m.has_text = kv_bool(g, "siglip2.has_text", true);

    m.vision.hidden_size = kv_u32(g, "siglip2.vision.embedding_length", 0);
    m.vision.num_attention_heads = kv_u32(g, "siglip2.vision.attention.head_count", 0);
    m.vision.num_hidden_layers = kv_u32(g, "siglip2.vision.block_count", 0);
    m.vision.patch_size = kv_u32(g, "siglip2.vision.patch_size", 16);
    m.vision.num_patches = kv_u32(g, "siglip2.vision.num_patches", 256);
    m.vision.num_channels = kv_u32(g, "siglip2.vision.num_channels", 3);
    m.vision.layer_norm_eps = kv_f32(g, "siglip2.vision.layer_norm_eps", 1e-6f);

    m.text.hidden_size = kv_u32(g, "siglip2.text.embedding_length", 0);
    m.text.num_attention_heads = kv_u32(g, "siglip2.text.attention.head_count", 0);
    m.text.num_hidden_layers = kv_u32(g, "siglip2.text.block_count", 0);
    m.text.max_position_embeddings = kv_u32(g, "siglip2.text.max_position_embeddings", 64);
    m.text.projection_size = kv_u32(g, "siglip2.text.projection_size", m.text.hidden_size);
    m.text.layer_norm_eps = kv_f32(g, "siglip2.text.layer_norm_eps", 1e-6f);

    m.preproc.default_max_num_patches = kv_u32(g, "siglip2.preproc.default_max_num_patches", 256);
    m.preproc.rescale_factor = kv_f32(g, "siglip2.preproc.rescale_factor", 1.0f / 255.0f);
    kv_f32_array(g, "siglip2.preproc.image_mean", m.preproc.image_mean, 3);
    kv_f32_array(g, "siglip2.preproc.image_std", m.preproc.image_std, 3);

    if (m.has_vision && (m.vision.hidden_size <= 0 || m.vision.num_hidden_layers <= 0)) {
        throw except("SigLIP2 GGUF is missing required vision config keys");
    }
    if (m.has_text && (m.text.hidden_size <= 0 || m.text.num_hidden_layers <= 0)) {
        throw except("SigLIP2 GGUF is missing required text config keys");
    }

    if (m.has_vision && m.has_text) {
        m.score.logit_scale = scalar_tensor(file.data.get(), "mm.logit_scale");
        m.score.logit_bias = scalar_tensor(file.data.get(), "mm.logit_bias");
    }

    // Weights are transferred verbatim: no float-type conversion, no layout
    // permutation. The graphs below assume the exact dtypes the converter
    // emitted (F16 matmul weights, F32 norms/biases), and embedding parity
    // against production is defined against those bytes.
    m.weights = visp::model_init(size_t(file.n_tensors()));
    visp::model_transfer(file, m.weights, device);
    m.backend = device;
    return m;
}

} // namespace siglip2
