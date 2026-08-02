#include "siglip2/siglip2.h"
#include "util/env.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace siglip2 {

namespace {

std::string blk_name(int il, char const* suffix) {
    char buf[64];
    snprintf(buf, sizeof(buf), "t.blk.%d.%s", il, suffix);
    return buf;
}

struct Block {
    ggml_tensor* ln1_w = nullptr;
    ggml_tensor* ln1_b = nullptr;
    ggml_tensor* qkv_w = nullptr;
    ggml_tensor* qkv_b = nullptr;
    ggml_tensor* o_w = nullptr;
    ggml_tensor* o_b = nullptr;
    ggml_tensor* ln2_w = nullptr;
    ggml_tensor* ln2_b = nullptr;
    ggml_tensor* up_w = nullptr;
    ggml_tensor* up_b = nullptr;
    ggml_tensor* down_w = nullptr;
    ggml_tensor* down_b = nullptr;
};

ggml_tensor* build_layernorm(
    ggml_context* ctx, ggml_tensor* cur, ggml_tensor* weight, ggml_tensor* bias, float eps) {
    cur = ggml_norm(ctx, cur, eps);
    if (weight) {
        cur = ggml_mul(ctx, cur, weight);
    }
    if (bias) {
        cur = ggml_add(ctx, cur, bias);
    }
    return cur;
}

// See vision.cpp: zero-extends an activation when the quantizer K-padded the
// matching weight.
ggml_tensor* pad_x_to_w(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w) {
    int64_t const k_w = w->ne[0];
    int64_t const k_x = x->ne[0];
    if (k_w == k_x) {
        return x;
    }
    GGML_ASSERT(k_w > k_x && "K-padded W must have larger ne[0] than activation");
    return ggml_pad(ctx, ggml_cont(ctx, x), (int)(k_w - k_x), 0, 0, 0);
}

constexpr int FA_TC_ALIGN = 16;

int fa_padded_d(int d_head) {
    static bool const no_pad = visp::env_flag("SIGLIP2_DISABLE_FA_PAD");
    if (no_pad) {
        return d_head;
    }
    return (d_head + FA_TC_ALIGN - 1) & ~(FA_TC_ALIGN - 1);
}

// Encoder block with an optional key-side padding mask. fa_mask_f16 is the F16
// cast of attn_mask, built once by the caller and shared across layers.
ggml_tensor* build_block(
    ggml_context* ctx,
    Block const& layer,
    ggml_tensor* inp,
    ggml_tensor* attn_mask,   // (n_pos_k, n_pos_q) F32 0/-inf, may be null
    ggml_tensor* fa_mask_f16, // same mask F16-cast for the FA path, may be null
    int n_pos,
    int d_head,
    int n_head,
    float ln_eps,
    float kq_scale,
    bool use_fa) {
    ggml_tensor* residual = inp;
    ggml_tensor* cur = build_layernorm(ctx, inp, layer.ln1_w, layer.ln1_b, ln_eps);

    int const H = d_head * n_head;
    size_t const es = sizeof(float);
    ggml_tensor* qkv =
        ggml_add(ctx, ggml_mul_mat(ctx, layer.qkv_w, pad_x_to_w(ctx, cur, layer.qkv_w)), layer.qkv_b);
    // Q/K/V are strided views: they interleave along the output axis, so a
    // reshape would not describe them.
    ggml_tensor* Q = ggml_view_3d(ctx, qkv, d_head, n_head, n_pos, d_head * es, 3 * H * es, 0);
    ggml_tensor* K = ggml_view_3d(ctx, qkv, d_head, n_head, n_pos, d_head * es, 3 * H * es, H * es);
    ggml_tensor* V =
        ggml_view_3d(ctx, qkv, d_head, n_head, n_pos, d_head * es, 3 * H * es, 2 * H * es);

    ggml_tensor* KQV;
    if (use_fa) {
        int const pad = fa_padded_d(d_head) - d_head;
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);
        if (pad > 0) {
            Q = ggml_pad(ctx, ggml_cont(ctx, Q), pad, 0, 0, 0);
            K = ggml_pad(ctx, ggml_cont(ctx, K), pad, 0, 0, 0);
            V = ggml_pad(ctx, ggml_cont(ctx, V), pad, 0, 0, 0);
        }
        ggml_tensor* K_f16 = ggml_cast(ctx, K, GGML_TYPE_F16);
        ggml_tensor* V_f16 = ggml_cast(ctx, V, GGML_TYPE_F16);
        KQV = ggml_flash_attn_ext(ctx, Q, K_f16, V_f16, fa_mask_f16, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(KQV, GGML_PREC_F32);
        if (pad > 0) {
            KQV = ggml_view_3d(ctx, KQV, d_head, n_head, n_pos, KQV->nb[1], KQV->nb[2], 0);
            KQV = ggml_cont(ctx, KQV);
        }
        KQV = ggml_reshape_2d(ctx, KQV, d_head * n_head, n_pos);
    } else {
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));
        ggml_tensor* KQ = ggml_mul_mat(ctx, K, Q);
        KQ = ggml_soft_max_ext(ctx, KQ, attn_mask, kq_scale, 0.0f);
        KQV = ggml_mul_mat(ctx, V, KQ);
        KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
        KQV = ggml_cont_2d(ctx, KQV, d_head * n_head, n_pos);
    }

    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.o_w, pad_x_to_w(ctx, KQV, layer.o_w)), layer.o_b);
    cur = ggml_add(ctx, cur, residual);
    residual = cur;

    cur = build_layernorm(ctx, cur, layer.ln2_w, layer.ln2_b, ln_eps);
    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.up_w, pad_x_to_w(ctx, cur, layer.up_w)), layer.up_b);
    cur = ggml_gelu(ctx, cur);
    cur = ggml_add(
        ctx, ggml_mul_mat(ctx, layer.down_w, pad_x_to_w(ctx, cur, layer.down_w)), layer.down_b);

    return ggml_add(ctx, cur, residual);
}

// Batched variant: activations are (H, n_pos, n_batch) and QKV/FA inputs 4D.
// For n_batch=1 the byte layout is identical to build_block's.
ggml_tensor* build_block_batched(
    ggml_context* ctx,
    Block const& layer,
    ggml_tensor* inp,
    int n_pos,
    int n_batch,
    int d_head,
    int n_head,
    float ln_eps,
    float kq_scale,
    bool use_fa) {
    int const H = d_head * n_head;
    size_t const es = sizeof(float);

    ggml_tensor* residual = inp;
    ggml_tensor* cur = build_layernorm(ctx, inp, layer.ln1_w, layer.ln1_b, ln_eps);

    ggml_tensor* qkv =
        ggml_add(ctx, ggml_mul_mat(ctx, layer.qkv_w, pad_x_to_w(ctx, cur, layer.qkv_w)), layer.qkv_b);
    ggml_tensor* Q = ggml_view_4d(
        ctx, qkv, d_head, n_head, n_pos, n_batch, d_head * es, 3 * H * es,
        3 * H * (size_t)n_pos * es, 0);
    ggml_tensor* K = ggml_view_4d(
        ctx, qkv, d_head, n_head, n_pos, n_batch, d_head * es, 3 * H * es,
        3 * H * (size_t)n_pos * es, H * es);
    ggml_tensor* V = ggml_view_4d(
        ctx, qkv, d_head, n_head, n_pos, n_batch, d_head * es, 3 * H * es,
        3 * H * (size_t)n_pos * es, 2 * H * es);

    ggml_tensor* KQV;
    if (use_fa) {
        int const pad = fa_padded_d(d_head) - d_head;
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);
        if (pad > 0) {
            Q = ggml_pad(ctx, ggml_cont(ctx, Q), pad, 0, 0, 0);
            K = ggml_pad(ctx, ggml_cont(ctx, K), pad, 0, 0, 0);
            V = ggml_pad(ctx, ggml_cont(ctx, V), pad, 0, 0, 0);
        }
        ggml_tensor* K_f16 = ggml_cast(ctx, K, GGML_TYPE_F16);
        ggml_tensor* V_f16 = ggml_cast(ctx, V, GGML_TYPE_F16);
        KQV = ggml_flash_attn_ext(ctx, Q, K_f16, V_f16, nullptr, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(KQV, GGML_PREC_F32);
        // FA result is (d_pad, n_head, n_pos, n_batch).
        if (pad > 0) {
            KQV = ggml_view_4d(
                ctx, KQV, d_head, n_head, n_pos, n_batch, KQV->nb[1], KQV->nb[2], KQV->nb[3], 0);
            KQV = ggml_cont(ctx, KQV);
        }
        KQV = ggml_reshape_3d(ctx, KQV, d_head * n_head, n_pos, n_batch);
    } else {
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));
        ggml_tensor* KQ = ggml_mul_mat(ctx, K, Q);
        KQ = ggml_soft_max_ext(ctx, KQ, nullptr, kq_scale, 0.0f);
        KQV = ggml_mul_mat(ctx, V, KQ);
        KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
        KQV = ggml_cont(ctx, KQV);
        KQV = ggml_reshape_3d(ctx, KQV, d_head * n_head, n_pos, n_batch);
    }

    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.o_w, pad_x_to_w(ctx, KQV, layer.o_w)), layer.o_b);
    cur = ggml_add(ctx, cur, residual);
    residual = cur;

    cur = build_layernorm(ctx, cur, layer.ln2_w, layer.ln2_b, ln_eps);
    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.up_w, pad_x_to_w(ctx, cur, layer.up_w)), layer.up_b);
    cur = ggml_gelu(ctx, cur);
    cur = ggml_add(
        ctx, ggml_mul_mat(ctx, layer.down_w, pad_x_to_w(ctx, cur, layer.down_w)), layer.down_b);

    return ggml_add(ctx, cur, residual);
}

// One cached graph per (n_pos, n_batch, has_mask), with a per-entry gallocr so
// tensor pointers stay stable across calls. See vision.cpp.
struct GraphCacheEntry {
    int n_pos = 0;
    int n_batch = 0;
    bool has_mask = false;

    std::vector<uint8_t> arena;
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t gallocr = nullptr;

    ggml_tensor* tok_inp = nullptr;
    ggml_tensor* pos_inp = nullptr;
    ggml_tensor* mask_inp = nullptr;
    ggml_tensor* pool_idx = nullptr;
    ggml_tensor* proj_out = nullptr;

    GraphCacheEntry() = default;
    GraphCacheEntry(GraphCacheEntry const&) = delete;
    GraphCacheEntry& operator=(GraphCacheEntry const&) = delete;
    GraphCacheEntry(GraphCacheEntry&& o) noexcept { *this = std::move(o); }
    GraphCacheEntry& operator=(GraphCacheEntry&& o) noexcept {
        if (this != &o) {
            if (gallocr) {
                ggml_gallocr_free(gallocr);
            }
            if (ctx) {
                ggml_free(ctx);
            }
            n_pos = o.n_pos;
            n_batch = o.n_batch;
            has_mask = o.has_mask;
            arena = std::move(o.arena);
            ctx = std::exchange(o.ctx, nullptr);
            gf = std::exchange(o.gf, nullptr);
            gallocr = std::exchange(o.gallocr, nullptr);
            tok_inp = std::exchange(o.tok_inp, nullptr);
            pos_inp = std::exchange(o.pos_inp, nullptr);
            mask_inp = std::exchange(o.mask_inp, nullptr);
            pool_idx = std::exchange(o.pool_idx, nullptr);
            proj_out = std::exchange(o.proj_out, nullptr);
        }
        return *this;
    }
    ~GraphCacheEntry() {
        if (gallocr) {
            ggml_gallocr_free(gallocr);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

} // namespace

struct TextEncoder::State {
    Model const* model = nullptr;
    TextConfig config;

    ggml_tensor* token_embd = nullptr;
    ggml_tensor* position_embd = nullptr;
    ggml_tensor* final_ln_w = nullptr;
    ggml_tensor* final_ln_b = nullptr;
    ggml_tensor* head_w = nullptr;
    ggml_tensor* head_b = nullptr;

    std::vector<Block> blocks;

    static constexpr size_t kCacheCap = 8;
    std::vector<GraphCacheEntry> graph_cache;
};

TextEncoder::TextEncoder() = default;

TextEncoder::~TextEncoder() {
    close();
}

void TextEncoder::close() {
    delete state_;
    state_ = nullptr;
}

bool TextEncoder::load(Model const& model) {
    close();
    if (!model.has_text) {
        error_msg_ = "model has no text tower";
        return false;
    }
    auto st = std::make_unique<State>();
    st->model = &model;
    st->config = model.text;

    try {
        st->token_embd = model.get("t.token_embd.weight");
        st->position_embd = model.get("t.position_embd.weight");
        st->final_ln_w = model.get("t.final_ln.weight");
        st->final_ln_b = model.get("t.final_ln.bias");
        st->head_w = model.get("t.head.weight");
        st->head_b = model.get("t.head.bias");

        st->blocks.resize(st->config.num_hidden_layers);
        for (int il = 0; il < st->config.num_hidden_layers; ++il) {
            Block& b = st->blocks[il];
            struct suffix_slot {
                ggml_tensor** dst;
                char const* suffix;
            };
            suffix_slot slots[] = {
                {&b.ln1_w, "ln1.weight"},       {&b.ln1_b, "ln1.bias"},
                {&b.qkv_w, "attn_qkv.weight"},  {&b.qkv_b, "attn_qkv.bias"},
                {&b.o_w, "attn_o.weight"},      {&b.o_b, "attn_o.bias"},
                {&b.ln2_w, "ln2.weight"},       {&b.ln2_b, "ln2.bias"},
                {&b.up_w, "ffn_up.weight"},     {&b.up_b, "ffn_up.bias"},
                {&b.down_w, "ffn_down.weight"}, {&b.down_b, "ffn_down.bias"},
            };
            for (auto& slot : slots) {
                *slot.dst = model.get(blk_name(il, slot.suffix).c_str());
            }
        }
    } catch (std::exception const& e) {
        error_msg_ = e.what();
        return false;
    }

    state_ = st.release();
    return true;
}

bool TextEncoder::encode_batch(
    int32_t const* token_ids,
    int n_tokens,
    int n_batch,
    int32_t const* attention_mask,
    std::vector<float>& out_embeddings) {
    if (!state_) {
        error_msg_ = "TextEncoder not loaded";
        return false;
    }
    if (n_batch <= 0) {
        error_msg_ = "n_batch must be > 0";
        return false;
    }
    TextConfig const& config = state_->config;
    if (n_tokens <= 0 || n_tokens > config.max_position_embeddings) {
        char buf[128];
        snprintf(
            buf, sizeof(buf), "n_tokens must be in (0, %d], got %d", config.max_position_embeddings,
            n_tokens);
        error_msg_ = buf;
        return false;
    }
    // A batched mask would need a (n_pos, n_pos, 1, n_batch) build; HF's path
    // passes no mask, so the batched graph doesn't carry one.
    bool const single = n_batch == 1;
    if (!single && attention_mask != nullptr) {
        error_msg_ = "encode_batch with attention_mask requires n_batch == 1";
        return false;
    }

    int const H = config.hidden_size;
    int const n_head = config.num_attention_heads;
    int const d_head = H / n_head;
    int const n_pos = n_tokens;
    int const proj = config.projection_size;
    float const kq_scale = 1.0f / std::sqrt((float)d_head);
    bool const has_mask = single && attention_mask != nullptr;

    static bool const use_fa = !visp::env_flag("SIGLIP2_DISABLE_FA");

    GraphCacheEntry* pe = nullptr;
    for (auto& ce : state_->graph_cache) {
        if (ce.n_pos == n_pos && ce.n_batch == n_batch && ce.has_mask == has_mask) {
            pe = &ce;
            break;
        }
    }
    bool const was_miss = pe == nullptr;
    if (was_miss) {
        if (state_->graph_cache.size() >= State::kCacheCap) {
            state_->graph_cache.erase(state_->graph_cache.begin());
        }
        state_->graph_cache.emplace_back();
        GraphCacheEntry& e = state_->graph_cache.back();
        e.n_pos = n_pos;
        e.n_batch = n_batch;
        e.has_mask = has_mask;

        int const max_nodes = 2048;
        e.arena.assign(
            ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(max_nodes, false), 0);
        ggml_init_params gp = {e.arena.size(), e.arena.data(), true};
        e.ctx = ggml_init(gp);
        if (!e.ctx) {
            state_->graph_cache.pop_back();
            error_msg_ = "ggml_init for graph context failed";
            return false;
        }
        e.gf = ggml_new_graph_custom(e.ctx, max_nodes, false);

        // ggml_get_rows asserts source->ne[2] == idx->ne[1], which doesn't fit
        // a shared vocab/position table, so indices stay 1D and the result is
        // reshaped afterwards.
        int64_t const n_idx = (int64_t)n_pos * n_batch;
        e.tok_inp = ggml_new_tensor_1d(e.ctx, GGML_TYPE_I32, n_idx);
        ggml_set_name(e.tok_inp, "token_ids");
        ggml_set_input(e.tok_inp);

        e.pos_inp = ggml_new_tensor_1d(e.ctx, GGML_TYPE_I32, n_idx);
        ggml_set_name(e.pos_inp, "position_ids");
        ggml_set_input(e.pos_inp);

        ggml_tensor* fa_mask_f16 = nullptr;
        if (has_mask) {
            e.mask_inp = ggml_new_tensor_2d(e.ctx, GGML_TYPE_F32, n_pos, n_pos);
            ggml_set_name(e.mask_inp, "attn_mask");
            ggml_set_input(e.mask_inp);
            if (use_fa) {
                fa_mask_f16 = ggml_cast(e.ctx, e.mask_inp, GGML_TYPE_F16);
            }
        }

        ggml_tensor* x = ggml_get_rows(e.ctx, state_->token_embd, e.tok_inp);
        // A K-padded token embedding table yields (K_padded, n_idx); slice back
        // to H before adding the position embedding.
        if (x->ne[0] != H) {
            GGML_ASSERT(x->ne[0] > H);
            x = ggml_cont(e.ctx, ggml_view_2d(e.ctx, x, H, x->ne[1], x->nb[1], 0));
        }
        ggml_tensor* p = ggml_get_rows(e.ctx, state_->position_embd, e.pos_inp);
        if (!single) {
            x = ggml_reshape_3d(e.ctx, x, H, n_pos, n_batch);
            p = ggml_reshape_3d(e.ctx, p, H, n_pos, n_batch);
        }
        x = ggml_add(e.ctx, x, p);

        for (int il = 0; il < config.num_hidden_layers; ++il) {
            if (single) {
                x = build_block(
                    e.ctx, state_->blocks[il], x, e.mask_inp, fa_mask_f16, n_pos, d_head, n_head,
                    config.layer_norm_eps, kq_scale, use_fa);
            } else {
                x = build_block_batched(
                    e.ctx, state_->blocks[il], x, n_pos, n_batch, d_head, n_head,
                    config.layer_norm_eps, kq_scale, use_fa);
            }
        }

        x = build_layernorm(
            e.ctx, x, state_->final_ln_w, state_->final_ln_b, config.layer_norm_eps);

        if (single) {
            e.pool_idx = ggml_new_tensor_1d(e.ctx, GGML_TYPE_I32, 1);
        } else {
            e.pool_idx = ggml_new_tensor_2d(e.ctx, GGML_TYPE_I32, 1, n_batch);
        }
        ggml_set_name(e.pool_idx, "pool_idx");
        ggml_set_input(e.pool_idx);
        ggml_tensor* pooled = ggml_get_rows(e.ctx, ggml_cont(e.ctx, x), e.pool_idx);
        if (!single) {
            pooled = ggml_reshape_2d(e.ctx, pooled, H, n_batch);
        }

        e.proj_out = ggml_add(
            e.ctx, ggml_mul_mat(e.ctx, state_->head_w, pad_x_to_w(e.ctx, pooled, state_->head_w)),
            state_->head_b);
        ggml_set_name(e.proj_out, "text_embed");
        ggml_set_output(e.proj_out);
        ggml_build_forward_expand(e.gf, e.proj_out);

        e.gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(state_->model->backend));
        if (!e.gallocr || !ggml_gallocr_reserve(e.gallocr, e.gf)) {
            state_->graph_cache.pop_back();
            error_msg_ = "ggml_gallocr_reserve failed";
            return false;
        }
        pe = &e;
    }
    GraphCacheEntry& ce = *pe;

    if (was_miss && !ggml_gallocr_alloc_graph(ce.gallocr, ce.gf)) {
        error_msg_ = "ggml_gallocr_alloc_graph failed";
        return false;
    }

    size_t const n_idx = (size_t)n_pos * (size_t)n_batch;
    ggml_backend_tensor_set(ce.tok_inp, token_ids, 0, sizeof(int32_t) * n_idx);
    {
        std::vector<int32_t> pos(n_idx);
        for (int b = 0; b < n_batch; ++b) {
            for (int i = 0; i < n_pos; ++i) {
                pos[(size_t)b * n_pos + i] = i;
            }
        }
        ggml_backend_tensor_set(ce.pos_inp, pos.data(), 0, sizeof(int32_t) * n_idx);
    }
    if (ce.mask_inp) {
        std::vector<float> mask((size_t)n_pos * n_pos, 0.0f);
        float const neg_inf = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < n_pos; ++k) {
            if (attention_mask[k] == 0) {
                for (int q = 0; q < n_pos; ++q) {
                    mask[(size_t)q * n_pos + k] = neg_inf;
                }
            }
        }
        ggml_backend_tensor_set(ce.mask_inp, mask.data(), 0, sizeof(float) * mask.size());
    }
    {
        std::vector<int32_t> idx(n_batch, (int32_t)(n_pos - 1));
        ggml_backend_tensor_set(ce.pool_idx, idx.data(), 0, sizeof(int32_t) * (size_t)n_batch);
    }

    ggml_status st = ggml_backend_graph_compute(state_->model->backend, ce.gf);
    if (st != GGML_STATUS_SUCCESS) {
        error_msg_ = "text graph compute failed status=" + std::to_string((int)st);
        return false;
    }

    out_embeddings.assign((size_t)proj * (size_t)n_batch, 0.0f);
    ggml_backend_tensor_get(
        ce.proj_out, out_embeddings.data(), 0, sizeof(float) * (size_t)proj * (size_t)n_batch);
    return true;
}

} // namespace siglip2
