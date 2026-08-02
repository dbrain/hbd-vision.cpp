#include "siglip2/siglip2.h"
#include "util/env.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace siglip2 {

namespace {

std::string blk_name(int il, char const* suffix) {
    char buf[64];
    snprintf(buf, sizeof(buf), "v.blk.%d.%s", il, suffix);
    return buf;
}

struct Block {
    ggml_tensor* ln1_w = nullptr;
    ggml_tensor* ln1_b = nullptr;
    // Fused Q/K/V: weight is (H, 3*H), bias is (3*H,). Q/K/V become strided
    // views afterwards, replacing three projections with one wider mul_mat.
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

// Siglip2MultiheadAttentionPoolingHead. HF packs in_proj_{weight,bias}; the
// converter splits them so this uses the same attention shape as a block.
struct ProbeHead {
    ggml_tensor* probe = nullptr;
    ggml_tensor* q_w = nullptr;
    ggml_tensor* q_b = nullptr;
    ggml_tensor* k_w = nullptr;
    ggml_tensor* k_b = nullptr;
    ggml_tensor* v_w = nullptr;
    ggml_tensor* v_b = nullptr;
    ggml_tensor* o_w = nullptr;
    ggml_tensor* o_b = nullptr;
    ggml_tensor* ln_w = nullptr;
    ggml_tensor* ln_b = nullptr;
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

// Q4_K/Q4_0 K-padding: the quantizer zero-pads a weight's innermost dim up to
// the quant block size, so the matching activation has to be zero-extended too.
// No-op for F16/Q8_0 weights.
ggml_tensor* pad_x_to_w(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w) {
    int64_t const k_w = w->ne[0];
    int64_t const k_x = x->ne[0];
    if (k_w == k_x) {
        return x;
    }
    GGML_ASSERT(k_w > k_x && "K-padded W must have larger ne[0] than activation");
    return ggml_pad(ctx, ggml_cont(ctx, x), (int)(k_w - k_x), 0, 0, 0);
}

bool fa_pad_disabled_env() {
    static bool const v = visp::env_flag("SIGLIP2_DISABLE_FA_PAD");
    return v;
}

// SigLIP2-so400m's d_head=72 doesn't divide 16, so the CUDA MMA/WMMA flash-attn
// tile templates reject it. Zero-padding the d-axis to the next multiple of 16
// is mathematically null (padded q·k = 0, padded v is sliced away) and routes
// to the tensor-core kernel. SIGLIP2_DISABLE_FA_PAD=1 skips the pad and lets
// ggml's dispatcher pick the tile kernel, which has a native case 72.
constexpr int FA_TC_ALIGN = 16;

int fa_padded_d(int d_head) {
    if (fa_pad_disabled_env()) {
        return d_head;
    }
    return (d_head + FA_TC_ALIGN - 1) & ~(FA_TC_ALIGN - 1);
}

ggml_tensor* fa_attn_pad_slice(
    ggml_context* ctx,
    ggml_tensor* Q, // (d_head, ?, n_head) F32, post-permute view
    ggml_tensor* K,
    ggml_tensor* V,
    ggml_tensor* fa_mask_f16,
    int d_head,
    int n_head,
    int n_pos_q,
    float kq_scale) {
    int const d_pad = fa_padded_d(d_head);
    if (d_pad != d_head) {
        int const pad = d_pad - d_head;
        // The pad op needs an F32 contiguous source, so pad before the F16 cast.
        Q = ggml_pad(ctx, ggml_cont(ctx, Q), pad, 0, 0, 0);
        K = ggml_pad(ctx, ggml_cont(ctx, K), pad, 0, 0, 0);
        V = ggml_pad(ctx, ggml_cont(ctx, V), pad, 0, 0, 0);
    }
    ggml_tensor* K_f16 = ggml_cast(ctx, K, GGML_TYPE_F16);
    ggml_tensor* V_f16 = ggml_cast(ctx, V, GGML_TYPE_F16);
    ggml_tensor* KQV =
        ggml_flash_attn_ext(ctx, Q, K_f16, V_f16, fa_mask_f16, kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(KQV, GGML_PREC_F32);
    if (d_pad != d_head) {
        KQV = ggml_view_3d(ctx, KQV, d_head, n_head, n_pos_q, KQV->nb[1], KQV->nb[2], 0);
        KQV = ggml_cont(ctx, KQV);
    }
    return ggml_reshape_2d(ctx, KQV, d_head * n_head, n_pos_q);
}

// Pre-LN attention + pre-LN MLP, both with residuals.
ggml_tensor* build_block(
    ggml_context* ctx,
    Block const& layer,
    ggml_tensor* inp,
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
    ggml_tensor* Q = ggml_view_3d(ctx, qkv, d_head, n_head, n_pos, d_head * es, 3 * H * es, 0);
    ggml_tensor* K = ggml_view_3d(ctx, qkv, d_head, n_head, n_pos, d_head * es, 3 * H * es, H * es);
    ggml_tensor* V =
        ggml_view_3d(ctx, qkv, d_head, n_head, n_pos, d_head * es, 3 * H * es, 2 * H * es);

    ggml_tensor* KQV;
    if (use_fa) {
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_permute(ctx, V, 0, 2, 1, 3); // NOT transposed under FA
        KQV = fa_attn_pad_slice(ctx, Q, K, V, nullptr, d_head, n_head, n_pos, kq_scale);
    } else {
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));
        ggml_tensor* KQ = ggml_mul_mat(ctx, K, Q);
        KQ = ggml_soft_max_ext(ctx, KQ, nullptr, kq_scale, 0.0f);
        KQV = ggml_mul_mat(ctx, V, KQ);
        KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
        KQV = ggml_cont_2d(ctx, KQV, d_head * n_head, n_pos);
    }

    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.o_w, pad_x_to_w(ctx, KQV, layer.o_w)), layer.o_b);
    cur = ggml_add(ctx, cur, residual);
    residual = cur;

    cur = build_layernorm(ctx, cur, layer.ln2_w, layer.ln2_b, ln_eps);
    cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.up_w, pad_x_to_w(ctx, cur, layer.up_w)), layer.up_b);
    cur = ggml_gelu(ctx, cur); // gelu_pytorch_tanh
    cur = ggml_add(
        ctx, ggml_mul_mat(ctx, layer.down_w, pad_x_to_w(ctx, cur, layer.down_w)), layer.down_b);

    return ggml_add(ctx, cur, residual);
}

// Cross-attention of a learnable probe (n_pos_q=1) against the last hidden
// state, then layernorm + MLP residual. Returns the pooled (H, 1) vector.
ggml_tensor* build_probe_head(
    ggml_context* ctx,
    ProbeHead const& head,
    ggml_tensor* last_hidden,
    int n_pos,
    int H,
    int d_head,
    int n_head,
    float ln_eps,
    float kq_scale,
    bool use_fa) {
    ggml_tensor* probe_in = ggml_reshape_2d(ctx, head.probe, H, 1);

    ggml_tensor* Q =
        ggml_add(ctx, ggml_mul_mat(ctx, head.q_w, pad_x_to_w(ctx, probe_in, head.q_w)), head.q_b);
    ggml_tensor* K = ggml_add(
        ctx, ggml_mul_mat(ctx, head.k_w, pad_x_to_w(ctx, last_hidden, head.k_w)), head.k_b);
    ggml_tensor* V = ggml_add(
        ctx, ggml_mul_mat(ctx, head.v_w, pad_x_to_w(ctx, last_hidden, head.v_w)), head.v_b);

    Q = ggml_reshape_3d(ctx, Q, d_head, n_head, 1);
    K = ggml_reshape_3d(ctx, K, d_head, n_head, n_pos);
    V = ggml_reshape_3d(ctx, V, d_head, n_head, n_pos);

    ggml_tensor* KQV;
    if (use_fa) {
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);
        KQV = fa_attn_pad_slice(ctx, Q, K, V, nullptr, d_head, n_head, 1, kq_scale);
    } else {
        Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx, K, 0, 2, 1, 3);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));
        ggml_tensor* KQ = ggml_mul_mat(ctx, K, Q);
        KQ = ggml_soft_max_ext(ctx, KQ, nullptr, kq_scale, 0.0f);
        KQV = ggml_mul_mat(ctx, V, KQ);
        KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
        KQV = ggml_cont_2d(ctx, KQV, d_head * n_head, 1);
    }

    ggml_tensor* attn =
        ggml_add(ctx, ggml_mul_mat(ctx, head.o_w, pad_x_to_w(ctx, KQV, head.o_w)), head.o_b);

    // HF: no layernorm before the MHA; output = attn + MLP(LN(attn)).
    ggml_tensor* normed = build_layernorm(ctx, attn, head.ln_w, head.ln_b, ln_eps);
    normed = ggml_add(
        ctx, ggml_mul_mat(ctx, head.up_w, pad_x_to_w(ctx, normed, head.up_w)), head.up_b);
    normed = ggml_gelu(ctx, normed);
    normed = ggml_add(
        ctx, ggml_mul_mat(ctx, head.down_w, pad_x_to_w(ctx, normed, head.down_w)), head.down_b);

    return ggml_add(ctx, attn, normed);
}

// One cached graph per (n_patches_h, n_patches_w, pooling). Bypasses
// ggml_backend_sched and uses a per-entry gallocr so tensor pointers stay
// stable across calls, which is what CUDA-graph capture needs to kick in.
struct GraphCacheEntry {
    int n_patches_h = 0;
    int n_patches_w = 0;
    Pooling pooling = Pooling::MEAN;

    std::vector<uint8_t> arena;
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t gallocr = nullptr;

    ggml_tensor* inp = nullptr;
    ggml_tensor* pooled = nullptr;

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
            n_patches_h = o.n_patches_h;
            n_patches_w = o.n_patches_w;
            pooling = o.pooling;
            arena = std::move(o.arena);
            ctx = std::exchange(o.ctx, nullptr);
            gf = std::exchange(o.gf, nullptr);
            gallocr = std::exchange(o.gallocr, nullptr);
            inp = std::exchange(o.inp, nullptr);
            pooled = std::exchange(o.pooled, nullptr);
        }
        return *this;
    }
    ~GraphCacheEntry() {
        // gallocr owns a buffer referencing tensors in ctx: free it first.
        if (gallocr) {
            ggml_gallocr_free(gallocr);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

} // namespace

struct VisionEncoder::State {
    Model const* model = nullptr;
    VisionConfig config;

    ggml_tensor* patch_embd_w = nullptr;
    ggml_tensor* patch_embd_b = nullptr;
    ggml_tensor* pos_embd = nullptr;
    ggml_tensor* post_ln_w = nullptr;
    ggml_tensor* post_ln_b = nullptr;

    std::vector<Block> blocks;
    ProbeHead head;

    // NaFlex resolves to a small finite set of grids per max_num_patches.
    static constexpr size_t kCacheCap = 4;
    std::vector<GraphCacheEntry> graph_cache;
};

VisionEncoder::VisionEncoder() = default;

VisionEncoder::~VisionEncoder() {
    close();
}

void VisionEncoder::close() {
    delete state_;
    state_ = nullptr;
}

bool VisionEncoder::load(Model const& model) {
    close();
    if (!model.has_vision) {
        error_msg_ = "model has no vision tower";
        return false;
    }
    auto st = std::make_unique<State>();
    st->model = &model;
    st->config = model.vision;

    try {
        st->patch_embd_w = model.get("v.patch_embd.weight");
        st->patch_embd_b = model.get("v.patch_embd.bias");
        st->pos_embd = model.get("v.position_embd.weight");
        st->post_ln_w = model.get("v.post_ln.weight");
        st->post_ln_b = model.get("v.post_ln.bias");

        ProbeHead& h = st->head;
        struct named_slot {
            ggml_tensor** dst;
            char const* name;
        };
        named_slot head_slots[] = {
            {&h.probe, "v.head.probe"},
            {&h.q_w, "v.head.attn_q.weight"},
            {&h.q_b, "v.head.attn_q.bias"},
            {&h.k_w, "v.head.attn_k.weight"},
            {&h.k_b, "v.head.attn_k.bias"},
            {&h.v_w, "v.head.attn_v.weight"},
            {&h.v_b, "v.head.attn_v.bias"},
            {&h.o_w, "v.head.attn_o.weight"},
            {&h.o_b, "v.head.attn_o.bias"},
            {&h.ln_w, "v.head.ln.weight"},
            {&h.ln_b, "v.head.ln.bias"},
            {&h.up_w, "v.head.ffn_up.weight"},
            {&h.up_b, "v.head.ffn_up.bias"},
            {&h.down_w, "v.head.ffn_down.weight"},
            {&h.down_b, "v.head.ffn_down.bias"},
        };
        for (auto& slot : head_slots) {
            *slot.dst = model.get(slot.name);
        }

        st->blocks.resize(st->config.num_hidden_layers);
        for (int il = 0; il < st->config.num_hidden_layers; ++il) {
            Block& b = st->blocks[il];
            struct suffix_slot {
                ggml_tensor** dst;
                char const* suffix;
            };
            suffix_slot slots[] = {
                {&b.ln1_w, "ln1.weight"},         {&b.ln1_b, "ln1.bias"},
                {&b.qkv_w, "attn_qkv.weight"},    {&b.qkv_b, "attn_qkv.bias"},
                {&b.o_w, "attn_o.weight"},        {&b.o_b, "attn_o.bias"},
                {&b.ln2_w, "ln2.weight"},         {&b.ln2_b, "ln2.bias"},
                {&b.up_w, "ffn_up.weight"},       {&b.up_b, "ffn_up.bias"},
                {&b.down_w, "ffn_down.weight"},   {&b.down_b, "ffn_down.bias"},
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

bool VisionEncoder::encode(
    float const* pixel_values,
    int n_patches_h,
    int n_patches_w,
    Pooling pooling,
    std::vector<float>& out_embedding) {
    if (!state_) {
        error_msg_ = "VisionEncoder not loaded";
        return false;
    }
    if (n_patches_h <= 0 || n_patches_w <= 0) {
        error_msg_ = "n_patches_h and n_patches_w must be positive";
        return false;
    }
    VisionConfig const& config = state_->config;
    int const n_per_side = (int)std::round(std::sqrt((double)config.num_patches));
    if (n_per_side * n_per_side != config.num_patches) {
        error_msg_ = "config.num_patches must be a perfect square (native grid)";
        return false;
    }

    int const H = config.hidden_size;
    int const n_head = config.num_attention_heads;
    int const d_head = H / n_head;
    int const n_pos = n_patches_h * n_patches_w;
    int const feat = config.num_channels * config.patch_size * config.patch_size;
    float const kq_scale = 1.0f / std::sqrt((float)d_head);

    static bool const use_fa = !visp::env_flag("SIGLIP2_DISABLE_FA");

    GraphCacheEntry* pe = nullptr;
    for (auto& ce : state_->graph_cache) {
        if (ce.n_patches_h == n_patches_h && ce.n_patches_w == n_patches_w &&
            ce.pooling == pooling) {
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
        e.n_patches_h = n_patches_h;
        e.n_patches_w = n_patches_w;
        e.pooling = pooling;

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

        e.inp = ggml_new_tensor_2d(e.ctx, GGML_TYPE_F32, feat, n_pos);
        ggml_set_name(e.inp, "pixel_values");
        ggml_set_input(e.inp);

        ggml_tensor* x =
            ggml_mul_mat(e.ctx, state_->patch_embd_w, pad_x_to_w(e.ctx, e.inp, state_->patch_embd_w));
        x = ggml_add(e.ctx, x, state_->patch_embd_b);

        ggml_tensor* pos_embd = state_->pos_embd;
        if (n_patches_h != n_per_side || n_patches_w != n_per_side) {
            pos_embd = ggml_reshape_3d(e.ctx, pos_embd, H, n_per_side, n_per_side);
            pos_embd = ggml_permute(e.ctx, pos_embd, 2, 0, 1, 3);
            pos_embd = ggml_interpolate(
                e.ctx,
                pos_embd,
                n_patches_w,
                n_patches_h,
                H,
                1,
                GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS);
            pos_embd = ggml_permute(e.ctx, pos_embd, 1, 2, 0, 3);
            pos_embd = ggml_cont_2d(e.ctx, pos_embd, H, n_pos);
        }
        x = ggml_add(e.ctx, x, pos_embd);

        for (int il = 0; il < config.num_hidden_layers; ++il) {
            x = build_block(
                e.ctx, state_->blocks[il], x, n_pos, d_head, n_head, config.layer_norm_eps,
                kq_scale, use_fa);
        }

        x = build_layernorm(e.ctx, x, state_->post_ln_w, state_->post_ln_b, config.layer_norm_eps);

        if (pooling == Pooling::MEAN) {
            e.pooled = ggml_cont(e.ctx, ggml_transpose(e.ctx, x));
            e.pooled = ggml_mean(e.ctx, e.pooled);
        } else {
            e.pooled = build_probe_head(
                e.ctx, state_->head, x, n_pos, H, d_head, n_head, config.layer_norm_eps, kq_scale,
                use_fa);
        }
        ggml_set_name(e.pooled, "pooled");
        ggml_set_output(e.pooled);
        ggml_build_forward_expand(e.gf, e.pooled);

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

    ggml_backend_tensor_set(ce.inp, pixel_values, 0, sizeof(float) * (size_t)feat * n_pos);

    ggml_status st = ggml_backend_graph_compute(state_->model->backend, ce.gf);
    if (st != GGML_STATUS_SUCCESS) {
        error_msg_ = "vision graph compute failed status=" + std::to_string((int)st);
        return false;
    }

    out_embedding.assign((size_t)H, 0.0f);
    ggml_backend_tensor_get(ce.pooled, out_embedding.data(), 0, sizeof(float) * (size_t)H);
    return true;
}

} // namespace siglip2
