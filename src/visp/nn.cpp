#include "nn.h"
#include "util/string.h"
#include <cstdlib>

namespace visp {

// VRAM lever: a k>1 conv routed through ggml_conv_2d materialises an im2col matrix of
// [IC*KH*KW, OH*OW] F32 elements. For the encoder's full-resolution high-channel convs
// this single buffer is up to ~1.15 GiB and dominates GPU peak VRAM. ggml_conv_2d_direct
// streams the conv with no im2col buffer (slower per-conv, tiny VRAM). This returns the
// im2col element count so the call site can pick direct vs im2col by a tunable threshold.
// Threshold via env VISP_IM2COL_MAX (in MiB of the F32 im2col matrix); 0/unset = always
// im2col (legacy behaviour). Latency is not a concern for the matting deploy, VRAM is.
static int64_t im2col_max_elems() {
    static int64_t cached = -1;
    if (cached < 0) {
        // Default 128 MiB caps GPU peak VRAM at ~2524 MiB (vs ~3596 unbounded) at full
        // 1024 quality and ~free latency (~712 vs ~691 ms). The matting deploy is a
        // worker-isolated lazy-evict GPU service that must fit the free-VRAM window when
        // the heavy GPU services are resident, so low peak is the priority. Set
        // VISP_IM2COL_MAX=0 to disable (always im2col, lowest latency, highest VRAM),
        // or a smaller MiB value to push more convs to the streaming direct kernel.
        char const* e = getenv("VISP_IM2COL_MAX"); // MiB
        double mib = e ? atof(e) : 128.0;
        cached = mib > 0 ? (int64_t)(mib * 1024.0 * 1024.0 / 4.0) : 0; // F32 elems
    }
    return cached;
}

// im2col matrix element count for a conv: IC*KH*KW (per output px) * OH*OW (output px).
static int64_t im2col_elems(int64_t ic, int64_t kh, int64_t kw, int64_t oh, int64_t ow) {
    return ic * kh * kw * oh * ow;
}

static int64_t conv_out_dim(int64_t in, int64_t k, int stride, int pad) {
    return (in + 2 * pad - k) / stride + 1;
}

// On the CUDA backend, loaded weights (bias/scale) live at the device "preferred
// float type" (F16), while conv/linear activations compute in F32. ggml's CUDA
// binary-broadcast ops (add/mul) assert matching element sizes and crash on a
// mixed F32+F16 operand (binbcast.cu: nb10 % sizeof(src1_t) == 0). Cast the
// weight operand up to the activation type before the op. The cast is lossless
// (F16->F32) and is skipped entirely on the CPU path (where both operands are
// already F32), so behaviour there is unchanged.
//
// NOTE: ggml_*_inplace cannot be used once the operand may need a cast, because
// the cast produces a fresh tensor and the in-place result must alias `x` (the
// activation), so we use the non-inplace ops here.
static tensor cast_like(model_ref m, tensor w, tensor ref) {
    return w->type == ref->type ? w : tensor(ggml_cast(m, w, ref->type));
}

tensor linear(model_ref m, tensor x) {
    x = ggml_mul_mat(m, m.weights("weight"), x);
    if (tensor bias = m.find("bias")) {
        x = ggml_add(m, x, cast_like(m, bias, x));
    }
    return x;
}

tensor layer_norm(model_ref m, tensor x, float eps) {
    x = ggml_norm(m, x, eps);
    x = ggml_mul(m, x, cast_like(m, m.weights("weight"), x));
    x = ggml_add(m, x, cast_like(m, m.weights("bias"), x));
    return named(m, x);
}

tensor permute_cwhn_to_whcn(model_ref m, tensor x) {
    return ggml_permute(m, x, 2, 0, 1, 3);
}

tensor permute_whcn_to_cwhn(model_ref m, tensor x) {
    return ggml_permute(m, x, 1, 2, 0, 3);
}

std::array<int64_t, 4> nelements_whcn(model_ref const& m, tensor t) {
    auto ne = nelements(t);
    return (m.flags & model_build_flag::cwhn) ? std::array{ne[1], ne[2], ne[0], ne[3]} : ne;
}

tensor cwhn_to_contiguous_2d(model_ref m, tensor x) {
    if (m.flags & model_build_flag::cwhn) {
        return x; // preferred 2D layout is CWHN too
    }
    return ggml_cont(m, permute_cwhn_to_whcn(m, x));
}

tensor whcn_to_contiguous_2d(model_ref m, tensor x) {
    if (m.flags & model_build_flag::cwhn) {
        return ggml_cont(m, permute_whcn_to_cwhn(m, x));
    }
    return x;
}

tensor contiguous_2d_to_cwhn(model_ref m, tensor x) {
    if (m.flags & model_build_flag::cwhn) {
        return x; // x is already CWHN
    }
    return ggml_cont(m, permute_whcn_to_cwhn(m, x));
}

tensor contiguous_2d_to_whcn(model_ref m, tensor x) {
    if (m.flags & model_build_flag::cwhn) {
        return ggml_cont(m, permute_cwhn_to_whcn(m, x));
    }
    return x;
}

tensor add_bias_2d(model_ref m, tensor x) {
    if (tensor bias = m.find("bias")) {
        if (!(m.flags & model_build_flag::cwhn)) {
            bias = ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1);
        }
        x = ggml_add(m, x, cast_like(m, bias, x));
    }
    return x;
}

tensor conv_2d(model_ref m, tensor x, int stride, int pad) {
    tensor weight = m.weights("weight");

    if (m.flags & model_build_flag::cwhn) {
        if (weight->ne[1] == 1 && weight->ne[2] == 1 && stride == 1) {
            auto [c, w, h, b] = nelements(x);
            weight = ggml_reshape_2d(m, weight, weight->ne[0], weight->ne[3]);
            x = ggml_reshape_2d(m, x, x->ne[0], w * h * b);
            x = ggml_mul_mat(m, weight, x);
            x = ggml_reshape_4d(m, x, weight->ne[1], w, h, b);

        } else if (m.flags & model_build_flag::conv_2d_direct_cwhn) {
            weight = permute_cwhn_to_whcn(m, weight);
            x = permute_cwhn_to_whcn(m, x);
            x = ggml_conv_2d_direct(m, weight, x, stride, stride, pad, pad, 1, 1);
            x = permute_whcn_to_cwhn(m, x);

        } else {
            // cwhn k>1 conv. weight is cwhn [IC, KW, KH, OC]; x is cwhn [C, W, H, B].
            // This is the ENCODER path (patch_embed sets cwhn) and holds the biggest
            // im2col buffers (full-res high-channel convs, up to ~1.15 GiB). When the
            // im2col matrix would exceed VISP_IM2COL_MAX, use the direct kernel (no
            // im2col buffer) to cap peak VRAM at a small latency cost.
            int64_t ic = weight->ne[0], kw = weight->ne[1], kh = weight->ne[2];
            int64_t oh = conv_out_dim(x->ne[2], kh, stride, pad);
            int64_t ow = conv_out_dim(x->ne[1], kw, stride, pad);
            int64_t mx = im2col_max_elems();
            weight = ggml_cont(m, permute_cwhn_to_whcn(m, weight));
            x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
            if (mx > 0 && im2col_elems(ic, kh, kw, oh, ow) > mx) {
                x = ggml_conv_2d_direct(m, weight, x, stride, stride, pad, pad, 1, 1);
            } else {
                x = ggml_conv_2d(m, weight, x, stride, stride, pad, pad, 1, 1);
            }
            x = ggml_cont(m, permute_whcn_to_cwhn(m, x));
        }
    } else { // WHCN layout
        if (weight->ne[0] == 1 && weight->ne[1] == 1 && stride == 1 && pad == 0) {
            // 1x1 conv == matmul over the channel dim. The whcn channel dim is
            // dim2 (W,H,C,B), so bring C innermost, GEMM, then restore. The
            // ggml_conv_2d_direct kernel is one-thread-per-output looping IC
            // with no reuse; routing 1x1 through mul_mat uses cuBLAS/MMQ.
            auto [w, h, c, b] = nelements(x);
            tensor w2 = ggml_reshape_2d(m, weight, weight->ne[2], weight->ne[3]); // [IC, OC]
            tensor xc = ggml_cont(m, permute_whcn_to_cwhn(m, x));                 // -> [C, W, H, B]
            xc = ggml_reshape_2d(m, xc, c, w * h * b);                            // [C, W*H*B]
            tensor y = ggml_mul_mat(m, w2, xc);                                   // [OC, W*H*B]
            y = ggml_reshape_4d(m, y, weight->ne[3], w, h, b);                    // [OC, W, H, B]
            x = ggml_cont(m, permute_cwhn_to_whcn(m, y));                         // -> [W, H, OC, B]
        } else {
            // k>1 conv (whcn = decoder). ggml_conv_2d does im2col + ggml_mul_mat
            // (cuBLAS/MMQ on CUDA). When the im2col matrix would exceed
            // VISP_IM2COL_MAX, fall back to the streaming direct kernel to cap
            // peak VRAM. (BiRefNet-only call site; shared CONV_2D kernel untouched.)
            // weight is whcn [KW, KH, IC, OC]; x is whcn [W, H, C, B].
            int64_t kw = weight->ne[0], kh = weight->ne[1], ic = weight->ne[2];
            int64_t ow = conv_out_dim(x->ne[0], kw, stride, pad);
            int64_t oh = conv_out_dim(x->ne[1], kh, stride, pad);
            int64_t mx = im2col_max_elems();
            if (mx > 0 && im2col_elems(ic, kh, kw, oh, ow) > mx) {
                x = ggml_conv_2d_direct(m, weight, x, stride, stride, pad, pad, 1, 1);
            } else {
                x = ggml_conv_2d(m, weight, x, stride, stride, pad, pad, 1, 1);
            }
        }
    }
    x = add_bias_2d(m, x);
    return x;
}

tensor conv_2d_depthwise(model_ref m, tensor x, int stride, int pad) {
    tensor weight = m.weights("weight");

    if (m.flags & model_build_flag::cwhn) {
        weight = ggml_permute(m, weight, 3, 2, 0, 1);
        x = permute_cwhn_to_whcn(m, x);
        x = ggml_conv_2d_dw_direct(m, weight, x, stride, stride, pad, pad, 1, 1);
        x = permute_whcn_to_cwhn(m, x);
    } else {
        x = ggml_conv_2d_dw_direct(m, weight, x, stride, stride, pad, pad, 1, 1);
    }
    x = add_bias_2d(m, x);
    return x;
}

tensor conv_transpose_2d(model_ref m, tensor x, int stride) {
    tensor weight = m.weights("weight");
    if (m.flags & model_build_flag::f16_conv_transpose) {
        // TODO: ggml_conv_transpose_2d_p0 expects fp16 weights (cpu backend)
        weight = ggml_cast(m, weight, GGML_TYPE_F16);
    }
    if (m.flags & model_build_flag::cwhn) {
        x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
    }
    x = ggml_conv_transpose_2d_p0(m, weight, x, stride);

    if (m.flags & model_build_flag::cwhn) {
        x = ggml_cont(m, permute_whcn_to_cwhn(m, x));
    }
    x = add_bias_2d(m, x);
    return x;
}

tensor conv_2d_deform(
    model_ref m, tensor x, tensor weight, tensor offset, tensor mask, int stride, int pad) {

    // CONV_2D_DEFORM has no CUDA kernel, so on the GPU backend the scheduler
    // offloads it to the CPU. The CPU deform kernel (ggml-cpu/ops.cpp) feeds the
    // weight straight into ggml_call_mul_mat() with a hardcoded GGML_TYPE_F32 and
    // has no F16 path. On the GPU path the model keeps its native F16 weights, so
    // those F16 bytes would be reinterpreted as F32 -> garbage GEMM -> the ASPP
    // decoder explodes to inf and the mask collapses. Cast the weight to F32 to
    // match what the kernel expects. No-op on the CPU backend (weights already F32)
    // and lossless (F16->F32). Mirrors conv_transpose_2d's f16 cast for its kernel.
    if (weight->type != GGML_TYPE_F32) {
        weight = ggml_cast(m, weight, GGML_TYPE_F32);
    }

    if (m.flags & model_build_flag::cwhn) {
        x = permute_cwhn_to_whcn(m, x);
        weight = permute_cwhn_to_whcn(m, weight);
        offset = permute_cwhn_to_whcn(m, offset);
        if (mask) {
            mask = permute_cwhn_to_whcn(m, mask);
        }
    }
    x = ggml_conv_2d_deform(m, weight, x, offset, mask, stride, stride, pad, pad);

    if (m.flags & model_build_flag::cwhn) {
        x = permute_whcn_to_cwhn(m, x);
    }
    return x;
}

tensor batch_norm_2d(model_ref m, tensor x) {
    // Batch norm is expected to be have been fused into mul+add. See convert.py
    ASSERT(m.find("running_mean") == nullptr, "Batch norm was not fused");
    ASSERT(m.find("running_var") == nullptr, "Batch norm was not fused");

    tensor weight = m.weights("weight");
    tensor bias = m.weights("bias");
    if (!(m.flags & model_build_flag::cwhn)) { // WHCN layout
        weight = ggml_reshape_4d(m, weight, 1, 1, weight->ne[0], 1);
        bias = ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1);
    }
    x = ggml_mul(m, x, cast_like(m, weight, x));
    x = ggml_add(m, x, cast_like(m, bias, x));
    return named(m, x);
}

tensor patch_embed(model_ref m, tensor x, int patch_size) {
    ASSERT(x->ne[1] % patch_size == 0 && x->ne[2] % patch_size == 0);
    char const* proj = m.find("proj.weight") ? "proj" : "projection";

    m.flags |= model_build_flag::cwhn;
    x = conv_2d(m[proj], x, patch_size);

    if (m.find("norm.weight")) {
        auto [c, w, h, b] = nelements(x);
        x = ggml_reshape_3d(m, x, c, w * h, b);
        x = layer_norm(m["norm"], x);
        x = ggml_reshape_4d(m, x, c, w, h, b);
    }
    return named(m, x);
}

attention_qkv split_qkv(model_ref m, tensor x, int n_heads, int split_dim) {
    auto [c, n, b, _] = nelements(x);

    tensor qkv = linear(m, x);
    switch (split_dim) {
        case 1:
            qkv = ggml_reshape_4d(m, qkv, c / n_heads, 3, n_heads * n, b);
            qkv = ggml_cont(m, ggml_permute(m, qkv, 0, 3, 1, 2));
            break;
        case 2:
            qkv = ggml_reshape_4d(m, qkv, c / n_heads, n_heads, 3, n * b);
            qkv = ggml_cont(m, ggml_permute(m, qkv, 0, 1, 3, 2));
            break;
        default: ASSERT(false, "Unsupported split_dim");
    }

    auto split = [&](tensor t, size_t index) mutable {
        t = slice(m, t, {}, {}, {}, index);
        t = ggml_reshape_4d(m, t, c / n_heads, n_heads, n, b);
        return t;
    };

    tensor q = split(qkv, 0);
    tensor k = split(qkv, 1);
    tensor v = split(qkv, 2);
    return {q, k, v};
}

tensor attention(
    model_ref m, tensor q, tensor k, tensor v, tensor mask, float scale, model_ref m_out) {

    q = ggml_permute(m, q, 0, 2, 1, 3);
    k = ggml_permute(m, k, 0, 2, 1, 3);

    tensor x = nullptr;
    if (m.flags & model_build_flag::flash_attention) {
        v = ggml_permute(m, v, 0, 2, 1, 3);

        // The CUDA flash-attn kernel has no head_dim=32 instance (Swin; crashes fattn.cu).
        // Zero-pad head_dim to 64: lossless (padded K dims are 0 -> no effect on Q.Kᵀ, so
        // `scale`=1/sqrt(orig_hd) is unchanged; padded V dims are 0 -> output tail is 0 and
        // sliced off). Combined with per-head-mask support in fattn (dbrain/ggml), this lets
        // the Swin encoder use FA, which avoids materialising the [n,n] per-window scores.
        int64_t head_dim = q->ne[0];
        bool pad_head = head_dim < 64;
        if (pad_head) {
            int64_t pd = 64 - head_dim;
            q = ggml_pad(m, ggml_cont(m, q), pd, 0, 0, 0);
            k = ggml_pad(m, ggml_cont(m, k), pd, 0, 0, 0);
            v = ggml_pad(m, ggml_cont(m, v), pd, 0, 0, 0);
        }

        k = ggml_cast(m, k, GGML_TYPE_F16);
        v = ggml_cast(m, v, GGML_TYPE_F16);
        if (mask && mask->type != GGML_TYPE_F16) {
            mask = ggml_cast(m, mask, GGML_TYPE_F16);
        }

        x = ggml_flash_attn_ext(m, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(x, GGML_PREC_F32);

        if (pad_head) { // slice padded head_dim back to the real size
            x = ggml_cont(m, ggml_view_4d(m, x, head_dim, x->ne[1], x->ne[2], x->ne[3],
                                          x->nb[1], x->nb[2], x->nb[3], 0));
        }

    } else {
        v = ggml_cont(m, ggml_permute(m, v, 1, 2, 0, 3));

        tensor attn = ggml_mul_mat(m, k, q);
        attn = ggml_soft_max_ext(m, attn, mask, scale, 0.0f);
        x = ggml_mul_mat(m, v, attn);

        x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    }

    // [head_dim, n_heads, n_patches, batch] -> [embed_dim, n_patches, batch]
    x = ggml_reshape_3d(m, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]);
    x = linear(m_out, x);

    return named(m, x);
}

} // namespace visp