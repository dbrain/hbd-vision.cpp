#include "visp/arch/birefnet.h"
#include "util/math.h"
#include "util/string.h"
#include "visp/arch/swin.h"
#include "visp/nn.h"
#include "visp/vision.h"

#include <ggml.h>

#include <cstdlib>

namespace visp {
namespace birefnet {

//
// Encoder
//

constexpr int32_t bilinear_align_corners = GGML_SCALE_MODE_BILINEAR |
    (int)GGML_SCALE_FLAG_ALIGN_CORNERS;

tensor upscale_to_whcn(model_ref m, tensor x, tensor target) {
    return interpolate(m, x, {target->ne[0], target->ne[1]}, bilinear_align_corners);
}

tensor upscale_to(model_ref m, tensor x, tensor target) {
    auto [target_width, target_height, c, n] = nelements_whcn(m, target);
    x = contiguous_2d_to_whcn(m, x);
    x = interpolate(m, x, {target_width, target_height}, bilinear_align_corners);
    x = whcn_to_contiguous_2d(m, x);
    return x;
}

tensor downscale_by_whcn(model_ref m, tensor x, int f) {
    return interpolate(m, x, {x->ne[0] / f, x->ne[1] / f}, bilinear_align_corners);
}

tensor downscale_by(model_ref m, tensor x, int f) {
    x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
    x = downscale_by_whcn(m, x, f);
    x = ggml_cont(m, permute_whcn_to_cwhn(m, x));
    return x;
}

swin_result encode_concat(model_ref m, swin_result& xs, swin_result& xs_low) {
    // TODO: implement cwhn upscale/interpolate which allows downscale & align_corners=True
    for (int i = 0; i < 4; ++i) {
        xs[i] = ggml_cont(m, permute_cwhn_to_whcn(m, xs[i]));
        xs_low[i] = permute_cwhn_to_whcn(m, xs_low[i]);
    }
    // clang-format off
    xs[0] = concat(m, {xs[0], upscale_to_whcn(m, xs_low[0], xs[0])}, 2);
    xs[1] = concat(m, {xs[1], upscale_to_whcn(m, xs_low[1], xs[1])}, 2);
    xs[2] = concat(m, {xs[2], upscale_to_whcn(m, xs_low[2], xs[2])}, 2);
    xs[3] = concat(m, {xs[3], upscale_to_whcn(m, xs_low[3], xs[3])}, 2);
    xs[3] = concat(m, {downscale_by_whcn(m, xs[0], 8),
                       downscale_by_whcn(m, xs[1], 4),
                       downscale_by_whcn(m, xs[2], 2),
                       xs[3]}, /*dim = */ 2);
    // clang-format on

    // whcn -> native
    for (int i = 0; i < 4; ++i) {
        xs[i] = whcn_to_contiguous_2d(m, xs[i]);
    }
    return xs;
}

swin_result encode(model_ref m, tensor x, swin_params const& p) {
    auto xs = swin_encode(m["bb"], x, p);
    auto x_low = downscale_by(m, x, 2);
    auto xs_low = swin_encode(m["bb"], x_low, p);
    encode_concat(m, xs, xs_low);
    return xs;
}

//
// Decoder
//

tensor conv_2d_batch_norm(model_ref m, tensor x, int stride = 1, int pad = 0) {
    return conv_2d(m, x, stride, pad); // batch_norm is fused into conv_2d at model conversion
}

tensor deformable_conv_2d(model_ref m, tensor x, int stride, int pad) {
    tensor offset = conv_2d(m["offset"], x, stride, pad);
    tensor modulator = conv_2d(m["modulator"], x, stride, pad);
    modulator = ggml_sigmoid_inplace(m, modulator);
    modulator = ggml_scale_inplace(m, modulator, 2.0f);

    x = conv_2d_deform(m, x, m.weights("conv.weight"), offset, modulator, stride, pad);
    return named(m, x);
}

tensor mean_2d(model_ref m, tensor x) {
    auto [w, h, c, n] = nelements_whcn(m, x);
    x = contiguous_2d_to_whcn(m, x);
    x = ggml_reshape_3d(m, x, w * h, c, n);
    x = ggml_mean(m, x);
    x = is_cwhn(m) ? ggml_reshape_4d(m, x, c, 1, 1, n) : ggml_reshape_4d(m, x, 1, 1, c, n);
    return x;
}

tensor global_avg_pool(model_ref m, tensor x) {
    x = mean_2d(m, x);
    x = conv_2d_batch_norm(m[1], x);
    x = ggml_relu_inplace(m, x);
    return named(m, x);
}

tensor aspp_module_deformable(model_ref m, tensor x, int padding) {
    x = deformable_conv_2d(m["conv"], x, 1, padding);
    x = batch_norm_2d(m["bn"], x);
    x = ggml_relu_inplace(m, x);
    return named(m, x);
}

// A 1x1 conv applied to a channel-wise concat of `parts` equals the sum of per-part
// 1x1 convs with the weight split along input channels:
//   conv1x1(concat(parts)) == sum_k conv1x1_k(parts[k])
// Computing it this way avoids ever materialising the (often large) concat tensor, and
// lets gallocr free each part right after its matmul instead of holding all of them plus
// the concat. whcn/GPU 1x1 only; lossless modulo FP add order. The caller guards layout.
static tensor conv_1x1_split_sum(model_ref m, model_ref conv, std::initializer_list<tensor> parts) {
    tensor cw = conv.weights("weight"); // whcn [1, 1, IC, OC]
    int64_t oc = cw->ne[3];
    tensor w2 = ggml_reshape_2d(m, cw, cw->ne[2], oc); // [IC, OC]
    tensor acc = nullptr;
    int64_t off = 0;
    for (tensor p : parts) {
        auto [w, h, c, b] = nelements(p);
        tensor ws = ggml_cont(m, ggml_view_2d(m, w2, c, oc, w2->nb[1], off * w2->nb[0]));
        tensor xc = ggml_cont(m, permute_whcn_to_cwhn(m, p)); // [C, W, H, B]
        xc = ggml_reshape_2d(m, xc, c, w * h * b);            // [C, W*H*B]
        tensor y = ggml_mul_mat(m, ws, xc);                  // [OC, W*H*B]
        y = ggml_reshape_4d(m, y, oc, w, h, b);              // [OC, W, H, B]
        y = ggml_cont(m, permute_cwhn_to_whcn(m, y));        // [W, H, OC, B]
        acc = acc ? tensor(ggml_add(m, acc, y)) : y;
        off += c;
    }
    if (tensor bias = conv.find("bias")) {
        bias = ggml_reshape_4d(m, bias, 1, 1, oc, 1);
        if (bias->type != acc->type) {
            bias = ggml_cast(m, bias, acc->type);
        }
        acc = ggml_add(m, acc, bias);
    }
    return acc;
}

tensor aspp_deformable(model_ref m, tensor x) {
    const int kernel_sizes[] = {1, 3, 7};
    const int channel_dim = is_cwhn(m) ? 0 : 2;

    tensor x1 = aspp_module_deformable(m["aspp1"], x);
    model_ref aspp_deforms = m["aspp_deforms"];
    tensor x_deforms[3];
    for (int i = 0; i < 3; ++i) {
        int padding = kernel_sizes[i] / 2;
        x_deforms[i] = aspp_module_deformable(aspp_deforms[i], x, padding);
    }
    tensor x5 = global_avg_pool(m["global_avg_pool"], x);
    auto [w1, h1, c, n] = nelements_whcn(m, x1);
    x5 = contiguous_2d_to_whcn(m, x5);
    x5 = interpolate(m, x5, {w1, h1}, bilinear_align_corners);
    x5 = whcn_to_contiguous_2d(m, x5);

    // conv1 is a 1x1 conv over the 5-way concat (e.g. [256,256,1280] = 320 MiB at the
    // highest decoder block). Split it so the concat is never materialised — this is the
    // graph's binding peak after the output-head fusion (bench/VRAM.md). whcn/GPU only.
    tensor cw1 = m["conv1"].weights("weight");
    if (is_whcn(m) && cw1->ne[0] == 1 && cw1->ne[1] == 1) {
        x = conv_1x1_split_sum(m, m["conv1"], {x1, x_deforms[0], x_deforms[1], x_deforms[2], x5});
    } else {
        x = concat(m, {x1, x_deforms[0], x_deforms[1], x_deforms[2], x5}, channel_dim);
        x = conv_2d_batch_norm(m["conv1"], x);
    }
    x = ggml_relu_inplace(m, x);
    return named(m, x);
}

tensor basic_decoder_block(model_ref m, tensor x) {
    x = conv_2d_batch_norm(m["conv_in"], x, 1, 1);
    x = ggml_relu_inplace(m, x);
    x = aspp_deformable(m["dec_att"], x);
    x = conv_2d_batch_norm(m["conv_out"], x, 1, 1);
    return named(m, x);
}

tensor simple_conv(model_ref m, tensor x) {
    x = conv_2d(m["conv1"], x, 1, 1);
    x = conv_2d(m["conv_out"], x, 1, 1);
    return named(m, x);
}

tensor image_to_patches(model_ref m, tensor x, int64_t out_w, int64_t out_h) {
    auto [w, h, c, b] = nelements(x);
    ASSERT(w % out_w == 0 && h % out_h == 0 && "Grid must divide image size");
    int64_t grid_w = w / out_w;
    int64_t grid_h = h / out_h;
    x = ggml_reshape_4d(m, x, out_w, grid_w, out_h, grid_h * c * b);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(m, x, out_w, out_h, grid_w * grid_h * c, b);
    return x;
}

tensor gdt_conv(model_ref m, tensor x) {
    x = conv_2d_batch_norm(m[0], x, 1, 1);
    x = ggml_relu_inplace(m, x);
    return x;
}

tensor decode(model_ref m, tensor x, swin_result const& features) {
    const int channel_dim = is_cwhn(m) ? 0 : 2;

    tensor x1 = features[0];
    tensor x2 = features[1];
    tensor x3 = features[2];
    tensor x4 = features[3];
    tensor x_whcn = ggml_cont(m, permute_cwhn_to_whcn(m, x));
    if (is_whcn(m)) {
        x = x_whcn;
    }
    {
        auto [w, h, c, n] = nelements_whcn(m, x4);
        tensor patches = image_to_patches(m, x_whcn, w, h);
        patches = whcn_to_contiguous_2d(m, patches);
        patches = simple_conv(m["ipt_blk5"], patches);
        x4 = ggml_concat(m, x4, patches, channel_dim);
    }
    tensor p4 = basic_decoder_block(m["block4"], x4);
    tensor p4_gdt = gdt_conv(m["gdt_convs_4"], p4);
    tensor gdt_attn_4 = conv_2d(m["gdt_convs_attn_4.0"], p4_gdt);
    gdt_attn_4 = ggml_sigmoid(m, gdt_attn_4);
    p4 = ggml_mul(m, p4, gdt_attn_4);

    x3 = conv_2d(m["lateral_block4.conv"], x3);
    tensor _p4 = upscale_to(m, p4, x3);
    // ggml-cuda's add kernel asserts element-aligned src1 strides (binbcast.cu);
    // under whcn the lateral-conv output can be non-contiguous -> force cont.
    tensor _p3 = ggml_add_inplace(m, _p4, ggml_cont(m, x3));

    {
        auto [w, h, c, n] = nelements_whcn(m, _p3);
        tensor patches = image_to_patches(m, x_whcn, w, h);
        patches = whcn_to_contiguous_2d(m, patches);
        patches = simple_conv(m["ipt_blk4"], patches);
        _p3 = ggml_concat(m, _p3, patches, channel_dim);
    }
    tensor p3 = basic_decoder_block(m["block3"], _p3);
    tensor p3_gdt = gdt_conv(m["gdt_convs_3"], p3);
    tensor gdt_attn_3 = conv_2d(m["gdt_convs_attn_3.0"], p3_gdt);
    gdt_attn_3 = ggml_sigmoid(m, gdt_attn_3);
    p3 = ggml_mul(m, p3, gdt_attn_3);

    _p3 = upscale_to(m, p3, x2);
    x2 = conv_2d(m["lateral_block3.conv"], x2);
    tensor _p2 = ggml_add_inplace(m, _p3, ggml_cont(m, x2));

    {
        auto [w, h, c, n] = nelements_whcn(m, _p2);
        tensor patches = image_to_patches(m, x_whcn, w, h);
        patches = whcn_to_contiguous_2d(m, patches);
        patches = simple_conv(m["ipt_blk3"], patches);
        _p2 = ggml_concat(m, _p2, patches, channel_dim);
    }
    tensor p2 = basic_decoder_block(m["block2"], _p2);
    tensor p2_gdt = gdt_conv(m["gdt_convs_2"], p2);
    tensor gdt_attn2 = conv_2d(m["gdt_convs_attn_2.0"], p2_gdt);
    gdt_attn2 = ggml_sigmoid(m, gdt_attn2);
    p2 = ggml_mul(m, p2, gdt_attn2);

    _p2 = upscale_to(m, p2, x1);
    x1 = conv_2d(m["lateral_block2.conv"], x1);
    tensor _p1 = ggml_add_inplace(m, _p2, ggml_cont(m, x1));

    {
        auto [w, h, c, n] = nelements_whcn(m, _p1);
        tensor patches = image_to_patches(m, x_whcn, w, h);
        patches = whcn_to_contiguous_2d(m, patches);
        patches = simple_conv(m["ipt_blk2"], patches);
        _p1 = ggml_concat(m, _p1, patches, channel_dim);
    }
    _p1 = basic_decoder_block(m["block1"], _p1);
    // NB: _p1 is NOT upscaled here — the output head below fuses the upscale into a
    // low-res 1x1 conv (see the commute argument). The plain-concat fallback upscales.
    tensor p1_ipt = simple_conv(m["ipt_blk1"], x);

    // Output head, VRAM-fused. conv_out1.0 is a 1x1 conv (240->1 for RMBG-2.0). Two
    // lossless algebraic identities collapse the most expensive full-res buffers:
    //   (1) 1x1-conv over a channel-concat == sum of two 1x1 convs on the split halves
    //         conv1x1(concat([A, B])) == conv1x1_a(A) + conv1x1_b(B)
    //       -> never materialise the [1024,1024,240] concat (960 MiB).
    //   (2) a 1x1 conv (channel-linear) commutes with bilinear upscale (spatial-linear),
    //       since they act on orthogonal axes:
    //         conv1x1_a(upscale(_p1)) == upscale(conv1x1_a(_p1))
    //       -> run conv1x1_a at the low (pre-upscale) resolution and upscale only its
    //          single output channel, never materialising the [1024,1024,192] upscale
    //          (768 MiB). _p1 stays at decoder res (~256^2).
    // Together these remove the graph's two largest transient buffers at full 1024
    // quality. VRAM is the deploy priority (worker-isolated service sharing the 12GB
    // card). Lossless modulo FP add/interp order. whcn/GPU path only; cwhn/CPU keeps the
    // plain concat (no VRAM pressure, different 1x1 layout).
    model_ref m_out = m["conv_out1.0"];
    tensor cw = m_out.weights("weight");
    tensor p1_out;
    if (is_whcn(m) && cw->ne[0] == 1 && cw->ne[1] == 1 &&
        cw->ne[2] == _p1->ne[2] + p1_ipt->ne[2]) {
        int64_t oc = cw->ne[3];
        tensor w2 = ggml_reshape_2d(m, cw, cw->ne[2], oc); // [IC, OC]
        int64_t c1 = _p1->ne[2], c2 = p1_ipt->ne[2];
        tensor wa = ggml_cont(m, ggml_view_2d(m, w2, c1, oc, w2->nb[1], 0));
        tensor wb = ggml_cont(m, ggml_view_2d(m, w2, c2, oc, w2->nb[1], c1 * w2->nb[0]));
        auto conv1x1 = [&](tensor x_in, tensor wt) {
            auto [w, h, c, b] = nelements(x_in);
            tensor xc = ggml_cont(m, permute_whcn_to_cwhn(m, x_in)); // [C, W, H, B]
            xc = ggml_reshape_2d(m, xc, c, w * h * b);               // [C, W*H*B]
            tensor y = ggml_mul_mat(m, wt, xc);                      // [OC, W*H*B]
            y = ggml_reshape_4d(m, y, oc, w, h, b);                  // [OC, W, H, B]
            return ggml_cont(m, permute_cwhn_to_whcn(m, y));         // [W, H, OC, B]
        };
        tensor y_a = upscale_to(m, conv1x1(_p1, wa), x); // conv at low res, upscale 1ch
        tensor y_b = conv1x1(p1_ipt, wb);                // p1_ipt already full-res
        p1_out = ggml_add(m, y_a, y_b);
        if (tensor bias = m_out.find("bias")) {
            bias = ggml_reshape_4d(m, bias, 1, 1, oc, 1);
            if (bias->type != p1_out->type) {
                bias = ggml_cast(m, bias, p1_out->type);
            }
            p1_out = ggml_add(m, p1_out, bias);
        }
    } else {
        _p1 = upscale_to(m, _p1, x);
        _p1 = ggml_concat(m, _p1, p1_ipt, channel_dim);
        p1_out = conv_2d(m["conv_out1.0"], _p1);
    }
    p1_out = ggml_sigmoid_inplace(m, p1_out);

    return named(m, p1_out);
}

} // namespace birefnet

tensor birefnet_predict(model_ref m, tensor image, birefnet_params const& p) {
    // Encoder
    swin_result features = birefnet::encode(m, image, p.encoder);
    // Squeeze block
    features[3] = birefnet::basic_decoder_block(m["squeeze_module.0"], features[3]);
    // Decoder
    tensor scaled_preds = birefnet::decode(m["decoder"], image, features);

    return compute_graph_output(m, scaled_preds);
}

image_data birefnet_process_input(image_view image, birefnet_params const& p) {
    constexpr f32x4 mean = f32x4{0.485f, 0.456f, 0.406f, 0.f};
    constexpr f32x4 std = f32x4{0.229f, 0.224f, 0.225f, 1.f};

    image_data resized;
    if (image.extent != p.image_extent) {
        resized = image_scale(image, p.image_extent);
        image = image_view(resized);
    }

    return image_u8_to_f32(image, image_format::rgb_f32, -mean, 1.f / std);
}

image_data birefnet_process_output(
    span<float const> mask_data, i32x2 target_extent, birefnet_params const& p) {

    image_view mask_output(p.image_extent, mask_data);
    image_data mask_resized;
    if (p.image_extent != target_extent) {
        mask_resized = image_scale(mask_output, target_extent);
        mask_output = mask_resized;
    }
    return image_f32_to_u8(mask_output, image_format::alpha_u8);
}

i32x2 birefnet_image_extent(i32x2 input_extent, birefnet_params const& p, size_t max_alloc) {
    i32x2 extent{p.image_size, p.image_size};
    if (p.image_size == -1) {
        ASSERT(input_extent[0] > 0 && input_extent[1] > 0);
        // largest layer in BiRefNet-dynamic is input for 240-channel conv-2d at full resolution
        size_t req_alloc = size_t(input_extent[0]) * input_extent[1] * 240ULL * sizeof(float);
        if (req_alloc > max_alloc) {
            float scale = std::sqrt(float(max_alloc) / float(req_alloc));
            input_extent = {
                std::max(1, int(input_extent[0] * scale) - p.image_multiple),
                std::max(1, int(input_extent[1] * scale) - p.image_multiple)};
        }
        extent = {
            next_multiple(input_extent[0], p.image_multiple),
            next_multiple(input_extent[1], p.image_multiple)};
    }
    return extent;
}

birefnet_params birefnet_detect_params(
    model_file const& f, i32x2 dynamic_extent, size_t max_alloc) {

    if (std::string_view arch = f.arch(); arch != "birefnet") {
        throw except("Architecture expected to be 'birefnet', but was '{}' ({})", arch, f.path);
    }
    birefnet_params p;
    p.image_size = f.get_int("birefnet.image_size");
    p.image_multiple = f.get_int("birefnet.image_multiple");
    // VISP_BIREFNET_RES overrides the inference resolution. RMBG-2.0 is native
    // 1024; CPU cost scales ~res^2, so lower values trade matte detail for speed.
    // Ignored for dynamic models (image_size == -1, sized from the input).
    if (char const* e = getenv("VISP_BIREFNET_RES"); e && p.image_size > 0) {
        int r = atoi(e);
        if (r >= 64) {
            p.image_size = r;
        }
    }
    p.image_extent = birefnet_image_extent(dynamic_extent, p, max_alloc);
    p.encoder = swin_detect_params(f);
    return p;
}

birefnet_buffers birefnet_precompute(model_ref m, birefnet_params const& p) {
    return swin_precompute(m, p.image_extent, p.encoder);
}

} // namespace visp
