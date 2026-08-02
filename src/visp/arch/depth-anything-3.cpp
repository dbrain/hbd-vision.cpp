#include "visp/arch/depth-anything-3.h"
#include "util/math.h"
#include "util/string.h"
#include "visp/arch/dino.h"
#include "visp/ml.h"
#include "visp/nn.h"

#include <algorithm>
#include <cmath>

namespace visp {
namespace da3 {

int32_t const bilinear_align_corners = int32_t(GGML_SCALE_MODE_BILINEAR) |
    GGML_SCALE_FLAG_ALIGN_CORNERS;

//
// Pre-computed buffers

// torch.nn.functional.interpolate(mode="bicubic", align_corners=False) uses the classic
// Keys cubic kernel with A = -0.75 and replicated borders.
void cubic_coefficients(float t, float (&w)[4]) {
    constexpr float a = -0.75f;
    auto near = [](float x) { return ((a + 2.f) * x - (a + 3.f)) * x * x + 1.f; };
    auto far = [](float x) { return ((a * x - 5.f * a) * x + 8.f * a) * x - 4.f * a; };
    w[0] = far(t + 1.f);
    w[1] = near(t);
    w[2] = near(1.f - t);
    w[3] = far(2.f - t);
}

// DINOv2 interpolates the position grid by passing an explicit scale factor of
// (target + 0.1) / grid rather than the target size, which shifts the sample grid by up to
// ~0.1 patch. Reproducing that in-graph is not possible with ggml's size-based interpolate,
// and skipping it costs ~6% RMS on the embedding, so resample on the host instead.
tensor_data create_pos_encoding(model_ref m, depthany3_params const& p) {
    tensor src = m.weights("backbone.embeddings.position_embeddings");
    ASSERT(src->type == GGML_TYPE_F32, "Expecting F32 position embeddings");
    tensor_data host = transfer_from_backend(src);
    span<float const> in = host.as_f32();

    int64_t dim = src->ne[0];
    int64_t grid = p.pos_embed_grid;
    int64_t pw = p.image_extent[0] / p.image_multiple;
    int64_t ph = p.image_extent[1] / p.image_multiple;
    ASSERT(src->ne[1] == 1 + grid * grid, "Position embedding size does not match grid");

    tensor_data result = tensor_alloc(ggml_new_tensor_3d(m, GGML_TYPE_F32, dim, 1 + pw * ph, 1));
    span<float> out = result.as_f32();
    std::copy(in.begin(), in.begin() + dim, out.begin()); // class token embedding

    float const offset = 0.1f;
    float scale_y = float(double(grid) / (double(ph) + offset));
    float scale_x = float(double(grid) / (double(pw) + offset));
    auto sample = [&](int64_t row, int64_t col, int64_t c) {
        row = clamp<int64_t>(row, 0, grid - 1);
        col = clamp<int64_t>(col, 0, grid - 1);
        return in[size_t((1 + row * grid + col) * dim + c)];
    };

    for (int64_t oy = 0; oy < ph; ++oy) {
        float real_y = scale_y * (float(oy) + 0.5f) - 0.5f;
        int64_t iy = int64_t(std::floor(real_y));
        float wy[4];
        cubic_coefficients(real_y - float(iy), wy);

        for (int64_t ox = 0; ox < pw; ++ox) {
            float real_x = scale_x * (float(ox) + 0.5f) - 0.5f;
            int64_t ix = int64_t(std::floor(real_x));
            float wx[4];
            cubic_coefficients(real_x - float(ix), wx);

            float* dst = &out[size_t((1 + oy * pw + ox) * dim)];
            for (int64_t c = 0; c < dim; ++c) {
                float v = 0.f;
                for (int j = 0; j < 4; ++j) {
                    float row_sum = 0.f;
                    for (int i = 0; i < 4; ++i) {
                        row_sum += wx[i] * sample(iy - 1 + j, ix - 1 + i, c);
                    }
                    v += wy[j] * row_sum;
                }
                dst[c] = v;
            }
        }
    }
    make_constant(result.x, "da3.pos_embed");
    return result;
}

// Token positions for 2D RoPE. Patch positions are 1-based; the camera/class token at index 0
// gets position 0. Global-attention blocks see a constant position, so relative rotation
// between patches vanishes but the camera token stays distinguishable.
std::array<tensor_data, 3> create_rope_positions(model_ref m, depthany3_params const& p) {
    int64_t pw = p.image_extent[0] / p.image_multiple;
    int64_t ph = p.image_extent[1] / p.image_multiple;
    int64_t n = 1 + pw * ph;

    auto alloc = [&](char const* name) {
        tensor_data t = tensor_alloc(ggml_new_tensor_1d(m, GGML_TYPE_I32, n));
        make_constant(t.x, name);
        return t;
    };
    std::array<tensor_data, 3> r{alloc("da3.rope_y"), alloc("da3.rope_x"), alloc("da3.rope_c")};
    span<int32_t> y = r[0].as_i32();
    span<int32_t> x = r[1].as_i32();
    span<int32_t> c = r[2].as_i32();
    y[0] = x[0] = c[0] = 0;
    for (int64_t i = 0; i < pw * ph; ++i) {
        y[size_t(i + 1)] = int32_t(i / pw + 1);
        x[size_t(i + 1)] = int32_t(i % pw + 1);
        c[size_t(i + 1)] = 1;
    }
    return r;
}

// Sinusoidal UV embedding added to every pyramid stage and again before the output head.
// Channels are laid out as [sin(u) cos(u) sin(v) cos(v)], a quarter of the channels each.
tensor_data create_uv_embed(
    model_ref m, i64x2 extent, int64_t channels, float aspect, depthany3_params const& p,
    tensor_name name) {

    ASSERT(channels % 4 == 0, "UV embedding requires a channel count divisible by 4");
    int64_t w = extent[0], h = extent[1], q = channels / 4;
    bool cwhn = is_cwhn(m);
    tensor t = cwhn ? ggml_new_tensor_4d(m, GGML_TYPE_F32, channels, w, h, 1)
                    : ggml_new_tensor_4d(m, GGML_TYPE_F32, w, h, channels, 1);
    tensor_data result = tensor_alloc(t);
    span<float> out = result.as_f32();

    float diag = std::sqrt(aspect * aspect + 1.f);
    auto axis = [](int64_t i, int64_t n, float half_range) {
        float lo = -half_range * float(n - 1) / float(n);
        return n > 1 ? lo + float(i) * (-2.f * lo) / float(n - 1) : lo;
    };
    std::vector<float> omega(size_t(q), 0.f);
    for (int64_t k = 0; k < q; ++k) {
        omega[size_t(k)] = 1.f / std::pow(p.head_pos_embed_omega, float(k) / float(q));
    }

    for (int64_t j = 0; j < h; ++j) {
        float v = axis(j, h, 1.f / diag);
        for (int64_t i = 0; i < w; ++i) {
            float u = axis(i, w, aspect / diag);
            for (int64_t c = 0; c < channels; ++c) {
                int64_t k = c % q;
                float angle = (c < 2 * q ? u : v) * omega[size_t(k)];
                float e = (c < q || (c >= 2 * q && c < 3 * q)) ? std::sin(angle) : std::cos(angle);
                size_t index = cwhn ? size_t((j * w + i) * channels + c)
                                    : size_t((c * h + j) * w + i);
                out[index] = e * p.head_pos_embed_ratio;
            }
        }
    }
    make_constant(result.x, name);
    return result;
}

//
// Encoder

std::vector<tensor> encode(model_ref m, tensor image, depthany3_params const& p) {
    tensor pos = ggml_get_tensor(m, "da3.pos_embed");
    tensor rope_y = ggml_get_tensor(m, "da3.rope_y");
    tensor rope_x = ggml_get_tensor(m, "da3.rope_x");
    tensor rope_c = ggml_get_tensor(m, "da3.rope_c");
    ASSERT(pos && rope_y && rope_x && rope_c, "Missing pre-computed buffers");

    tensor x = dino::prepare_tokens(m["embeddings"], image, p.dino.patch_size, pos);
    tensor local = x;

    std::vector<tensor> features;
    model_ref encoder = m["encoder.layer"];
    for (int i = 0; i < p.dino.n_layers; ++i) {
        if (i == p.alt_start) {
            // slot 0 of the learned camera token is the reference view; source-view slot 1 is
            // unused because a single input image is always the reference
            tensor cam = slice(m, m.weights("embeddings.camera_token"), {}, {0}, {}, {});
            tensor rest = ggml_cont(m, slice(m, x, {}, {1, x->ne[1]}, {}, {}));
            x = concat(m, {ggml_cont(m, cam), rest}, 1);
        }
        bool global = i >= p.alt_start && i % 2 == 1;

        dino::block_params bp;
        bp.qk_norm = i >= p.qknorm_start;
        bp.rope_frequency = p.rope_frequency;
        if (i >= p.rope_start) {
            bp.rope_pos_y = global ? rope_c : rope_y;
            bp.rope_pos_x = global ? rope_c : rope_x;
        }
        x = dino::layer(encoder[i], x, p.dino, bp);
        if (!global) {
            local = x;
        }

        if (std::find(p.feature_layers.begin(), p.feature_layers.end(), i) !=
            p.feature_layers.end()) {
            // the local half passes through raw, only the global half is normalized
            tensor f = concat(m, {local, layer_norm(m["layernorm"], x)}, 0);
            f = ggml_cont(m, slice(m, f, {}, {1, f->ne[1]}, {}, {}));
            ggml_format_name(f, "da3_layer_%d", i);
            ggml_build_forward_expand(m.graph, f);
            features.push_back(f);
        }
    }
    return features;
}

//
// DualDPT head (depth chain only; the ray/camera chain is not built)

tensor residual_unit(model_ref m, tensor x) {
    tensor out = ggml_relu(m, x);
    out = conv_2d(m["conv1"], out, 1, 1);
    out = ggml_relu(m, out);
    out = conv_2d(m["conv2"], out, 1, 1);
    return named(m, ggml_add(m, x, out));
}

tensor fusion(model_ref m, tensor x0, tensor x1, int64_t const* size) {
    tensor x = x0;
    if (x1) {
        x = ggml_add(m, x, residual_unit(m["resConfUnit1"], x1));
    }
    x = residual_unit(m["resConfUnit2"], x);

    int const dim = is_cwhn(m) ? 1 : 0;
    int64_t w = size ? size[dim + 0] : x->ne[dim + 0] * 2;
    int64_t h = size ? size[dim + 1] : x->ne[dim + 1] * 2;
    x = contiguous_2d_to_whcn(m, x);
    x = interpolate(m, x, {w, h}, bilinear_align_corners);
    x = whcn_to_contiguous_2d(m, x);

    x = conv_2d(m["out_conv"], x);
    return named(m, x);
}

depthany3_prediction head(model_ref m, span<tensor> features, depthany3_params const& p) {
    ASSERT(features.size() == 4);
    i64x2 extent{p.image_extent[0], p.image_extent[1]};
    i64x2 patch = extent / p.image_multiple;

    std::array<tensor, 4> stages;
    for (int i = 0; i < 4; ++i) {
        tensor x = layer_norm(m["norm"], features[i]);
        x = ggml_reshape_4d(m, x, x->ne[0], patch[0], patch[1], 1);

        model_ref proj = m["projects"][i];
        proj.flags |= model_build_flag::cwhn;
        x = conv_2d(proj, x); // 1x1 conv, keep CWHN layout and directly use mul_mat

        x = cwhn_to_contiguous_2d(m, x);
        x = ggml_add(m, x, ggml_get_tensor(m, format<tensor_name>("da3.uv_{}", i).c_str()));

        model_ref resize = m["resize_layers"];
        switch (i) {
            case 0: x = conv_transpose_2d(resize[0], x, 4); break;
            case 1: x = conv_transpose_2d(resize[1], x, 2); break;
            case 3: x = conv_2d(resize[3], x, 2, 1); break;
        }
        stages[i] = x;
    }

    model_ref s = m["scratch"];
    for (int i = 0; i < 4; ++i) {
        stages[i] = conv_2d(s[format<tensor_name>("layer{}_rn", i + 1)], stages[i], 1, 1);
    }

    tensor out = fusion(s["refinenet4"], stages[3], nullptr, stages[2]->ne);
    out = fusion(s["refinenet3"], out, stages[2], stages[1]->ne);
    out = fusion(s["refinenet2"], out, stages[1], stages[0]->ne);
    out = fusion(s["refinenet1"], out, stages[0]);
    out = conv_2d(s["output_conv1"], out, 1, 1);

    out = contiguous_2d_to_whcn(m, out);
    out = interpolate(m, out, extent, bilinear_align_corners);
    out = whcn_to_contiguous_2d(m, out);
    out = ggml_add(m, out, ggml_get_tensor(m, "da3.uv_out"));

    model_ref output_conv = s["output_conv2"];
    out = conv_2d(output_conv[0], out, 1, 1);
    out = ggml_relu_inplace(m, out);
    out = conv_2d(output_conv[2], out);

    out = contiguous_2d_to_whcn(m, out);
    tensor depth = ggml_exp(m, ggml_cont(m, slice(m, out, {}, {}, {0}, {})));
    tensor conf = ggml_exp(m, ggml_cont(m, slice(m, out, {}, {}, {1}, {})));
    return {depth, ggml_scale_bias(m, conf, 1.f, 1.f)};
}

} // namespace da3

depthany3_buffers depthany3_precompute(model_ref m, depthany3_params const& p) {
    i64x2 extent{p.image_extent[0], p.image_extent[1]};
    i64x2 patch = extent / p.image_multiple;
    float aspect = float(p.image_extent[0]) / float(p.image_extent[1]);

    depthany3_buffers buffers;
    buffers.push_back(da3::create_pos_encoding(m, p));
    for (tensor_data& t : da3::create_rope_positions(m, p)) {
        buffers.push_back(std::move(t));
    }
    for (int i = 0; i < 4; ++i) {
        buffers.push_back(da3::create_uv_embed(
            m, patch, p.head_out_channels[i], aspect, p, format<tensor_name>("da3.uv_{}", i)));
    }
    buffers.push_back(
        da3::create_uv_embed(m, extent, p.head_features / 2, aspect, p, "da3.uv_out"));
    return buffers;
}

depthany3_prediction depthany3_predict(model_ref m, tensor image, depthany3_params const& p) {
    std::vector<tensor> features = da3::encode(m["backbone"], image, p);
    depthany3_prediction out = da3::head(m["head"], features, p);
    return {
        compute_graph_output(m, out.depth, "depth"),
        compute_graph_output(m, out.confidence, "confidence")};
}

i32x2 depthany3_image_extent(i32x2 extent, depthany3_params const& p) {
    int longest = std::max(extent[0], extent[1]);
    double scale = double(p.image_size) / double(longest);
    auto round_to_patch = [&](int value) {
        int scaled = std::max(1, int(std::lround(double(value) * scale)));
        int down = (scaled / p.image_multiple) * p.image_multiple;
        int up = down + p.image_multiple;
        return (up - scaled <= scaled - down) ? up : std::max(down, p.image_multiple);
    };
    return {round_to_patch(extent[0]), round_to_patch(extent[1])};
}

depthany3_params depthany3_detect_params(model_file const& file, i32x2 input_extent) {
    depthany3_params p;
    p.dino = dino_detect_params(file);
    p.image_size = file.get_int("depthanything3.image_size");
    p.image_multiple = file.get_int("depthanything3.image_multiple");
    p.pos_embed_grid = file.get_int("depthanything3.pos_embed_grid");
    p.alt_start = file.get_int("depthanything3.alt_start");
    p.qknorm_start = file.get_int("depthanything3.qknorm_start");
    p.rope_start = file.get_int("depthanything3.rope_start");
    p.head_features = file.get_int("depthanything3.head_features");
    file.get_array("depthanything3.head_out_channels", p.head_out_channels);
    file.get_array("depthanything3.feature_layers", p.feature_layers);
    if (input_extent[0] > 0 && input_extent[1] > 0) {
        p.image_extent = depthany3_image_extent(input_extent, p);
    }
    return p;
}

image_data depthany3_process_input(image_view image, depthany3_params const& p) {
    constexpr f32x4 mean = f32x4{0.485f, 0.456f, 0.406f, 0.f};
    constexpr f32x4 std = f32x4{0.229f, 0.224f, 0.225f, 1.f};

    image_data resized;
    if (image.extent != p.image_extent) {
        resized = image_scale(image, p.image_extent);
        image = image_view(resized);
    }
    return image_u8_to_f32(image, image_format::rgb_f32, -mean, 1.f / std);
}

image_data depthany3_process_output(
    span<float const> data, i32x2 extent, depthany3_params const& p) {

    image_view plane(p.image_extent, data);
    if (plane.extent != extent) {
        return image_scale(plane, extent);
    }
    image_data result = image_alloc(extent, image_format::alpha_f32);
    std::copy(data.begin(), data.end(), reinterpret_cast<float*>(result.data.get()));
    return result;
}

} // namespace visp
