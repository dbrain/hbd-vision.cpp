#include "server/worker.h"

#include "sam3.h"
#include "siglip2/siglip2.h"
#include "util/string.h"
#include "visp/vision.h"

#include "nlohmann/json.hpp"
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

using json = nlohmann::json;
using namespace visp;

namespace vserver {

namespace {

void l2_normalize(std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) {
        s += (double)x * x;
    }
    float const inv = (float)(1.0 / (std::sqrt(s) + 1e-12));
    for (auto& x : v) {
        x *= inv;
    }
}

void send_error(int fd, uint32_t req_id, std::string const& msg) {
    send_frame(fd, Frame::ERR, req_id, msg.data(), msg.size());
}

// Everything the worker owns: one device, and one lazily-loaded slot per model
// family. A further family slots in the same way.
struct WorkerModels {
    ServerConfig cfg;
    std::unique_ptr<backend_device> device;

    std::unique_ptr<birefnet_model> matting;
    int matting_native_size = 0;

    std::unique_ptr<depthany_model> depth2;
    int depth2_native_size = 0;

    std::unique_ptr<depthany3_model> depth3;
    int depth3_native_size = 0;

    std::unique_ptr<siglip2::Model> siglip;
    std::unique_ptr<siglip2::VisionEncoder> siglip_vision;
    std::unique_ptr<siglip2::TextEncoder> siglip_text;
    std::unique_ptr<siglip2::Tokenizer> siglip_tokenizer;

    // sam3.cpp owns its own ggml backend, so it does not go through
    // `device`. On this tree that backend is always CPU: sam3.cpp only ever
    // calls ggml_backend_metal_init() and falls back to the CPU backend
    // everywhere else. Pointing it at CUDA is not a build flag — its ViT uses
    // WIN_PART/WIN_UNPART, its PCS scoring head uses POOL_1D, and its
    // fusion/DETR/mask-decoder attention runs at head_dim 32; none of the four
    // have a CUDA path, and sam3.cpp computes on a single backend with no
    // sched fallback, so each is a hard abort rather than a slow path.
    std::shared_ptr<sam3_model> sam3;
    sam3_state_ptr sam3_state;
    int sam3_threads = 0;

    void load_matting() {
        if (matting || cfg.matting_model.empty()) {
            return;
        }
        matting = std::make_unique<birefnet_model>(
            birefnet_load_model(cfg.matting_model.c_str(), *device));
        matting_native_size = matting->params.image_size;
    }

    void load_depth2() {
        if (depth2) {
            return;
        }
        if (cfg.depth_model.empty()) {
            throw except("depth model not configured (DEPTH_MODEL_PATH / --depth-model)");
        }
        depth2 = std::make_unique<depthany_model>(
            depthany_load_model(cfg.depth_model.c_str(), *device));
        depth2_native_size = depth2->params.image_size;
        fprintf(
            stderr, "[vision-worker] depth-anything-v2 loaded: %s (image_size %d)\n",
            cfg.depth_model.c_str(), depth2_native_size);
    }

    void load_depth3() {
        if (depth3) {
            return;
        }
        if (cfg.depth3_model.empty()) {
            throw except("depth-anything-3 not configured (DEPTH3_MODEL_PATH / --depth3-model)");
        }
        depth3 = std::make_unique<depthany3_model>(
            depthany3_load_model(cfg.depth3_model.c_str(), *device));
        depth3_native_size = depth3->params.image_size;
        fprintf(
            stderr, "[vision-worker] depth-anything-3 loaded: %s (image_size %d)\n",
            cfg.depth3_model.c_str(), depth3_native_size);
    }

    void load_siglip() {
        if (siglip) {
            return;
        }
        if (cfg.siglip_model.empty()) {
            throw except("siglip model not configured (SIGLIP_MODEL_PATH / --siglip-model)");
        }
        siglip = std::make_unique<siglip2::Model>(
            siglip2::load_model(cfg.siglip_model, *device));
        siglip_vision = std::make_unique<siglip2::VisionEncoder>();
        if (!siglip_vision->load(*siglip)) {
            throw except("siglip vision load: {}", siglip_vision->last_error());
        }
        siglip_text = std::make_unique<siglip2::TextEncoder>();
        if (!siglip_text->load(*siglip)) {
            throw except("siglip text load: {}", siglip_text->last_error());
        }
        if (cfg.siglip_tokenizer.empty()) {
            throw except("siglip tokenizer not configured (SIGLIP_TOKENIZER_PATH)");
        }
        siglip_tokenizer = std::make_unique<siglip2::Tokenizer>();
        if (!siglip_tokenizer->load(cfg.siglip_tokenizer)) {
            throw except("siglip tokenizer load: {}", siglip_tokenizer->last_error());
        }
        fprintf(
            stderr, "[vision-worker] siglip2 loaded: %s (hidden %d)\n", cfg.siglip_model.c_str(),
            siglip->vision.hidden_size);
    }

    void load_sam3() {
        if (sam3) {
            return;
        }
        if (cfg.sam3_model.empty()) {
            throw except("sam3 model not configured (SAM3_MODEL_PATH / --sam3-model)");
        }
        sam3_params p;
        p.model_path = cfg.sam3_model;
        p.use_gpu = false;
        p.n_threads = sam3_threads;
        sam3 = sam3_load_model(p);
        if (!sam3) {
            throw except("sam3 load failed: {}", cfg.sam3_model);
        }
        if (sam3_is_visual_only(*sam3)) {
            sam3.reset();
            throw except("sam3 model is visual-only — /parts needs the text (PCS) head");
        }
        sam3_state = sam3_create_state(*sam3, p);
        if (!sam3_state) {
            sam3.reset();
            throw except("sam3 state alloc failed");
        }
        fprintf(stderr, "[vision-worker] sam3 loaded: %s\n", cfg.sam3_model.c_str());
    }

    std::vector<float> embed_image(
        std::string const& bytes, int max_num_patches, siglip2::Pooling pooling) {
        int w = 0;
        int h = 0;
        int channels = 0;
        uint8_t* px = stbi_load_from_memory(
            reinterpret_cast<stbi_uc const*>(bytes.data()), (int)bytes.size(), &w, &h, &channels,
            3);
        if (!px) {
            throw except(
                "image decode failed: {}", stbi_failure_reason() ? stbi_failure_reason() : "?");
        }
        siglip2::PreprocResult pp;
        std::string err;
        bool ok = siglip2::preprocess_image_rgb(
            px, h, w, max_num_patches, siglip->vision.patch_size, siglip->preproc.rescale_factor,
            siglip->preproc.image_mean, siglip->preproc.image_std, pp, err);
        stbi_image_free(px);
        if (!ok) {
            throw except("preprocess: {}", err);
        }
        std::vector<float> emb;
        if (!siglip_vision->encode(
                pp.pixel_values.data(), pp.n_patches_h, pp.n_patches_w, pooling, emb)) {
            throw except("vision encode: {}", siglip_vision->last_error());
        }
        return emb;
    }

    // Tokenizes to max_position_embeddings with pad_token_id and no attention
    // mask, matching HF Siglip2Processor: the live service does
    // `model.text_model(**inputs)` with input_ids only, so pad embeddings flow
    // through attention as real tokens. Passing a mask diverges by ~0.24 cosine
    // on short prompts.
    std::vector<std::vector<float>> embed_texts(std::vector<std::string> const& prompts) {
        int const n_batch = (int)prompts.size();
        int const max_pos = siglip->text.max_position_embeddings;
        int const proj = siglip->text.projection_size;

        std::vector<int32_t> ids_flat((size_t)max_pos * n_batch);
        for (int b = 0; b < n_batch; ++b) {
            std::vector<int32_t> ids;
            std::vector<int32_t> mask;
            if (!siglip_tokenizer->encode(prompts[b], max_pos, ids, mask)) {
                throw except("tokenize: {}", siglip_tokenizer->last_error());
            }
            std::memcpy(
                ids_flat.data() + (size_t)b * max_pos, ids.data(), sizeof(int32_t) * max_pos);
        }
        std::vector<float> flat;
        if (!siglip_text->encode_batch(ids_flat.data(), max_pos, n_batch, nullptr, flat)) {
            throw except("text encode: {}", siglip_text->last_error());
        }
        std::vector<std::vector<float>> out(n_batch, std::vector<float>((size_t)proj));
        for (int b = 0; b < n_batch; ++b) {
            std::memcpy(
                out[b].data(), flat.data() + (size_t)b * proj, sizeof(float) * (size_t)proj);
        }
        return out;
    }
};

json handle_siglip(WorkerModels& m, std::vector<uint8_t> const& payload) {
    if (payload.size() < 4) {
        throw except("short siglip payload");
    }
    uint32_t json_len = 0;
    std::memcpy(&json_len, payload.data(), 4);
    if (payload.size() < 4 + (size_t)json_len) {
        throw except("truncated siglip payload");
    }
    json req = json::parse(payload.begin() + 4, payload.begin() + 4 + json_len);

    std::vector<std::string> images;
    {
        size_t off = 4 + (size_t)json_len;
        for (auto const& sz : req.value("image_sizes", json::array())) {
            size_t n = sz.get<size_t>();
            if (off + n > payload.size()) {
                throw except("truncated siglip image blob");
            }
            images.emplace_back((char const*)payload.data() + off, n);
            off += n;
        }
    }

    m.load_siglip();

    std::string const op = req.value("op", "");
    int const max_num_patches =
        req.value("max_num_patches", m.cfg.siglip_default_max_num_patches);
    std::string const pooling_str = req.value("pooling", std::string("probe"));
    siglip2::Pooling pooling = siglip2::Pooling::PROBE;
    if (pooling_str == "mean") {
        pooling = siglip2::Pooling::MEAN;
    } else if (!(pooling_str.empty() || pooling_str == "probe" || pooling_str == "pooler")) {
        throw except("pooling must be one of: pooler, probe, mean (got '{}')", pooling_str);
    }

    std::vector<std::string> prompts;
    for (auto const& p : req.value("prompts", json::array())) {
        prompts.push_back(p.get<std::string>());
    }

    if (op == "embed") {
        std::vector<std::vector<float>> embs;
        embs.reserve(images.size());
        for (auto const& img : images) {
            auto e = m.embed_image(img, max_num_patches, pooling);
            l2_normalize(e);
            embs.push_back(std::move(e));
        }
        return json{{"embeddings", embs}};
    }
    if (op == "text") {
        auto embs = m.embed_texts(prompts);
        for (auto& e : embs) {
            l2_normalize(e);
        }
        return json{{"embeddings", embs}};
    }

    // classify / classify_from_embeddings: score_image_text L2-normalizes both
    // sides itself, so pre-normalized client embeddings score identically.
    int const H = m.siglip->vision.hidden_size;
    int n_img = 0;
    std::vector<float> img_buf;
    if (op == "classify") {
        n_img = (int)images.size();
        img_buf.assign((size_t)n_img * H, 0.0f);
        for (int i = 0; i < n_img; ++i) {
            auto e = m.embed_image(images[i], max_num_patches, siglip2::Pooling::PROBE);
            std::memcpy(img_buf.data() + (size_t)i * H, e.data(), sizeof(float) * H);
        }
    } else if (op == "cfe") {
        auto const& rows = req.at("image_embeddings");
        n_img = (int)rows.size();
        img_buf.assign((size_t)n_img * H, 0.0f);
        for (int i = 0; i < n_img; ++i) {
            if ((int)rows[i].size() != H) {
                throw except("image_embedding row has wrong dim (expected {})", H);
            }
            for (int c = 0; c < H; ++c) {
                img_buf[(size_t)i * H + c] = rows[i][c].get<float>();
            }
        }
    } else {
        throw except("unknown siglip op '{}'", op);
    }

    auto txt_embs = m.embed_texts(prompts);
    int const n_txt = (int)txt_embs.size();
    std::vector<float> txt_buf((size_t)n_txt * H);
    for (int j = 0; j < n_txt; ++j) {
        std::memcpy(txt_buf.data() + (size_t)j * H, txt_embs[j].data(), sizeof(float) * H);
    }

    std::vector<float> logits((size_t)n_img * n_txt);
    std::vector<float> probs((size_t)n_img * n_txt);
    siglip2::score_image_text(
        img_buf.data(), n_img, txt_buf.data(), n_txt, H, m.siglip->score, logits.data(),
        probs.data());

    std::vector<std::vector<float>> scores_out(n_img, std::vector<float>(n_txt));
    std::vector<std::vector<float>> logits_out(n_img, std::vector<float>(n_txt));
    for (int i = 0; i < n_img; ++i) {
        for (int j = 0; j < n_txt; ++j) {
            scores_out[i][j] = probs[(size_t)i * n_txt + j];
            logits_out[i][j] = logits[(size_t)i * n_txt + j];
        }
    }
    json out = {{"scores", scores_out}};
    if (req.value("return_logits", false)) {
        out["logits"] = logits_out;
    }
    return out;
}

void handle_matte(WorkerModels& m, int fd, FrameHeader const& hdr, std::vector<uint8_t>& payload) {
    if (payload.size() < 12) {
        send_error(fd, hdr.req_id, "short infer payload");
        return;
    }
    int32_t process_res;
    int32_t w;
    int32_t h;
    std::memcpy(&process_res, payload.data() + 0, 4);
    std::memcpy(&w, payload.data() + 4, 4);
    std::memcpy(&h, payload.data() + 8, 4);
    size_t need = 12 + (size_t)w * h * 4;
    if (w <= 0 || h <= 0 || payload.size() != need) {
        send_error(fd, hdr.req_id, "bad infer dimensions");
        return;
    }
    m.load_matting();
    if (!m.matting) {
        send_error(fd, hdr.req_id, "matting model not configured (MODEL_PATH / --model)");
        return;
    }
    if (m.matting_native_size > 0) {
        m.matting->params.image_size = process_res > 0 ? process_res : m.matting_native_size;
    }
    i32x2 ext{w, h};
    image_view input(ext, image_format::rgba_u8, payload.data() + 12);
    image_data out = birefnet_compute(*m.matting, input);
    uint8_t const* mask = out.data.get();
    if (!mask) {
        send_error(fd, hdr.req_id, "inference produced no mask");
        return;
    }
    std::vector<uint8_t> resp(12 + (size_t)w * h);
    std::memcpy(resp.data() + 0, &w, 4);
    std::memcpy(resp.data() + 4, &h, 4);
    int32_t one = 1;
    std::memcpy(resp.data() + 8, &one, 4); // reserved
    std::memcpy(resp.data() + 12, mask, (size_t)w * h);
    send_frame(fd, Frame::MATTE_RESP, hdr.req_id, resp.data(), resp.size());
}

void handle_depth(WorkerModels& m, int fd, FrameHeader const& hdr, std::vector<uint8_t>& payload) {
    if (payload.size() < 16) {
        send_error(fd, hdr.req_id, "short depth payload");
        return;
    }
    int32_t model;
    int32_t image_size;
    int32_t w;
    int32_t h;
    std::memcpy(&model, payload.data() + 0, 4);
    std::memcpy(&image_size, payload.data() + 4, 4);
    std::memcpy(&w, payload.data() + 8, 4);
    std::memcpy(&h, payload.data() + 12, 4);
    size_t need = 16 + (size_t)w * h * 4;
    if (w <= 0 || h <= 0 || payload.size() != need) {
        send_error(fd, hdr.req_id, "bad depth dimensions");
        return;
    }

    i32x2 ext{w, h};
    image_view input(ext, image_format::rgba_u8, payload.data() + 16);
    size_t const npix = (size_t)w * h;

    image_data depth;
    image_data confidence;
    if ((DepthModel)model == DepthModel::v3) {
        m.load_depth3();
        m.depth3->params.image_size = image_size > 0 ? image_size : m.depth3_native_size;
        depthany3_images out = depthany3_compute(*m.depth3, input);
        depth = std::move(out.depth);
        confidence = std::move(out.confidence);
        if (!confidence.data) {
            send_error(fd, hdr.req_id, "inference produced no confidence");
            return;
        }
    } else {
        m.load_depth2();
        m.depth2->params.image_size = image_size > 0 ? image_size : m.depth2_native_size;
        depth = depthany_compute(*m.depth2, input);
    }
    if (!depth.data) {
        send_error(fd, hdr.req_id, "inference produced no depth");
        return;
    }

    int32_t const n_maps = confidence.data ? 2 : 1;
    std::vector<uint8_t> resp(12 + npix * 4 * (size_t)n_maps);
    std::memcpy(resp.data() + 0, &w, 4);
    std::memcpy(resp.data() + 4, &h, 4);
    std::memcpy(resp.data() + 8, &n_maps, 4);
    std::memcpy(resp.data() + 12, depth.data.get(), npix * 4);
    if (n_maps == 2) {
        std::memcpy(resp.data() + 12 + npix * 4, confidence.data.get(), npix * 4);
    }
    send_frame(fd, Frame::DEPTH_RESP, hdr.req_id, resp.data(), resp.size());
}

// One image encode, then one cheap prompt-conditioned decode per noun phrase.
// That split is the whole reason a multi-part rig is affordable: on this CPU the
// 848M ViT encode is ~80-95 s and each additional PCS decode is ~13-17 s, so a
// five-part creature costs ~1.5 encodes, not five.
void handle_parts(WorkerModels& m, int fd, FrameHeader const& hdr, std::vector<uint8_t>& payload) {
    if (payload.size() < 4) {
        send_error(fd, hdr.req_id, "short parts payload");
        return;
    }
    uint32_t json_len = 0;
    std::memcpy(&json_len, payload.data(), 4);
    if (payload.size() < 4 + (size_t)json_len) {
        send_error(fd, hdr.req_id, "truncated parts payload");
        return;
    }
    json req = json::parse(payload.begin() + 4, payload.begin() + 4 + json_len);

    int const w = req.at("width").get<int>();
    int const h = req.at("height").get<int>();
    size_t const npix = (size_t)w * h;
    if (w <= 0 || h <= 0 || payload.size() != 4 + (size_t)json_len + npix * 4) {
        send_error(fd, hdr.req_id, "bad parts dimensions");
        return;
    }
    std::vector<std::string> prompts;
    for (auto const& p : req.at("prompts")) {
        prompts.push_back(p.get<std::string>());
    }
    float const threshold = req.value("threshold", 0.35f);
    float const nms = req.value("nms", 0.1f);
    int const max_instances = req.value("max_instances", 0);

    m.load_sam3();

    sam3_image img;
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.data.resize(npix * 3);
    uint8_t const* rgba = payload.data() + 4 + json_len;
    for (size_t i = 0; i < npix; ++i) {
        img.data[i * 3 + 0] = rgba[i * 4 + 0];
        img.data[i * 3 + 1] = rgba[i * 4 + 1];
        img.data[i * 3 + 2] = rgba[i * 4 + 2];
    }

    int64_t const t_enc0 = ggml_time_us();
    if (!sam3_encode_image(*m.sam3_state, *m.sam3, img)) {
        send_error(fd, hdr.req_id, "sam3 image encode failed");
        return;
    }
    double const encode_seconds = (ggml_time_us() - t_enc0) / 1e6;

    json parts = json::array();
    json decode_seconds = json::array();
    std::vector<uint8_t> masks;
    for (auto const& prompt : prompts) {
        sam3_pcs_params pcs;
        pcs.text_prompt = prompt;
        pcs.score_threshold = threshold;
        pcs.nms_threshold = nms;
        int64_t const t0 = ggml_time_us();
        sam3_result r = sam3_segment_pcs(*m.sam3_state, *m.sam3, pcs);
        decode_seconds.push_back((ggml_time_us() - t0) / 1e6);

        std::stable_sort(
            r.detections.begin(), r.detections.end(),
            [](sam3_detection const& a, sam3_detection const& b) { return a.score > b.score; });
        int kept = 0;
        for (auto const& d : r.detections) {
            if (max_instances > 0 && kept >= max_instances) {
                break;
            }
            if (d.mask.width != w || d.mask.height != h || d.mask.data.size() != npix) {
                send_error(fd, hdr.req_id, "sam3 returned a mask at an unexpected resolution");
                return;
            }
            size_t area = 0;
            for (uint8_t v : d.mask.data) {
                area += v ? 1 : 0;
            }
            if (area == 0) {
                continue; // a box with no pixels animates nothing
            }
            parts.push_back(json{
                {"name", prompt},
                {"score", d.score},
                {"box", {d.box.x0, d.box.y0, d.box.x1, d.box.y1}},
                {"area", (uint64_t)area}});
            masks.insert(masks.end(), d.mask.data.begin(), d.mask.data.end());
            ++kept;
        }
    }

    json resp = {
        {"width", w},
        {"height", h},
        {"parts", std::move(parts)},
        {"encode_seconds", encode_seconds},
        {"decode_seconds", std::move(decode_seconds)}};
    std::string rs = resp.dump();
    std::vector<uint8_t> out(4 + rs.size() + masks.size());
    uint32_t const rlen = (uint32_t)rs.size();
    std::memcpy(out.data(), &rlen, 4);
    std::memcpy(out.data() + 4, rs.data(), rs.size());
    std::memcpy(out.data() + 4 + rs.size(), masks.data(), masks.size());
    send_frame(fd, Frame::PARTS_RESP, hdr.req_id, out.data(), out.size());
}

} // namespace

int run_worker(ServerConfig const& cfg) {
    int fd = cfg.worker_fd;

    // BiRefNet's Swin attention has head_dim=32: flash-attn works but is slower
    // than the F16 encoder here. Overridable from the container env.
    ::setenv("VISP_FLASH_ATTENTION", "0", 0);

    WorkerModels m;
    m.cfg = cfg;
    m.sam3_threads =
        cfg.n_threads > 0 ? cfg.n_threads : (int)std::max(1u, std::thread::hardware_concurrency());
    std::string backend_name;
    try {
        m.device = std::make_unique<backend_device>(backend_init(cfg.default_backend));
        if (cfg.n_threads > 0) {
            backend_set_n_threads(*m.device, cfg.n_threads);
        }
        backend_name = std::string(to_string(m.device->type()));
        // F16 encoder activations are near-lossless and cut both time and peak
        // VRAM, but only CUDA has the kernels: ggml's CPU backend has no F16
        // GGML_OP_NORM and Vulkan no F16 unary, and both hard-abort mid-graph.
        // Vulkan is backend_type::vulkan, not ::gpu, so it takes the force-off
        // branch.
        //
        // HARD DEPENDENCY on three ggml CUDA pieces the v0.18 consolidation
        // rewrite dropped, all re-ported on top of it (see depend/ggml):
        //   F16 SOFT_MAX and F16 ROLL — without either, the worker ABORTS on the
        //     first /remove (GGML_ASSERT(... == GGML_TYPE_F32)), seen as a 500.
        //   ggml_cuda_mul_mat_f16_dst — without it the graph completes and every
        //     surface signal looks right (HTTP 200, correct extent, RGB
        //     byte-identical, VRAM and latency on target) while the matte comes
        //     back BLANK: mean alpha 13 vs 181, alpha RMSE 203/255. The generic
        //     "non-F32 dst -> ggml_cuda_mul_mat_cublas" branch does run, but it
        //     is not an equivalent substitute.
        // Do not move depend/ggml back before those without forcing this off,
        // and gate any change here on the ALPHA channel against a reference
        // matte — the loud failure is a 500, but the quiet one is a blank cutout
        // that no status code, VRAM figure or RGB comparison will catch.
        if (m.device->type() == backend_type::gpu) {
            ::setenv("VISP_F16_ENCODER", "1", 0);
        } else {
            ::unsetenv("VISP_F16_ENCODER");
        }
        m.load_matting();
    } catch (std::exception const& ex) {
        std::string msg = std::string("worker load failed: ") + ex.what();
        send_frame(fd, Frame::ERR, 0, msg.data(), msg.size());
        return 1;
    }

    json hello = {{"backend", backend_name}, {"native_size", m.matting_native_size}};
    std::string hs = hello.dump();
    if (!send_frame(fd, Frame::HELLO, 0, hs.data(), hs.size())) {
        return 1;
    }
    fprintf(
        stderr, "[vision-worker] pid=%d ready on %s (matting native %d)\n", (int)getpid(),
        backend_name.c_str(), m.matting_native_size);

    for (;;) {
        FrameHeader hdr{};
        std::vector<uint8_t> payload;
        IpcError e;
        if (!recv_frame(fd, &hdr, &payload, &e)) {
            break; // parent closed / EOF
        }
        Frame ft = (Frame)hdr.type;
        if (ft == Frame::SHUTDOWN) {
            break;
        }
        try {
            if (ft == Frame::MATTE_REQ) {
                handle_matte(m, fd, hdr, payload);
            } else if (ft == Frame::DEPTH_REQ) {
                handle_depth(m, fd, hdr, payload);
            } else if (ft == Frame::SIGLIP_REQ) {
                std::string out = handle_siglip(m, payload).dump();
                send_frame(fd, Frame::SIGLIP_RESP, hdr.req_id, out.data(), out.size());
            } else if (ft == Frame::PARTS_REQ) {
                handle_parts(m, fd, hdr, payload);
            }
        } catch (std::exception const& ex) {
            send_error(fd, hdr.req_id, ex.what());
        }
    }
    return 0;
}

} // namespace vserver
