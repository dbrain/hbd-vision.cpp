//
// SigLIP2 endpoints — wire-compatible with kobbler-vision's FastAPI service
// and with hbd-siglip2.cpp's siglip2-server, which koblem's api/src/vision.rs
// calls in production.
//
//   POST /v1/embeddings                multipart images -> {"embeddings": [[f32]]}
//   POST /v1/classify                  multipart images + prompts -> {"scores", ["logits"]}
//   POST /v1/text_embeddings           form prompts -> {"embeddings": [[f32]]}
//   POST /v1/classify_from_embeddings  JSON {image_embeddings, prompts} -> {"scores", ["logits"]}
//
// Scalar knobs (pooling, max_num_patches, return_logits, return_last_hidden)
// are read from the query string / urlencoded body ONLY, never from multipart
// parts. cpp-httplib files multipart fields under req.files and never under
// req.params, so the siglip2-server these endpoints replace has always ignored
// them when koblem sends them as multipart text — which it does. Reading them
// from req.files here would silently change every embedding koblem has already
// persisted, so the asymmetry is deliberate. `prompts` is the exception: it was
// always collected from both sources.
//

#include "server/endpoints.h"

#include <cstring>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace vserver {

namespace {

std::vector<std::string> collect_prompts(httplib::Request const& req) {
    std::vector<std::string> out;
    auto fr = req.files.equal_range("prompts");
    for (auto it = fr.first; it != fr.second; ++it) {
        out.push_back(it->second.content);
    }
    auto pr = req.params.equal_range("prompts");
    for (auto it = pr.first; it != pr.second; ++it) {
        out.push_back(it->second);
    }
    return out;
}

// [u32 json_len][json][concatenated image bytes]
std::vector<uint8_t> build_payload(json const& envelope, std::vector<std::string> const& images) {
    std::string const head = envelope.dump();
    size_t total = 4 + head.size();
    for (auto const& img : images) {
        total += img.size();
    }
    std::vector<uint8_t> buf(total);
    uint32_t len = (uint32_t)head.size();
    std::memcpy(buf.data(), &len, 4);
    std::memcpy(buf.data() + 4, head.data(), head.size());
    size_t off = 4 + head.size();
    for (auto const& img : images) {
        std::memcpy(buf.data() + off, img.data(), img.size());
        off += img.size();
    }
    return buf;
}

// Round-trips one siglip request through the worker, keeping it warm for the
// whole call. Throws std::runtime_error.
json call_worker(
    ServerState& st, httplib::Request const& req, json envelope,
    std::vector<std::string> const& images) {
    json sizes = json::array();
    for (auto const& img : images) {
        sizes.push_back(img.size());
    }
    envelope["image_sizes"] = sizes;
    std::vector<uint8_t> payload = build_payload(envelope, images);

    std::scoped_lock lk(st.mtx);
    take_request_gpu_locked(st, req);
    st.ensure_worker_locked(st.cfg.default_backend);
    std::vector<uint8_t> resp =
        st.request_locked(Frame::SIGLIP_REQ, payload.data(), payload.size(), Frame::SIGLIP_RESP);
    return json::parse(resp.begin(), resp.end());
}

bool require_configured(ServerState& st, httplib::Response& res) {
    if (st.cfg.siglip_model.empty()) {
        send_json(res, 503, {{"error", "siglip model not configured (SIGLIP_MODEL_PATH)"}});
        return false;
    }
    return true;
}

int param_max_num_patches(httplib::Request const& req, int dflt) {
    if (!req.has_param("max_num_patches")) {
        return dflt;
    }
    std::string const v = req.get_param_value("max_num_patches");
    return v.empty() ? dflt : std::atoi(v.c_str());
}

} // namespace

void register_siglip_routes(httplib::Server& srv, ServerState& st) {
    srv.Post("/v1/embeddings", [&st](httplib::Request const& req, httplib::Response& res) {
        InFlightGuard guard(st);
        st.touch();
        if (!require_configured(st, res)) {
            return;
        }
        auto files = req.get_file_values("images");
        if (files.empty()) {
            send_json(res, 400, {{"error", "No images provided"}});
            return;
        }
        std::string const pooling = req.get_param_value("pooling");
        if (!(pooling.empty() || pooling == "pooler" || pooling == "probe" || pooling == "mean")) {
            send_json(
                res, 400,
                {{"error", "pooling must be one of: pooler, probe, mean (got '" + pooling + "')"}});
            return;
        }
        if (req.get_param_value("return_last_hidden") == "true") {
            send_json(res, 501, {{"error", "return_last_hidden not implemented"}});
            return;
        }

        std::vector<std::string> images;
        images.reserve(files.size());
        for (auto const& f : files) {
            images.push_back(f.content);
        }
        json envelope = {
            {"op", "embed"},
            {"pooling", pooling.empty() ? "probe" : pooling},
            {"max_num_patches",
             param_max_num_patches(req, st.cfg.siglip_default_max_num_patches)}};
        try {
            send_json(res, 200, call_worker(st, req, std::move(envelope), images));
        } catch (std::exception const& e) {
            send_json(res, 500, {{"error", e.what()}});
        }
        st.touch();
    });

    srv.Post("/v1/text_embeddings", [&st](httplib::Request const& req, httplib::Response& res) {
        InFlightGuard guard(st);
        st.touch();
        if (!require_configured(st, res)) {
            return;
        }
        std::vector<std::string> prompts = collect_prompts(req);
        if (prompts.empty()) {
            send_json(res, 400, {{"error", "No prompts provided"}});
            return;
        }
        json envelope = {{"op", "text"}, {"prompts", prompts}};
        try {
            send_json(res, 200, call_worker(st, req, std::move(envelope), {}));
        } catch (std::exception const& e) {
            send_json(res, 500, {{"error", e.what()}});
        }
        st.touch();
    });

    srv.Post("/v1/classify", [&st](httplib::Request const& req, httplib::Response& res) {
        InFlightGuard guard(st);
        st.touch();
        if (!require_configured(st, res)) {
            return;
        }
        auto files = req.get_file_values("images");
        std::vector<std::string> prompts = collect_prompts(req);
        if (files.empty()) {
            send_json(res, 400, {{"error", "No images provided"}});
            return;
        }
        if (prompts.empty()) {
            send_json(res, 400, {{"error", "No prompts provided"}});
            return;
        }
        std::vector<std::string> images;
        images.reserve(files.size());
        for (auto const& f : files) {
            images.push_back(f.content);
        }
        json envelope = {
            {"op", "classify"},
            {"prompts", prompts},
            {"return_logits", req.get_param_value("return_logits") == "true"},
            {"max_num_patches",
             param_max_num_patches(req, st.cfg.siglip_default_max_num_patches)}};
        try {
            send_json(res, 200, call_worker(st, req, std::move(envelope), images));
        } catch (std::exception const& e) {
            send_json(res, 500, {{"error", e.what()}});
        }
        st.touch();
    });

    srv.Post(
        "/v1/classify_from_embeddings",
        [&st](httplib::Request const& req, httplib::Response& res) {
            InFlightGuard guard(st);
            st.touch();
            if (!require_configured(st, res)) {
                return;
            }
            json body;
            try {
                body = json::parse(req.body);
            } catch (std::exception const& e) {
                send_json(res, 400, {{"error", std::string("invalid JSON: ") + e.what()}});
                return;
            }
            if (!body.contains("image_embeddings") || !body.contains("prompts")) {
                send_json(res, 400, {{"error", "JSON must contain image_embeddings and prompts"}});
                return;
            }
            json envelope = {
                {"op", "cfe"},
                {"image_embeddings", body["image_embeddings"]},
                {"prompts", body["prompts"]},
                {"return_logits", body.value("return_logits", false)}};
            try {
                send_json(res, 200, call_worker(st, req, std::move(envelope), {}));
            } catch (std::exception const& e) {
                send_json(res, 500, {{"error", e.what()}});
            }
            st.touch();
        });
}

} // namespace vserver
