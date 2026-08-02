//
// Parts endpoint — SAM 3 promptable concept segmentation (PCS).
//
//   POST /parts   image bytes + noun phrases in -> one instance mask per match.
//
//   image / file            multipart file part; a raw body also works
//   gpu                     multipart field or query — placement target
//   prompts                 JSON array of noun phrases, or a comma/newline
//                           separated list. "arm", "leg", "head", "wheel", ...
//   threshold               float, default 0.35 — PCS score floor
//   nms                     float, default 0.1
//   max_instances           int, cap per phrase (0 = uncapped)
//
// The default threshold is BELOW SAM 3's own 0.5 on purpose. Whole objects score
// 0.8-0.95, but a part inside one scores lower the less it stands out: measured
// on the 14-sprite corpus, a bear's forelimb is 0.67, a robot's side-bar arm is
// 0.48 and a wizard's arm inside a robe sleeve is 0.41 — all three located
// exactly right. At 0.5 those last two come back empty. Pair a low floor with
// max_instances set to the count the character sheet asked for ("2 arms"), which
// is a far better filter than a score cut: the sheet knows how many there are.
//
// One image, not N: the caller sends every phrase it wants in a single request
// because SAM 3 encodes the image once and then runs a cheap prompt-conditioned
// decode per phrase. Splitting the phrases across requests pays the whole encode
// again each time, so the API deliberately has no single-prompt shape.
//
// Masks come back as 8-bit greyscale PNGs (0 or 255) at the INPUT resolution,
// one per located instance, each tagged with the phrase that found it. Alpha on
// the input is ignored — SAM 3 wants the natural render, and the krea2 sprites
// keep their original background under the cut-out.
//
// Backend: sam3.cpp drives its own ggml backend rather than sharing the worker's,
// so it is handed cpu-or-not and picks the device itself. `gpu` still selects the
// card for worker placement, which is what keeps the gate's disabled-card
// enforcement working.
//

#include "server/endpoints.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace visp;

namespace vserver {

namespace {

struct InputImage {
    std::unique_ptr<unsigned char, void (*)(void*)> px{nullptr, stbi_image_free};
    int w = 0;
    int h = 0;
};

bool decode_image(std::string const& bytes, InputImage* out) {
    int w = 0;
    int h = 0;
    int c = 0;
    unsigned char* px = stbi_load_from_memory(
        reinterpret_cast<stbi_uc const*>(bytes.data()), (int)bytes.size(), &w, &h, &c, 4);
    if (!px) {
        return false;
    }
    out->px = std::unique_ptr<unsigned char, void (*)(void*)>(px, stbi_image_free);
    out->w = w;
    out->h = h;
    return true;
}

std::vector<unsigned char> encode_png8_gray(unsigned char const* px, int w, int h) {
    std::vector<unsigned char> out;
    stbi_write_png_to_func(
        [](void* ctx, void* data, int len) {
            auto* v = static_cast<std::vector<unsigned char>*>(ctx);
            v->insert(v->end(), (unsigned char*)data, (unsigned char*)data + len);
        },
        &out, w, h, 1, px, w);
    return out;
}

// "arm, leg, head" / one per line / a JSON array all mean the same thing.
std::vector<std::string> parse_prompts(std::string const& raw) {
    std::vector<std::string> out;
    std::string s = raw;
    auto trim = [](std::string v) {
        size_t a = v.find_first_not_of(" \t\r\n\"");
        size_t b = v.find_last_not_of(" \t\r\n\"");
        return a == std::string::npos ? std::string() : v.substr(a, b - a + 1);
    };
    if (!s.empty() && s.front() == '[') {
        json j = json::parse(s, nullptr, false);
        if (j.is_array()) {
            for (auto const& e : j) {
                std::string v = trim(e.is_string() ? e.get<std::string>() : e.dump());
                if (!v.empty()) {
                    out.push_back(v);
                }
            }
            return out;
        }
    }
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == ',' || s[i] == '\n') {
            std::string v = trim(s.substr(start, i - start));
            if (!v.empty()) {
                out.push_back(v);
            }
            start = i + 1;
        }
    }
    return out;
}

backend_type parse_backend_or(std::string const& s, backend_type dflt) {
    if (s == "cpu") return backend_type::cpu;
    if (s == "gpu" || s == "cuda") return backend_type::gpu;
    return dflt;
}

std::string field(httplib::Request const& req, char const* key) {
    if (req.has_file(key)) {
        return req.get_file_value(key).content;
    }
    return req.get_param_value(key);
}

struct PartsResult {
    json meta;
    std::vector<uint8_t> masks; // w*h per part, in `meta["parts"]` order
};

// One inference round-trip. Caller holds st.mtx.
PartsResult parts_locked(
    ServerState& st, json const& spec, int w, int h, unsigned char const* rgba) {
    std::string js = spec.dump();
    size_t const npix = (size_t)w * h;
    std::vector<uint8_t> reqbuf(4 + js.size() + npix * 4);
    uint32_t const jl = (uint32_t)js.size();
    std::memcpy(reqbuf.data(), &jl, 4);
    std::memcpy(reqbuf.data() + 4, js.data(), js.size());
    std::memcpy(reqbuf.data() + 4 + js.size(), rgba, npix * 4);

    std::vector<uint8_t> resp =
        st.request_locked(Frame::PARTS_REQ, reqbuf.data(), reqbuf.size(), Frame::PARTS_RESP);
    if (resp.size() < 4) {
        st.kill_worker_locked();
        throw std::runtime_error("short worker response");
    }
    uint32_t rl = 0;
    std::memcpy(&rl, resp.data(), 4);
    if (resp.size() < 4 + (size_t)rl) {
        st.kill_worker_locked();
        throw std::runtime_error("truncated worker response");
    }
    PartsResult out;
    out.meta = json::parse(resp.begin() + 4, resp.begin() + 4 + rl);
    size_t const n_parts = out.meta.at("parts").size();
    if (resp.size() != 4 + (size_t)rl + n_parts * npix) {
        st.kill_worker_locked();
        throw std::runtime_error("worker mask payload size mismatch");
    }
    out.masks.assign(resp.begin() + 4 + rl, resp.end());
    return out;
}

} // namespace

void register_parts_routes(httplib::Server& srv, ServerState& st) {
    srv.Post("/parts", [&st](httplib::Request const& req, httplib::Response& res) {
        InFlightGuard guard(st);
        st.touch();
        if (st.cfg.sam3_model.empty()) {
            send_json(res, 503, {{"error", "sam3 not configured (SAM3_MODEL_PATH / --sam3-model)"}});
            return;
        }
        std::vector<std::string> prompts = parse_prompts(field(req, "prompts"));
        if (prompts.empty()) {
            send_json(
                res, 400,
                {{"error",
                  "prompts is required — a JSON array or a comma/newline separated list of noun "
                  "phrases, e.g. [\"arm\",\"leg\",\"head\"]"}});
            return;
        }

        backend_type const backend = req.has_param("backend")
            ? parse_backend_or(req.get_param_value("backend"), st.cfg.default_backend)
            : st.cfg.default_backend;

        json spec = {{"prompts", prompts}};
        {
            std::string t = field(req, "threshold");
            std::string n = field(req, "nms");
            std::string mi = field(req, "max_instances");
            if (!t.empty()) spec["threshold"] = (float)std::atof(t.c_str());
            if (!n.empty()) spec["nms"] = (float)std::atof(n.c_str());
            if (!mi.empty()) spec["max_instances"] = std::atoi(mi.c_str());
        }

        std::string body;
        if (req.is_multipart_form_data()) {
            for (char const* key : {"image", "file"}) {
                for (auto const& f : req.get_file_values(key)) {
                    body = f.content;
                    break;
                }
                if (!body.empty()) {
                    break;
                }
            }
        }
        if (body.empty() && !req.body.empty()) {
            body = req.body;
        }
        if (body.empty()) {
            send_json(res, 400, {{"error", "empty image body"}});
            return;
        }

        InputImage img;
        if (!decode_image(body, &img)) {
            send_json(res, 400, {{"error", "failed to decode image"}});
            return;
        }
        spec["width"] = img.w;
        spec["height"] = img.h;

        PartsResult pr;
        int64_t t0 = ggml_time_us();
        try {
            std::scoped_lock lk(st.mtx);
            take_request_gpu_locked(st, req);
            st.ensure_worker_locked(backend);
            pr = parts_locked(st, spec, img.w, img.h, img.px.get());
        } catch (std::exception const& e) {
            send_json(res, 500, {{"error", e.what()}});
            return;
        }
        double infer_elapsed = (ggml_time_us() - t0) / 1e6;
        st.touch();

        size_t const npix = (size_t)img.w * img.h;
        json parts = json::array();
        for (size_t i = 0; i < pr.meta.at("parts").size(); ++i) {
            json p = pr.meta["parts"][i];
            std::vector<unsigned char> png =
                encode_png8_gray(pr.masks.data() + i * npix, img.w, img.h);
            if (png.empty()) {
                send_json(res, 500, {{"error", "failed to encode mask PNG"}});
                return;
            }
            p["mask_png_base64"] = base64_encode(png.data(), png.size());
            parts.push_back(std::move(p));
        }

        std::string const backend_name{to_string(backend)};
        res.set_header("X-Parts-Elapsed-Seconds", std::to_string(infer_elapsed));
        res.set_header("X-Parts-Backend", backend_name);
        send_json(
            res, 200,
            {{"width", img.w},
             {"height", img.h},
             {"prompts", prompts},
             {"count", (int)parts.size()},
             {"parts", std::move(parts)},
             {"model", "sam3"},
             {"encode_seconds", pr.meta.value("encode_seconds", 0.0)},
             {"decode_seconds", pr.meta.value("decode_seconds", json::array())},
             {"elapsed_seconds", infer_elapsed},
             {"backend", backend_name}});
    });
}

} // namespace vserver
