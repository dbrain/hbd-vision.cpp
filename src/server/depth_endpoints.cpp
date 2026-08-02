//
// Depth endpoint — Depth-Anything V2 (default) or Depth-Anything 3.
//
//   POST /depth   image bytes in -> JSON with a 16-bit greyscale depth PNG per
//                 image, plus a 16-bit confidence PNG when the model produces
//                 one. Always JSON: two maps cannot share one image/png body.
//
//   image / file / images   multipart file part(s); a raw body also works
//   gpu                     multipart field or query — placement target
//   backend                 str  cpu | gpu (default: server default)
//   model                   str  depth-anything-v2 (default) | depth-anything-3
//                                (short forms: v2 | v3)
//   process_res_short       int  V2 ONLY — floor on the SHORTEST side
//   process_res_long        int  DA3 ONLY — bound on the LONGEST side
//   format                  str  "json" (the only shape; accepted for uniformity)
//
// The two families do not agree on what an inference resolution is, so there is
// no `process_res`: V2's knob is a FLOOR on the shortest side (it never
// downscales below the input; native 518) and DA3's is a BOUND on the longest
// (native 504). One name for both would silently mean two different pictures, so
// each axis is named and the wrong one for the chosen model is a 400.
//
// Per-model output conventions, reported as `depth_polarity`:
//   depth-anything-v2  "disparity" — larger is NEARER, min/max normalized to
//                      [0,1] by the model pipeline, and no confidence map.
//   depth-anything-3   "distance"  — larger is FARTHER, unnormalized, plus a
//                      per-pixel confidence map.
// A consumer that ignores this reads every sprite inside out.
//
// Every PNG is min/max normalized over the map it encodes, and the pre-
// normalization range is reported per image so the raw values are recoverable:
//   value = depth_min + (px / 65535) * (depth_max - depth_min)
//

#include "server/endpoints.h"

#include "stb_image.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// stb_image_write's deflate compressor. Declared in the implementation, not the
// public header, but exported with C linkage from libstb.
extern "C" unsigned char* stbi_zlib_compress(
    unsigned char* data, int data_len, int* out_len, int quality);

using json = nlohmann::json;
using namespace visp;

namespace vserver {

namespace {

uint32_t crc32_bytes(unsigned char const* buf, size_t len, uint32_t crc) {
    static uint32_t table[256];
    static bool built = false;
    if (!built) {
        for (uint32_t n = 0; n < 256; ++n) {
            uint32_t c = n;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
            }
            table[n] = c;
        }
        built = true;
    }
    crc ^= 0xffffffffu;
    for (size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ buf[i]) & 0xff] ^ (crc >> 8);
    }
    return crc ^ 0xffffffffu;
}

void put_be32(std::vector<unsigned char>& v, uint32_t x) {
    v.push_back((unsigned char)(x >> 24));
    v.push_back((unsigned char)(x >> 16));
    v.push_back((unsigned char)(x >> 8));
    v.push_back((unsigned char)x);
}

void put_chunk(
    std::vector<unsigned char>& out, char const tag[4], unsigned char const* data, size_t len) {
    put_be32(out, (uint32_t)len);
    size_t const crc_at = out.size();
    out.insert(out.end(), tag, tag + 4);
    out.insert(out.end(), data, data + len);
    put_be32(out, crc32_bytes(out.data() + crc_at, 4 + len, 0));
}

// stb only writes 8-bit PNGs; depth needs 16. Greyscale, colour type 0, Sub
// filter (bpp 2) — smooth maps compress far better than with no filter.
std::vector<unsigned char> encode_png16_gray(uint16_t const* px, int w, int h) {
    size_t const row = (size_t)w * 2;
    std::vector<unsigned char> raw((row + 1) * (size_t)h);
    for (int y = 0; y < h; ++y) {
        unsigned char* dst = raw.data() + (row + 1) * (size_t)y;
        dst[0] = 1;
        uint16_t const* src = px + (size_t)y * w;
        for (int x = 0; x < w; ++x) {
            unsigned char hi = (unsigned char)(src[x] >> 8);
            unsigned char lo = (unsigned char)(src[x] & 0xff);
            unsigned char phi = x > 0 ? (unsigned char)(src[x - 1] >> 8) : 0;
            unsigned char plo = x > 0 ? (unsigned char)(src[x - 1] & 0xff) : 0;
            dst[1 + x * 2 + 0] = (unsigned char)(hi - phi);
            dst[1 + x * 2 + 1] = (unsigned char)(lo - plo);
        }
    }

    int zlen = 0;
    unsigned char* z = stbi_zlib_compress(raw.data(), (int)raw.size(), &zlen, 8);
    if (!z) {
        return {};
    }

    std::vector<unsigned char> out;
    out.reserve((size_t)zlen + 128);
    unsigned char const sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    out.insert(out.end(), sig, sig + 8);

    std::vector<unsigned char> ihdr;
    put_be32(ihdr, (uint32_t)w);
    put_be32(ihdr, (uint32_t)h);
    ihdr.push_back(16); // bit depth
    ihdr.push_back(0);  // colour type: greyscale
    ihdr.push_back(0);  // deflate
    ihdr.push_back(0);  // adaptive filtering
    ihdr.push_back(0);  // no interlace
    put_chunk(out, "IHDR", ihdr.data(), ihdr.size());
    put_chunk(out, "IDAT", z, (size_t)zlen);
    put_chunk(out, "IEND", nullptr, 0);
    free(z);
    return out;
}

struct Normalized {
    std::vector<uint16_t> px;
    float min = 0.f;
    float max = 0.f;
};

Normalized normalize_u16(float const* v, size_t n) {
    Normalized out;
    out.px.resize(n);
    float lo = v[0];
    float hi = v[0];
    for (size_t i = 1; i < n; ++i) {
        lo = std::fmin(lo, v[i]);
        hi = std::fmax(hi, v[i]);
    }
    out.min = lo;
    out.max = hi;
    float const scale = hi > lo ? 65535.f / (hi - lo) : 0.f;
    for (size_t i = 0; i < n; ++i) {
        float s = (v[i] - lo) * scale;
        s = s < 0.f ? 0.f : (s > 65535.f ? 65535.f : s);
        out.px[i] = (uint16_t)std::lround(s);
    }
    return out;
}

struct InputImage {
    std::unique_ptr<unsigned char, void (*)(void*)> px{nullptr, stbi_image_free};
    int w = 0;
    int h = 0;
};

// Depth models read a black field as a hole and a white one as sky, and either
// drags the subject's relief along with it. The game always sends a matted
// cut-out on transparency, and those pixels are discarded via alpha downstream,
// so mid-grey is the only neutral choice.
void composite_on_mid_grey(unsigned char* px, size_t npix) {
    for (size_t i = 0; i < npix; ++i) {
        unsigned a = px[i * 4 + 3];
        if (a == 255) {
            continue;
        }
        for (int k = 0; k < 3; ++k) {
            unsigned v = px[i * 4 + k];
            px[i * 4 + k] = (unsigned char)((v * a + 128u * (255u - a) + 127u) / 255u);
        }
        px[i * 4 + 3] = 255;
    }
}

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
    composite_on_mid_grey(out->px.get(), (size_t)w * h);
    return true;
}

backend_type parse_backend_or(std::string const& s, backend_type dflt) {
    if (s == "cpu") return backend_type::cpu;
    if (s == "gpu" || s == "cuda") return backend_type::gpu;
    return dflt;
}

struct DepthResult {
    std::vector<float> depth;
    std::vector<float> confidence; // empty when the model has none
};

// One inference round-trip. Caller holds st.mtx.
DepthResult depth_locked(
    ServerState& st, DepthModel model, int process_res, int w, int h, unsigned char const* rgba) {
    std::vector<uint8_t> req(16 + (size_t)w * h * 4);
    int32_t mm = (int32_t)model;
    int32_t pr = process_res;
    int32_t ww = w;
    int32_t hh = h;
    std::memcpy(req.data() + 0, &mm, 4);
    std::memcpy(req.data() + 4, &pr, 4);
    std::memcpy(req.data() + 8, &ww, 4);
    std::memcpy(req.data() + 12, &hh, 4);
    std::memcpy(req.data() + 16, rgba, (size_t)w * h * 4);

    std::vector<uint8_t> resp =
        st.request_locked(Frame::DEPTH_REQ, req.data(), req.size(), Frame::DEPTH_RESP);
    if (resp.size() < 12) {
        st.kill_worker_locked();
        throw std::runtime_error("short worker response");
    }
    int32_t rw;
    int32_t rh;
    int32_t n_maps;
    std::memcpy(&rw, resp.data() + 0, 4);
    std::memcpy(&rh, resp.data() + 4, 4);
    std::memcpy(&n_maps, resp.data() + 8, 4);
    size_t const npix = (size_t)rw * rh;
    if (rw != w || rh != h || (n_maps != 1 && n_maps != 2) ||
        12 + npix * 4 * (size_t)n_maps != resp.size()) {
        st.kill_worker_locked();
        throw std::runtime_error("worker response size mismatch");
    }
    DepthResult out;
    out.depth.resize(npix);
    std::memcpy(out.depth.data(), resp.data() + 12, npix * 4);
    if (n_maps == 2) {
        out.confidence.resize(npix);
        std::memcpy(out.confidence.data(), resp.data() + 12 + npix * 4, npix * 4);
    }
    return out;
}

struct ModelSpec {
    DepthModel id;
    char const* name;
    char const* polarity;
    char const* res_param;   // the ONE resolution parameter this model accepts
    char const* other_param; // the other model's — a 400, never reinterpreted
};

constexpr ModelSpec kV2{
    DepthModel::v2, "depth-anything-v2", "disparity", "process_res_short", "process_res_long"};
constexpr ModelSpec kV3{
    DepthModel::v3, "depth-anything-3", "distance", "process_res_long", "process_res_short"};

ModelSpec const* parse_model(std::string const& s) {
    if (s.empty() || s == "v2" || s == "depth-anything-v2") return &kV2;
    if (s == "v3" || s == "3" || s == "depth-anything-3") return &kV3;
    return nullptr;
}

} // namespace

void register_depth_routes(httplib::Server& srv, ServerState& st) {
    srv.Post("/depth", [&st](httplib::Request const& req, httplib::Response& res) {
        InFlightGuard guard(st);
        st.touch();
        ModelSpec const* spec = parse_model(req.get_param_value("model"));
        if (!spec) {
            send_json(
                res, 400,
                {{"error", "unknown model — use depth-anything-v2 or depth-anything-3"}});
            return;
        }
        std::string const& path =
            spec->id == DepthModel::v3 ? st.cfg.depth3_model : st.cfg.depth_model;
        if (path.empty()) {
            send_json(
                res, 503,
                {{"error",
                  std::string(spec->name) + " not configured (" +
                      (spec->id == DepthModel::v3 ? "DEPTH3_MODEL_PATH" : "DEPTH_MODEL_PATH") +
                      ")"}});
            return;
        }
        if (req.has_param("process_res")) {
            send_json(
                res, 400,
                {{"error",
                  "process_res is ambiguous across depth models — use process_res_short "
                  "(depth-anything-v2, floor on the shortest side) or process_res_long "
                  "(depth-anything-3, bound on the longest side)"}});
            return;
        }
        if (req.has_param(spec->other_param)) {
            send_json(
                res, 400,
                {{"error",
                  std::string(spec->other_param) + " is not a " + spec->name + " parameter — it " +
                      "takes " + spec->res_param}});
            return;
        }
        backend_type const backend =
            req.has_param("backend")
                ? parse_backend_or(req.get_param_value("backend"), st.cfg.default_backend)
                : st.cfg.default_backend;
        int const process_res = req.has_param(spec->res_param)
            ? std::atoi(req.get_param_value(spec->res_param).c_str())
            : 0;

        std::vector<std::string> bodies;
        if (req.is_multipart_form_data()) {
            for (char const* key : {"image", "file", "images"}) {
                for (auto const& f : req.get_file_values(key)) {
                    bodies.push_back(f.content);
                }
            }
        }
        if (bodies.empty() && !req.body.empty()) {
            bodies.push_back(req.body);
        }
        if (bodies.empty()) {
            send_json(res, 400, {{"error", "empty image body"}});
            return;
        }

        std::vector<InputImage> imgs(bodies.size());
        for (size_t i = 0; i < bodies.size(); ++i) {
            if (!decode_image(bodies[i], &imgs[i])) {
                send_json(res, 400, {{"error", "failed to decode image " + std::to_string(i)}});
                return;
            }
        }

        std::vector<DepthResult> preds(imgs.size());
        int64_t t0 = ggml_time_us();
        try {
            std::scoped_lock lk(st.mtx);
            take_request_gpu_locked(st, req);
            st.ensure_worker_locked(backend);
            for (size_t i = 0; i < imgs.size(); ++i) {
                preds[i] = depth_locked(
                    st, spec->id, process_res, imgs[i].w, imgs[i].h, imgs[i].px.get());
            }
        } catch (std::exception const& e) {
            send_json(res, 500, {{"error", e.what()}});
            return;
        }
        double infer_elapsed = (ggml_time_us() - t0) / 1e6;
        st.touch();

        json results = json::array();
        for (size_t i = 0; i < imgs.size(); ++i) {
            Normalized d = normalize_u16(preds[i].depth.data(), preds[i].depth.size());
            std::vector<unsigned char> dp = encode_png16_gray(d.px.data(), imgs[i].w, imgs[i].h);
            if (dp.empty()) {
                send_json(res, 500, {{"error", "failed to encode 16-bit PNG"}});
                return;
            }
            json out = {
                {"width", imgs[i].w},
                {"height", imgs[i].h},
                {"depth_png_base64", base64_encode(dp.data(), dp.size())},
                {"depth_min", d.min},
                {"depth_max", d.max}};
            // Absent, never defaulted: a constant here would read as a model that
            // was uniformly sure. V2 has no confidence head at all, and DA3's is
            // a per-image rank rather than a probability, so a consumer that
            // needs a blend weight should derive one (the alpha distance field
            // is what DA3's confidence tracked anyway).
            if (!preds[i].confidence.empty()) {
                Normalized c =
                    normalize_u16(preds[i].confidence.data(), preds[i].confidence.size());
                std::vector<unsigned char> cp =
                    encode_png16_gray(c.px.data(), imgs[i].w, imgs[i].h);
                if (cp.empty()) {
                    send_json(res, 500, {{"error", "failed to encode 16-bit PNG"}});
                    return;
                }
                out["confidence_png_base64"] = base64_encode(cp.data(), cp.size());
                out["confidence_min"] = c.min;
                out["confidence_max"] = c.max;
            }
            results.push_back(std::move(out));
        }

        res.set_header("X-Depth-Elapsed-Seconds", std::to_string(infer_elapsed));
        res.set_header("X-Depth-Backend", st.worker_backend_name);
        res.set_header("X-Depth-Model", spec->name);
        send_json(
            res, 200,
            {{"count", (int)results.size()},
             {"results", results},
             {"model", spec->name},
             {"depth_polarity", spec->polarity},
             {"elapsed_seconds", infer_elapsed},
             {"backend", st.worker_backend_name}});
    });
}

} // namespace vserver
