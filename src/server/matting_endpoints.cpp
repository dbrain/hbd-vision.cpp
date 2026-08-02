//
// Matting (BiRefNet / RMBG-2.0) endpoints.
//
//   POST /remove   image bytes in -> PNG out (single) OR JSON of base64 PNGs
//                  (multipart with multiple images). All knobs via query:
//     backend       str    cpu | gpu (default: server default) — per request
//     process_res   int    inference resolution (model native if omitted)
//     sensitivity   float  mask strength / threshold, [0,1], default 1.0
//     mask_blur     int    box-blur radius applied to the mask, px
//     mask_offset   int    dilate (+) / erode (-) the mask, px
//     refine        0|1    edge decontamination via foreground estimation
//     invert        0|1    invert the mask
//     bg_mode       str    "alpha" (RGBA cutout, default) | "color"
//     bg_color      str    "#rrggbb" background for bg_mode=color
//     mask_only     0|1    return the grayscale mask instead of a cutout
//     format        str    "png" (single, default) | "json"
//

#include "server/endpoints.h"

#include "visp/image.h"

#include "stb_image.h"
#include "stb_image_write.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace visp;

namespace vserver {

namespace {

void png_writer(void* ctx, void* data, int size) {
    auto* out = static_cast<std::vector<unsigned char>*>(ctx);
    auto* p = static_cast<unsigned char*>(data);
    out->insert(out->end(), p, p + size);
}

std::vector<unsigned char> encode_png(unsigned char const* data, int w, int h, int comp) {
    std::vector<unsigned char> out;
    stbi_write_png_to_func(png_writer, &out, w, h, comp, data, w * comp);
    return out;
}

float clamp01(float x) {
    return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

void mask_offset(std::vector<float>& mask, i32x2 ext, int r) {
    if (r == 0) {
        return;
    }
    std::vector<float> tmp(mask.size());
    if (r < 0) {
        image_view src(ext, span<float const>(mask.data(), mask.size()));
        image_span dst(ext, span<float>(tmp.data(), tmp.size()));
        image_erosion(src, dst, -r);
        mask.swap(tmp);
    } else {
        for (auto& v : mask) {
            v = 1.f - v;
        }
        image_view src(ext, span<float const>(mask.data(), mask.size()));
        image_span dst(ext, span<float>(tmp.data(), tmp.size()));
        image_erosion(src, dst, r);
        for (size_t i = 0; i < tmp.size(); ++i) {
            mask[i] = 1.f - tmp[i];
        }
    }
}

void mask_blur(std::vector<float>& mask, i32x2 ext, int r) {
    if (r <= 0) {
        return;
    }
    std::vector<float> tmp(mask.size());
    image_view src(ext, span<float const>(mask.data(), mask.size()));
    image_span dst(ext, span<float>(tmp.data(), tmp.size()));
    image_blur(src, dst, r);
    mask.swap(tmp);
}

void mask_sensitivity(std::vector<float>& mask, float s) {
    if (s >= 0.999f) {
        return;
    }
    float cut = 1.f - s;
    float inv = 1.f / std::max(s, 1e-4f);
    for (auto& v : mask) {
        v = clamp01((v - cut) * inv);
    }
}

bool parse_hex_color(std::string const& s, float rgb[3]) {
    std::string h = s;
    if (!h.empty() && h[0] == '#') {
        h = h.substr(1);
    }
    if (h.size() != 6) {
        return false;
    }
    auto hx = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (int i = 0; i < 3; ++i) {
        int hi = hx(h[2 * i]);
        int lo = hx(h[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            return false;
        }
        rgb[i] = float(hi * 16 + lo) / 255.f;
    }
    return true;
}

struct RemoveParams {
    backend_type backend = backend_type::gpu;
    int process_res = 0; // 0 => model native
    float sensitivity = 1.0f;
    int mask_blur = 0;
    int mask_offset = 0;
    bool refine = false;
    bool invert = false;
    std::string bg_mode = "alpha";
    float bg_color[3] = {0, 0, 0};
    bool mask_only = false;
    bool force_json = false;
};

int qi(httplib::Request const& r, char const* k, int d) {
    return r.has_param(k) ? std::atoi(r.get_param_value(k).c_str()) : d;
}
float qf(httplib::Request const& r, char const* k, float d) {
    return r.has_param(k) ? float(std::atof(r.get_param_value(k).c_str())) : d;
}
bool qb(httplib::Request const& r, char const* k, bool d) {
    if (!r.has_param(k)) {
        return d;
    }
    std::string v = r.get_param_value(k);
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

backend_type parse_backend_or(std::string const& s, backend_type dflt) {
    if (s == "cpu") return backend_type::cpu;
    if (s == "gpu" || s == "cuda") return backend_type::gpu;
    return dflt;
}

RemoveParams parse_remove_params(httplib::Request const& req, backend_type dflt) {
    RemoveParams p;
    p.backend =
        req.has_param("backend") ? parse_backend_or(req.get_param_value("backend"), dflt) : dflt;
    p.process_res = qi(req, "process_res", 0);
    p.sensitivity = qf(req, "sensitivity", 1.0f);
    p.mask_blur = qi(req, "mask_blur", 0);
    p.mask_offset = qi(req, "mask_offset", 0);
    p.refine = qb(req, "refine", false);
    p.invert = qb(req, "invert", false);
    p.mask_only = qb(req, "mask_only", false);
    p.force_json = req.get_param_value("format") == "json";
    if (req.has_param("bg_mode")) {
        p.bg_mode = req.get_param_value("bg_mode");
    }
    if (req.has_param("bg_color")) {
        parse_hex_color(req.get_param_value("bg_color"), p.bg_color);
    }
    return p;
}

std::vector<unsigned char> render_png(
    RemoveParams const& p, std::vector<float>& mask, unsigned char const* px, int w, int h) {
    i32x2 ext{w, h};
    size_t const npix = (size_t)w * h;
    if (p.invert) {
        for (auto& v : mask) {
            v = 1.f - v;
        }
    }
    mask_sensitivity(mask, p.sensitivity);
    mask_offset(mask, ext, p.mask_offset);
    mask_blur(mask, ext, p.mask_blur);

    if (p.mask_only) {
        std::vector<unsigned char> g(npix);
        for (size_t i = 0; i < npix; ++i) {
            g[i] = (unsigned char)std::lround(clamp01(mask[i]) * 255.f);
        }
        return encode_png(g.data(), w, h, 1);
    }
    if (p.bg_mode == "color") {
        std::vector<unsigned char> rgb(npix * 3);
        for (size_t i = 0; i < npix; ++i) {
            float a = clamp01(mask[i]);
            for (int k = 0; k < 3; ++k) {
                float fg = px[i * 4 + k] / 255.f;
                rgb[i * 3 + k] =
                    (unsigned char)std::lround(clamp01(fg * a + p.bg_color[k] * (1.f - a)) * 255.f);
            }
        }
        return encode_png(rgb.data(), w, h, 3);
    }
    std::vector<unsigned char> rgba(npix * 4);
    if (p.refine) {
        std::vector<float> img_f(npix * 4);
        for (size_t i = 0; i < npix * 4; ++i) {
            img_f[i] = px[i] / 255.f;
        }
        image_view img_v(ext, image_format::rgba_f32, img_f.data());
        image_view msk_v(ext, span<float const>(mask.data(), mask.size()));
        image_data fg = image_estimate_foreground(img_v, msk_v);
        float const* f = reinterpret_cast<float const*>(fg.data.get());
        for (size_t i = 0; i < npix; ++i) {
            for (int k = 0; k < 3; ++k) {
                rgba[i * 4 + k] = (unsigned char)std::lround(clamp01(f[i * 4 + k]) * 255.f);
            }
            rgba[i * 4 + 3] = (unsigned char)std::lround(clamp01(mask[i]) * 255.f);
        }
    } else {
        for (size_t i = 0; i < npix; ++i) {
            rgba[i * 4 + 0] = px[i * 4 + 0];
            rgba[i * 4 + 1] = px[i * 4 + 1];
            rgba[i * 4 + 2] = px[i * 4 + 2];
            rgba[i * 4 + 3] = (unsigned char)std::lround(clamp01(mask[i]) * 255.f);
        }
    }
    return encode_png(rgba.data(), w, h, 4);
}

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

// One inference round-trip. Caller holds st.mtx.
std::vector<uint8_t> matte_locked(
    ServerState& st, int process_res, int w, int h, unsigned char const* rgba, int* out_w,
    int* out_h) {
    std::vector<uint8_t> req(12 + (size_t)w * h * 4);
    int32_t pr = process_res;
    int32_t ww = w;
    int32_t hh = h;
    std::memcpy(req.data() + 0, &pr, 4);
    std::memcpy(req.data() + 4, &ww, 4);
    std::memcpy(req.data() + 8, &hh, 4);
    std::memcpy(req.data() + 12, rgba, (size_t)w * h * 4);

    std::vector<uint8_t> resp =
        st.request_locked(Frame::MATTE_REQ, req.data(), req.size(), Frame::MATTE_RESP);
    if (resp.size() < 12) {
        st.kill_worker_locked();
        throw std::runtime_error("short worker response");
    }
    int32_t rw;
    int32_t rh;
    std::memcpy(&rw, resp.data() + 0, 4);
    std::memcpy(&rh, resp.data() + 4, 4);
    if ((size_t)rw * rh + 12 != resp.size()) {
        st.kill_worker_locked();
        throw std::runtime_error("worker response size mismatch");
    }
    *out_w = rw;
    *out_h = rh;
    return std::vector<uint8_t>(resp.begin() + 12, resp.end());
}

} // namespace

void register_matting_routes(httplib::Server& srv, ServerState& st) {
    srv.Post("/remove", [&st](httplib::Request const& req, httplib::Response& res) {
        InFlightGuard guard(st);
        st.touch();
        RemoveParams p = parse_remove_params(req, st.cfg.default_backend);

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

        // Decode on the parent: cheap, and keeps it off the worker lock.
        std::vector<InputImage> imgs(bodies.size());
        for (size_t i = 0; i < bodies.size(); ++i) {
            if (!decode_image(bodies[i], &imgs[i])) {
                send_json(res, 400, {{"error", "failed to decode image " + std::to_string(i)}});
                return;
            }
        }

        // One lock-hold: ensure the worker (per-request backend), run all N
        // inferences while warm. A backend switch, if any, happens once here.
        std::vector<std::vector<uint8_t>> masks(imgs.size());
        int64_t t0 = ggml_time_us();
        try {
            std::scoped_lock lk(st.mtx);
            take_request_gpu_locked(st, req);
            st.ensure_worker_locked(p.backend);
            for (size_t i = 0; i < imgs.size(); ++i) {
                int rw = 0;
                int rh = 0;
                masks[i] =
                    matte_locked(st, p.process_res, imgs[i].w, imgs[i].h, imgs[i].px.get(), &rw, &rh);
                if (rw != imgs[i].w || rh != imgs[i].h) {
                    send_json(res, 500, {{"error", "mask/extent mismatch"}});
                    return;
                }
            }
        } catch (std::exception const& e) {
            send_json(res, 500, {{"error", e.what()}});
            return;
        }
        double infer_elapsed = (ggml_time_us() - t0) / 1e6;
        st.touch();

        std::vector<std::vector<unsigned char>> pngs(imgs.size());
        for (size_t i = 0; i < imgs.size(); ++i) {
            size_t const npix = (size_t)imgs[i].w * imgs[i].h;
            std::vector<float> mf(npix);
            for (size_t j = 0; j < npix; ++j) {
                mf[j] = masks[i][j] / 255.f;
            }
            pngs[i] = render_png(p, mf, imgs[i].px.get(), imgs[i].w, imgs[i].h);
        }

        res.set_header("X-BG-Elapsed-Seconds", std::to_string(infer_elapsed));
        res.set_header("X-BG-Backend", st.worker_backend_name);
        if (pngs.size() == 1 && !p.force_json) {
            res.set_content(
                reinterpret_cast<char const*>(pngs[0].data()), pngs[0].size(), "image/png");
        } else {
            json out = json::array();
            for (auto& png : pngs) {
                out.push_back({{"png_base64", base64_encode(png.data(), png.size())}});
            }
            send_json(
                res, 200,
                {{"count", (int)pngs.size()},
                 {"results", out},
                 {"elapsed_seconds", infer_elapsed},
                 {"backend", st.worker_backend_name}});
        }
    });
}

} // namespace vserver
