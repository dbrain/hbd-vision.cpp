//
// birefnet_server — resident HTTP background-removal (matting) microservice.
//
// Wraps vision.cpp's BiRefNet (RMBG-2.0 / BiRefNet-*) inference in a small
// httplib server, mirroring the style of the other ~/dev/*.cpp services
// (siglip2-server, sd-server): env+CLI config, a resident model that is kept
// warm for fast turnaround, an explicit unload endpoint, and an idle-unload
// watchdog.
//
// Endpoints
//   GET  /health              — liveness + model/backend info
//   POST /remove              — raw image bytes in -> PNG out (all knobs via query)
//   POST /unload              — free the model (alias of /v1/admin/unload)
//   POST /v1/admin/unload     — free the model
//   POST /v1/admin/load       — (re)load the model
//
// /remove query parameters (mirrors the reference bg_server.py contract)
//   process_res   int    inference resolution (model native if omitted; 256..2048)
//   sensitivity   float  mask strength / threshold, [0,1], default 1.0 (passthrough)
//   mask_blur     int    box-blur radius applied to the mask, px (default 0)
//   mask_offset   int    dilate (+) / erode (-) the mask, px (default 0)
//   refine        0|1    edge-decontamination via foreground estimation (default 0)
//   invert        0|1    invert the mask (default 0)
//   bg_mode       str    "alpha" (RGBA cutout, default) | "color" (flatten onto bg_color)
//   bg_color      str    "#rrggbb" background for bg_mode=color (default #000000)
//   mask_only     0|1    return the grayscale mask instead of a cutout (default 0)
//
// Response carries header  X-BG-Elapsed-Seconds  (inference+post-proc wall time).
//
// Config (env, overridable by CLI flags)
//   MODEL_PATH / --model         path to the *.gguf BiRefNet model
//   HOST       / --host          bind host (default 0.0.0.0)
//   PORT       / --port          bind port (default 8898)
//   MATTING_BACKEND / --backend  cpu | gpu | auto (default auto)
//   MATTING_THREADS / --threads  CPU thread count (default: ggml auto)
//   IDLE_UNLOAD_SECONDS          free the model after N idle seconds (0 = never)
//   LAZY_LOAD                    skip eager load at startup (load on first request)
//

#include "visp/vision.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

// stb implementations are provided by the linked `stb` static lib; we only
// need the declarations here.
#include "stb_image.h"
#include "stb_image_write.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
using namespace visp;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct ServerConfig {
    std::string model_path;
    std::string host = "0.0.0.0";
    int         port = 8898;
    std::optional<backend_type> backend; // nullopt => auto
    int         n_threads = 0;           // 0 => ggml default
    int         idle_unload_seconds = 0; // 0 => never
    bool        lazy_load = false;
};

static std::string env_str(const char* key, const char* dflt) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : std::string(dflt);
}
static bool env_truthy(const char* key) {
    const char* v = std::getenv(key);
    if (!v) return false;
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "on";
}

static std::optional<backend_type> parse_backend(const std::string& s) {
    if (s == "cpu") return backend_type::cpu;
    if (s == "gpu" || s == "cuda") return backend_type::gpu;
    return std::nullopt; // auto / unknown
}

static ServerConfig load_config(int argc, char** argv) {
    ServerConfig c;
    c.model_path          = env_str("MODEL_PATH", "");
    c.host                = env_str("HOST", "0.0.0.0");
    c.port                = std::atoi(env_str("PORT", "8898").c_str());
    c.backend             = parse_backend(env_str("MATTING_BACKEND", "auto"));
    c.n_threads           = std::atoi(env_str("MATTING_THREADS", "0").c_str());
    c.idle_unload_seconds = std::atoi(env_str("IDLE_UNLOAD_SECONDS", "0").c_str());
    c.lazy_load           = env_truthy("LAZY_LOAD");

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing argument after %s\n", what); exit(2); }
            return argv[++i];
        };
        if      (a == "--model"   || a == "-m") c.model_path = next("--model");
        else if (a == "--host")                 c.host = next("--host");
        else if (a == "--port"   || a == "-p")  c.port = std::atoi(next("--port").c_str());
        else if (a == "--backend"|| a == "-b")  c.backend = parse_backend(next("--backend"));
        else if (a == "--threads"|| a == "-t")  c.n_threads = std::atoi(next("--threads").c_str());
        else if (a == "--idle-unload-seconds")  c.idle_unload_seconds = std::atoi(next("--idle").c_str());
        else if (a == "--lazy-load")            c.lazy_load = true;
        else if (a == "--help"   || a == "-h") {
            printf("Usage: %s --model <gguf> [--port 8898] [--backend cpu|gpu|auto]\n"
                   "          [--threads N] [--idle-unload-seconds N] [--lazy-load]\n"
                   "  env: MODEL_PATH HOST PORT MATTING_BACKEND MATTING_THREADS\n"
                   "       IDLE_UNLOAD_SECONDS LAZY_LOAD\n", argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            exit(2);
        }
    }
    return c;
}

// ---------------------------------------------------------------------------
// Resident model state
// ---------------------------------------------------------------------------

static long long now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

struct ServerState {
    ServerConfig cfg;
    std::mutex   mtx;                       // serializes load/unload + inference
    std::unique_ptr<backend_device> backend;
    std::unique_ptr<birefnet_model> model;
    std::atomic<bool>      loaded{false};
    std::atomic<long long> last_request_ms{0};
    std::string            backend_name;
    int                    native_image_size = 0; // model's default inference res

    void touch() { last_request_ms.store(now_ms()); }

    // caller must hold mtx
    void load_locked() {
        if (loaded.load()) return;
        backend_type bt = cfg.backend.value_or(backend_type::gpu);
        if (!cfg.backend) {
            // auto: prefer gpu when available, else cpu
            bt = backend_is_available(backend_type::gpu) ? backend_type::gpu : backend_type::cpu;
        }
        backend = std::make_unique<backend_device>(backend_init(bt));
        if (cfg.n_threads > 0) backend_set_n_threads(*backend, cfg.n_threads);
        backend_name = std::string(to_string(backend->type()));
        model = std::make_unique<birefnet_model>(
            birefnet_load_model(cfg.model_path.c_str(), *backend));
        native_image_size = model->params.image_size;
        loaded.store(true);
        fprintf(stderr, "[matting] model loaded on %s from %s\n",
                backend_name.c_str(), cfg.model_path.c_str());
    }

    // caller must hold mtx
    bool unload_locked() {
        bool was = loaded.exchange(false);
        model.reset();
        backend.reset();
        if (was) fprintf(stderr, "[matting] model unloaded\n");
        return was;
    }
};

// ---------------------------------------------------------------------------
// Small image helpers (PNG codec via stb, post-processing)
// ---------------------------------------------------------------------------

static void png_writer(void* ctx, void* data, int size) {
    auto* out = static_cast<std::vector<unsigned char>*>(ctx);
    auto* p   = static_cast<unsigned char*>(data);
    out->insert(out->end(), p, p + size);
}
static std::vector<unsigned char> encode_png(const unsigned char* data, int w, int h, int comp) {
    std::vector<unsigned char> out;
    stbi_write_png_to_func(png_writer, &out, w, h, comp, data, w * comp);
    return out;
}

static float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

// dilate (+r) / erode (-r) an alpha_f32 mask in place via lib erosion.
static void mask_offset(std::vector<float>& mask, i32x2 ext, int r) {
    if (r == 0) return;
    std::vector<float> tmp(mask.size());
    if (r < 0) {
        image_view src(ext, span<const float>(mask.data(), mask.size()));
        image_span dst(ext, span<float>(tmp.data(), tmp.size()));
        image_erosion(src, dst, -r);
        mask.swap(tmp);
    } else {
        // dilation = invert -> erode -> invert
        for (auto& v : mask) v = 1.f - v;
        image_view src(ext, span<const float>(mask.data(), mask.size()));
        image_span dst(ext, span<float>(tmp.data(), tmp.size()));
        image_erosion(src, dst, r);
        for (size_t i = 0; i < tmp.size(); ++i) mask[i] = 1.f - tmp[i];
    }
}

static void mask_blur(std::vector<float>& mask, i32x2 ext, int r) {
    if (r <= 0) return;
    std::vector<float> tmp(mask.size());
    image_view src(ext, span<const float>(mask.data(), mask.size()));
    image_span dst(ext, span<float>(tmp.data(), tmp.size()));
    image_blur(src, dst, r);
    mask.swap(tmp);
}

// sensitivity in [0,1]: 1.0 = passthrough; < 1 raises the cutoff, mapping
// [1-s, 1] -> [0, 1] (values below the cutoff go to 0). A simple, monotone
// "mask strength" knob — to be A/B-tuned against the PyTorch reference later.
static void mask_sensitivity(std::vector<float>& mask, float s) {
    if (s >= 0.999f) return;
    float cut = 1.f - s;
    float inv = 1.f / std::max(s, 1e-4f);
    for (auto& v : mask) v = clamp01((v - cut) * inv);
}

static bool parse_hex_color(const std::string& s, float rgb[3]) {
    std::string h = s;
    if (!h.empty() && h[0] == '#') h = h.substr(1);
    if (h.size() != 6) return false;
    auto hx = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (int i = 0; i < 3; ++i) {
        int hi = hx(h[2 * i]), lo = hx(h[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        rgb[i] = float(hi * 16 + lo) / 255.f;
    }
    return true;
}

// ---------------------------------------------------------------------------
// /remove handler
// ---------------------------------------------------------------------------

struct RemoveParams {
    int   process_res = 0;     // 0 => model native
    float sensitivity = 1.0f;
    int   mask_blur   = 0;
    int   mask_offset = 0;
    bool  refine      = false;
    bool  invert      = false;
    std::string bg_mode = "alpha";
    float bg_color[3] = {0, 0, 0};
    bool  mask_only   = false;
};

static int qi(const httplib::Request& r, const char* k, int d) {
    return r.has_param(k) ? std::atoi(r.get_param_value(k).c_str()) : d;
}
static float qf(const httplib::Request& r, const char* k, float d) {
    return r.has_param(k) ? float(std::atof(r.get_param_value(k).c_str())) : d;
}
static bool qb(const httplib::Request& r, const char* k, bool d) {
    if (!r.has_param(k)) return d;
    std::string v = r.get_param_value(k);
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

static RemoveParams parse_remove_params(const httplib::Request& req) {
    RemoveParams p;
    p.process_res = qi(req, "process_res", 0);
    p.sensitivity = qf(req, "sensitivity", 1.0f);
    p.mask_blur   = qi(req, "mask_blur", 0);
    p.mask_offset = qi(req, "mask_offset", 0);
    p.refine      = qb(req, "refine", false);
    p.invert      = qb(req, "invert", false);
    p.mask_only   = qb(req, "mask_only", false);
    if (req.has_param("bg_mode"))  p.bg_mode = req.get_param_value("bg_mode");
    if (req.has_param("bg_color")) parse_hex_color(req.get_param_value("bg_color"), p.bg_color);
    return p;
}

static void send_json(httplib::Response& res, int code, const json& body) {
    res.status = code;
    res.set_content(body.dump(), "application/json");
}

int main(int argc, char** argv) {
    ggml_time_init();

    ServerState st;
    st.cfg = load_config(argc, argv);
    if (st.cfg.model_path.empty()) {
        fprintf(stderr, "error: MODEL_PATH / --model is required\n");
        return 2;
    }

    if (!st.cfg.lazy_load) {
        std::scoped_lock lk(st.mtx);
        st.load_locked();
    }

    httplib::Server srv;
    srv.set_payload_max_length(64ull * 1024 * 1024);
    srv.set_read_timeout(120);
    srv.set_write_timeout(120);

    // httplib v0.20 keep-alive: a body-less POST (e.g. /unload, /v1/admin/*)
    // without Content-Length can stall waiting for a body. Inject 0 so the
    // route fires immediately. (Same workaround as the siglip2/sd servers.)
    srv.set_pre_routing_handler([](const httplib::Request& req, httplib::Response&) {
        if ((req.method == "POST" || req.method == "PUT" || req.method == "PATCH") &&
            !req.has_header("Content-Length") &&
            req.get_header_value("Transfer-Encoding") != "chunked") {
            const_cast<httplib::Request&>(req).set_header("Content-Length", "0");
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        json body = {
            {"status", "ok"},
            {"model", st.cfg.model_path},
            {"model_loaded", st.loaded.load()},
            {"backend", st.backend_name},
        };
        send_json(res, 200, body);
    });

    auto do_unload = [&](httplib::Response& res) {
        bool was;
        { std::scoped_lock lk(st.mtx); was = st.unload_locked(); }
        send_json(res, 200, {{"unloaded", was}, {"model_loaded", st.loaded.load()}});
    };
    srv.Post("/unload",          [&](const httplib::Request&, httplib::Response& res){ do_unload(res); });
    srv.Post("/v1/admin/unload", [&](const httplib::Request&, httplib::Response& res){ do_unload(res); });
    srv.Post("/v1/admin/load",   [&](const httplib::Request&, httplib::Response& res){
        std::string err;
        try { std::scoped_lock lk(st.mtx); st.load_locked(); }
        catch (const std::exception& e) { send_json(res, 500, {{"error", e.what()}}); return; }
        send_json(res, 200, {{"model_loaded", st.loaded.load()}, {"backend", st.backend_name}});
    });

    srv.Post("/remove", [&](const httplib::Request& req, httplib::Response& res) {
        st.touch();
        RemoveParams p = parse_remove_params(req);

        // Input bytes: raw body (image/png etc) or multipart "image"/"file".
        const std::string* body = &req.body;
        std::string mp;
        if (req.is_multipart_form_data()) {
            for (const char* key : {"image", "file", "images"}) {
                if (req.has_file(key)) { mp = req.get_file_value(key).content; body = &mp; break; }
            }
        }
        if (body->empty()) { send_json(res, 400, {{"error", "empty image body"}}); return; }

        // Decode to RGBA8.
        int w = 0, h = 0, c = 0;
        unsigned char* px = stbi_load_from_memory(
            reinterpret_cast<const stbi_uc*>(body->data()), (int)body->size(), &w, &h, &c, 4);
        if (!px) { send_json(res, 400, {{"error", "failed to decode image"}}); return; }
        std::unique_ptr<unsigned char, void(*)(void*)> px_guard(px, stbi_image_free);
        i32x2 ext{w, h};
        const size_t npix = (size_t)w * h;

        int64_t t0 = ggml_time_us();
        std::vector<float> mask;
        try {
            std::scoped_lock lk(st.mtx);
            if (!st.loaded.load()) st.load_locked();

            // Resolution knob: override per request, else restore native.
            // birefnet_compute rebuilds the cached graph when the effective
            // extent changes (so the first call at a new res is slower).
            // Dynamic models (native image_size == -1) ignore process_res.
            if (st.native_image_size > 0)
                st.model->params.image_size = p.process_res > 0 ? p.process_res : st.native_image_size;

            image_view input(ext, image_format::rgba_u8, px);
            // birefnet_compute returns an alpha_u8 mask (0..255) already scaled
            // back to the original extent. Normalize to f32 [0,1] for post-proc.
            image_data out = birefnet_compute(*st.model, input);
            const uint8_t* m = out.data.get();
            if (!m) { send_json(res, 500, {{"error", "inference produced no mask"}}); return; }
            mask.resize(npix);
            for (size_t i = 0; i < npix; ++i) mask[i] = m[i] / 255.f;
        } catch (const std::exception& e) {
            send_json(res, 500, {{"error", e.what()}});
            return;
        }

        // --- post-processing (outside the inference lock) ---
        if (p.invert) for (auto& v : mask) v = 1.f - v;
        mask_sensitivity(mask, p.sensitivity);
        mask_offset(mask, ext, p.mask_offset);
        mask_blur(mask, ext, p.mask_blur);

        std::vector<unsigned char> png;

        if (p.mask_only) {
            std::vector<unsigned char> g(npix);
            for (size_t i = 0; i < npix; ++i) g[i] = (unsigned char)std::lround(clamp01(mask[i]) * 255.f);
            png = encode_png(g.data(), w, h, 1);
        } else if (p.bg_mode == "color") {
            std::vector<unsigned char> rgb(npix * 3);
            for (size_t i = 0; i < npix; ++i) {
                float a = clamp01(mask[i]);
                for (int k = 0; k < 3; ++k) {
                    float fg = px[i * 4 + k] / 255.f;
                    float v  = fg * a + p.bg_color[k] * (1.f - a);
                    rgb[i * 3 + k] = (unsigned char)std::lround(clamp01(v) * 255.f);
                }
            }
            png = encode_png(rgb.data(), w, h, 3);
        } else { // alpha cutout (RGBA)
            std::vector<unsigned char> rgba(npix * 4);
            if (p.refine) {
                // foreground decontamination at the mask border
                std::vector<float> img_f(npix * 4);
                for (size_t i = 0; i < npix * 4; ++i) img_f[i] = px[i] / 255.f;
                image_view img_v(ext, image_format::rgba_f32, img_f.data());
                image_view msk_v(ext, span<const float>(mask.data(), mask.size()));
                image_data fg = image_estimate_foreground(img_v, msk_v);
                const float* f = reinterpret_cast<const float*>(fg.data.get());
                for (size_t i = 0; i < npix; ++i) {
                    for (int k = 0; k < 3; ++k)
                        rgba[i * 4 + k] = (unsigned char)std::lround(clamp01(f[i * 4 + k]) * 255.f);
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
            png = encode_png(rgba.data(), w, h, 4);
        }

        double elapsed = (ggml_time_us() - t0) / 1e6;
        res.set_header("X-BG-Elapsed-Seconds", std::to_string(elapsed));
        res.set_content(reinterpret_cast<const char*>(png.data()), png.size(), "image/png");
    });

    // Idle-unload watchdog.
    if (st.cfg.idle_unload_seconds > 0) {
        std::thread([&st]() {
            const int secs = st.cfg.idle_unload_seconds;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(10));
                if (!st.loaded.load()) continue;
                long long idle = now_ms() - st.last_request_ms.load();
                if (idle >= (long long)secs * 1000) {
                    std::scoped_lock lk(st.mtx);
                    if (st.loaded.load() && now_ms() - st.last_request_ms.load() >= (long long)secs * 1000)
                        st.unload_locked();
                }
            }
        }).detach();
        fprintf(stderr, "[matting] idle-unload: %d s threshold, 10 s poll\n", st.cfg.idle_unload_seconds);
    }

    fprintf(stderr, "[matting] listening on %s:%d\n", st.cfg.host.c_str(), st.cfg.port);
    if (!srv.listen(st.cfg.host.c_str(), st.cfg.port)) {
        fprintf(stderr, "error: failed to bind %s:%d\n", st.cfg.host.c_str(), st.cfg.port);
        return 1;
    }
    return 0;
}
