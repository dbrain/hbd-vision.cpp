//
// birefnet_server — worker-isolated HTTP background-removal (matting) microservice.
//
// Wraps vision.cpp's BiRefNet (RMBG-2.0 / BiRefNet-*) inference in a small
// httplib server. Unlike a plain resident server, the model runs in a forked
// WORKER process; the parent (HTTP front-end) never initializes a GPU backend,
// so it holds NO CUDA context and the worker can be SIGKILL'd to reclaim 100%
// of VRAM (true-0) between requests. Mirrors the qwen3-tts.cpp isolation shape.
//
// Key properties (user-locked deploy shape):
//   * Worker isolation — fork + execv `--worker <fd>`; SIGKILL on evict = true-0 VRAM.
//   * Per-request backend — `backend=cpu|gpu` (GPU default). Switching respawns the worker.
//   * Serialized execution — one inference at a time (a single warm worker, a mutex).
//   * No evict while busy/queued — in_flight atomic gates eviction; queued requests
//     keep the worker warm (no load/unload churn for overlapping requests).
//   * Evict policy — IDLE_UNLOAD_SECONDS=0 (default) evicts the instant the server goes
//     idle (drop to 0 after each request); >0 keeps the worker warm N idle seconds then
//     drops to 0.
//   * Batch — POST many images in one request -> one load, N inferences, then the normal
//     idle-evict. Amortizes the cold load across a batch.
//
// Endpoints
//   GET  /health              — liveness + model/backend/worker info
//   POST /remove              — image bytes in -> PNG out (single) OR JSON of base64
//                               PNGs (multipart with multiple images). All knobs via query.
//   POST /unload              — kill the worker (free all VRAM)  (alias /v1/admin/unload)
//   POST /v1/admin/unload     — kill the worker
//   POST /v1/admin/load       — (re)spawn the worker (default backend)
//
// /remove query parameters
//   backend       str    cpu | gpu  (default: server default, usually gpu) — per request
//   process_res   int    inference resolution (model native if omitted; 256..2048)
//   sensitivity   float  mask strength / threshold, [0,1], default 1.0 (passthrough)
//   mask_blur     int    box-blur radius applied to the mask, px (default 0)
//   mask_offset   int    dilate (+) / erode (-) the mask, px (default 0)
//   refine        0|1    edge-decontamination via foreground estimation (default 0)
//   invert        0|1    invert the mask (default 0)
//   bg_mode       str    "alpha" (RGBA cutout, default) | "color" (flatten onto bg_color)
//   bg_color      str    "#rrggbb" background for bg_mode=color (default #000000)
//   mask_only     0|1    return the grayscale mask instead of a cutout (default 0)
//   format        str    "png" (single, default) | "json" (force a JSON base64 response)
//
// Config (env, overridable by CLI flags)
//   MODEL_PATH / --model         path to the *.gguf BiRefNet model
//   HOST       / --host          bind host (default 0.0.0.0)
//   PORT       / --port          bind port (default 8898)
//   MATTING_BACKEND / --backend  cpu | gpu (default gpu) — the DEFAULT per-request backend
//   MATTING_THREADS / --threads  CPU thread count (default: ggml auto)
//   IDLE_UNLOAD_SECONDS          drop to 0 after N idle seconds (0 = immediately when idle)
//   MATTING_KEEP_RESIDENT        if set, never auto-evict (worker stays warm until /unload)
//

#include "visp/vision.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include "stb_image.h"
#include "stb_image_write.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::json;
using namespace visp;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct ServerConfig {
    std::string model_path;
    std::string host = "0.0.0.0";
    int         port = 8898;
    backend_type default_backend = backend_type::gpu; // default per-request backend
    int         n_threads = 0;            // 0 => ggml default (CPU worker)
    int         idle_unload_seconds = 0;  // 0 => evict the instant idle; >0 => warm N s
    bool        keep_resident = false;    // never auto-evict
    int         worker_fd = -1;           // >=0 => this process is a worker
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

static backend_type parse_backend_or(const std::string& s, backend_type dflt) {
    if (s == "cpu") return backend_type::cpu;
    if (s == "gpu" || s == "cuda") return backend_type::gpu;
    return dflt;
}

static ServerConfig load_config(int argc, char** argv) {
    ServerConfig c;
    c.model_path          = env_str("MODEL_PATH", "");
    c.host                = env_str("HOST", "0.0.0.0");
    c.port                = std::atoi(env_str("PORT", "8898").c_str());
    c.default_backend     = parse_backend_or(env_str("MATTING_BACKEND", "gpu"), backend_type::gpu);
    c.n_threads           = std::atoi(env_str("MATTING_THREADS", "0").c_str());
    c.idle_unload_seconds = std::atoi(env_str("IDLE_UNLOAD_SECONDS", "0").c_str());
    c.keep_resident       = env_truthy("MATTING_KEEP_RESIDENT");

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing argument after %s\n", what); exit(2); }
            return argv[++i];
        };
        if      (a == "--model"   || a == "-m") c.model_path = next("--model");
        else if (a == "--host")                 c.host = next("--host");
        else if (a == "--port"   || a == "-p")  c.port = std::atoi(next("--port").c_str());
        else if (a == "--backend"|| a == "-b")  c.default_backend = parse_backend_or(next("--backend"), backend_type::gpu);
        else if (a == "--threads"|| a == "-t")  c.n_threads = std::atoi(next("--threads").c_str());
        else if (a == "--idle-unload-seconds")  c.idle_unload_seconds = std::atoi(next("--idle").c_str());
        else if (a == "--keep-resident")        c.keep_resident = true;
        else if (a == "--worker")               c.worker_fd = std::atoi(next("--worker").c_str());
        else if (a == "--help"   || a == "-h") {
            printf("Usage: %s --model <gguf> [--port 8898] [--backend cpu|gpu]\n"
                   "          [--threads N] [--idle-unload-seconds N] [--keep-resident]\n"
                   "  env: MODEL_PATH HOST PORT MATTING_BACKEND MATTING_THREADS\n"
                   "       IDLE_UNLOAD_SECONDS MATTING_KEEP_RESIDENT\n", argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            exit(2);
        }
    }
    return c;
}

static long long now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// base64 (for JSON batch responses)
// ---------------------------------------------------------------------------

static std::string base64_encode(const unsigned char* data, size_t len) {
    static const char tbl[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 2 < len; i += 3) {
        uint32_t n = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | data[i + 2];
        out.push_back(tbl[(n >> 18) & 63]);
        out.push_back(tbl[(n >> 12) & 63]);
        out.push_back(tbl[(n >> 6) & 63]);
        out.push_back(tbl[n & 63]);
    }
    if (i < len) {
        uint32_t n = uint32_t(data[i]) << 16;
        if (i + 1 < len) n |= uint32_t(data[i + 1]) << 8;
        out.push_back(tbl[(n >> 18) & 63]);
        out.push_back(tbl[(n >> 12) & 63]);
        out.push_back(i + 1 < len ? tbl[(n >> 6) & 63] : '=');
        out.push_back('=');
    }
    return out;
}

// ---------------------------------------------------------------------------
// Worker IPC — 12-byte length-prefixed frames over an AF_UNIX socketpair.
// (Shape borrowed from qwen3-tts.cpp worker_ipc.)
// ---------------------------------------------------------------------------

enum class Frame : uint32_t {
    HELLO      = 0x01, // worker->parent: JSON {backend, native_size} once model is loaded
    INFER_REQ  = 0x20, // parent->worker: [i32 process_res][i32 w][i32 h][rgba8 w*h*4]
    INFER_RESP = 0x21, // worker->parent: [i32 w][i32 h][u8 mask w*h]
    INFER_ERR  = 0x2F, // worker->parent: utf8 error message
    SHUTDOWN   = 0x30, // parent->worker: clean exit
};

struct FrameHeader {
    uint32_t type;
    uint32_t len;     // payload bytes that follow (excludes header)
    uint32_t req_id;
};
static_assert(sizeof(FrameHeader) == 12, "FrameHeader must stay 12 bytes");

static constexpr size_t MAX_FRAME_PAYLOAD = 512ull * 1024 * 1024; // generous (raw RGBA @ 2048 ~16MB)

enum class IpcError { OK, EofClean, EofMidFrame, SocketError, PayloadTooBig };

static bool read_exact(int fd, void* buf, size_t len, IpcError* e = nullptr) {
    char* p = static_cast<char*>(buf);
    size_t got = 0;
    while (got < len) {
        ssize_t r = ::read(fd, p + got, len - got);
        if (r > 0) { got += (size_t)r; continue; }
        if (r == 0) { if (e) *e = got == 0 ? IpcError::EofClean : IpcError::EofMidFrame; return false; }
        if (errno == EINTR) continue;
        if (e) *e = IpcError::SocketError;
        return false;
    }
    if (e) *e = IpcError::OK;
    return true;
}

static bool write_exact(int fd, const void* buf, size_t len) {
    const char* p = static_cast<const char*>(buf);
    size_t sent = 0;
    while (sent < len) {
        ssize_t w = ::write(fd, p + sent, len - sent);
        if (w > 0) { sent += (size_t)w; continue; }
        if (w < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}

static bool send_frame(int fd, Frame type, uint32_t req_id, const void* payload, size_t len) {
    if (len > MAX_FRAME_PAYLOAD) return false;
    FrameHeader hdr{ (uint32_t)type, (uint32_t)len, req_id };
    if (len == 0) return write_exact(fd, &hdr, sizeof(hdr));
    iovec iov[2];
    iov[0].iov_base = &hdr;                       iov[0].iov_len = sizeof(hdr);
    iov[1].iov_base = const_cast<void*>(payload); iov[1].iov_len = len;
    size_t total = sizeof(hdr) + len, sent = 0;
    while (sent < total) {
        iovec cur[2]; int n = 0;
        size_t hskip = sent < sizeof(hdr) ? sent : sizeof(hdr);
        if (hskip < sizeof(hdr)) { cur[n].iov_base = (char*)&hdr + hskip; cur[n].iov_len = sizeof(hdr) - hskip; n++; }
        size_t pskip = sent > sizeof(hdr) ? sent - sizeof(hdr) : 0;
        cur[n].iov_base = (char*)payload + pskip; cur[n].iov_len = len - pskip; n++;
        ssize_t w = ::writev(fd, cur, n);
        if (w > 0) { sent += (size_t)w; continue; }
        if (w < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}

static bool recv_frame(int fd, FrameHeader* hdr, std::vector<uint8_t>* payload, IpcError* e = nullptr) {
    if (!read_exact(fd, hdr, sizeof(*hdr), e)) return false;
    if (hdr->len > MAX_FRAME_PAYLOAD) { if (e) *e = IpcError::PayloadTooBig; return false; }
    payload->resize(hdr->len);
    if (hdr->len == 0) { if (e) *e = IpcError::OK; return true; }
    return read_exact(fd, payload->data(), hdr->len, e);
}

// fork + execv self with `--worker <fd> --model <m> --backend <bt> [--threads N]`.
// Returns child pid, sets *out_fd to the parent end of the socketpair. -1 on failure.
static pid_t spawn_worker(const char* self_argv0, const ServerConfig& cfg,
                          backend_type bt, int* out_fd) {
    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return -1;
    int parent_fd = sv[0], worker_fd = sv[1];
    int flags = ::fcntl(parent_fd, F_GETFD);
    if (flags >= 0) ::fcntl(parent_fd, F_SETFD, flags | FD_CLOEXEC);

    pid_t pid = ::fork();
    if (pid < 0) { ::close(sv[0]); ::close(sv[1]); return -1; }
    if (pid == 0) {
        ::close(parent_fd);
        char fdbuf[16]; std::snprintf(fdbuf, sizeof(fdbuf), "%d", worker_fd);
        char thbuf[16]; std::snprintf(thbuf, sizeof(thbuf), "%d", cfg.n_threads);
        const char* btstr = (bt == backend_type::cpu) ? "cpu" : "gpu";
        std::vector<std::string> a = { self_argv0, "--worker", fdbuf,
                                       "--model", cfg.model_path,
                                       "--backend", btstr };
        if (cfg.n_threads > 0) { a.push_back("--threads"); a.push_back(thbuf); }
        std::vector<char*> argv_p;
        for (auto& s : a) argv_p.push_back(s.data());
        argv_p.push_back(nullptr);
        ::execv(self_argv0, argv_p.data());
        std::fprintf(stderr, "execv failed: %s\n", strerror(errno));
        ::_exit(127);
    }
    ::close(worker_fd);
    if (out_fd) *out_fd = parent_fd;
    return pid;
}

// ---------------------------------------------------------------------------
// Worker process — owns the GPU/CPU backend + model, serves inference frames.
// ---------------------------------------------------------------------------

static int run_worker(const ServerConfig& cfg) {
    int fd = cfg.worker_fd;
    backend_type bt = cfg.default_backend;

    // BiRefNet's Swin attention has head_dim=32, which crashes the CUDA flash-attn
    // kernel (fattn.cu). Default FA off (still overridable via the env var) so the
    // worker matches the verified-correct CLI config. Must be set before backend init.
    ::setenv("VISP_FLASH_ATTENTION", "0", 0);

    std::unique_ptr<backend_device> backend;
    std::unique_ptr<birefnet_model> model;
    int native_size = 0;
    std::string backend_name;
    try {
        backend = std::make_unique<backend_device>(backend_init(bt));
        if (cfg.n_threads > 0) backend_set_n_threads(*backend, cfg.n_threads);
        backend_name = std::string(to_string(backend->type()));
        model = std::make_unique<birefnet_model>(birefnet_load_model(cfg.model_path.c_str(), *backend));
        native_size = model->params.image_size;
    } catch (const std::exception& ex) {
        std::string msg = std::string("worker load failed: ") + ex.what();
        send_frame(fd, Frame::INFER_ERR, 0, msg.data(), msg.size());
        return 1;
    }

    json hello = {{"backend", backend_name}, {"native_size", native_size}};
    std::string hs = hello.dump();
    if (!send_frame(fd, Frame::HELLO, 0, hs.data(), hs.size())) return 1;
    fprintf(stderr, "[matting-worker] pid=%d loaded on %s (native %d)\n",
            (int)getpid(), backend_name.c_str(), native_size);

    for (;;) {
        FrameHeader hdr{}; std::vector<uint8_t> payload; IpcError e;
        if (!recv_frame(fd, &hdr, &payload, &e)) break; // parent closed/EOF -> exit
        Frame ft = (Frame)hdr.type;
        if (ft == Frame::SHUTDOWN) break;
        if (ft != Frame::INFER_REQ) continue;

        // [i32 process_res][i32 w][i32 h][rgba8 ...]
        if (payload.size() < 12) {
            const char* m = "short infer payload";
            send_frame(fd, Frame::INFER_ERR, hdr.req_id, m, strlen(m));
            continue;
        }
        int32_t process_res, w, h;
        std::memcpy(&process_res, payload.data() + 0, 4);
        std::memcpy(&w,           payload.data() + 4, 4);
        std::memcpy(&h,           payload.data() + 8, 4);
        size_t need = 12 + (size_t)w * h * 4;
        if (w <= 0 || h <= 0 || payload.size() != need) {
            const char* m = "bad infer dimensions";
            send_frame(fd, Frame::INFER_ERR, hdr.req_id, m, strlen(m));
            continue;
        }
        try {
            if (native_size > 0)
                model->params.image_size = process_res > 0 ? process_res : native_size;
            i32x2 ext{w, h};
            image_view input(ext, image_format::rgba_u8, payload.data() + 12);
            image_data out = birefnet_compute(*model, input);
            const uint8_t* m = out.data.get();
            if (!m) {
                const char* em = "inference produced no mask";
                send_frame(fd, Frame::INFER_ERR, hdr.req_id, em, strlen(em));
                continue;
            }
            std::vector<uint8_t> resp(12 + (size_t)w * h);
            std::memcpy(resp.data() + 0, &w, 4);
            std::memcpy(resp.data() + 4, &h, 4);
            int32_t one = 1; std::memcpy(resp.data() + 8, &one, 4); // reserved
            std::memcpy(resp.data() + 12, m, (size_t)w * h);
            send_frame(fd, Frame::INFER_RESP, hdr.req_id, resp.data(), resp.size());
        } catch (const std::exception& ex) {
            std::string msg = ex.what();
            send_frame(fd, Frame::INFER_ERR, hdr.req_id, msg.data(), msg.size());
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Small image helpers (PNG codec via stb, post-processing) — parent side
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

static void mask_offset(std::vector<float>& mask, i32x2 ext, int r) {
    if (r == 0) return;
    std::vector<float> tmp(mask.size());
    if (r < 0) {
        image_view src(ext, span<const float>(mask.data(), mask.size()));
        image_span dst(ext, span<float>(tmp.data(), tmp.size()));
        image_erosion(src, dst, -r);
        mask.swap(tmp);
    } else {
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
// /remove parameters
// ---------------------------------------------------------------------------

struct RemoveParams {
    backend_type backend = backend_type::gpu;
    int   process_res = 0;     // 0 => model native
    float sensitivity = 1.0f;
    int   mask_blur   = 0;
    int   mask_offset = 0;
    bool  refine      = false;
    bool  invert      = false;
    std::string bg_mode = "alpha";
    float bg_color[3] = {0, 0, 0};
    bool  mask_only   = false;
    bool  force_json  = false;
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

static RemoveParams parse_remove_params(const httplib::Request& req, backend_type dflt) {
    RemoveParams p;
    p.backend     = req.has_param("backend") ? parse_backend_or(req.get_param_value("backend"), dflt) : dflt;
    p.process_res = qi(req, "process_res", 0);
    p.sensitivity = qf(req, "sensitivity", 1.0f);
    p.mask_blur   = qi(req, "mask_blur", 0);
    p.mask_offset = qi(req, "mask_offset", 0);
    p.refine      = qb(req, "refine", false);
    p.invert      = qb(req, "invert", false);
    p.mask_only   = qb(req, "mask_only", false);
    p.force_json  = req.get_param_value("format") == "json";
    if (req.has_param("bg_mode"))  p.bg_mode = req.get_param_value("bg_mode");
    if (req.has_param("bg_color")) parse_hex_color(req.get_param_value("bg_color"), p.bg_color);
    return p;
}

static void send_json(httplib::Response& res, int code, const json& body) {
    res.status = code;
    res.set_content(body.dump(), "application/json");
}

// ---------------------------------------------------------------------------
// Parent server state — manages the (single) worker, serializes inference,
// evicts when idle.
// ---------------------------------------------------------------------------

struct ServerState {
    ServerConfig cfg;
    std::string  self_path;

    std::mutex   mtx;                  // serializes worker access (one infer at a time)
    pid_t        worker_pid = -1;
    int          worker_fd  = -1;
    backend_type worker_bt  = backend_type::gpu;
    std::string  worker_backend_name;
    int          worker_native_size = 0;
    uint32_t     next_req_id = 1;

    std::atomic<int>       in_flight{0};      // running OR queued requests
    std::atomic<long long> last_request_ms{0};
    std::condition_variable idle_cv;

    void touch() { last_request_ms.store(now_ms()); }

    // caller holds mtx
    void kill_worker_locked() {
        if (worker_pid > 0) {
            ::kill(worker_pid, SIGKILL);
            int ws = 0; ::waitpid(worker_pid, &ws, 0);
            fprintf(stderr, "[matting] killed worker pid=%d -> VRAM reclaimed\n", (int)worker_pid);
        }
        if (worker_fd >= 0) { ::close(worker_fd); worker_fd = -1; }
        worker_pid = -1;
        worker_backend_name.clear();
        worker_native_size = 0;
    }

    bool worker_alive() const { return worker_pid > 0; }

    // caller holds mtx. Ensures a live worker on backend `bt`; respawns on mismatch.
    // throws std::runtime_error on failure.
    void ensure_worker_locked(backend_type bt) {
        if (worker_pid > 0 && worker_bt == bt) return;
        if (worker_pid > 0) {
            fprintf(stderr, "[matting] backend switch %s -> %s, respawning worker\n",
                    worker_backend_name.c_str(), bt == backend_type::cpu ? "cpu" : "gpu");
            kill_worker_locked();
        }
        int fd = -1;
        pid_t pid = spawn_worker(self_path.c_str(), cfg, bt, &fd);
        if (pid < 0) throw std::runtime_error("failed to spawn worker");
        // Wait for HELLO (or an error / EOF if load crashed).
        FrameHeader hdr{}; std::vector<uint8_t> payload; IpcError e;
        if (!recv_frame(fd, &hdr, &payload, &e) || (Frame)hdr.type != Frame::HELLO) {
            ::kill(pid, SIGKILL); int ws = 0; ::waitpid(pid, &ws, 0); ::close(fd);
            std::string emsg = "worker failed to load";
            if ((Frame)hdr.type == Frame::INFER_ERR && !payload.empty())
                emsg = std::string(payload.begin(), payload.end());
            throw std::runtime_error(emsg);
        }
        try {
            json h = json::parse(payload.begin(), payload.end());
            worker_backend_name = h.value("backend", std::string(bt == backend_type::cpu ? "CPU" : "GPU"));
            worker_native_size  = h.value("native_size", 0);
        } catch (...) {}
        worker_pid = pid; worker_fd = fd; worker_bt = bt;
        fprintf(stderr, "[matting] worker pid=%d ready on %s (native %d)\n",
                (int)pid, worker_backend_name.c_str(), worker_native_size);
    }

    // caller holds mtx. One inference round-trip. Returns mask (w*h u8). throws on error.
    std::vector<uint8_t> infer_locked(int process_res, int w, int h, const unsigned char* rgba,
                                      int* out_w, int* out_h) {
        std::vector<uint8_t> req(12 + (size_t)w * h * 4);
        int32_t pr = process_res, ww = w, hh = h;
        std::memcpy(req.data() + 0, &pr, 4);
        std::memcpy(req.data() + 4, &ww, 4);
        std::memcpy(req.data() + 8, &hh, 4);
        std::memcpy(req.data() + 12, rgba, (size_t)w * h * 4);
        uint32_t rid = next_req_id++;
        if (!send_frame(worker_fd, Frame::INFER_REQ, rid, req.data(), req.size())) {
            kill_worker_locked();
            throw std::runtime_error("worker send failed (worker died)");
        }
        FrameHeader hdr{}; std::vector<uint8_t> resp; IpcError e;
        if (!recv_frame(worker_fd, &hdr, &resp, &e)) {
            kill_worker_locked();
            throw std::runtime_error("worker recv failed (worker died)");
        }
        if ((Frame)hdr.type == Frame::INFER_ERR) {
            throw std::runtime_error(std::string(resp.begin(), resp.end()));
        }
        if ((Frame)hdr.type != Frame::INFER_RESP || resp.size() < 12) {
            kill_worker_locked();
            throw std::runtime_error("unexpected worker response");
        }
        int32_t rw, rh;
        std::memcpy(&rw, resp.data() + 0, 4);
        std::memcpy(&rh, resp.data() + 4, 4);
        if ((size_t)rw * rh + 12 != resp.size()) {
            kill_worker_locked();
            throw std::runtime_error("worker response size mismatch");
        }
        *out_w = rw; *out_h = rh;
        return std::vector<uint8_t>(resp.begin() + 12, resp.end());
    }
};

// RAII: count an in-flight request (running OR queued). On the last one leaving,
// nudge the idle watchdog so eviction can fire promptly.
struct InFlightGuard {
    ServerState& st;
    explicit InFlightGuard(ServerState& s) : st(s) { st.in_flight.fetch_add(1); }
    ~InFlightGuard() {
        if (st.in_flight.fetch_sub(1) == 1) st.idle_cv.notify_all();
    }
};

// Render one (mask, original rgba) pair to a PNG per the post-processing knobs.
static std::vector<unsigned char> render_png(const RemoveParams& p, std::vector<float>& mask,
                                             const unsigned char* px, int w, int h) {
    i32x2 ext{w, h};
    const size_t npix = (size_t)w * h;
    if (p.invert) for (auto& v : mask) v = 1.f - v;
    mask_sensitivity(mask, p.sensitivity);
    mask_offset(mask, ext, p.mask_offset);
    mask_blur(mask, ext, p.mask_blur);

    if (p.mask_only) {
        std::vector<unsigned char> g(npix);
        for (size_t i = 0; i < npix; ++i) g[i] = (unsigned char)std::lround(clamp01(mask[i]) * 255.f);
        return encode_png(g.data(), w, h, 1);
    }
    if (p.bg_mode == "color") {
        std::vector<unsigned char> rgb(npix * 3);
        for (size_t i = 0; i < npix; ++i) {
            float a = clamp01(mask[i]);
            for (int k = 0; k < 3; ++k) {
                float fg = px[i * 4 + k] / 255.f;
                float v  = fg * a + p.bg_color[k] * (1.f - a);
                rgb[i * 3 + k] = (unsigned char)std::lround(clamp01(v) * 255.f);
            }
        }
        return encode_png(rgb.data(), w, h, 3);
    }
    // alpha cutout (RGBA)
    std::vector<unsigned char> rgba(npix * 4);
    if (p.refine) {
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
    return encode_png(rgba.data(), w, h, 4);
}

// A decoded input image (RGBA8), owned.
struct InputImage {
    std::unique_ptr<unsigned char, void(*)(void*)> px{nullptr, stbi_image_free};
    int w = 0, h = 0;
};

static bool decode_image(const std::string& bytes, InputImage* out) {
    int w = 0, h = 0, c = 0;
    unsigned char* px = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc*>(bytes.data()), (int)bytes.size(), &w, &h, &c, 4);
    if (!px) return false;
    out->px = std::unique_ptr<unsigned char, void(*)(void*)>(px, stbi_image_free);
    out->w = w; out->h = h;
    return true;
}

int main(int argc, char** argv) {
    ggml_time_init();
    ::signal(SIGPIPE, SIG_IGN);

    ServerConfig cfg = load_config(argc, argv);
    if (cfg.model_path.empty()) {
        fprintf(stderr, "error: MODEL_PATH / --model is required\n");
        return 2;
    }

    // Worker role: load the model + serve inference frames, then exit.
    if (cfg.worker_fd >= 0) {
        return run_worker(cfg);
    }

    // ---- Parent (HTTP front-end). Holds NO backend / CUDA context. ----
    ServerState st;
    st.cfg = cfg;
    st.self_path = argv[0];
    st.worker_bt = cfg.default_backend;

    httplib::Server srv;
    srv.set_payload_max_length(256ull * 1024 * 1024);
    srv.set_read_timeout(300);
    srv.set_write_timeout(300);

    srv.set_pre_routing_handler([](const httplib::Request& req, httplib::Response&) {
        if ((req.method == "POST" || req.method == "PUT" || req.method == "PATCH") &&
            !req.has_header("Content-Length") &&
            req.get_header_value("Transfer-Encoding") != "chunked") {
            const_cast<httplib::Request&>(req).set_header("Content-Length", "0");
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        bool alive; std::string bname; int native;
        { std::scoped_lock lk(st.mtx); alive = st.worker_alive(); bname = st.worker_backend_name; native = st.worker_native_size; }
        json body = {
            {"status", "ok"},
            {"model", st.cfg.model_path},
            {"worker_alive", alive},
            {"backend", bname},
            {"native_size", native},
            {"in_flight", st.in_flight.load()},
            {"default_backend", st.cfg.default_backend == backend_type::cpu ? "cpu" : "gpu"},
            {"idle_unload_seconds", st.cfg.idle_unload_seconds},
            {"keep_resident", st.cfg.keep_resident},
        };
        send_json(res, 200, body);
    });

    auto do_unload = [&](httplib::Response& res) {
        bool was;
        { std::scoped_lock lk(st.mtx); was = st.worker_alive(); st.kill_worker_locked(); }
        send_json(res, 200, {{"unloaded", was}});
    };
    srv.Post("/unload",          [&](const httplib::Request&, httplib::Response& res){ do_unload(res); });
    srv.Post("/v1/admin/unload", [&](const httplib::Request&, httplib::Response& res){ do_unload(res); });
    srv.Post("/v1/admin/load",   [&](const httplib::Request&, httplib::Response& res){
        try { std::scoped_lock lk(st.mtx); st.ensure_worker_locked(st.cfg.default_backend); }
        catch (const std::exception& e) { send_json(res, 500, {{"error", e.what()}}); return; }
        send_json(res, 200, {{"worker_alive", true}, {"backend", st.worker_backend_name}});
    });

    srv.Post("/remove", [&](const httplib::Request& req, httplib::Response& res) {
        InFlightGuard guard(st);   // counts this request as live (gates eviction)
        st.touch();
        RemoveParams p = parse_remove_params(req, st.cfg.default_backend);

        // Collect input images: multipart files (image/file/images, possibly many),
        // else the raw body as a single image.
        std::vector<std::string> bodies;
        if (req.is_multipart_form_data()) {
            for (const char* key : {"image", "file", "images"}) {
                for (const auto& f : req.get_file_values(key)) bodies.push_back(f.content);
            }
        }
        if (bodies.empty() && !req.body.empty()) bodies.push_back(req.body);
        if (bodies.empty()) { send_json(res, 400, {{"error", "empty image body"}}); return; }

        // Decode all inputs first (parent side; cheap, no GPU).
        std::vector<InputImage> imgs(bodies.size());
        for (size_t i = 0; i < bodies.size(); ++i) {
            if (!decode_image(bodies[i], &imgs[i])) {
                send_json(res, 400, {{"error", "failed to decode image " + std::to_string(i)}});
                return;
            }
        }

        // One lock-hold: ensure the worker (per-request backend), run all N inferences
        // while warm, collect masks. Backend switch (if any) happens once here.
        std::vector<std::vector<uint8_t>> masks(imgs.size());
        int64_t t0 = ggml_time_us();
        try {
            std::scoped_lock lk(st.mtx);
            st.ensure_worker_locked(p.backend);
            for (size_t i = 0; i < imgs.size(); ++i) {
                int rw = 0, rh = 0;
                masks[i] = st.infer_locked(p.process_res, imgs[i].w, imgs[i].h, imgs[i].px.get(), &rw, &rh);
                if (rw != imgs[i].w || rh != imgs[i].h) {
                    send_json(res, 500, {{"error", "mask/extent mismatch"}});
                    return;
                }
            }
        } catch (const std::exception& e) {
            send_json(res, 500, {{"error", e.what()}});
            return;
        }
        double infer_elapsed = (ggml_time_us() - t0) / 1e6;
        st.touch();

        // Post-process + encode (outside the inference lock).
        std::vector<std::vector<unsigned char>> pngs(imgs.size());
        for (size_t i = 0; i < imgs.size(); ++i) {
            const size_t npix = (size_t)imgs[i].w * imgs[i].h;
            std::vector<float> mf(npix);
            for (size_t j = 0; j < npix; ++j) mf[j] = masks[i][j] / 255.f;
            pngs[i] = render_png(p, mf, imgs[i].px.get(), imgs[i].w, imgs[i].h);
        }

        res.set_header("X-BG-Elapsed-Seconds", std::to_string(infer_elapsed));
        res.set_header("X-BG-Backend", st.worker_backend_name);
        if (pngs.size() == 1 && !p.force_json) {
            res.set_content(reinterpret_cast<const char*>(pngs[0].data()), pngs[0].size(), "image/png");
        } else {
            json out = json::array();
            for (auto& png : pngs)
                out.push_back({{"png_base64", base64_encode(png.data(), png.size())}});
            send_json(res, 200, {{"count", (int)pngs.size()}, {"results", out},
                                 {"elapsed_seconds", infer_elapsed}, {"backend", st.worker_backend_name}});
        }
    });

    // Idle-evict watchdog: when the server goes idle (no running/queued requests),
    // wait `idle_unload_seconds` (0 = immediately) then SIGKILL the worker -> true-0 VRAM.
    // Skips entirely in keep-resident mode.
    if (!cfg.keep_resident) {
        std::thread([&st]() {
            const long long thresh_ms = (long long)st.cfg.idle_unload_seconds * 1000;
            std::mutex cvm;
            for (;;) {
                std::unique_lock<std::mutex> ul(cvm);
                // Wake on idle-notify, or poll at least every 1s.
                st.idle_cv.wait_for(ul, std::chrono::milliseconds(1000));
                if (st.in_flight.load() != 0) continue;
                long long idle = now_ms() - st.last_request_ms.load();
                if (idle < thresh_ms) continue;
                // Try to evict without blocking an arriving request.
                std::unique_lock<std::mutex> lk(st.mtx, std::try_to_lock);
                if (!lk.owns_lock()) continue;
                if (st.in_flight.load() == 0 && st.worker_alive() &&
                    now_ms() - st.last_request_ms.load() >= thresh_ms) {
                    st.kill_worker_locked();
                }
            }
        }).detach();
        fprintf(stderr, "[matting] idle-evict: %d s threshold (0 = evict when idle)\n",
                cfg.idle_unload_seconds);
    } else {
        fprintf(stderr, "[matting] keep-resident: worker stays warm until /unload\n");
    }

    fprintf(stderr, "[matting] listening on %s:%d (default backend %s, worker-isolated)\n",
            cfg.host.c_str(), cfg.port, cfg.default_backend == backend_type::cpu ? "cpu" : "gpu");
    if (!srv.listen(cfg.host.c_str(), cfg.port)) {
        fprintf(stderr, "error: failed to bind %s:%d\n", cfg.host.c_str(), cfg.port);
        return 1;
    }
    return 0;
}
