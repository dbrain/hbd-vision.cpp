//
// vision-server — worker-isolated HTTP service fronting visp's models.
//
// The parent (HTTP front-end) never initializes a compute backend, so it holds
// no GPU context; a forked worker owns the device and every model, which means
// SIGKILLing it reclaims 100% of VRAM (true-0). Matting, depth and SigLIP2 share
// that one worker, one device and one VRAM budget. A further family registers
// its routes the same way.
//
// Endpoints
//   GET  /health                      liveness + model/backend/worker info
//   GET  /v1/gpu/status               GPU residency announce for the koblem gate
//   POST /unload                      kill the worker (alias /v1/admin/unload)
//   POST /v1/admin/unload
//   POST /v1/admin/load               (re)spawn the worker
//   POST /remove                      matting (see matting_endpoints.cpp)
//   POST /depth                       Depth-Anything V2 / 3 (see depth_endpoints.cpp)
//   POST /parts                       SAM 3 text-prompted parts (see parts_endpoints.cpp)
//   POST /v1/embeddings               siglip2 (see siglip_endpoints.cpp)
//   POST /v1/classify
//   POST /v1/text_embeddings
//   POST /v1/classify_from_embeddings
//
// Config (env, overridable by CLI flags)
//   MODEL_PATH / --model                   BiRefNet matting *.gguf
//   DEPTH_MODEL_PATH / --depth-model       Depth-Anything-V2 *.gguf (/depth default)
//   DEPTH3_MODEL_PATH / --depth3-model     Depth-Anything-3 *.gguf (?model=depth-anything-3)
//   SIGLIP_MODEL_PATH / --siglip-model     SigLIP2 *.gguf
//   SIGLIP_TOKENIZER_PATH / --siglip-tokenizer   sentencepiece tokenizer.model
//   SAM3_MODEL_PATH / --sam3-model         SAM 3 *.ggml (/parts; CPU-only)
//   DEFAULT_MAX_NUM_PATCHES                siglip2 NaFlex budget (default 256)
//   HOST / --host, PORT / --port
//   MATTING_BACKEND / --backend            cpu | gpu — the DEFAULT per-request backend
//   MATTING_THREADS / --threads            CPU thread count
//   IDLE_UNLOAD_SECONDS                    drop to 0 after N idle seconds (0 = when idle)
//   MATTING_KEEP_RESIDENT                  never auto-evict
//   WORKER_DEFAULT_GPU                     CUDA_VISIBLE_DEVICES for un-targeted requests
//

#include "server/endpoints.h"
#include "server/worker.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include <signal.h>

using json = nlohmann::json;
using namespace visp;

namespace vserver {

void send_json(httplib::Response& res, int code, json const& body) {
    res.status = code;
    res.set_content(body.dump(), "application/json");
}

void take_request_gpu_locked(ServerState& st, httplib::Request const& req) {
    std::string g =
        req.has_file("gpu") ? req.get_file_value("gpu").content : req.get_param_value("gpu");
    if (!g.empty()) {
        st.next_gpu = g;
    }
}

namespace {

std::string env_str(char const* key, char const* dflt) {
    char const* v = std::getenv(key);
    return v ? std::string(v) : std::string(dflt);
}

bool env_truthy(char const* key) {
    char const* v = std::getenv(key);
    if (!v) {
        return false;
    }
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "on";
}

backend_type parse_backend_or(std::string const& s, backend_type dflt) {
    if (s == "cpu") return backend_type::cpu;
    if (s == "gpu" || s == "cuda") return backend_type::gpu;
    return dflt;
}

ServerConfig load_config(int argc, char** argv) {
    ServerConfig c;
    c.matting_model = env_str("MODEL_PATH", "");
    c.depth_model = env_str("DEPTH_MODEL_PATH", "");
    c.depth3_model = env_str("DEPTH3_MODEL_PATH", "");
    c.siglip_model = env_str("SIGLIP_MODEL_PATH", "");
    c.siglip_tokenizer = env_str("SIGLIP_TOKENIZER_PATH", "");
    c.sam3_model = env_str("SAM3_MODEL_PATH", "");
    c.siglip_default_max_num_patches = std::atoi(env_str("DEFAULT_MAX_NUM_PATCHES", "256").c_str());
    c.host = env_str("HOST", "0.0.0.0");
    c.port = std::atoi(env_str("PORT", "8898").c_str());
    c.default_backend = parse_backend_or(env_str("MATTING_BACKEND", "gpu"), backend_type::gpu);
    c.n_threads = std::atoi(env_str("MATTING_THREADS", "0").c_str());
    c.idle_unload_seconds = std::atoi(env_str("IDLE_UNLOAD_SECONDS", "0").c_str());
    c.keep_resident = env_truthy("MATTING_KEEP_RESIDENT");

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](char const* what) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing argument after %s\n", what);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--model" || a == "-m") c.matting_model = next("--model");
        else if (a == "--depth-model") c.depth_model = next("--depth-model");
        else if (a == "--depth3-model") c.depth3_model = next("--depth3-model");
        else if (a == "--siglip-model") c.siglip_model = next("--siglip-model");
        else if (a == "--siglip-tokenizer") c.siglip_tokenizer = next("--siglip-tokenizer");
        else if (a == "--sam3-model") c.sam3_model = next("--sam3-model");
        else if (a == "--default-max-num-patches")
            c.siglip_default_max_num_patches = std::atoi(next("--default-max-num-patches").c_str());
        else if (a == "--host") c.host = next("--host");
        else if (a == "--port" || a == "-p") c.port = std::atoi(next("--port").c_str());
        else if (a == "--backend" || a == "-b")
            c.default_backend = parse_backend_or(next("--backend"), backend_type::gpu);
        else if (a == "--threads" || a == "-t") c.n_threads = std::atoi(next("--threads").c_str());
        else if (a == "--idle-unload-seconds")
            c.idle_unload_seconds = std::atoi(next("--idle-unload-seconds").c_str());
        else if (a == "--keep-resident") c.keep_resident = true;
        else if (a == "--worker") c.worker_fd = std::atoi(next("--worker").c_str());
        else if (a == "--help" || a == "-h") {
            printf(
                "Usage: %s [--model <birefnet.gguf>] [--depth-model <da2.gguf>]\n"
                "          [--depth3-model <da3.gguf>]\n"
                "          [--siglip-model <siglip2.gguf>]\n"
                "          [--siglip-tokenizer <tokenizer.model>] [--sam3-model <sam3.ggml>]\n"
                "          [--port 8898]\n"
                "          [--backend cpu|gpu] [--threads N] [--idle-unload-seconds N]\n"
                "          [--keep-resident] [--default-max-num-patches N]\n"
                "  env: MODEL_PATH DEPTH_MODEL_PATH DEPTH3_MODEL_PATH SIGLIP_MODEL_PATH\n"
                "       SIGLIP_TOKENIZER_PATH SAM3_MODEL_PATH HOST PORT\n"
                "       MATTING_BACKEND MATTING_THREADS IDLE_UNLOAD_SECONDS\n"
                "       MATTING_KEEP_RESIDENT DEFAULT_MAX_NUM_PATCHES WORKER_DEFAULT_GPU\n",
                argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            exit(2);
        }
    }
    return c;
}

} // namespace
} // namespace vserver

int main(int argc, char** argv) {
    using namespace vserver;

    ggml_time_init();
    ::signal(SIGPIPE, SIG_IGN);

    ServerConfig cfg = load_config(argc, argv);
    if (cfg.matting_model.empty() && cfg.depth_model.empty() && cfg.depth3_model.empty() &&
        cfg.siglip_model.empty() && cfg.sam3_model.empty()) {
        fprintf(
            stderr,
            "error: at least one of MODEL_PATH / DEPTH_MODEL_PATH / DEPTH3_MODEL_PATH / "
            "SIGLIP_MODEL_PATH / SAM3_MODEL_PATH is required\n");
        return 2;
    }

    if (cfg.worker_fd >= 0) {
        return run_worker(cfg);
    }

    ServerState st;
    st.cfg = cfg;
    st.self_path = argv[0];
    if (char const* g = std::getenv("WORKER_DEFAULT_GPU")) {
        st.default_gpu = g;
        fprintf(stderr, "[vision] default GPU = %s\n", g);
    }
    st.worker_bt = cfg.default_backend;

    if ((!cfg.siglip_model.empty() || !cfg.depth_model.empty() || !cfg.depth3_model.empty() ||
         !cfg.sam3_model.empty()) &&
        cfg.idle_unload_seconds == 0 &&
        !cfg.keep_resident) {
        fprintf(
            stderr,
            "[vision] WARNING: siglip2/depth/sam3 enabled with IDLE_UNLOAD_SECONDS=0 — the worker "
            "is "
            "evicted the moment the server goes idle, so every request pays a full model "
            "reload. Set IDLE_UNLOAD_SECONDS>0 or MATTING_KEEP_RESIDENT=1.\n");
    }

    httplib::Server srv;
    srv.set_payload_max_length(256ull * 1024 * 1024);
    srv.set_read_timeout(300);
    srv.set_write_timeout(300);

    srv.set_pre_routing_handler([](httplib::Request const& req, httplib::Response&) {
        if ((req.method == "POST" || req.method == "PUT" || req.method == "PATCH") &&
            !req.has_header("Content-Length") &&
            req.get_header_value("Transfer-Encoding") != "chunked") {
            const_cast<httplib::Request&>(req).set_header("Content-Length", "0");
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    srv.Get("/health", [&st](httplib::Request const&, httplib::Response& res) {
        bool alive;
        std::string bname;
        int native;
        {
            std::scoped_lock lk(st.mtx);
            alive = st.worker_alive();
            bname = st.worker_backend_name;
            native = st.worker_native_size;
        }
        send_json(
            res, 200,
            {{"status", "ok"},
             {"model", st.cfg.matting_model},
             {"depth_model", st.cfg.depth_model},
             {"depth3_model", st.cfg.depth3_model},
             {"siglip_model", st.cfg.siglip_model},
             {"sam3_model", st.cfg.sam3_model},
             {"worker_alive", alive},
             {"model_loaded", alive},
             {"backend", bname},
             {"native_size", native},
             {"in_flight", st.in_flight.load()},
             {"default_backend", st.cfg.default_backend == backend_type::cpu ? "cpu" : "gpu"},
             {"idle_unload_seconds", st.cfg.idle_unload_seconds},
             {"keep_resident", st.cfg.keep_resident}});
    });

    srv.Get("/v1/gpu/status", [&st](httplib::Request const&, httplib::Response& res) {
        bool alive;
        std::string gpu;
        {
            std::scoped_lock lk(st.mtx);
            alive = st.worker_alive();
            gpu = st.worker_gpu;
        }
        json body = {{"loaded", alive}};
        if (alive && !gpu.empty()) {
            body["gpu"] = gpu;
        } else {
            body["gpu"] = nullptr;
        }
        send_json(res, 200, body);
    });

    auto do_unload = [&st](httplib::Response& res) {
        bool was;
        {
            std::scoped_lock lk(st.mtx);
            was = st.worker_alive();
            st.kill_worker_locked();
        }
        send_json(res, 200, {{"unloaded", was}, {"model_loaded", false}});
    };
    srv.Post("/unload", [&](httplib::Request const&, httplib::Response& res) { do_unload(res); });
    srv.Post(
        "/v1/admin/unload", [&](httplib::Request const&, httplib::Response& res) { do_unload(res); });
    srv.Post("/v1/admin/load", [&st](httplib::Request const&, httplib::Response& res) {
        try {
            std::scoped_lock lk(st.mtx);
            st.ensure_worker_locked(st.cfg.default_backend);
        } catch (std::exception const& e) {
            send_json(res, 500, {{"error", e.what()}});
            return;
        }
        send_json(
            res, 200,
            {{"worker_alive", true}, {"model_loaded", true}, {"backend", st.worker_backend_name}});
    });

    register_matting_routes(srv, st);
    register_depth_routes(srv, st);
    register_siglip_routes(srv, st);
    register_parts_routes(srv, st);

    // Idle-evict watchdog: when nothing is running or queued, wait
    // idle_unload_seconds then SIGKILL the worker -> true-0 VRAM.
    if (!cfg.keep_resident) {
        std::thread([&st]() {
            long long const thresh_ms = (long long)st.cfg.idle_unload_seconds * 1000;
            std::mutex cvm;
            for (;;) {
                std::unique_lock<std::mutex> ul(cvm);
                st.idle_cv.wait_for(ul, std::chrono::milliseconds(1000));
                if (st.in_flight.load() != 0) {
                    continue;
                }
                if (now_ms() - st.last_request_ms.load() < thresh_ms) {
                    continue;
                }
                // Try to evict without blocking an arriving request.
                std::unique_lock<std::mutex> lk(st.mtx, std::try_to_lock);
                if (!lk.owns_lock()) {
                    continue;
                }
                if (st.in_flight.load() == 0 && st.worker_alive() &&
                    now_ms() - st.last_request_ms.load() >= thresh_ms) {
                    st.kill_worker_locked();
                }
            }
        }).detach();
        fprintf(
            stderr, "[vision] idle-evict: %d s threshold (0 = evict when idle)\n",
            cfg.idle_unload_seconds);
    } else {
        fprintf(stderr, "[vision] keep-resident: worker stays warm until /unload\n");
    }

    fprintf(
        stderr, "[vision] listening on %s:%d (default backend %s, worker-isolated)\n",
        cfg.host.c_str(), cfg.port, cfg.default_backend == backend_type::cpu ? "cpu" : "gpu");
    if (!srv.listen(cfg.host.c_str(), cfg.port)) {
        fprintf(stderr, "error: failed to bind %s:%d\n", cfg.host.c_str(), cfg.port);
        return 1;
    }
    return 0;
}
