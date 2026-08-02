#pragma once

#include "server/server_ipc.h"
#include "visp/ml.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

namespace vserver {

struct ServerConfig {
    std::string matting_model;
    std::string depth_model;  // Depth-Anything V2 — /depth's default
    std::string depth3_model; // Depth-Anything 3 — ?model=depth-anything-3
    std::string siglip_model;
    std::string siglip_tokenizer;
    int siglip_default_max_num_patches = 256;

    std::string host = "0.0.0.0";
    int port = 8898;
    visp::backend_type default_backend = visp::backend_type::gpu;
    int n_threads = 0;           // 0 => visp default
    int idle_unload_seconds = 0; // 0 => evict the instant idle
    bool keep_resident = false;
    int worker_fd = -1; // >=0 => this process is the worker
};

// Parent-side worker manager. One worker process owns the compute device and
// every model; requests are serialized through `mtx` so a single warm worker
// serves matting, depth and siglip against one VRAM budget.
struct ServerState {
    ServerConfig cfg;
    std::string self_path;

    std::mutex mtx;
    pid_t worker_pid = -1;
    int worker_fd = -1;
    visp::backend_type worker_bt = visp::backend_type::gpu;
    std::string worker_backend_name;
    std::string default_gpu; // WORKER_DEFAULT_GPU for un-targeted requests
    std::string next_gpu;    // pending per-request target (guarded by mtx)
    std::string worker_gpu;  // GPU the live worker is pinned to (guarded by mtx)
    int worker_native_size = 0;
    uint32_t next_req_id = 1;

    std::atomic<int> in_flight{0}; // running OR queued
    std::atomic<long long> last_request_ms{0};
    std::condition_variable idle_cv;

    void touch() { last_request_ms.store(now_ms()); }
    bool worker_alive() const { return worker_pid > 0; }

    // All three require the caller to hold mtx. They throw std::runtime_error.
    void kill_worker_locked();
    void ensure_worker_locked(visp::backend_type bt);
    std::vector<uint8_t> request_locked(
        Frame type, void const* payload, size_t len, Frame expect);
};

// Counts a request as live (running OR queued) so the idle watchdog can't
// evict underneath it, and nudges the watchdog when the last one leaves.
struct InFlightGuard {
    ServerState& st;
    explicit InFlightGuard(ServerState& s) : st(s) { st.in_flight.fetch_add(1); }
    ~InFlightGuard() {
        if (st.in_flight.fetch_sub(1) == 1) {
            st.idle_cv.notify_all();
        }
    }
};

} // namespace vserver
