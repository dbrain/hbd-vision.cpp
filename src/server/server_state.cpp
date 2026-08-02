#include "server/server_state.h"

#include "nlohmann/json.hpp"

#include <cstdio>
#include <stdexcept>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::json;

namespace vserver {

void ServerState::kill_worker_locked() {
    if (worker_pid > 0) {
        ::kill(worker_pid, SIGKILL);
        int ws = 0;
        ::waitpid(worker_pid, &ws, 0);
        fprintf(stderr, "[vision] killed worker pid=%d -> VRAM reclaimed\n", (int)worker_pid);
    }
    if (worker_fd >= 0) {
        ::close(worker_fd);
        worker_fd = -1;
    }
    worker_pid = -1;
    worker_backend_name.clear();
    worker_native_size = 0;
}

void ServerState::ensure_worker_locked(visp::backend_type bt) {
    std::string const want_gpu = next_gpu.empty() ? default_gpu : next_gpu;
    if (worker_pid > 0 && worker_bt == bt && worker_gpu == want_gpu) {
        return;
    }
    if (worker_pid > 0) {
        fprintf(
            stderr, "[vision] respawn worker (backend %s->%s, gpu '%s'->'%s')\n",
            worker_backend_name.c_str(), bt == visp::backend_type::cpu ? "cpu" : "gpu",
            worker_gpu.c_str(), want_gpu.c_str());
        kill_worker_locked();
    }

    std::vector<std::string> args = {
        "--backend", bt == visp::backend_type::cpu ? "cpu" : "gpu"};
    if (!cfg.matting_model.empty()) {
        args.push_back("--model");
        args.push_back(cfg.matting_model);
    }
    if (!cfg.depth_model.empty()) {
        args.push_back("--depth-model");
        args.push_back(cfg.depth_model);
    }
    if (!cfg.depth3_model.empty()) {
        args.push_back("--depth3-model");
        args.push_back(cfg.depth3_model);
    }
    if (!cfg.siglip_model.empty()) {
        args.push_back("--siglip-model");
        args.push_back(cfg.siglip_model);
    }
    if (!cfg.siglip_tokenizer.empty()) {
        args.push_back("--siglip-tokenizer");
        args.push_back(cfg.siglip_tokenizer);
    }
    if (cfg.n_threads > 0) {
        args.push_back("--threads");
        args.push_back(std::to_string(cfg.n_threads));
    }

    int fd = -1;
    pid_t pid = spawn_worker(self_path.c_str(), args, want_gpu, &fd);
    worker_gpu = want_gpu;
    if (pid < 0) {
        throw std::runtime_error("failed to spawn worker");
    }
    FrameHeader hdr{};
    std::vector<uint8_t> payload;
    IpcError e;
    if (!recv_frame(fd, &hdr, &payload, &e) || (Frame)hdr.type != Frame::HELLO) {
        ::kill(pid, SIGKILL);
        int ws = 0;
        ::waitpid(pid, &ws, 0);
        ::close(fd);
        std::string emsg = "worker failed to load";
        if ((Frame)hdr.type == Frame::ERR && !payload.empty()) {
            emsg = std::string(payload.begin(), payload.end());
        }
        throw std::runtime_error(emsg);
    }
    try {
        json h = json::parse(payload.begin(), payload.end());
        worker_backend_name =
            h.value("backend", std::string(bt == visp::backend_type::cpu ? "CPU" : "GPU"));
        worker_native_size = h.value("native_size", 0);
    } catch (...) {
    }
    worker_pid = pid;
    worker_fd = fd;
    worker_bt = bt;
    fprintf(
        stderr, "[vision] worker pid=%d ready on %s (matting native %d)\n", (int)pid,
        worker_backend_name.c_str(), worker_native_size);
}

std::vector<uint8_t> ServerState::request_locked(
    Frame type, void const* payload, size_t len, Frame expect) {
    uint32_t rid = next_req_id++;
    if (!send_frame(worker_fd, type, rid, payload, len)) {
        kill_worker_locked();
        throw std::runtime_error("worker send failed (worker died)");
    }
    FrameHeader hdr{};
    std::vector<uint8_t> resp;
    IpcError e;
    if (!recv_frame(worker_fd, &hdr, &resp, &e)) {
        kill_worker_locked();
        throw std::runtime_error("worker recv failed (worker died)");
    }
    if ((Frame)hdr.type == Frame::ERR) {
        throw std::runtime_error(std::string(resp.begin(), resp.end()));
    }
    if ((Frame)hdr.type != expect) {
        kill_worker_locked();
        throw std::runtime_error("unexpected worker response frame");
    }
    return resp;
}

} // namespace vserver
