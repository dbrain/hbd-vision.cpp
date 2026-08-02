#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <sys/types.h>

namespace vserver {

// 12-byte length-prefixed frames over an AF_UNIX socketpair. Shape borrowed
// from qwen3-tts.cpp worker_ipc.
enum class Frame : uint32_t {
    HELLO = 0x01,      // worker->parent: JSON {backend, native_size} once loaded
    MATTE_REQ = 0x20,  // parent->worker: [i32 process_res][i32 w][i32 h][rgba8 w*h*4]
    MATTE_RESP = 0x21, // worker->parent: [i32 w][i32 h][i32 reserved][u8 mask w*h]
    ERR = 0x2F,        // worker->parent: utf8 error message
    SHUTDOWN = 0x30,   // parent->worker: clean exit
    SIGLIP_REQ = 0x40, // parent->worker: [u32 json_len][json][concatenated image bytes]
    SIGLIP_RESP = 0x41, // worker->parent: json
    DEPTH_REQ = 0x50,   // parent->worker: [i32 model][i32 image_size][i32 w][i32 h][rgba8 w*h*4]
    DEPTH_RESP = 0x51,  // worker->parent: [i32 w][i32 h][i32 n_maps][f32 depth][f32 conf if n_maps==2]
    PARTS_REQ = 0x60,   // parent->worker: [u32 json_len][json][rgba8 w*h*4]
    PARTS_RESP = 0x61   // worker->parent: [u32 json_len][json][u8 mask w*h per part, in order]
};

// Which depth family serves a DEPTH_REQ. V2 is the default: on flat-lit matted
// sprites it resolves 4-10x more relief than DA3-Small (HANDOFF §9.3). DA3 stays
// reachable because its port is validated to F16 noise and it is the only one
// with a confidence map.
enum class DepthModel : int32_t {
    v2 = 0,
    v3 = 1,
};

struct FrameHeader {
    uint32_t type;
    uint32_t len; // payload bytes following the header
    uint32_t req_id;
};
static_assert(sizeof(FrameHeader) == 12, "FrameHeader must stay 12 bytes");

enum class IpcError { OK, EofClean, EofMidFrame, SocketError, PayloadTooBig };

bool send_frame(int fd, Frame type, uint32_t req_id, void const* payload, size_t len);
bool recv_frame(int fd, FrameHeader* hdr, std::vector<uint8_t>* payload, IpcError* e = nullptr);

// fork + execv self with `--worker <fd>` plus `extra_args`. Returns the child
// pid and sets *out_fd to the parent end of the socketpair; -1 on failure.
// A non-empty `gpu` is exported as CUDA_VISIBLE_DEVICES in the child.
pid_t spawn_worker(
    char const* self_argv0,
    std::vector<std::string> const& extra_args,
    std::string const& gpu,
    int* out_fd);

std::string base64_encode(unsigned char const* data, size_t len);

long long now_ms();

} // namespace vserver
