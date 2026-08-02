#include "server/server_ipc.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

namespace vserver {

namespace {

constexpr size_t MAX_FRAME_PAYLOAD = 512ull * 1024 * 1024;

bool read_exact(int fd, void* buf, size_t len, IpcError* e) {
    char* p = static_cast<char*>(buf);
    size_t got = 0;
    while (got < len) {
        ssize_t r = ::read(fd, p + got, len - got);
        if (r > 0) {
            got += (size_t)r;
            continue;
        }
        if (r == 0) {
            if (e) {
                *e = got == 0 ? IpcError::EofClean : IpcError::EofMidFrame;
            }
            return false;
        }
        if (errno == EINTR) {
            continue;
        }
        if (e) {
            *e = IpcError::SocketError;
        }
        return false;
    }
    if (e) {
        *e = IpcError::OK;
    }
    return true;
}

bool write_exact(int fd, void const* buf, size_t len) {
    char const* p = static_cast<char const*>(buf);
    size_t sent = 0;
    while (sent < len) {
        ssize_t w = ::write(fd, p + sent, len - sent);
        if (w > 0) {
            sent += (size_t)w;
            continue;
        }
        if (w < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
    return true;
}

} // namespace

bool send_frame(int fd, Frame type, uint32_t req_id, void const* payload, size_t len) {
    if (len > MAX_FRAME_PAYLOAD) {
        return false;
    }
    FrameHeader hdr{(uint32_t)type, (uint32_t)len, req_id};
    if (len == 0) {
        return write_exact(fd, &hdr, sizeof(hdr));
    }
    size_t total = sizeof(hdr) + len;
    size_t sent = 0;
    while (sent < total) {
        iovec cur[2];
        int n = 0;
        size_t hskip = sent < sizeof(hdr) ? sent : sizeof(hdr);
        if (hskip < sizeof(hdr)) {
            cur[n].iov_base = (char*)&hdr + hskip;
            cur[n].iov_len = sizeof(hdr) - hskip;
            n++;
        }
        size_t pskip = sent > sizeof(hdr) ? sent - sizeof(hdr) : 0;
        cur[n].iov_base = (char*)const_cast<void*>(payload) + pskip;
        cur[n].iov_len = len - pskip;
        n++;
        ssize_t w = ::writev(fd, cur, n);
        if (w > 0) {
            sent += (size_t)w;
            continue;
        }
        if (w < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
    return true;
}

bool recv_frame(int fd, FrameHeader* hdr, std::vector<uint8_t>* payload, IpcError* e) {
    if (!read_exact(fd, hdr, sizeof(*hdr), e)) {
        return false;
    }
    if (hdr->len > MAX_FRAME_PAYLOAD) {
        if (e) {
            *e = IpcError::PayloadTooBig;
        }
        return false;
    }
    payload->resize(hdr->len);
    if (hdr->len == 0) {
        if (e) {
            *e = IpcError::OK;
        }
        return true;
    }
    return read_exact(fd, payload->data(), hdr->len, e);
}

pid_t spawn_worker(
    char const* self_argv0,
    std::vector<std::string> const& extra_args,
    std::string const& gpu,
    int* out_fd) {
    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        return -1;
    }
    int parent_fd = sv[0];
    int worker_fd = sv[1];
    int flags = ::fcntl(parent_fd, F_GETFD);
    if (flags >= 0) {
        ::fcntl(parent_fd, F_SETFD, flags | FD_CLOEXEC);
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(sv[0]);
        ::close(sv[1]);
        return -1;
    }
    if (pid == 0) {
        ::close(parent_fd);
        if (!gpu.empty()) {
            ::setenv("CUDA_VISIBLE_DEVICES", gpu.c_str(), 1);
        }
        char fdbuf[16];
        std::snprintf(fdbuf, sizeof(fdbuf), "%d", worker_fd);
        std::vector<std::string> a = {self_argv0, "--worker", fdbuf};
        a.insert(a.end(), extra_args.begin(), extra_args.end());
        std::vector<char*> argv_p;
        argv_p.reserve(a.size() + 1);
        for (auto& s : a) {
            argv_p.push_back(s.data());
        }
        argv_p.push_back(nullptr);
        ::execv(self_argv0, argv_p.data());
        std::fprintf(stderr, "execv failed: %s\n", strerror(errno));
        ::_exit(127);
    }
    ::close(worker_fd);
    if (out_fd) {
        *out_fd = parent_fd;
    }
    return pid;
}

std::string base64_encode(unsigned char const* data, size_t len) {
    static char const tbl[] =
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
        if (i + 1 < len) {
            n |= uint32_t(data[i + 1]) << 8;
        }
        out.push_back(tbl[(n >> 18) & 63]);
        out.push_back(tbl[(n >> 12) & 63]);
        out.push_back(i + 1 < len ? tbl[(n >> 6) & 63] : '=');
        out.push_back('=');
    }
    return out;
}

long long now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

} // namespace vserver
