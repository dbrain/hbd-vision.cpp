#pragma once

#include "server/server_state.h"

namespace vserver {

// Runs the GPU-owning child process: loads models on `cfg.default_backend`,
// then serves inference frames until the parent closes the socket. The parent
// never initializes a backend, so SIGKILLing this process reclaims all VRAM.
int run_worker(ServerConfig const& cfg);

} // namespace vserver
