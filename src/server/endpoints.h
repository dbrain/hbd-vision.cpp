#pragma once

#include "server/server_state.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

namespace vserver {

void send_json(httplib::Response& res, int code, nlohmann::json const& body);

// Reads the per-request GPU target (gate placement) into st.next_gpu.
// Caller must hold st.mtx.
void take_request_gpu_locked(ServerState& st, httplib::Request const& req);

void register_matting_routes(httplib::Server& srv, ServerState& st);
void register_depth_routes(httplib::Server& srv, ServerState& st);
void register_siglip_routes(httplib::Server& srv, ServerState& st);

} // namespace vserver
