# vision-server: resident HTTP microservice fronting visp's models (matting +
# Depth-Anything 3 + SigLIP2). One worker process, one backend device, one VRAM
# budget.
#
# Fetches header-only cpp-httplib + nlohmann_json. Links the in-tree visioncpp
# and siglip2 libraries, ggml, and stb (for the in-memory PNG/JPEG codecs used
# on the HTTP boundary).

include(FetchContent)

FetchContent_Declare(
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG        v0.20.0)
FetchContent_MakeAvailable(httplib)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(vision-server)
target_sources(vision-server PRIVATE
  depth_endpoints.cpp
  main.cpp
  matting_endpoints.cpp
  server_ipc.cpp
  server_state.cpp
  siglip_endpoints.cpp
  worker.cpp
)
target_include_directories(vision-server PRIVATE ..)
target_compile_definitions(vision-server PRIVATE ${VISP_ASSERT} ${VISP_DEFINITIONS})
target_compile_options(vision-server PRIVATE ${VISP_WARNINGS} ${VISP_COMP_OPTIONS})
target_link_options(vision-server PRIVATE ${VISP_LINK_OPTIONS})
target_link_libraries(vision-server PRIVATE
  ggml visioncpp siglip2 stb httplib::httplib nlohmann_json::nlohmann_json ${VISP_FMT_LINK})
set_target_properties(vision-server PROPERTIES CXX_STANDARD 20)
