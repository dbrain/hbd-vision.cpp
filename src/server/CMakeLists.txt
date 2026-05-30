# matting-server: resident HTTP background-removal microservice (BiRefNet).
#
# Fetches header-only cpp-httplib + nlohmann_json (matching the siglip2/sd
# server style). Links the in-tree visioncpp library, ggml, and stb (for the
# in-memory PNG decode/encode used on the HTTP boundary).

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

add_executable(matting-server)
target_sources(matting-server PRIVATE birefnet_server.cpp)
target_include_directories(matting-server PRIVATE ..)
target_compile_definitions(matting-server PRIVATE ${VISP_ASSERT} ${VISP_DEFINITIONS})
target_compile_options(matting-server PRIVATE ${VISP_WARNINGS} ${VISP_COMP_OPTIONS})
target_link_options(matting-server PRIVATE ${VISP_LINK_OPTIONS})
target_link_libraries(matting-server PRIVATE
  ggml visioncpp stb httplib::httplib nlohmann_json::nlohmann_json ${VISP_FMT_LINK})
set_target_properties(matting-server PROPERTIES CXX_STANDARD 20)
