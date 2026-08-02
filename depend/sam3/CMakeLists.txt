# sam3.cpp — SAM 3 promptable concept segmentation, built against THIS tree's
# ggml rather than the submodule's own pinned ggml (which we never initialize).
#
# The upstream single translation unit compiles the stb image/write
# implementations into itself. vision.cpp already ships those symbols in the
# `stb` target, so force-including the headers first trips their include guards
# and sam3.cpp emits declarations only. No source change, no duplicate symbols.

set(SAM3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sam3.cpp)

add_library(sam3 STATIC ${SAM3_DIR}/sam3.cpp)
target_include_directories(sam3 PUBLIC ${SAM3_DIR})
target_link_libraries(sam3 PUBLIC ggml stb)
target_compile_features(sam3 PUBLIC cxx_std_17)
target_compile_options(sam3 PRIVATE ${VISP_NO_WARNINGS})
set_target_properties(sam3 PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  target_compile_options(sam3 PRIVATE /FIstb_image.h /FIstb_image_write.h)
else()
  # SHELL: keeps each flag/argument pair together — CMake otherwise de-duplicates
  # the two bare `-include` tokens and leaves the headers as input files.
  target_compile_options(
    sam3 PRIVATE "SHELL:-include stb_image.h" "SHELL:-include stb_image_write.h")
endif()
