# SigLIP2 image/text embedding towers, sharing visp's backend_device + GGUF
# loading so a process can hold matting, depth and siglip2 against one device
# and one VRAM budget.

include(FetchContent)

set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_USE_BUILTIN_PROTOBUF ON CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG        v0.2.0)
# sentencepiece v0.2.0 still declares cmake_minimum_required(VERSION 3.1),
# which CMake >= 4 refuses outright.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(sentencepiece)
unset(CMAKE_POLICY_VERSION_MINIMUM)
foreach(spm_target sentencepiece-static sentencepiece_train-static spm_encode spm_decode
                   spm_normalize spm_train spm_export_vocab)
  if(TARGET ${spm_target})
    if(NOT MSVC)
      # v0.2.0 uses uint32_t in sentencepiece_processor.h without including
      # <cstdint>, which GCC >= 13 / libstdc++ no longer pulls in transitively.
      target_compile_options(${spm_target} PRIVATE -include cstdint -w)
    endif()
    set_target_properties(${spm_target} PROPERTIES EXCLUDE_FROM_ALL ON)
  endif()
endforeach()

add_library(siglip2 STATIC)
target_sources(siglip2 PRIVATE
  model.cpp
  preproc.cpp
  score.cpp
  text.cpp
  tokenizer.cpp
  vision.cpp
)
target_compile_features(siglip2 PUBLIC cxx_std_20)
target_compile_definitions(siglip2 PRIVATE ${VISP_ASSERT} ${VISP_DEFINITIONS})
target_compile_options(siglip2 PRIVATE ${VISP_WARNINGS} ${VISP_COMP_OPTIONS})
target_include_directories(siglip2
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
  PRIVATE ${sentencepiece_SOURCE_DIR}/src ${sentencepiece_BINARY_DIR}
)
target_link_libraries(siglip2
  PUBLIC visioncpp ggml
  PRIVATE sentencepiece-static ${VISP_FMT_LINK}
)
set_target_properties(siglip2 PROPERTIES POSITION_INDEPENDENT_CODE ON)
