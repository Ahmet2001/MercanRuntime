cmake_minimum_required(VERSION 3.16)
project(mercan VERSION 0.1.2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MERCAN_LLAMA_DIR "${CMAKE_SOURCE_DIR}/build/_deps/llama-src" CACHE PATH "Path to patched llama.cpp source")
option(MERCAN_BUILD_SHARED "Build libmercan as a shared library" OFF)
option(MERCAN_BUILD_SDK_TESTS "Build Mercan SDK ABI regression tests" OFF)

if(NOT EXISTS "${MERCAN_LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Patched llama.cpp not found at ${MERCAN_LLAMA_DIR}. Run scripts/prepare_llama.sh first or set MERCAN_LLAMA_DIR.")
endif()

# The downloadable Mercan CLI must not require separate llama/ggml shared libraries.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

add_subdirectory("${MERCAN_LLAMA_DIR}" "${CMAKE_BINARY_DIR}/llama" EXCLUDE_FROM_ALL)

set(MERCAN_RUNTIME_SOURCES
    runtime/libmercan/src/mercan.cpp
    core/model_loader.cpp
    core/generic_backend.cpp
    core/architecture_registry.cpp
    core/tokenizer_registry.cpp
    core/plugin_loader.cpp
    core/builtin_plugins.cpp
    architectures/nedolm/architecture.cpp
    architectures/nedolm/tokenizer.cpp
)

if(MERCAN_BUILD_SHARED)
    add_library(libmercan SHARED ${MERCAN_RUNTIME_SOURCES})
    target_compile_definitions(libmercan PRIVATE MERCAN_BUILD_SHARED=1)
else()
    add_library(libmercan STATIC ${MERCAN_RUNTIME_SOURCES})
endif()
set_target_properties(libmercan PROPERTIES OUTPUT_NAME mercan)
target_include_directories(libmercan
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime/libmercan/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        "${CMAKE_SOURCE_DIR}/core"
        "${MERCAN_LLAMA_DIR}/include"
        "${MERCAN_LLAMA_DIR}/ggml/include")
target_link_libraries(libmercan PUBLIC llama ${CMAKE_DL_LIBS})

add_executable(mercan cli/main.cpp)
target_link_libraries(mercan PRIVATE libmercan)
if(GGML_CUDA)
    target_compile_definitions(mercan PRIVATE MERCAN_CUDA_BUILD=1)
endif()

if(MERCAN_BUILD_SDK_TESTS)
    enable_testing()
    add_executable(mercan-sdk-prefix-test tests/sdk_prefix_compat.cpp)
    target_link_libraries(mercan-sdk-prefix-test PRIVATE libmercan)
    add_test(NAME mercan-sdk-prefix-compat COMMAND mercan-sdk-prefix-test)

    add_executable(mercan-tensor-abi-test tests/tensor_abi_compile.cpp)
    target_include_directories(mercan-tensor-abi-test PRIVATE runtime/libmercan/include)
    add_test(NAME mercan-tensor-abi-v1 COMMAND mercan-tensor-abi-test)

    add_executable(mercan-graph-attention-abi-test tests/graph_attention_abi_compile.cpp)
    target_include_directories(mercan-graph-attention-abi-test PRIVATE runtime/libmercan/include)
    add_test(NAME mercan-graph-attention-abi-v1 COMMAND mercan-graph-attention-abi-test)

    add_executable(mercan-kv-abi-test tests/kv_abi_compile.cpp)
    target_include_directories(mercan-kv-abi-test PRIVATE runtime/libmercan/include)
    add_test(NAME mercan-kv-abi-v1 COMMAND mercan-kv-abi-test)

    add_executable(mercan-graph-output-abi-test tests/graph_output_abi_compile.cpp)
    target_include_directories(mercan-graph-output-abi-test PRIVATE runtime/libmercan/include)
    add_test(NAME mercan-graph-output-abi-v1 COMMAND mercan-graph-output-abi-test)

    add_library(mercan_arch_anka MODULE examples/plugins/anka/plugin.cpp)
    target_include_directories(mercan_arch_anka PRIVATE runtime/libmercan/include)
    set_target_properties(mercan_arch_anka PROPERTIES OUTPUT_NAME "mercan_arch_anka")

    add_executable(mercan-plugin-loader-test tests/plugin_loader_test.cpp)
    target_link_libraries(mercan-plugin-loader-test PRIVATE libmercan)
    add_test(NAME mercan-plugin-loader-v1 COMMAND mercan-plugin-loader-test $<TARGET_FILE:mercan_arch_anka>)

    add_executable(mercan-generic-plugin-backend-test tests/generic_plugin_backend_test.cpp)
    target_include_directories(mercan-generic-plugin-backend-test PRIVATE runtime/libmercan/include "${MERCAN_LLAMA_DIR}/ggml/include")
    target_link_libraries(mercan-generic-plugin-backend-test PRIVATE libmercan ggml)
    add_test(NAME mercan-generic-plugin-backend-v1 COMMAND mercan-generic-plugin-backend-test $<TARGET_FILE:mercan_arch_anka> ${CMAKE_CURRENT_BINARY_DIR}/anka-tiny.mercan)
    if(GGML_CUDA)
        add_test(NAME mercan-generic-plugin-backend-gpu-v1 COMMAND mercan-generic-plugin-backend-test $<TARGET_FILE:mercan_arch_anka> ${CMAKE_CURRENT_BINARY_DIR}/anka-tiny-gpu.mercan gpu)
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS mercan libmercan
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
    runtime/libmercan/include/mercan.h
    runtime/libmercan/include/mercan_arch.h
    runtime/libmercan/include/mercan_tokenizer.h
    runtime/libmercan/include/mercan_tensor.h
    runtime/libmercan/include/mercan_graph.h
    runtime/libmercan/include/mercan_kv.h
    runtime/libmercan/include/mercan_plugin.h
    runtime/nedo004-ffi/include/nedo004.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
