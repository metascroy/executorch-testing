# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

project(executorch_testing)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

include(FetchContent)
FetchContent_Declare(
googletest
URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# Set executorch build options
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
set(EXECUTORCH_BUILD_COREML ON CACHE BOOL "Enable CoreML backend for ExecutorTorch" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "Enable CoreML backend for ExecutorTorch" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "Enable CoreML backend for ExecutorTorch" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "Enable CoreML backend for ExecutorTorch" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "Enable CoreML backend for ExecutorTorch" FORCE)
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON CACHE BOOL "Enable CoreML backend for ExecutorTorch" FORCE)

# ET_EVENT_TRACER_ENABLED is defined by this
set(EXECUTORCH_ENABLE_EVENT_TRACER OFF CACHE BOOL "Enable CoreML backend for ExecutorTorch" FORCE)
set(EXECUTORCH_BUILD_DEVTOOLS OFF CACHE BOOL "Enable CoreML backend for ExecutorTorch" FORCE)
add_compile_definitions("-Dprotobuf_BUILD_TESTS=OFF")


add_subdirectory("third-party/executorch")

add_executable(test_model test_model.cpp)

target_link_libraries(
    test_model
    PRIVATE executorch
    extension_module_static
    extension_flat_tensor
    extension_tensor
    optimized_native_cpu_ops_lib
    coremldelegate
    GTest::gtest_main
)


include(GoogleTest)
gtest_discover_tests(test_model)
