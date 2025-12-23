cmake_minimum_required(VERSION 3.16)
project(NextCompiler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Project options (independent of third-party libraries)
option(NEXTCOMPILER_BUILD_TESTS "Build tests for NextCompiler" ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/src)

# ============================================================
# Third-party dependencies
# ============================================================
include(FetchContent)

# Disable all third-party tests/docs globally before FetchContent
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)

# nlohmann/json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# IR library
add_library(ir STATIC
    src/ir/Type.cpp
    src/ir/Value.cpp
    src/ir/Operation.cpp
    src/ir/Block.cpp
    src/ir/Region.cpp
    src/ir/Module.cpp
    src/ir/IRBuilder.cpp
    src/ir/IRPrinter.cpp
)
target_link_libraries(ir PUBLIC nlohmann_json::nlohmann_json)

# ============================================================
# Tests (project-specific, not affected by BUILD_TESTING)
# ============================================================
if(NEXTCOMPILER_BUILD_TESTS)
    enable_testing()
    add_executable(test_ir tests/test_ir.cpp)
    target_link_libraries(test_ir ir)
    add_test(NAME test_ir COMMAND test_ir)
endif()
