cmake_minimum_required(VERSION 3.16)
project(batched-qsim CXX)

include(FetchContent)

FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG        4.1.00
)
FetchContent_MakeAvailable(Kokkos)

FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG        v2.9
)
FetchContent_MakeAvailable(argparse)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
)
FetchContent_MakeAvailable(doctest)

add_executable(qsim-bench src/main.cpp src/gate.cpp)
target_link_libraries(qsim-bench kokkos argparse)

enable_testing()
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

add_executable(qsim-test src/test.cpp src/gate.cpp)
target_link_libraries(qsim-test kokkos doctest)

doctest_discover_tests(qsim-test)
