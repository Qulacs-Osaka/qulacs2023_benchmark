cmake_minimum_required(VERSION 3.16)
project(MyProject)

set(KOKKOS_PATH "/benchmarks/kokkos")
set(CMAKE_PREFIX_PATH ${KOKKOS_PATH};${CMAKE_PREFIX_PATH})  

find_package(Kokkos REQUIRED)
include_directories(${Kokkos_INCLUDE_DIRS_RET})

add_executable(myTarget /benchmarks/main.cpp)
target_link_libraries(myTarget Kokkos::kokkos)

target_compile_definitions(myTarget PUBLIC KOKKOS_ENABLE_CUDA=1)
