cmake_minimum_required(VERSION 3.16)
project(MyProject)

set(KOKKOS_PATH "/benchmarks/kokkos_cpu")
set(CMAKE_PREFIX_PATH ${KOKKOS_PATH};${CMAKE_PREFIX_PATH})  

find_package(Kokkos REQUIRED)
include_directories(${Kokkos_INCLUDE_DIRS_RET})

add_executable(myTarget /benchmarks/main.cpp)
target_link_libraries(myTarget Kokkos::kokkos)
