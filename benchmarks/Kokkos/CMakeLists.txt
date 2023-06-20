cmake_minimum_required(VERSION 3.16)
project(MyProject)

set(KOKKOS_PATH "/app/include/kokkos")
set(CMAKE_PREFIX_PATH ${KOKKOS_PATH};${CMAKE_PREFIX_PATH})
find_package(Kokkos REQUIRED)

add_executable(myTarget main.cpp)
target_link_libraries(myTarget Kokkos::kokkos)
