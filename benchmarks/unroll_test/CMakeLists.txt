cmake_minimum_required(VERSION 3.16)
project(MyProject)

set(KOKKOS_PATH "/app/kokkos")
set(CMAKE_PREFIX_PATH ${KOKKOS_PATH};${CMAKE_PREFIX_PATH})  

find_package(Kokkos REQUIRED)
include_directories(${Kokkos_INCLUDE_DIRS_RET})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(myTarget src/main.cpp)
target_link_libraries(myTarget Kokkos::kokkos)
