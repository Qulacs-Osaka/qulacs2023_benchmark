cmake_minimum_required(VERSION 3.10)
project(MyProject)

find_package(Kokkos REQUIRED)

add_executable(myTarget main.cpp)
target_link_libraries(myTarget Kokkos::kokkos)
