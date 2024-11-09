cmake_minimum_required(VERSION 3.29)
project(Astatine LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
add_executable(Astatine nugget.cu
        NUGGET.cuh
        test.cu)

set_target_properties(Astatine PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
