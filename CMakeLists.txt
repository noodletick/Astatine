cmake_minimum_required(VERSION 3.29)
project(Astatine CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Astatine CUTRIX.cu
        nugget.cu
        NUGGET.cuh)

set_target_properties(Astatine PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
