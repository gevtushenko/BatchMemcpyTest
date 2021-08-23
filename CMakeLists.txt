cmake_minimum_required(VERSION 3.20)
project(BatchMemcpyTest CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(BatchMemcpyTest main.cu)

set_target_properties(BatchMemcpyTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(BatchMemcpyTest PRIVATE
        C:/Users/evtus/Documents/GitHub/senior-zero/cub
        C:/Users/evtus/Documents/GitHub/senior-zero/thrust)
