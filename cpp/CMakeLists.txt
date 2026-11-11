cmake_minimum_required(VERSION 3.16)
project(nozzle_cfd_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(maccormack_cfd maccormack_cfd.cpp)
if (MSVC)
    target_compile_options(maccormack_cfd PRIVATE /O2 /W4)
else()
    target_compile_options(maccormack_cfd PRIVATE -O3 -Wall -Wextra)
endif()
