cmake_minimum_required(VERSION 3.16)
project(nozzle_cfd_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(maccormack_cfd maccormack_cfd.cpp)
add_executable(hllc_fvm_cfd hllc_fvm_solver.cpp)
add_library(nozzle_fvm_core SHARED hllc_fvm_solver.cpp)
target_compile_definitions(nozzle_fvm_core PRIVATE NOZZLE_FVM_BUILD_DLL)

foreach(tgt maccormack_cfd hllc_fvm_cfd nozzle_fvm_core)
    if (MSVC)
        target_compile_options(${tgt} PRIVATE /O2 /W4)
    else()
        target_compile_options(${tgt} PRIVATE -O3 -Wall -Wextra)
    endif()
endforeach()
