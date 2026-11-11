cmake_minimum_required(VERSION 3.16)
project(nozzle_cfd_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

add_executable(maccormack_cfd maccormack_cfd.cpp)
add_executable(hllc_fvm_cfd hllc_fvm_solver.cpp)
add_library(nozzle_fvm_core SHARED hllc_fvm_solver.cpp)
target_compile_definitions(nozzle_fvm_core PRIVATE NOZZLE_FVM_BUILD_DLL)

add_executable(axisymmetric_ns2d axisymmetric_ns2d_solver.cpp)
add_library(nozzle_fvm2d_core SHARED axisymmetric_ns2d_solver.cpp)
target_compile_definitions(nozzle_fvm2d_core PRIVATE NOZZLE_FVM2D_BUILD_DLL)

if (OpenMP_CXX_FOUND)
    target_link_libraries(axisymmetric_ns2d PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(nozzle_fvm2d_core PRIVATE OpenMP::OpenMP_CXX)
endif()

foreach(tgt maccormack_cfd hllc_fvm_cfd nozzle_fvm_core axisymmetric_ns2d nozzle_fvm2d_core)
    if (MSVC)
        target_compile_options(${tgt} PRIVATE /O2 /W4)
    else()
        target_compile_options(${tgt} PRIVATE -O3 -Wall -Wextra)
    endif()
endforeach()
