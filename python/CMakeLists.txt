cmake_minimum_required(VERSION 3.20)
project(HypercubeESNPython LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── pybind11 ──
find_package(pybind11 CONFIG REQUIRED)

# ── Optimization flags (match main project) ──
# HYPERCUBE_ARCH controls -march. Defaults to "native" for local dev builds.
# cibuildwheel overrides to "x86-64-v2" (x86_64) or "none" (ARM, MSVC).
set(HYPERCUBE_ARCH "native" CACHE STRING "Target architecture for -march (native, x86-64-v2, none)")

if(MSVC)
    add_compile_options(/O2 /fp:fast)
else()
    add_compile_options(-O3 -ffast-math)
    if(NOT HYPERCUBE_ARCH STREQUAL "none")
        if(HYPERCUBE_ARCH STREQUAL "native")
            add_compile_options(-march=native -mtune=native)
        else()
            add_compile_options(-march=${HYPERCUBE_ARCH} -mtune=generic)
        endif()
    endif()
    add_compile_options(-Wall -Wextra -Wno-unknown-pragmas)
endif()

# ── Core sources compiled directly into the module (PIC required) ──
set(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CORE_SOURCES
    ${CORE_DIR}/Reservoir.cpp
    ${CORE_DIR}/ESN.cpp
    ${CORE_DIR}/Readout.cpp
)

# ── HypercubeCNN dependency ──
# Vendored read-only snapshot at ../third_party/HypercubeCNN (see VENDORED.md).
# No sibling checkout, no pre-built .a, no network fetch — offline & version-pinned.
# third_party/ lies outside this source tree, so a binary dir arg is required.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/HypercubeCNN
                 ${CMAKE_BINARY_DIR}/HypercubeCNN-build)
set(HCNN_LIBS HypercubeCNNCore)

# ── Build the Python extension module ──
# HypercubeCNNCore's PUBLIC include dir propagates through the link below, so no
# explicit HCNN include path is needed here.
pybind11_add_module(_core bindings.cpp ${CORE_SOURCES})
target_include_directories(_core PRIVATE ${CORE_DIR})
target_link_libraries(_core PRIVATE ${HCNN_LIBS})

# ── Linking ──
if(MINGW)
    # Statically link MinGW runtime DLLs so the .pyd is self-contained.
    target_link_options(_core PRIVATE -static-libgcc -static-libstdc++)
    target_link_libraries(_core PRIVATE -Wl,-Bstatic pthread winpthread -Wl,-Bdynamic)
    # Posix-model MinGW always needs libwinpthread-1.dll at runtime.
    # Ship it inside the package so the .pyd loads without MinGW on PATH.
    find_file(WINPTHREAD_DLL libwinpthread-1.dll PATHS ENV PATH NO_DEFAULT_PATH)
    if(WINPTHREAD_DLL)
        install(FILES ${WINPTHREAD_DLL} DESTINATION hypercube_esn)
    endif()
endif()

# ── Install into the hypercube_esn package directory ──
install(TARGETS _core DESTINATION hypercube_esn)
