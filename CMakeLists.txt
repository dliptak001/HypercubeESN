cmake_minimum_required(VERSION 4.1)
project(HypercubeESN VERSION 0.2.4)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Prefer -std=c++23 over GNU extensions

# ——— Static library: core pipeline (Reservoir, ESN, Readout) ———
add_library(HypercubeESNCore STATIC
        ${PROJECT_SOURCE_DIR}/Reservoir.cpp
    ${PROJECT_SOURCE_DIR}/ESN.cpp
    ${PROJECT_SOURCE_DIR}/Readout.cpp
)

# Public headers — consumers need these via target_link_libraries
set(PUBLIC_HEADERS
    ${PROJECT_SOURCE_DIR}/ESN.h
        ${PROJECT_SOURCE_DIR}/Reservoir.h
    ${PROJECT_SOURCE_DIR}/Readout.h
)

# ——— HypercubeCNN dependency (CNN readout) ———
# Vendored read-only snapshot at third_party/HypercubeCNN (see VENDORED.md).
# No sibling checkout or network fetch needed; the in-tree shim CMakeLists
# builds only its HypercubeCNNCore static lib.
add_subdirectory(third_party/HypercubeCNN)

# Include directories: PROJECT_SOURCE_DIR for in-tree builds,
# install prefix include/HypercubeESN for installed consumers
target_include_directories(HypercubeESNCore
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/HypercubeESN>
)

target_link_libraries(HypercubeESNCore PUBLIC $<BUILD_INTERFACE:HypercubeCNNCore>)

# ——— Optimization and warning flags (library) ———
# Note: -flto is unavailable here — CLion 2024.3.2's bundled MinGW gcc 13.1
# was built without LTO support (cc1plus: "LTO support has not been enabled
# in this configuration").  Would need a different MinGW distribution.
target_compile_options(HypercubeESNCore PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
        $<$<CONFIG:Debug>:-O0 -g>
        -Wall -Wextra -Wno-unknown-pragmas
)

if(MINGW)
    target_link_options(HypercubeESNCore PRIVATE -static-libgcc -static-libstdc++)
endif()

# ——— Executables: only when building standalone (not via FetchContent) ———
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_executable(HypercubeESN main.cpp)
    add_executable(BasicPrediction examples/BasicPrediction.cpp)
    add_executable(SignalClassification examples/SignalClassification.cpp)
    add_executable(StreamingAnomaly examples/StreamingAnomaly.cpp)
    add_executable(MemoryCapacity examples/MemoryCapacity/MemoryCapacity.cpp)
    add_executable(NARMA examples/NARMA/NARMA.cpp)
    file(GLOB STREAMING_TEXT_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/examples/StreamingText/*.cpp)
    add_executable(StreamingText ${STREAMING_TEXT_SOURCES})

    set(EXAMPLE_TARGETS BasicPrediction SignalClassification StreamingAnomaly MemoryCapacity NARMA StreamingText)
    set(EXE_TARGETS HypercubeESN ${EXAMPLE_TARGETS})

    foreach(tgt IN LISTS EXE_TARGETS)
        target_link_libraries(${tgt} PRIVATE HypercubeESNCore)
    endforeach()

    foreach(tgt IN LISTS EXE_TARGETS)
        target_compile_options(${tgt} PRIVATE
                $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math>
                $<$<CONFIG:Debug>:-O0 -g>
                -Wall -Wextra -Wno-unknown-pragmas
        )

        if(MINGW)
            target_link_options(${tgt} PRIVATE -static-libgcc -static-libstdc++)
            # Workaround for a gcc 13.1 MinGW bug where
            # `std::type_info::operator==` in <typeinfo> is emitted out-of-line
            # at -O0 (Debug) and collides with libstdc++.a(tinfo.o). Release
            # (-O3) inlines it and is unaffected. Fixed upstream in gcc 13.2,
            # but CLion 2024.3.2 / 2026.1 both still bundle 13.1. Debug-only
            # so we don't mask real multi-def bugs in Release.
            target_link_options(${tgt} PRIVATE
                $<$<CONFIG:Debug>:-Wl,--allow-multiple-definition>)
        endif()
    endforeach()
endif()

# ——— Install rules ———
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the static library
install(TARGETS HypercubeESNCore
    EXPORT HypercubeESNTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install public headers into include/HypercubeESN/
install(FILES ${PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/HypercubeESN
)

# Generate and install CMake package config
install(EXPORT HypercubeESNTargets
    FILE HypercubeESNTargets.cmake
    NAMESPACE HypercubeESN::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HypercubeESN
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/HypercubeESNConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeESNConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HypercubeESN
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeESNConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeESNConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/HypercubeESNConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HypercubeESN
)
