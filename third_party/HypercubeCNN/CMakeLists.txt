# Vendored read-only snapshot of dliptak001/HypercubeCNN — see VENDORED.md.
# The .h/.cpp here are an upstream snapshot: do NOT edit, re-vendor instead.
# This CMakeLists is HypercubeESN build glue (ours), not upstream.
#
# Deliberately minimal vs. upstream's CMakeLists: no install/export rules and no
# hcnn_compile_flags interface (that is where upstream's -march=native / -O3 lived).
# The core inherits the parent project's compile flags instead, so anything we
# publish stays portable across CPUs. The HypercubeCNNCore target name is preserved
# so consumers link it unchanged.

find_package(Threads REQUIRED)

add_library(HypercubeCNNCore STATIC
    HCNNConv.cpp
    HCNNPool.cpp
    HCNNNetwork.cpp
    HCNNReadout.cpp
    HCNN.cpp
)
add_library(HypercubeCNN::HypercubeCNNCore ALIAS HypercubeCNNCore)

target_compile_features(HypercubeCNNCore PUBLIC cxx_std_23)
set_target_properties(HypercubeCNNCore PROPERTIES CXX_EXTENSIONS OFF)

# Build-only include: consumers link $<BUILD_INTERFACE:HypercubeCNNCore>, so no
# INSTALL_INTERFACE path is needed.
target_include_directories(HypercubeCNNCore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(HypercubeCNNCore PUBLIC Threads::Threads)

# Upstream default (Padé fast-tanh). PRIVATE: affects only HCNNConv.cpp's TU;
# public headers are untouched, so consumers link whichever variant was built.
target_compile_definitions(HypercubeCNNCore PRIVATE HCNN_FAST_TANH)
