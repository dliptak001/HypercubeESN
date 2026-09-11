# HypercubeESN: an echo-state net on a Boolean hypercube

David C. Liptak — https://github.com/dliptak001/HypercubeESN

`pip install hypercube-esn`  ·  Apache-2.0  ·  C++23 + Python

HypercubeESN is a reservoir computer whose graph is not stored. Vertices are bitstrings. A neighbor is one XOR of a single bit (Hamming-1). With a seed and a few scalars the whole reservoir reconstructs, so there is no adjacency matrix to allocate or cache.

That buys three things ESNs usually pay for with a random sparse graph:

1. Homogeneity. Every vertex has the same degree and the same local world.
2. Cheap walks. A neighbor is arithmetic on the index, not a pointer chase.
3. A topology-native readout. The output layer is a CNN on the same vertices, so the data never leaves the cube it was born on.

Author-run C++ campaigns (one operating point): NARMA-30/50/70 NRMSE 0.0441 / 0.0751 / 0.1251. Jaeger memory capacity up to 1380 on a 4096-node reservoir.

Siblings on the same cube, for static fields rather than streams: HypercubeWTF, HypercubeCascade. Also in the family: HypercubeCNN, HypercubeHopfield, HypercubeEtalon. HypercubeESN is not a spiking net; a hypercube SNN is planned, not shipped.

This note is the public artifact behind the software announcements on UAI, Connectionists (held for moderator), and robotics-worldwide.
