#pragma once
#include "LorenzAttractor.h"
#include "JanusShuttle.h"

/// @brief The Lorenz instantiation of the generic @ref JanusShuttle driver: two
/// LorenzAttractors (one forward, one backward) shuttled back-and-forth along a
/// reflecting 1D scan. LorenzAttractor satisfies @ref Steppable (nested State,
/// reset(State), step(double), public state).
using LorenzAttractorPong = JanusShuttle<LorenzAttractor>;
