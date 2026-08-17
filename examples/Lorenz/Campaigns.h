#pragma once

#include "Lorenz.h"

#include <cstddef>
#include <cstdint>
#include <string>

// ---------------------------------------------------------------------------
// Campaign entry points -- plain functions. Call them from main.cpp (copy/paste).
// No CLI framework: pick what to run by editing main.
// First argument is always reservoir DIM (N = 2^DIM); restored on exit.
//
// Pipeline (keepers):
//   SeedSweep  →  Train  →  OrbitSweep  →  FreeRun
//   (+ DefaultWeightStem for shared weight paths)
// ---------------------------------------------------------------------------

/// Default readout path (no extension) under config::MODEL_SAVE_DIR:
///   {MODEL_SAVE_DIR}/lorenz_seed{esn}_D{dim}_M{M}
/// Train / OrbitSweep / FreeRun share this when @p weights_stem is null/empty.
/// On disk: path + ".hcnw" and path + ".arch.json".
[[nodiscard]] std::string DefaultWeightStem(uint64_t esn_seed,
                                            size_t dim,
                                            size_t history_depth);

/// Parallel train+freerun seed search (no weight save). Spawns @p num_threads
/// workers over @p num_seeds ESN seeds derived from @p base_esn_seed via
/// SplitMix64 (decorrelated; not base+i). Each worker: Train in memory →
/// @p freerun_runs Unseen freeruns; aggregates use top-10% freerun pool per
/// metric. Interim seed reports suppressed; stderr heartbeats only (mutexed).
/// @p base_orbit_seed is the **shared remix root** for train and freerun
/// (each phase advances its own orbit stream via mix64 — not a single orbit).
/// Final report: full seed table + top_k highlights for VPT, duty, and
/// VPT*duty (stdout and surveys/*.csv + *.txt). Dynamics overrides:
/// @p spectral_radius / @p input_scaling >0 set for the run (RAII restore);
/// 0 keeps config. Always trains (no load / no save). Refuses
/// LOAD_TRAINED_WEIGHTS and SAVE_TRAINED_WEIGHTS. Requires
/// Lorenz::kReadoutNumThreads == 1 (HCNN single-threaded per network).
/// @p num_threads is capped to hardware_concurrency and to @p num_seeds.
int SeedSweep(size_t dim,
              size_t history_depth,
              uint64_t base_esn_seed,
              size_t num_seeds,
              size_t num_threads,
              size_t epochs,
              int freerun_runs,
              uint64_t base_orbit_seed = 9333312947715283458ull,
              int top_k = 10,
              float spectral_radius = 0.f,
              float input_scaling = 0.f);

/// Train-only: remixed orbits for @p epochs starting from @p target_orbit remix
/// seed; save readout to @p weights_stem (or DefaultWeightStem). No free-run.
/// Restores DIM, HISTORY_DEPTH, EPOCHS on exit.
int Train(size_t dim,
          size_t history_depth,
          uint64_t esn_seed,
          uint64_t target_orbit,
          size_t epochs,
          const char* weights_stem = nullptr);

/// Parallel orbit ranking for one ESN seed (**load-only**; no train).
/// Loads readout from @p weights_stem (or config::LOAD_WEIGHTS_STEM if null/empty),
/// then freeruns @p num_orbits fixed orbits in parallel:
/// orbit_i = Mix64(base_orbit ^ FNV*(i+1)). One freerun per orbit (raw metrics,
/// no top-10% pool). Ranks by VPT / duty / VPT*duty. Survey TXT only (no CSV);
/// stdout + file keep top 100 and bottom 10 by VPT*duty. Top-k sections include
/// attractor IC (x,y,z) for FreeRun. Pipeline: Train → OrbitSweep → FreeRun
/// (same weights path). Requires HCNN threads=1; @p num_threads capped to HW
/// and @p num_orbits. Optional SR/IS overrides as FreeRun.
int OrbitSweep(size_t dim,
               size_t history_depth,
               uint64_t base_esn_seed,
               float spectral_radius,
               float input_scaling,
               uint64_t base_orbit_seed,
               size_t num_orbits,
               size_t num_threads,
               const char* weights_stem = nullptr,
               int top_k = 10);

/// Load readout weights + one free-run for @p orbit_seed (same key OrbitSweep
/// ranks). Does **not** train. Rebuilds IC via IcFromOrbitSeed(orbit_seed) at
/// full double precision — do **not** paste printed IC floats (chaotic discard
/// over ~TW steps makes 6-digit ICs diverge from the survey path).
/// Weights stem: @p weights_stem if non-null/non-empty, else
/// config::LOAD_WEIGHTS_STEM. Writes CSV under RUNS_DIR/traces/ (plot with
/// plot_freerun_overlay.py). Restores DIM / HISTORY_DEPTH / SPECTRAL_RADIUS /
/// INPUT_SCALING on exit. Channel gains fixed (`config::INPUT_SCALE_CH`).
/// Distinct from member @c Lorenz::FreeRun (this is the campaign wrapper).
/// @param spectral_radius If > 0, set config::SPECTRAL_RADIUS; <= 0 keeps config.
/// @param input_scaling   If > 0, set config::INPUT_SCALING; <= 0 keeps config.
/// @param orbit_seed      Mix64 orbit key from OrbitSweep (not a rounded IC).
/// @param freerun_steps   Generative runway length; 0 → config::FREE_RUN_WINDOW_SIZE
///                        (default 2000). Use larger values for long plots; OrbitSweep
///                        ranking still uses the default window.
int FreeRun(size_t dim,
            size_t history_depth,
            uint64_t esn_seed,
            float spectral_radius,
            float input_scaling,
            uint64_t orbit_seed,
            const char* weights_stem = nullptr,
            size_t freerun_steps = 0);
