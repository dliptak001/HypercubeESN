#include "Campaigns.h"
#include "Lorenz.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{
    namespace fs = std::filesystem;

    void FormatWallTime(char* buf, size_t n, const double seconds)
    {
        if (seconds < 60.0)
            std::snprintf(buf, n, "%.1f s", seconds);
        else if (seconds < 3600.0)
            std::snprintf(buf, n, "%.1f s (%.1f min)", seconds, seconds / 60.0);
        else
            std::snprintf(buf, n, "%.1f s (%.2f h)", seconds, seconds / 3600.0);
    }

    // RAII: assign a config value for the campaign duration, restore on exit.
    template <typename T>
    struct ConfigRestore
    {
        T& slot;
        T saved;
        ConfigRestore(T& ref, T value) : slot(ref), saved(ref) { slot = value; }
        ~ConfigRestore() { slot = saved; }
        ConfigRestore(const ConfigRestore&) = delete;
        ConfigRestore& operator=(const ConfigRestore&) = delete;
    };

    using ConfigSizeRestore = ConfigRestore<size_t>;
    using ConfigFloatRestore = ConfigRestore<float>;

    // Reservoir requires 5 <= dim <= 16 (see Reservoir.cpp).
    bool ValidateDim(size_t dim, const char* campaign)
    {
        if (dim >= 5 && dim <= 16)
            return true;
        std::fprintf(stderr, "[%s] refused: dim=%zu (reservoir requires 5 <= dim <= 16)\n",
                     campaign, dim);
        return false;
    }

    bool ValidateHistoryDepth(size_t M, const char* campaign)
    {
        if (M >= 1 && M <= 64)
            return true;
        std::fprintf(stderr, "[%s] refused: M=%zu (reservoir requires 1 <= M <= 64)\n",
                     campaign, M);
        return false;
    }

    bool ValidateSpectralRadius(float sr, const char* campaign)
    {
        if (std::isfinite(sr) && sr > 0.0f)
            return true;
        std::fprintf(stderr, "[%s] refused: spectral_radius=%g (need finite and > 0)\n",
                     campaign, static_cast<double>(sr));
        return false;
    }

    bool ValidateInputScaling(float is, const char* campaign)
    {
        if (std::isfinite(is) && is > 0.0f)
            return true;
        std::fprintf(stderr, "[%s] refused: input_scaling=%g (need finite and > 0)\n",
                     campaign, static_cast<double>(is));
        return false;
    }

    // Format locked config::INPUT_SCALE_CH for banners / metadata.
    std::string FormatDriveGains()
    {
        std::ostringstream o;
        o << '[';
        for (size_t i = 0; i < kNumDriveChannels; ++i)
        {
            if (i)
                o << ',';
            o << config::INPUT_SCALE_CH[i];
        }
        o << ']';
        return o.str();
    }

    // Optional SR / IS (FreeRun, SeedSweep, OrbitSweep).
    // >0 sets; 0 keeps current config.
    bool ValidateDynamicsOverrides(float spectral_radius, float input_scaling,
                                   const char* campaign)
    {
        if (spectral_radius < 0.0f)
        {
            std::fprintf(stderr,
                         "[%s] refused: spectral_radius=%g (use > 0 to set, or 0 to keep config)\n",
                         campaign, static_cast<double>(spectral_radius));
            return false;
        }
        if (input_scaling < 0.0f)
        {
            std::fprintf(stderr,
                         "[%s] refused: input_scaling=%g (use > 0 to set, or 0 to keep config)\n",
                         campaign, static_cast<double>(input_scaling));
            return false;
        }
        if (spectral_radius > 0.0f && !ValidateSpectralRadius(spectral_radius, campaign))
            return false;
        if (input_scaling > 0.0f && !ValidateInputScaling(input_scaling, campaign))
            return false;
        return true;
    }

    std::string TimestampNow()
    {
        using clock = std::chrono::system_clock;
        const std::time_t tt = clock::to_time_t(clock::now());
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &tt);
#else
        localtime_r(&tt, &tm);
#endif
        char buf[32];
        std::strftime(buf, sizeof buf, "%Y%m%d_%H%M%S", &tm);
        return buf;
    }

    // ---- Shared paths --------------------------------------------------------
    // C:\HypercubeESN\results\{traces|surveys}

    fs::path RunsRoot() { return config::RUNS_DIR; }
    fs::path TracesDir() { return RunsRoot() / "traces"; }
    fs::path SurveysDir() { return RunsRoot() / "surveys"; }

    // Create dir; log and return false on failure. tag e.g. "train", "freerun".
    bool EnsureDir(const fs::path& dir, const char* tag)
    {
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec)
        {
            std::fprintf(stderr, "[%s] create_directories(%s) failed: %s\n",
                         tag, dir.string().c_str(), ec.message().c_str());
            return false;
        }
        return true;
    }

    void ReportWrote(const char* tag, const fs::path& path)
    {
        std::error_code ec;
        const auto sz = fs::file_size(path, ec);
        if (ec)
            std::printf("[%s] wrote %s\n", tag, path.string().c_str());
        else
            std::printf("[%s] wrote %s  (%llu bytes)\n",
                        tag, path.string().c_str(),
                        static_cast<unsigned long long>(sz));
        std::fflush(stdout);
    }

    void ReportDone(const char* tag, double wall_seconds)
    {
        char time_buf[64];
        FormatWallTime(time_buf, sizeof time_buf, wall_seconds);
        std::printf("[%s] done  wall time: %s\n", tag, time_buf);
        std::fflush(stdout);
    }

    // Primary freerun score line (all campaigns that emit freerun metrics).
    void ReportFreerunScores(const char* tag,
                             double vpt_lt, double duty, double vpt_x_duty, double rmse)
    {
        std::printf("[%s] VPT=%.2f lt  duty=%.3f  VPT*duty=%.3f  RMSE=%.6f\n",
                    tag, vpt_lt, duty, vpt_x_duty, rmse);
        std::fflush(stdout);
    }

    void ReportBanner(const char* campaign_name)
    {
        std::printf("=== HypercubeESN: Lorenz / %s ===\n", campaign_name);
        std::fflush(stdout);
    }

    // Sample mean + sample std (n-1). Empty => mean=std=0.
    void MeanStd(const std::vector<double>& v, double& mean, double& stdev)
    {
        mean = 0;
        stdev = 0;
        if (v.empty())
            return;
        double sum = 0;
        for (double x : v) sum += x;
        mean = sum / static_cast<double>(v.size());
        if (v.size() < 2)
            return;
        double var = 0;
        for (double x : v)
        {
            const double d = x - mean;
            var += d * d;
        }
        stdev = std::sqrt(var / static_cast<double>(v.size() - 1));
    }

    double MeanOf(const std::vector<double>& v)
    {
        double m = 0, s = 0;
        MeanStd(v, m, s);
        return m;
    }

    /// Keep count for freerun pool: top 10% of n, at least 1 when n >= 1.
    /// max(1, ceil(n/10)); n=1000 -> 100.
    size_t Top10PoolKeep(size_t n)
    {
        if (n == 0)
            return 0;
        const size_t keep = (n + 9) / 10; // ceil(n/10)
        return keep < 1 ? 1 : keep;
    }

    /// Top 10% of a higher-is-better sample (largest values).
    std::vector<double> Top10HigherIsBetter(std::vector<double> v)
    {
        if (v.size() <= 1)
            return v;
        std::sort(v.begin(), v.end());
        const size_t keep = Top10PoolKeep(v.size());
        return std::vector<double>(v.end() - static_cast<std::ptrdiff_t>(keep), v.end());
    }

    /// Top 10% of a lower-is-better sample (smallest values).
    std::vector<double> Top10LowerIsBetter(std::vector<double> v)
    {
        if (v.size() <= 1)
            return v;
        std::sort(v.begin(), v.end());
        const size_t keep = Top10PoolKeep(v.size());
        return std::vector<double>(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(keep));
    }

} // namespace

// Default readout path (no extension): {MODEL_SAVE_DIR}/lorenz_seed{S}_D{dim}_M{M}
std::string DefaultWeightStem(uint64_t esn_seed, size_t dim, size_t history_depth)
{
    namespace fs = std::filesystem;
    return (fs::path(config::MODEL_SAVE_DIR) /
        ("lorenz_seed" + std::to_string(esn_seed) +
            "_D" + std::to_string(dim) +
            "_M" + std::to_string(history_depth))).string();
}

// ---------------------------------------------------------------------------
// Train (campaign: remixed train + save weights; no freerun)
// ---------------------------------------------------------------------------
// Distinct from Lorenz::Train (member).
int Train(size_t dim, size_t history_depth, uint64_t esn_seed,
          uint64_t target_orbit, size_t epochs, const char* weights_stem)
{
    if (!ValidateDim(dim, "train"))
        return 2;
    if (!ValidateHistoryDepth(history_depth, "train"))
        return 2;
    if (epochs < 1)
    {
        std::fprintf(stderr, "[train] refused: epochs=%zu (need >= 1)\n", epochs);
        return 2;
    }

    ConfigSizeRestore dim_restore(config::DIM, dim);
    ConfigSizeRestore m_restore(config::HISTORY_DEPTH, history_depth);
    ConfigSizeRestore epochs_restore(config::EPOCHS, epochs);

    const std::string stem =
        (weights_stem && weights_stem[0] != '\0')
            ? std::string(weights_stem)
            : DefaultWeightStem(esn_seed, dim, history_depth);

    ReportBanner("Train (save-only; no freerun)");
    std::printf("[train] DIM=%zu (N=%zu)  M=%zu  esn_seed=%llu  target_orbit=%llu  epochs=%zu\n",
                config::DIM, size_t{1} << config::DIM, config::HISTORY_DEPTH,
                static_cast<unsigned long long>(esn_seed),
                static_cast<unsigned long long>(target_orbit),
                epochs);
    std::printf("[train] save stem: %s  (.hcnw + .arch.json)\n", stem.c_str());
    std::printf("[train] note: config banner may say save=off -- that is only\n"
        "        config::SAVE_TRAINED_WEIGHTS (auto-save inside Lorenz::Train).\n"
        "        This campaign always calls SaveTrainedWeights after train.\n");
    std::printf("[train] remixed orbits x %zu epochs, then save weights\n", epochs);
    std::fflush(stdout);

    if (config::LOAD_TRAINED_WEIGHTS)
    {
        std::fprintf(stderr,
                     "[train] refused: LOAD_TRAINED_WEIGHTS is on "
                     "(this campaign always trains from scratch)\n");
        return 2;
    }

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    config::ENABLE_PRINTF = true;
    Lorenz lorenz(esn_seed, target_orbit);
    std::cout << lorenz.ReadoutArchSummary();
    lorenz.Train(); // uses config::EPOCHS; auto-save only if SAVE_TRAINED_WEIGHTS

    try
    {
        lorenz.SaveTrainedWeights(stem.c_str());
        ReportWrote("train", fs::path(stem + ".hcnw"));
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "[train] save failed: %s\n", e.what());
        return 1;
    }

    ReportDone("train", std::chrono::duration<double>(clock::now() - t0).count());
    return 0;
}

// ---------------------------------------------------------------------------
// FreeRun (campaign: load weights + freerun one OrbitSweep orbit_seed)
// ---------------------------------------------------------------------------
// Same path as OrbitSweep: IcFromOrbitSeed(orbit_seed) at full double precision.
// Do not pass rounded IC floats — free-run stream discards ~TW RK4 steps first,
// so 6-digit ICs diverge chaotically from the survey trajectory.
// CSV under RUNS_DIR/traces/. Distinct from Lorenz::FreeRun (member).
// Plotting is external (plot_freerun_overlay.py).
int FreeRun(size_t dim,
            size_t history_depth,
            uint64_t esn_seed,
            float spectral_radius,
            float input_scaling,
            uint64_t orbit_seed,
            const char* weights_stem,
            size_t freerun_steps)
{
    if (!ValidateDim(dim, "freerun"))
        return 2;
    if (!ValidateHistoryDepth(history_depth, "freerun"))
        return 2;
    if (orbit_seed == 0)
    {
        std::fprintf(stderr,
                     "[freerun] refused: orbit_seed=0 "
                     "(pass the orbit_seed from OrbitSweep, not a rounded IC)\n");
        return 2;
    }
    if (freerun_steps == 0)
        freerun_steps = config::FREE_RUN_WINDOW_SIZE;
    if (freerun_steps < 1)
    {
        std::fprintf(stderr, "[freerun] refused: freerun_steps=%zu (need >= 1)\n",
                     freerun_steps);
        return 2;
    }
    if (!ValidateDynamicsOverrides(spectral_radius, input_scaling, "freerun"))
        return 2;

    const char* stem = (weights_stem && weights_stem[0] != '\0')
                           ? weights_stem
                           : config::LOAD_WEIGHTS_STEM;
    if (stem == nullptr || stem[0] == '\0')
    {
        std::fprintf(stderr,
                     "[freerun] refused: no weights stem "
                     "(pass weights_stem or set config::LOAD_WEIGHTS_STEM)\n");
        return 2;
    }

    ConfigSizeRestore dim_restore(config::DIM, dim);
    ConfigSizeRestore m_restore(config::HISTORY_DEPTH, history_depth);
    ConfigFloatRestore sr_restore(
        config::SPECTRAL_RADIUS,
        spectral_radius > 0.0f ? spectral_radius : config::SPECTRAL_RADIUS);
    ConfigFloatRestore is_restore(
        config::INPUT_SCALING,
        input_scaling > 0.0f ? input_scaling : config::INPUT_SCALING);

    const size_t n_in = kNumDriveChannels;

    const fs::path trace_dir = TracesDir();
    if (!EnsureDir(trace_dir, "freerun"))
        return 2;

    const LorenzAttractor::State ic = Lorenz::IcFromOrbitSeed(orbit_seed);

    const fs::path out_path = trace_dir /
        ("seed" + std::to_string(esn_seed) +
         "_orbit" + std::to_string(orbit_seed) + ".csv");

    ReportBanner("FreeRun (load-only)");
    std::printf("[freerun] DIM=%zu (N=%zu)  M=%zu  esn_seed=%llu\n",
                config::DIM, size_t{1} << config::DIM, config::HISTORY_DEPTH,
                static_cast<unsigned long long>(esn_seed));
    std::printf("[freerun] SR=%.4f%s  input_scaling=%.4f%s\n",
                config::SPECTRAL_RADIUS,
                spectral_radius > 0.0f ? " (override)" : " (config)",
                config::INPUT_SCALING,
                input_scaling > 0.0f ? " (override)" : " (config)");
    {
        const std::string ch = FormatDriveGains();
        std::printf("[freerun] drive=[x,y,z,xz]  n_in=%zu  drive_ch=%s\n",
                    n_in, ch.c_str());
    }
    std::printf("[freerun] orbit_seed=%llu  (IcFromOrbitSeed; same as OrbitSweep)\n",
                static_cast<unsigned long long>(orbit_seed));
    std::printf("[freerun] IC=(%.6f, %.6f, %.6f)  (display only; full double used)\n",
                ic.x, ic.y, ic.z);
    std::printf("[freerun] freerun_steps=%zu  (%.2f lt @ dt=%.3f lambda=%.4f)\n",
                freerun_steps,
                freerun_steps * config::LYAPUNOV_EXPONENT * config::DT,
                config::DT, config::LYAPUNOV_EXPONENT);
    std::printf("[freerun] load stem: %s\n", stem);
    std::printf("[freerun] CSV dir: %s\n", trace_dir.string().c_str());
    std::fflush(stdout);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    config::ENABLE_PRINTF = true;
    Lorenz lorenz(esn_seed, /*orbit_seed=*/0);
    std::cout << lorenz.ReadoutArchSummary();

    try
    {
        lorenz.LoadTrainedWeights(stem);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "[freerun] load failed: %s\n", e.what());
        return 1;
    }

    std::printf("[freerun] free-run fixed orbit_seed (no train)\n");
    std::fflush(stdout);

    // fixed_orbit_seed path — same seating as OrbitSweep; runway length optional.
    FreeRunResult r = lorenz.FreeRun(/*verbose=*/true, out_path.string().c_str(),
                                     /*warmup_steps=*/0, orbit_seed,
                                     /*fixed_ic=*/nullptr, freerun_steps);
    if (!r.valid)
    {
        std::printf("[freerun] free-run invalid\n");
        return 1;
    }

    ReportFreerunScores("freerun", r.vpt_lt, r.duty, r.vpt_x_duty, r.rmse);
    std::printf("%s", r.row.c_str());

    {
        std::error_code e;
        if (!fs::exists(out_path, e) || e || fs::file_size(out_path, e) == 0)
        {
            std::fprintf(stderr, "[freerun] CSV missing/empty after free-run: %s\n",
                         out_path.string().c_str());
            return 1;
        }
        ReportWrote("freerun", out_path);
    }

    std::printf("[freerun] plot:\n"
                "  python examples/Lorenz/plot_freerun_overlay.py \"%s\"\n",
                out_path.string().c_str());

    ReportDone("freerun", std::chrono::duration<double>(clock::now() - t0).count());
    return 0;
}

// ---------------------------------------------------------------------------
// SeedSweep (train+freerun in memory; parallel ESN seeds; no weight I/O)
// ---------------------------------------------------------------------------
namespace
{
    // SplitMix64 finalizer (same constants as Lorenz.cpp / Reservoir.cpp).
    inline uint64_t Mix64(uint64_t x)
    {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    // Decorrelated ESN seed for index i in [0, num_seeds). Labeled substream of base.
    inline uint64_t ParallelEsnSeed(uint64_t base_esn_seed, size_t i)
    {
        return Mix64(base_esn_seed ^ (0x100000001B3ULL * (static_cast<uint64_t>(i) + 1ULL)));
    }

    struct ParSeedRow
    {
        size_t index = 0;
        uint64_t esn_seed = 0;
        bool ok = false;
        double mean_vpt = 0;
        double mean_duty = 0;
        double mean_vpt_x_duty = 0;
        double mean_rmse = 0;
        size_t n_valid = 0;
    };

    // Train in memory + freerun survey; top-10% means. No save, no per-seed stdout.
    ParSeedRow EvaluateParSeed(size_t index, uint64_t esn_seed, uint64_t base_orbit_seed,
                               int freerun_runs)
    {
        ParSeedRow row;
        row.index = index;
        row.esn_seed = esn_seed;

        // Train remixes from base_orbit_seed (RebuildDatastream advances its chain).
        Lorenz lorenz(esn_seed, base_orbit_seed);
        lorenz.Train();

        std::vector<double> vpt_lts, rmses, duties, vpt_x_duties;
        vpt_lts.reserve(static_cast<size_t>(freerun_runs));
        rmses.reserve(static_cast<size_t>(freerun_runs));
        duties.reserve(static_cast<size_t>(freerun_runs));
        vpt_x_duties.reserve(static_cast<size_t>(freerun_runs));

        // Freerun ICs: independent remix stream rooted at the same base_orbit_seed
        // (not a continuation of the train chain; not a single fixed orbit).
        for (int i = 0; i < freerun_runs; ++i)
        {
            const uint64_t freerun_os = Mix64(
                base_orbit_seed ^ (0x100000001B3ULL * (static_cast<uint64_t>(i) + 1ULL)));
            const FreeRunResult r = lorenz.FreeRun(false, nullptr, 0, freerun_os);
            if (!r.valid)
                continue;
            vpt_lts.push_back(r.vpt_lt);
            rmses.push_back(r.rmse);
            duties.push_back(r.duty);
            vpt_x_duties.push_back(r.vpt_x_duty);
        }

        row.n_valid = vpt_lts.size();
        row.ok = row.n_valid > 0;
        if (!row.ok)
            return row;

        row.mean_vpt = MeanOf(Top10HigherIsBetter(std::move(vpt_lts)));
        row.mean_duty = MeanOf(Top10HigherIsBetter(std::move(duties)));
        row.mean_vpt_x_duty = MeanOf(Top10HigherIsBetter(std::move(vpt_x_duties)));
        row.mean_rmse = MeanOf(Top10LowerIsBetter(std::move(rmses)));
        return row;
    }
} // namespace

int SeedSweep(size_t dim, size_t history_depth,
              uint64_t base_esn_seed, size_t num_seeds, size_t num_threads,
              size_t epochs, int freerun_runs, uint64_t base_orbit_seed,
              int top_k, float spectral_radius, float input_scaling)
{
    if (!ValidateDim(dim, "par-seed-sweep"))
        return 2;
    if (!ValidateHistoryDepth(history_depth, "par-seed-sweep"))
        return 2;
    if (num_seeds < 1)
    {
        std::fprintf(stderr, "[par-seed-sweep] refused: num_seeds=%zu (need >= 1)\n",
                     num_seeds);
        return 2;
    }
    if (num_threads < 1)
    {
        std::fprintf(stderr, "[par-seed-sweep] refused: num_threads=%zu (need >= 1)\n",
                     num_threads);
        return 2;
    }
    if (epochs < 1)
    {
        std::fprintf(stderr, "[par-seed-sweep] refused: epochs=%zu (need >= 1)\n",
                     epochs);
        return 2;
    }
    if (freerun_runs < 1)
    {
        std::fprintf(stderr, "[par-seed-sweep] refused: freerun_runs=%d (need >= 1)\n",
                     freerun_runs);
        return 2;
    }
    if (top_k < 1)
        top_k = 1;
    if (!ValidateDynamicsOverrides(spectral_radius, input_scaling, "par-seed-sweep"))
        return 2;
    if (config::LOAD_TRAINED_WEIGHTS)
    {
        std::fprintf(stderr,
                     "[par-seed-sweep] refused: LOAD_TRAINED_WEIGHTS is on "
                     "(this campaign always trains from scratch, no load)\n");
        return 2;
    }
    if (config::SAVE_TRAINED_WEIGHTS)
    {
        std::fprintf(stderr,
                     "[par-seed-sweep] refused: SAVE_TRAINED_WEIGHTS is on "
                     "(parallel in-memory train must not auto-save weights)\n");
        return 2;
    }
    // Host-parallelism requires HCNN single-threaded per Lorenz (no nested pools).
    static_assert(Lorenz::kReadoutNumThreads == 1,
                  "SeedSweep requires Lorenz::kReadoutNumThreads == 1");

    const size_t hw = std::thread::hardware_concurrency()
                          ? std::thread::hardware_concurrency()
                          : 1;
    if (num_threads > hw)
    {
        std::fprintf(stderr,
                     "[par-seed-sweep] num_threads=%zu > hardware_concurrency=%zu; "
                     "capping to %zu\n",
                     num_threads, hw, hw);
        num_threads = hw;
    }
    if (num_threads > num_seeds)
        num_threads = num_seeds;

    ConfigSizeRestore dim_restore(config::DIM, dim);
    ConfigSizeRestore m_restore(config::HISTORY_DEPTH, history_depth);
    ConfigSizeRestore epochs_restore(config::EPOCHS, epochs);
    ConfigFloatRestore sr_restore(
        config::SPECTRAL_RADIUS,
        spectral_radius > 0.0f ? spectral_radius : config::SPECTRAL_RADIUS);
    ConfigFloatRestore is_restore(
        config::INPUT_SCALING,
        input_scaling > 0.0f ? input_scaling : config::INPUT_SCALING);

    const size_t n_in = kNumDriveChannels;

    const fs::path survey_dir = SurveysDir();
    if (!EnsureDir(survey_dir, "par-seed-sweep"))
        return 2;

    const std::string ts = TimestampNow();
    const fs::path rank_csv = survey_dir /
    ("par_seed_sweep_D" + std::to_string(dim) +
        "_M" + std::to_string(history_depth) +
        "_n" + std::to_string(freerun_runs) +
        "_seeds" + std::to_string(num_seeds) +
        "_" + ts + ".csv");
    const fs::path rank_txt = survey_dir /
    ("par_seed_sweep_D" + std::to_string(dim) +
        "_M" + std::to_string(history_depth) +
        "_n" + std::to_string(freerun_runs) +
        "_seeds" + std::to_string(num_seeds) +
        "_" + ts + ".txt");

    // Quiet workers: no per-epoch / per-freerun spam; campaign heartbeats only.
    const bool saved_printf = config::ENABLE_PRINTF;
    const bool saved_progress = config::ENABLE_PROGRESS;
    config::ENABLE_PRINTF = false;
    config::ENABLE_PROGRESS = false;

    ReportBanner("SeedSweep");
    std::printf("[par-seed-sweep] DIM=%zu  M=%zu  num_seeds=%zu  num_threads=%zu  "
                "epochs=%zu  freerun_runs=%d  top_k=%d\n",
                dim, history_depth, num_seeds, num_threads, epochs, freerun_runs,
                top_k);
    std::printf("[par-seed-sweep] base_esn_seed=%llu  (ESN seeds = Mix64(base ^ "
                "FNV* (i+1)); not base+i)\n",
                static_cast<unsigned long long>(base_esn_seed));
    std::printf("[par-seed-sweep] base_orbit_seed=%llu  (shared remix root for train "
                "and freerun; each phase advances its own orbit stream)\n",
                static_cast<unsigned long long>(base_orbit_seed));
    std::printf("[par-seed-sweep] SR=%.4f%s  input_scaling=%.4f%s\n",
                static_cast<double>(config::SPECTRAL_RADIUS),
                spectral_radius > 0.0f ? " (override)" : " (config)",
                static_cast<double>(config::INPUT_SCALING),
                input_scaling > 0.0f ? " (override)" : " (config)");
    {
        const std::string ch = FormatDriveGains();
        std::printf("[par-seed-sweep] drive=[x,y,z,xz]  n_in=%zu  drive_ch=%s\n",
                    n_in, ch.c_str());
    }
    std::printf("[par-seed-sweep] always train in memory (no weight save/load)\n");
    std::printf("[par-seed-sweep] freerun pool = top 10%% per metric; heartbeats on stderr\n");
    std::printf("[par-seed-sweep] report: %s\n", rank_txt.string().c_str());
    std::printf("[par-seed-sweep] CSV:    %s\n", rank_csv.string().c_str());
    std::fflush(stdout);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    std::vector<ParSeedRow> rows(num_seeds);
    std::atomic<size_t> next_job{0};
    std::atomic<size_t> done_count{0};
    // Serialize multi-field stderr lines so heartbeats/failures do not interleave.
    std::mutex stderr_mu;

    {
        std::vector<std::jthread> pool;
        pool.reserve(num_threads);
        for (size_t t = 0; t < num_threads; ++t)
        {
            pool.emplace_back([&]()
            {
                for (;;)
                {
                    const size_t i = next_job.fetch_add(1, std::memory_order_relaxed);
                    if (i >= num_seeds)
                        break;
                    const uint64_t esn = ParallelEsnSeed(base_esn_seed, i);
                    try
                    {
                        rows[i] = EvaluateParSeed(i, esn, base_orbit_seed, freerun_runs);
                    }
                    catch (const std::exception& e)
                    {
                        rows[i].index = i;
                        rows[i].esn_seed = esn;
                        rows[i].ok = false;
                        std::lock_guard<std::mutex> lock(stderr_mu);
                        std::fprintf(stderr,
                                     "[par-seed-sweep] seed idx=%zu esn=%llu FAILED: %s\n",
                                     i, static_cast<unsigned long long>(esn), e.what());
                        std::fflush(stderr);
                    }
                    catch (...)
                    {
                        rows[i].index = i;
                        rows[i].esn_seed = esn;
                        rows[i].ok = false;
                        std::lock_guard<std::mutex> lock(stderr_mu);
                        std::fprintf(stderr,
                                     "[par-seed-sweep] seed idx=%zu esn=%llu FAILED: "
                                     "unknown exception\n",
                                     i, static_cast<unsigned long long>(esn));
                        std::fflush(stderr);
                    }
                    const size_t d = done_count.fetch_add(1, std::memory_order_relaxed) + 1;
                    {
                        std::lock_guard<std::mutex> lock(stderr_mu);
                        std::fprintf(stderr,
                                     "[par-seed-sweep] heartbeat %zu/%zu  "
                                     "(idx=%zu esn=%llu ok=%d)\n",
                                     d, num_seeds, i,
                                     static_cast<unsigned long long>(esn),
                                     rows[i].ok ? 1 : 0);
                        std::fflush(stderr);
                    }
                }
            });
        }
    } // jthread join: all seed jobs complete before report

    {
        std::lock_guard<std::mutex> lock(stderr_mu);
        std::fprintf(stderr, "[par-seed-sweep] all seeds finished (%zu/%zu)\n",
                     done_count.load(std::memory_order_relaxed), num_seeds);
        std::fflush(stderr);
    }
    std::fflush(stdout);

    config::ENABLE_PRINTF = saved_printf;
    config::ENABLE_PROGRESS = saved_progress;

    // Orderings among successful seeds (higher-is-better).
    auto make_order = [&](auto metric)
    {
        std::vector<size_t> ord;
        ord.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i)
            if (rows[i].ok)
                ord.push_back(i);
        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b)
        {
            return metric(rows[a]) > metric(rows[b]);
        });
        return ord;
    };
    const auto by_vxd = make_order([](const ParSeedRow& r) { return r.mean_vpt_x_duty; });
    const auto by_vpt = make_order([](const ParSeedRow& r) { return r.mean_vpt; });
    const auto by_duty = make_order([](const ParSeedRow& r) { return r.mean_duty; });

    auto rank_of = [](const std::vector<size_t>& ord, size_t idx) -> size_t
    {
        for (size_t r = 0; r < ord.size(); ++r)
            if (ord[r] == idx)
                return r + 1;
        return 0;
    };

    std::ostringstream report;
    auto emit = [&](const char* s)
    {
        report << s;
        std::fputs(s, stdout);
    };
    auto emitf = [&](const char* fmt, auto... args)
    {
        char buf[512];
        std::snprintf(buf, sizeof buf, fmt, args...);
        emit(buf);
    };

    emit("\n========================================================================\n");
    emit("=== SeedSweep final report (top-10% freerun means) ===\n");
    emit("========================================================================\n");
    emitf("DIM=%zu  M=%zu  num_seeds=%zu  threads=%zu  epochs=%zu  freerun_runs=%d\n",
          dim, history_depth, num_seeds, num_threads, epochs, freerun_runs);
    emitf("base_esn_seed=%llu  base_orbit_seed=%llu\n",
          static_cast<unsigned long long>(base_esn_seed),
          static_cast<unsigned long long>(base_orbit_seed));
    emitf("ok=%zu / %zu\n\n", by_vxd.size(), num_seeds);

    // Full table sorted by VPT*duty (primary).
    emit("--- All seeds (sorted by mean VPT*duty) ---\n");
    emit("note: VxD_mn is mean of top-10% (VPT*duty) products, not VPT_mn*duty_mn "
        "(top-10% pools are independent per metric).\n");
    emitf("%-4s %6s %20s %10s %10s %10s %10s %6s %6s %6s\n",
          "rank", "idx", "esn_seed", "VxD_mn", "VPT_mn", "duty_mn", "RMSE_mn",
          "r_VxD", "r_VPT", "r_duty");
    if (by_vxd.empty())
    {
        emit("(no successful seeds)\n");
    }
    else
    {
        for (size_t r = 0; r < by_vxd.size(); ++r)
        {
            const size_t i = by_vxd[r];
            const auto& s = rows[i];
            emitf("%-4zu %6zu %20llu %10.3f %10.2f %10.3f %10.6f %6zu %6zu %6zu\n",
                  r + 1, s.index,
                  static_cast<unsigned long long>(s.esn_seed),
                  s.mean_vpt_x_duty, s.mean_vpt, s.mean_duty, s.mean_rmse,
                  rank_of(by_vxd, i), rank_of(by_vpt, i), rank_of(by_duty, i));
        }
    }

    auto emit_top = [&](const char* title, const std::vector<size_t>& ord)
    {
        emitf("\n--- Top %d by %s ---\n", top_k, title);
        if (ord.empty())
        {
            emit("(none)\n");
            return;
        }
        const size_t n = std::min(static_cast<size_t>(top_k), ord.size());
        // Same columns for all tops; section title says sort key (no duplicate metric col).
        emitf("%-4s %6s %20s %10s %10s %10s\n",
              "rank", "idx", "esn_seed", "VxD_mn", "VPT_mn", "duty_mn");
        for (size_t r = 0; r < n; ++r)
        {
            const auto& s = rows[ord[r]];
            emitf("%-4zu %6zu %20llu %10.4f %10.4f %10.4f\n",
                  r + 1, s.index,
                  static_cast<unsigned long long>(s.esn_seed),
                  s.mean_vpt_x_duty, s.mean_vpt, s.mean_duty);
        }
    };
    emit_top("mean VPT*duty", by_vxd);
    emit_top("mean VPT", by_vpt);
    emit_top("mean duty", by_duty);

    // Failed seeds (if any).
    size_t n_fail = 0;
    for (const auto& s : rows)
        if (!s.ok)
            ++n_fail;
    if (n_fail > 0)
    {
        emitf("\n--- Failed / empty freerun seeds (%zu) ---\n", n_fail);
        for (const auto& s : rows)
        {
            if (s.ok)
                continue;
            emitf("  idx=%zu  esn_seed=%llu  n_valid=%zu\n",
                  s.index, static_cast<unsigned long long>(s.esn_seed), s.n_valid);
        }
    }

    emit("\nCherry-pick: use esn_seed from the tables above with Train / "
        "OrbitSweep / FreeRun (same dim/M/dynamics). Weights were not saved here.\n");
    std::fflush(stdout);

    // CSV: one row per seed with ranks on each metric.
    // Windows: must close the ofstream before rename (open handle => I/O error).
    {
        const fs::path tmp = fs::path(rank_csv.string() + ".tmp");
        bool wrote_ok = false;
        {
            std::ofstream csv(tmp, std::ios::out | std::ios::trunc);
            if (!csv)
            {
                std::fprintf(stderr, "[par-seed-sweep] failed to open %s\n",
                             tmp.string().c_str());
            }
            else
            {
                csv << "# SeedSweep\n"
                    << "# freerun_pool=top_10pct_per_metric\n"
                    << "# dim=" << dim << " history_depth=" << history_depth
                    << " epochs=" << epochs << " freerun_runs=" << freerun_runs
                    << " num_seeds=" << num_seeds << " num_threads=" << num_threads << "\n"
                    << "# base_esn_seed=" << base_esn_seed
                    << " base_orbit_seed=" << base_orbit_seed << "\n"
                    << "# note: mean_vpt_x_duty = mean of top-10% (VPT*duty) products; "
                    "not mean_vpt*mean_duty (pools independent per metric)\n"
                    << "idx,esn_seed,ok,mean_vpt_x_duty,mean_vpt,mean_duty,mean_rmse,"
                    "n_valid,rank_vpt_x_duty,rank_vpt,rank_duty\n";
                for (size_t i : by_vxd)
                {
                    const auto& s = rows[i];
                    csv << s.index << ',' << s.esn_seed << ",1,"
                        << s.mean_vpt_x_duty << ',' << s.mean_vpt << ','
                        << s.mean_duty << ',' << s.mean_rmse << ','
                        << s.n_valid << ','
                        << rank_of(by_vxd, i) << ','
                        << rank_of(by_vpt, i) << ','
                        << rank_of(by_duty, i) << '\n';
                }
                for (const auto& s : rows)
                {
                    if (s.ok)
                        continue;
                    csv << s.index << ',' << s.esn_seed << ",0,,,,"
                        << s.n_valid << ",,,\n";
                }
                csv.flush();
                wrote_ok = csv.good();
            }
        } // close handle before rename (required on Windows)
        if (wrote_ok)
        {
            std::error_code ec;
            fs::remove(rank_csv, ec);
            fs::rename(tmp, rank_csv, ec);
            if (ec)
            {
                // Fallback: leave data at .tmp so the overnight run is not lost.
                std::fprintf(stderr,
                             "[par-seed-sweep] rename CSV failed: %s  "
                             "(data left at %s)\n",
                             ec.message().c_str(), tmp.string().c_str());
            }
            else
                ReportWrote("par-seed-sweep", rank_csv);
        }
    }

    {
        std::ofstream txt(rank_txt, std::ios::out | std::ios::trunc);
        if (!txt)
        {
            std::fprintf(stderr, "[par-seed-sweep] failed to write %s\n",
                         rank_txt.string().c_str());
        }
        else
        {
            txt << report.str();
            txt.flush();
            if (txt.good())
                ReportWrote("par-seed-sweep", rank_txt);
        }
    }

    ReportDone("par-seed-sweep",
               std::chrono::duration<double>(clock::now() - t0).count());
    return by_vxd.empty() ? 1 : 0;
}

// ---------------------------------------------------------------------------
// OrbitSweep (load-only; parallel one-freerun-per-orbit ranking)
// ---------------------------------------------------------------------------
int OrbitSweep(size_t dim,
               size_t history_depth,
               uint64_t base_esn_seed,
               float spectral_radius,
               float input_scaling,
               uint64_t base_orbit_seed,
               size_t num_orbits,
               size_t num_threads,
               const char* weights_stem,
               int top_k)
{
    if (!ValidateDim(dim, "par-orbit-sweep"))
        return 2;
    if (!ValidateHistoryDepth(history_depth, "par-orbit-sweep"))
        return 2;
    if (num_orbits < 1)
    {
        std::fprintf(stderr, "[par-orbit-sweep] refused: num_orbits=%zu (need >= 1)\n",
                     num_orbits);
        return 2;
    }
    if (num_threads < 1)
    {
        std::fprintf(stderr, "[par-orbit-sweep] refused: num_threads=%zu (need >= 1)\n",
                     num_threads);
        return 2;
    }
    if (top_k < 1)
        top_k = 1;
    if (!ValidateDynamicsOverrides(spectral_radius, input_scaling, "par-orbit-sweep"))
        return 2;

    const char* stem_arg = (weights_stem && weights_stem[0] != '\0')
                               ? weights_stem
                               : config::LOAD_WEIGHTS_STEM;
    if (stem_arg == nullptr || stem_arg[0] == '\0')
    {
        std::fprintf(stderr,
                     "[par-orbit-sweep] refused: no weights path "
                     "(pass weights_stem or set config::LOAD_WEIGHTS_STEM; "
                     "run Train first)\n");
        return 2;
    }
    const std::string stem_str(stem_arg);
    {
        std::error_code ec;
        if (!fs::exists(fs::path(stem_str + ".hcnw"), ec) || ec)
        {
            std::fprintf(stderr,
                         "[par-orbit-sweep] refused: missing %s.hcnw  (Train first)\n",
                         stem_str.c_str());
            return 2;
        }
    }

    static_assert(Lorenz::kReadoutNumThreads == 1,
                  "OrbitSweep requires Lorenz::kReadoutNumThreads == 1");

    const size_t hw = std::thread::hardware_concurrency()
                          ? std::thread::hardware_concurrency()
                          : 1;
    if (num_threads > hw)
    {
        std::fprintf(stderr,
                     "[par-orbit-sweep] num_threads=%zu > hardware_concurrency=%zu; "
                     "capping to %zu\n",
                     num_threads, hw, hw);
        num_threads = hw;
    }
    if (num_threads > num_orbits)
        num_threads = num_orbits;

    ConfigSizeRestore dim_restore(config::DIM, dim);
    ConfigSizeRestore m_restore(config::HISTORY_DEPTH, history_depth);
    ConfigFloatRestore sr_restore(
        config::SPECTRAL_RADIUS,
        spectral_radius > 0.0f ? spectral_radius : config::SPECTRAL_RADIUS);
    ConfigFloatRestore is_restore(
        config::INPUT_SCALING,
        input_scaling > 0.0f ? input_scaling : config::INPUT_SCALING);

    const size_t n_in = kNumDriveChannels;

    const fs::path survey_dir = SurveysDir();
    if (!EnsureDir(survey_dir, "par-orbit-sweep"))
        return 2;

    const std::string ts = TimestampNow();
    const fs::path rank_txt = survey_dir /
    ("par_orbit_sweep_D" + std::to_string(dim) +
        "_M" + std::to_string(history_depth) +
        "_orbits" + std::to_string(num_orbits) +
        "_" + ts + ".txt");

    const bool saved_printf = config::ENABLE_PRINTF;
    const bool saved_progress = config::ENABLE_PROGRESS;
    config::ENABLE_PRINTF = false;
    config::ENABLE_PROGRESS = false;

    ReportBanner("OrbitSweep (load-only)");
    std::printf("[par-orbit-sweep] DIM=%zu  M=%zu  num_orbits=%zu  num_threads=%zu  "
                "top_k=%d\n",
                dim, history_depth, num_orbits, num_threads, top_k);
    std::printf("[par-orbit-sweep] base_esn_seed=%llu\n",
                static_cast<unsigned long long>(base_esn_seed));
    std::printf("[par-orbit-sweep] base_orbit_seed=%llu  (orbit_i = Mix64(base ^ FNV*(i+1)))\n",
                static_cast<unsigned long long>(base_orbit_seed));
    std::printf("[par-orbit-sweep] SR=%.4f%s  input_scaling=%.4f%s\n",
                static_cast<double>(config::SPECTRAL_RADIUS),
                spectral_radius > 0.0f ? " (override)" : " (config)",
                static_cast<double>(config::INPUT_SCALING),
                input_scaling > 0.0f ? " (override)" : " (config)");
    {
        const std::string ch = FormatDriveGains();
        std::printf("[par-orbit-sweep] drive=[x,y,z,xz]  n_in=%zu  drive_ch=%s\n",
                    n_in, ch.c_str());
    }
    std::printf("[par-orbit-sweep] load: %s  (.hcnw + .arch.json)\n", stem_str.c_str());
    std::printf("[par-orbit-sweep] freerun %zu orbits in parallel (quiet per-job load)\n",
                num_orbits);
    std::printf("[par-orbit-sweep] report: %s\n", rank_txt.string().c_str());
    std::fflush(stdout);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    struct OrbitRow
    {
        size_t index = 0;
        uint64_t orbit_seed = 0;
        bool ok = false;
        double vpt = 0;
        double duty = 0;
        double vpt_x_duty = 0;
        double rmse = 0;
        double ic_x = 0, ic_y = 0, ic_z = 0;
    };
    std::vector<OrbitRow> rows(num_orbits);
    std::atomic<size_t> next_job{0};
    std::mutex stderr_mu; // FAILED lines only (workers stay quiet on success)

    {
        std::vector<std::jthread> pool;
        pool.reserve(num_threads);
        for (size_t t = 0; t < num_threads; ++t)
        {
            pool.emplace_back([&]()
            {
                for (;;)
                {
                    const size_t i = next_job.fetch_add(1, std::memory_order_relaxed);
                    if (i >= num_orbits)
                        break;
                    const uint64_t orbit = Mix64(
                        base_orbit_seed ^
                        (0x100000001B3ULL * (static_cast<uint64_t>(i) + 1ULL)));
                    OrbitRow row;
                    row.index = i;
                    row.orbit_seed = orbit;
                    try
                    {
                        Lorenz lorenz(base_esn_seed, base_orbit_seed);
                        lorenz.LoadTrainedWeights(stem_str.c_str(), /*log_load=*/false);
                        const FreeRunResult r =
                            lorenz.FreeRun(false, nullptr, 0, orbit);
                        if (r.valid)
                        {
                            row.ok = true;
                            row.vpt = r.vpt_lt;
                            row.duty = r.duty;
                            row.vpt_x_duty = r.vpt_x_duty;
                            row.rmse = r.rmse;
                            const auto ic = Lorenz::IcFromOrbitSeed(orbit);
                            row.ic_x = ic.x;
                            row.ic_y = ic.y;
                            row.ic_z = ic.z;
                        }
                    }
                    catch (const std::exception& e)
                    {
                        row.ok = false;
                        std::lock_guard<std::mutex> lock(stderr_mu);
                        std::fprintf(stderr,
                                     "[par-orbit-sweep] orbit idx=%zu seed=%llu FAILED: %s\n",
                                     i, static_cast<unsigned long long>(orbit), e.what());
                        std::fflush(stderr);
                    }
                    catch (...)
                    {
                        row.ok = false;
                        std::lock_guard<std::mutex> lock(stderr_mu);
                        std::fprintf(stderr,
                                     "[par-orbit-sweep] orbit idx=%zu seed=%llu FAILED: "
                                     "unknown exception\n",
                                     i, static_cast<unsigned long long>(orbit));
                        std::fflush(stderr);
                    }
                    rows[i] = row;
                }
            });
        }
    }

    std::fprintf(stderr, "[par-orbit-sweep] all orbits finished (%zu)\n", num_orbits);
    std::fflush(stderr);
    std::fflush(stdout);

    config::ENABLE_PRINTF = saved_printf;
    config::ENABLE_PROGRESS = saved_progress;

    auto make_order = [&](auto metric)
    {
        std::vector<size_t> ord;
        ord.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i)
            if (rows[i].ok)
                ord.push_back(i);
        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b)
        {
            return metric(rows[a]) > metric(rows[b]);
        });
        return ord;
    };
    const auto by_vxd = make_order([](const OrbitRow& r) { return r.vpt_x_duty; });
    const auto by_vpt = make_order([](const OrbitRow& r) { return r.vpt; });
    const auto by_duty = make_order([](const OrbitRow& r) { return r.duty; });

    auto rank_of = [](const std::vector<size_t>& ord, size_t idx) -> size_t
    {
        for (size_t r = 0; r < ord.size(); ++r)
            if (ord[r] == idx)
                return r + 1;
        return 0;
    };

    // Large N (e.g. 10k orbits): keep only extremes for stdout + TXT report.
    // Ranking stays over all successful freeruns; the file is not the full table.
    constexpr size_t kFileKeepTop = 100;
    constexpr size_t kFileKeepBottom = 10;
    const size_t n_ok = by_vxd.size();
    const size_t n_top = std::min(kFileKeepTop, n_ok);
    const size_t bot_start =
        (n_ok > kFileKeepBottom) ? (n_ok - kFileKeepBottom) : 0;
    size_t n_fail = 0;
    for (const auto& s : rows)
        if (!s.ok)
            ++n_fail;

    std::ostringstream report;
    auto emit = [&](const char* s)
    {
        report << s;
        std::fputs(s, stdout);
    };
    auto emitf = [&](const char* fmt, auto... args)
    {
        char buf[640];
        std::snprintf(buf, sizeof buf, fmt, args...);
        emit(buf);
    };

    auto emit_orbit_row = [&](size_t rank1, size_t i)
    {
        const auto& s = rows[i];
        emitf("%-4zu %6zu %20llu %10.3f %10.2f %10.3f %10.6f %6zu %6zu %6zu\n",
              rank1, s.index,
              static_cast<unsigned long long>(s.orbit_seed),
              s.vpt_x_duty, s.vpt, s.duty, s.rmse,
              rank_of(by_vxd, i), rank_of(by_vpt, i), rank_of(by_duty, i));
    };

    emit("\n========================================================================\n");
    emit("=== OrbitSweep final report (load-only; one freerun per orbit) ===\n");
    emit("========================================================================\n");
    emitf("DIM=%zu  M=%zu  esn_seed=%llu  num_orbits=%zu  threads=%zu\n",
          dim, history_depth, static_cast<unsigned long long>(base_esn_seed),
          num_orbits, num_threads);
    emitf("base_orbit_seed=%llu\n", static_cast<unsigned long long>(base_orbit_seed));
    emitf("weights=%s\n", stem_str.c_str());
    emitf("ok=%zu / %zu  failed=%zu\n", n_ok, num_orbits, n_fail);
    emitf("file/stdout keep: top %zu + bottom %zu by VPT*duty (not full table)\n\n",
          kFileKeepTop, kFileKeepBottom);

    emit("note: one freerun per orbit; VxD = VPT*duty for that run.\n");
    auto emit_orbit_header = [&]() {
        emitf("%-4s %6s %20s %10s %10s %10s %10s %6s %6s %6s\n",
              "rank", "idx", "orbit_seed", "VxD", "VPT", "duty", "RMSE",
              "r_VxD", "r_VPT", "r_duty");
    };
    if (by_vxd.empty())
    {
        emit("(no successful freeruns)\n");
    }
    else
    {
        emitf("\n--- Top %zu by VPT*duty ---\n", n_top);
        emit_orbit_header();
        for (size_t r = 0; r < n_top; ++r)
            emit_orbit_row(r + 1, by_vxd[r]);

        // Bottom block when the full table is larger than the top keep set.
        if (n_ok > n_top)
        {
            emitf("\n--- Bottom %zu by VPT*duty (worst%s) ---\n",
                  n_ok - bot_start,
                  (bot_start < n_top) ? "; overlaps top" : "");
            emit_orbit_header();
            for (size_t r = bot_start; r < n_ok; ++r)
                emit_orbit_row(r + 1, by_vxd[r]);
        }
    }

    auto emit_top = [&](const char* title, const std::vector<size_t>& ord)
    {
        emitf("\n--- Top %d by %s ---\n", top_k, title);
        if (ord.empty())
        {
            emit("(none)\n");
            return;
        }
        const size_t n = std::min(static_cast<size_t>(top_k), ord.size());
        emitf("%-4s %6s %20s %10s %10s %10s %10s %10s %10s\n",
              "rank", "idx", "orbit_seed", "VxD", "VPT", "duty",
              "ic_x", "ic_y", "ic_z");
        for (size_t r = 0; r < n; ++r)
        {
            const auto& s = rows[ord[r]];
            emitf("%-4zu %6zu %20llu %10.4f %10.4f %10.4f %10.6f %10.6f %10.6f\n",
                  r + 1, s.index,
                  static_cast<unsigned long long>(s.orbit_seed),
                  s.vpt_x_duty, s.vpt, s.duty,
                  s.ic_x, s.ic_y, s.ic_z);
        }
    };
    emit_top("VPT*duty", by_vxd);
    emit_top("VPT", by_vpt);
    emit_top("duty", by_duty);

    if (!by_vxd.empty())
    {
        const auto& best = rows[by_vxd[0]];
        emitf("\nBest orbit_seed=%llu  VxD=%.4f  VPT=%.4f  duty=%.4f\n",
              static_cast<unsigned long long>(best.orbit_seed),
              best.vpt_x_duty, best.vpt, best.duty);
        emitf("  IC=(%.6f, %.6f, %.6f)  (display only)\n",
              best.ic_x, best.ic_y, best.ic_z);
        emitf("  FreeRun(%zu, %zu, %llu, /*SR*/0.f, /*IS*/0.f, %llu, \"%s\");\n",
              dim, history_depth,
              static_cast<unsigned long long>(base_esn_seed),
              static_cast<unsigned long long>(best.orbit_seed),
              stem_str.c_str());
    }

    emit("\nCherry-pick: FreeRun with orbit_seed above (not rounded IC floats), "
         "same esn_seed and weights path.\n");
    std::fflush(stdout);

    {
        std::ofstream txt(rank_txt, std::ios::out | std::ios::trunc);
        if (!txt)
        {
            std::fprintf(stderr, "[par-orbit-sweep] failed to write %s\n",
                         rank_txt.string().c_str());
        }
        else
        {
            txt << report.str();
            txt.flush();
            if (txt.good())
                ReportWrote("par-orbit-sweep", rank_txt);
        }
    }

    ReportDone("par-orbit-sweep",
               std::chrono::duration<double>(clock::now() - t0).count());
    return by_vxd.empty() ? 1 : 0;
}
