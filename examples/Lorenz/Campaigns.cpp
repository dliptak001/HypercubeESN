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
#include <windows.h>

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

// Optional SR / IS (SeedSweep, FreeRun, FreeRunSurvey, Parallel*).
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

// ---- Shared paths (all campaigns) -----------------------------------------
// C:\HypercubeESNRuns\results\{traces|surveys|campaigns}

fs::path RunsRoot() { return config::RUNS_DIR; }
fs::path TracesDir() { return RunsRoot() / "traces"; }
fs::path SurveysDir() { return RunsRoot() / "surveys"; }
fs::path CampaignsDir() { return config::RESULTS_DIR; } // .../campaigns

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

fs::path EnsureResultsDir()
{
    EnsureDir(CampaignsDir(), "results");
    return CampaignsDir();
}

// Atomic write: full content to path via path.tmp then rename. Returns false on fail.
bool WriteAtomicFile(const fs::path& path, const std::string& content, const char* tag)
{
    if (!EnsureDir(path.parent_path(), tag))
        return false;
    const fs::path tmp = fs::path(path.string() + ".tmp");
    {
        std::ofstream out(tmp, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!out)
        {
            std::fprintf(stderr, "[%s] failed to open %s\n", tag, tmp.string().c_str());
            return false;
        }
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
        out.flush();
        if (!out.good())
        {
            std::fprintf(stderr, "[%s] write failed for %s\n", tag, tmp.string().c_str());
            return false;
        }
    }
    std::error_code ec;
    fs::remove(path, ec);
    fs::rename(tmp, path, ec);
    if (ec)
    {
        std::fprintf(stderr, "[%s] rename %s -> %s failed: %s\n",
                     tag, tmp.string().c_str(), path.string().c_str(),
                     ec.message().c_str());
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

// Default weight stem: {MODEL_SAVE_DIR}/lorenz_seed{S}_D{dim}_M{M}
std::string DefaultWeightStem(uint64_t esn_seed, size_t dim, size_t history_depth)
{
    return (fs::path(config::MODEL_SAVE_DIR) /
            ("lorenz_seed" + std::to_string(esn_seed) +
             "_D" + std::to_string(dim) +
             "_M" + std::to_string(history_depth))).string();
}

const char* ActivationName(ReadoutActivation a)
{
    switch (a)
    {
    case ReadoutActivation::TANH:       return "TANH";
    case ReadoutActivation::RELU:       return "RELU";
    case ReadoutActivation::LEAKY_RELU: return "LEAKY_RELU";
    default:                            return "NONE";
    }
}


// Full run metadata for post-analysis (comment lines in CSV; prose in .txt).
void WriteMetadataBlock(std::ostream& o, const char* job, const std::string& timestamp,
                        uint64_t base_seed, uint64_t orbit_seed,
                        size_t num_trials, int num_runs,
                        size_t history_depth)
{
    o << "# HypercubeESN Lorenz results\n"
      << "# job=" << job << "\n"
      << "# timestamp=" << timestamp << "\n"
      << "# freerun=edge_warmup_then_past_span\n"
      << "# dim=" << config::DIM
      << "  N=" << (size_t{1} << config::DIM)
      << "  history_depth_M=" << history_depth << "\n"
      << "# epochs=" << config::EPOCHS
      << "  warmup=" << config::WARMUP_STEPS
      << "  train_window=" << config::TRAINING_WINDOW_SIZE
      << "  freerun_window=" << config::FREE_RUN_WINDOW_SIZE << "\n"
      << "# theta=" << config::VPT_THRESHOLD
      << "  input_scaling=" << config::INPUT_SCALING
      << "  SR=" << config::SPECTRAL_RADIUS
      << "  leak=" << config::LEAK_RATE << "\n"
      << "# conv_channels=" << config::CONV_CHANNELS
      << "  readout_slices=" << config::READOUT_SLICES
      << "  pooling=" << (config::USE_POOLING ? "on" : "off")
      << "  num_layers=" << config::NUM_LAYERS
      << "  activation=" << ActivationName(config::READOUT_ACTIVATION)
      << "\n"
      << "# base_seed=" << base_seed
      << "  orbit_seed=" << orbit_seed
      << "  num_trials=" << num_trials
      << "  freeruns_per_trial=" << num_runs << "\n"
      << "# load_weights=" << (config::LOAD_TRAINED_WEIGHTS ? "on" : "off")
      << "  save_weights=" << (config::SAVE_TRAINED_WEIGHTS ? "on" : "off") << "\n"
      << "# drive=[x,y,z,xz]  num_inputs=" << kNumDriveChannels << "\n";
    {
        const size_t n_in = kNumDriveChannels;
        o << "# drive_ch=[";
        for (size_t i = 0; i < n_in; ++i)
        {
            if (i)
                o << ',';
            o << config::INPUT_SCALE_CH[i];
        }
        o << "] (x input_scaling; [x,y,z,xz])\n";
    }
    o << "# metrics=mean_of_trial_means (sample std across trials)\n"
      << "# score_vpt_x_duty=VPT_lt*duty\n"
      << "# freerun_metrics=VPT,duty,VPT*duty,RMSE\n"
      << "# freerun_pool=top_10pct_per_metric (max(1,ceil(n/10)); weak ICs discarded; "
         "higher: VPT duty VPT*duty; lower: RMSE)\n";
}

const char* SurveyCsvHeader()
{
    return "M,mean_vpt,std_vpt,mean_rmse,std_rmse,mean_duty,std_duty,"
           "mean_vpt_x_duty,std_vpt_x_duty,"
           "n_trials_ok,num_trials,num_runs,wall_seconds,ok\n";
}

void WriteSurveyCsvRow(std::ostream& o, const SurveySummary& s)
{
    o << s.history_depth << ','
      << s.mean_vpt << ',' << s.std_vpt << ','
      << s.mean_rmse << ',' << s.std_rmse << ','
      << s.mean_duty << ',' << s.std_duty << ','
      << s.mean_vpt_x_duty << ',' << s.std_vpt_x_duty << ','
      << s.n_trials_ok << ',' << s.num_trials << ',' << s.num_runs << ','
      << s.wall_seconds << ','
      << (s.ok ? 1 : 0) << '\n';
}

// (1)+(2)+(3): one survey -> CSV + TXT under RESULTS_DIR with metadata.
void WriteSurveyResultFiles(const SurveySummary& s)
{
    const fs::path dir = EnsureResultsDir();
    if (!fs::exists(dir))
        return;

    const std::string ts = TimestampNow();
    const fs::path csv_path = dir / ("Survey_" + ts + "_M" + std::to_string(s.history_depth) + ".csv");
    const fs::path txt_path = dir / ("Survey_" + ts + "_M" + std::to_string(s.history_depth) + ".txt");

    {
        std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
        if (!csv)
        {
            std::fprintf(stderr, "[results] failed to write %s\n", csv_path.string().c_str());
            return;
        }
        WriteMetadataBlock(csv, "survey", ts, s.base_seed, s.orbit_seed,
                           s.num_trials, s.num_runs, s.history_depth);
        csv << SurveyCsvHeader();
        WriteSurveyCsvRow(csv, s);
    }
    {
        std::ofstream txt(txt_path, std::ios::out | std::ios::trunc);
        if (!txt)
        {
            std::fprintf(stderr, "[results] failed to write %s\n", txt_path.string().c_str());
            return;
        }
        WriteMetadataBlock(txt, "survey", ts, s.base_seed, s.orbit_seed,
                           s.num_trials, s.num_runs, s.history_depth);
        txt << "\nSurvey aggregate (mean of trial-means)\n";
        txt << "  M=" << s.history_depth
            << "  trials_ok=" << s.n_trials_ok << "/" << s.num_trials
            << "  freeruns/trial=" << s.num_runs << "\n";
        txt << "  VPT   mean=" << s.mean_vpt << "  std=" << s.std_vpt
            << "  (top 10% freeruns)\n";
        txt << "  RMSE  mean=" << s.mean_rmse << "  std=" << s.std_rmse
            << "  (top 10% freeruns, lower-is-better)\n";
        txt << "  duty  mean=" << s.mean_duty << "  std=" << s.std_duty
            << "  (top 10% freeruns)\n";
        txt << "  VPT*duty mean=" << s.mean_vpt_x_duty << "  std=" << s.std_vpt_x_duty
            << "  (top 10% freeruns)\n";
        txt << "  wall_seconds=" << s.wall_seconds << "  ok=" << (s.ok ? 1 : 0) << "\n";
    }

    // stdout (not stderr): CLion/Windows consoles merge streams asynchronously and will
    // splice stderr mid-line into printf tables if we log "wrote" on a different stream.
    ReportWrote("results", csv_path);
    ReportWrote("results", txt_path);
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

/// One trial: report text + per-trial free-run means (valid free-runs only).
struct TrialBundle
{
    std::string report;
    bool ok = false; ///< at least one valid free-run
    double mean_vpt = 0;
    double mean_rmse = 0;
    double mean_duty = 0;
    double mean_vpt_x_duty = 0;
    size_t n_valid = 0; ///< valid freeruns before top-10% filter
    double wall_seconds = 0;
};

TrialBundle RunTrial(uint64_t esn_seed, uint64_t orbit_seed, int num_runs)
{
    TrialBundle tb;
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    auto progress = [esn_seed](const char* phase, int done, int total)
    {
        if (!config::ENABLE_PROGRESS)
            return;
        std::fprintf(stderr, "[seed %llu] %s %d/%d\n",
                     static_cast<unsigned long long>(esn_seed), phase, done, total);
        std::fflush(stderr);
    };

    Lorenz lorenz(esn_seed, orbit_seed);
    if (config::LOAD_TRAINED_WEIGHTS)
    {
        progress("load weights", 0, 1);
        lorenz.LoadTrainedWeights();
        progress("load done; start free-runs", 0, num_runs);
    }
    else
    {
        progress("start train", 0, static_cast<int>(config::EPOCHS));
        lorenz.Train();
        progress("train done; start free-runs", 0, num_runs);
    }

    std::vector<FreeRunResult> results;
    results.reserve(num_runs);
    const int prog_every = (num_runs <= 20) ? 1 : std::max(10, num_runs / 20);
    for (int i = 0; i < num_runs; i++)
    {
        results.push_back(lorenz.FreeRun(false, nullptr, 0));
        if ((i + 1) % prog_every == 0 || i + 1 == num_runs)
            progress("free-run", i + 1, num_runs);
    }

    std::string out;
    char buf[384];
    auto emit = [&](const char* s) { out += s; };

    std::snprintf(buf, sizeof buf,
                  "\n=== ESN seed %llu : %d free-runs (orbit seed %llu) ===\n",
                  static_cast<unsigned long long>(esn_seed), num_runs,
                  static_cast<unsigned long long>(orbit_seed));
    emit(buf);

    std::vector<double> vpt_lts, rmses, duties, vpt_x_duties;
    size_t censored = 0, invalid = 0;
    for (const auto& r : results)
    {
        if (!r.valid)
        {
            ++invalid;
            continue;
        }
        vpt_lts.push_back(r.vpt_lt);
        rmses.push_back(r.rmse);
        duties.push_back(r.duty);
        vpt_x_duties.push_back(r.vpt_x_duty);
        if (!r.crossed) ++censored;
    }

    // Top 10% of ICs per metric (independent). Higher: VPT, duty, VPT*duty.
    // Lower: RMSE. Weak ICs discarded on purpose.
    const size_t n_full = vpt_lts.size();
    const std::vector<double> vpt_best = Top10HigherIsBetter(vpt_lts);
    const std::vector<double> duty_best = Top10HigherIsBetter(duties);
    const std::vector<double> vxd_best = Top10HigherIsBetter(vpt_x_duties);
    const std::vector<double> rmse_best = Top10LowerIsBetter(rmses);

    tb.n_valid = n_full;
    tb.ok = n_full > 0;
    if (tb.ok)
    {
        tb.mean_vpt = MeanOf(vpt_best);
        tb.mean_rmse = MeanOf(rmse_best);
        tb.mean_duty = MeanOf(duty_best);
        tb.mean_vpt_x_duty = MeanOf(vxd_best);
    }

    auto report = [&](const char* label, std::vector<double> v, int prec,
                      size_t n_pool = 0)
    {
        if (v.empty())
        {
            std::snprintf(buf, sizeof buf, "  %-20s (no valid runs)\n", label);
            emit(buf);
            return;
        }
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        double mean = 0, sd = 0;
        MeanStd(v, mean, sd);
        const double median = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
        if (n_pool > 0 && n_pool != n)
            std::snprintf(buf, sizeof buf,
                          "  %-20s n=%2zu (top 10%% of %zu)  min=%.*f  max=%.*f  "
                          "mean=%.*f  median=%.*f  std=%.*f\n",
                          label, n, n_pool,
                          prec, v.front(), prec, v.back(),
                          prec, mean, prec, median, prec, sd);
        else
            std::snprintf(buf, sizeof buf,
                          "  %-20s n=%2zu  min=%.*f  max=%.*f  mean=%.*f  median=%.*f  std=%.*f\n",
                          label, n, prec, v.front(), prec, v.back(),
                          prec, mean, prec, median, prec, sd);
        emit(buf);
    };

    std::snprintf(buf, sizeof buf, "\n=== Free-run stats (%d runs) ===\n", num_runs);
    emit(buf);
    report("VPT (lt)", vpt_best, 2, n_full);
    report("free-run RMSE", rmse_best, 6, n_full);
    report("duty (<=theta)", duty_best, 3, n_full);
    report("VPT*duty (lt)", vxd_best, 3, n_full);
    std::snprintf(buf, sizeof buf,
                  "  note: VPT / duty use theta=VPT_THRESHOLD=%.2f; "
                  "VPT*duty = VPT_lt * duty\n",
                  config::VPT_THRESHOLD);
    emit(buf);
    emit("  note: freerun stats = VPT, duty, VPT*duty, RMSE only; top 10% of ICs per "
         "metric (keep max(1,ceil(n/10)); higher: VPT duty VPT*duty; lower: RMSE). "
         "Weak ICs discarded on purpose; see examples/Lorenz/README.md\n");
    if (config::LOAD_TRAINED_WEIGHTS)
        emit("  note: readout loaded from disk (Train skipped)\n");
    if (censored)
    {
        std::snprintf(buf, sizeof buf,
                      "  note: %zu/%zu run(s) never crossed VPT_THRESHOLD=%.2f; "
                      "their VPT is counted at the window floor (a lower bound)\n",
                      censored, n_full, config::VPT_THRESHOLD);
        emit(buf);
    }
    if (invalid)
    {
        std::snprintf(buf, sizeof buf, "  note: %zu run(s) scored 0 steps and are excluded from the stats\n", invalid);
        emit(buf);
    }

    // Leaderboards: VPT, duty, VPT*duty (no RMSE top list -- aggregates cover RMSE).
    std::vector<const FreeRunResult*> valid;
    valid.reserve(results.size());
    for (const auto& r : results)
        if (r.valid) valid.push_back(&r);

    const size_t top_n = std::min<size_t>(5, valid.size());

    std::snprintf(buf, sizeof buf, "\n=== Top %zu highest VPT (lt) ===\n", top_n);
    emit(buf);
    std::sort(valid.begin(), valid.end(),
              [](const FreeRunResult* a, const FreeRunResult* b) { return a->vpt_lt > b->vpt_lt; });
    for (size_t i = 0; i < top_n; i++)
        emit(valid[i]->row.c_str());

    std::snprintf(buf, sizeof buf, "\n=== Top %zu highest duty (<=theta) ===\n", top_n);
    emit(buf);
    std::sort(valid.begin(), valid.end(),
              [](const FreeRunResult* a, const FreeRunResult* b) { return a->duty > b->duty; });
    for (size_t i = 0; i < top_n; i++)
        emit(valid[i]->row.c_str());

    std::snprintf(buf, sizeof buf, "\n=== Top %zu highest VPT*duty (lt) ===\n", top_n);
    emit(buf);
    std::sort(valid.begin(), valid.end(),
              [](const FreeRunResult* a, const FreeRunResult* b) {
                  return a->vpt_x_duty > b->vpt_x_duty;
              });
    for (size_t i = 0; i < top_n; i++)
        emit(valid[i]->row.c_str());

    tb.wall_seconds = std::chrono::duration<double>(clock::now() - t0).count();
    char time_buf[64];
    FormatWallTime(time_buf, sizeof time_buf, tb.wall_seconds);
    std::snprintf(buf, sizeof buf, "\n=== Trial wall time: %s ===\n", time_buf);
    emit(buf);

    tb.report = std::move(out);
    return tb;
}
} // namespace

// ---------------------------------------------------------------------------
// Campaign_SeedSurvey
// ---------------------------------------------------------------------------
int Campaign_SeedSurvey(size_t dim, size_t num_threads, int num_runs, uint64_t base_seed,
                        uint64_t orbit_seed, bool completion_beep, SurveySummary* out)
{
    if (!ValidateDim(dim, "survey"))
        return 2;
    ConfigSizeRestore dim_restore(config::DIM, dim);

    ReportBanner("SeedSurvey");

    config::ENABLE_PRINTF = false;

    {
        Lorenz probe(base_seed, orbit_seed);
        std::cout << probe.ReadoutArchSummary();
    }

    const size_t hw = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 1;
    const size_t max_threads = 4 * hw;
    if (num_threads == 0)
        num_threads = hw;
    num_threads = std::min(num_threads, max_threads);
    if (num_runs < 1)
        num_runs = 1;

    const long long train_steps_per_trial =
        static_cast<long long>(config::EPOCHS) *
        static_cast<long long>(config::TRAINING_WINDOW_SIZE);
    const long long freerun_steps_per_trial =
        static_cast<long long>(num_runs) *
        (static_cast<long long>(config::WARMUP_STEPS) +
         static_cast<long long>(config::FREE_RUN_WINDOW_SIZE));
    std::printf("[survey] load_weights=%s  (edge-warmup free-run; ext-fb off)\n",
                config::LOAD_TRAINED_WEIGHTS ? "on" : "off");
    if (config::LOAD_TRAINED_WEIGHTS)
        std::printf("[survey] load stem: %s\n", config::LOAD_WEIGHTS_STEM);
    std::printf("[survey] %zu trial(s) x %d free-run(s)  DIM=%zu N=%zu  history_depth(M)=%zu  "
                "epochs=%zu  train_window=%d  warmup=%zu  freerun_window=%zu\n",
                num_threads, num_runs, config::DIM, size_t{1} << config::DIM,
                config::HISTORY_DEPTH, config::EPOCHS, config::TRAINING_WINDOW_SIZE,
                config::WARMUP_STEPS, config::FREE_RUN_WINDOW_SIZE);
    std::printf("[survey] ~%lld train reservoir-steps/trial + ~%lld free-run "
                "reservoir-steps/trial (warmup+score); progress on stderr\n",
                train_steps_per_trial, freerun_steps_per_trial);
    if (num_runs >= 500)
        std::printf("[survey] NOTE: NUM_RUNS=%d is a heavy survey -- expect multi-hour wall "
                    "clock at this DIM/epoch/window setting\n",
                    num_runs);
    std::fflush(stdout);

    using clock = std::chrono::steady_clock;
    const auto survey_t0 = clock::now();

    std::vector<TrialBundle> trials(num_threads);
    {
        std::vector<std::jthread> pool;
        pool.reserve(num_threads);
        for (size_t t = 0; t < num_threads; t++)
        {
            pool.emplace_back([&, t]
            {
                const uint64_t esn_seed = base_seed + t;
                try
                {
                    trials[t] = RunTrial(esn_seed, orbit_seed, num_runs);
                }
                catch (const std::exception& e)
                {
                    char buf[512];
                    std::snprintf(buf, sizeof buf,
                                  "\n=== ESN seed %llu FAILED ===\n  %s\n",
                                  static_cast<unsigned long long>(esn_seed), e.what());
                    trials[t].report = buf;
                    trials[t].ok = false;
                }
                catch (...)
                {
                    char buf[256];
                    std::snprintf(buf, sizeof buf,
                                  "\n=== ESN seed %llu FAILED ===\n  unknown exception\n",
                                  static_cast<unsigned long long>(esn_seed));
                    trials[t].report = buf;
                    trials[t].ok = false;
                }
            });
        }
    }

    for (const auto& tr : trials)
        std::fputs(tr.report.c_str(), stdout);

    // Mean-of-trial-means (and std across trials) -- same style as historical TRACKING.
    std::vector<double> trial_vpt, trial_rmse, trial_duty, trial_vxd;
    trial_vpt.reserve(num_threads);
    trial_rmse.reserve(num_threads);
    trial_duty.reserve(num_threads);
    trial_vxd.reserve(num_threads);
    for (const auto& tr : trials)
    {
        if (!tr.ok)
            continue;
        trial_vpt.push_back(tr.mean_vpt);
        trial_rmse.push_back(tr.mean_rmse);
        trial_duty.push_back(tr.mean_duty);
        trial_vxd.push_back(tr.mean_vpt_x_duty);
    }

    SurveySummary sum;
    sum.history_depth = config::HISTORY_DEPTH;
    sum.num_inputs = kNumDriveChannels;
    sum.num_trials = num_threads;
    sum.num_runs = num_runs;
    sum.n_trials_ok = trial_vpt.size();
    sum.base_seed = base_seed;
    sum.orbit_seed = orbit_seed;
    MeanStd(trial_vpt, sum.mean_vpt, sum.std_vpt);
    MeanStd(trial_rmse, sum.mean_rmse, sum.std_rmse);
    MeanStd(trial_duty, sum.mean_duty, sum.std_duty);
    MeanStd(trial_vxd, sum.mean_vpt_x_duty, sum.std_vpt_x_duty);
    sum.wall_seconds = std::chrono::duration<double>(clock::now() - survey_t0).count();
    sum.ok = sum.n_trials_ok > 0;

    char time_buf[64];
    FormatWallTime(time_buf, sizeof time_buf, sum.wall_seconds);
    std::printf("\n=== Survey wall time: %s (%zu trial(s), %zu ok) ===\n",
                time_buf, num_threads, sum.n_trials_ok);
    if (sum.ok)
    {
        std::printf("[survey] aggregate (mean of %zu trial-means; top-10%% ICs):  "
                    "VPT mean=%.3f std=%.3f  RMSE mean=%.6f std=%.6f  "
                    "duty mean=%.3f std=%.3f  VPT*duty mean=%.3f std=%.3f  M=%zu\n",
                    sum.n_trials_ok,
                    sum.mean_vpt, sum.std_vpt,
                    sum.mean_rmse, sum.std_rmse,
                    sum.mean_duty, sum.std_duty,
                    sum.mean_vpt_x_duty, sum.std_vpt_x_duty,
                    sum.history_depth);
    }
    std::fflush(stdout);

    WriteSurveyResultFiles(sum);

    if (out)
        *out = sum;

    if (completion_beep)
        Beep(2500, 3000);
    return sum.ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Campaign_Trace
// ---------------------------------------------------------------------------
// Train one ESN seed, free-run, write per-step CSV (pred + true xyz).
// Output dir is absolute: {config::RESULTS_DIR}/traces/ (CWD-independent --
// CLion runs with build-dir CWD, so relative paths are unreliable).
// When target_orbit != 0: free-run that orbit once and print every generative step.
int Campaign_Trace(size_t dim, uint64_t esn_seed, int max_freeruns, uint64_t target_orbit,
                   uint64_t orbit_seed)
{
    if (!ValidateDim(dim, "trace"))
        return 2;
    ConfigSizeRestore dim_restore(config::DIM, dim);

    if (max_freeruns < 1)
        max_freeruns = 1;

    const fs::path trace_dir = TracesDir();
    if (!EnsureDir(trace_dir, "trace"))
        return 2;

    ReportBanner("Trace");
    std::printf("[trace] DIM=%zu (N=%zu)  M=%zu  esn_seed=%llu  max_freeruns=%d  "
                "target_orbit=%llu\n",
                config::DIM, size_t{1} << config::DIM, config::HISTORY_DEPTH,
                static_cast<unsigned long long>(esn_seed), max_freeruns,
                static_cast<unsigned long long>(target_orbit));
    std::printf("[trace] train %zu epochs then free-run; CSV dir: %s\n",
                config::EPOCHS, trace_dir.string().c_str());
    if (target_orbit != 0)
        std::printf("[trace] fixed orbit %llu -- one free-run, every step printed + CSV\n",
                    static_cast<unsigned long long>(target_orbit));
    std::fflush(stdout);

    config::ENABLE_PRINTF = true;
    Lorenz lorenz(esn_seed, orbit_seed);
    std::cout << lorenz.ReadoutArchSummary();
    if (config::LOAD_TRAINED_WEIGHTS)
        lorenz.LoadTrainedWeights();
    else
        lorenz.Train();

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    std::error_code ec;

    auto verify_csv = [](const fs::path& path) -> bool {
        std::error_code e;
        if (!fs::exists(path, e) || e)
        {
            std::fprintf(stderr, "[trace] CSV missing after free-run: %s\n",
                         path.string().c_str());
            return false;
        }
        const auto sz = fs::file_size(path, e);
        if (e || sz == 0)
        {
            std::fprintf(stderr, "[trace] CSV empty or unreadable: %s\n",
                         path.string().c_str());
            return false;
        }
        ReportWrote("trace", path);
        return true;
    };

    int dumped = 0;

    // Fixed orbit: one free-run, verbose step table + CSV (pred/true xyz).
    if (target_orbit != 0)
    {
        const fs::path out_path = trace_dir /
            ("seed" + std::to_string(esn_seed) + "_orbit" +
             std::to_string(target_orbit) + ".csv");
        std::printf("[trace] free-run fixed orbit=%llu  CSV=%s\n",
                    static_cast<unsigned long long>(target_orbit),
                    out_path.string().c_str());
        std::fflush(stdout);

        // ENABLE_PRINTF stays true so every generative step is printed.
        FreeRunResult r = lorenz.FreeRun(/*verbose=*/true, out_path.string().c_str(),
                                         0, target_orbit);
        if (!r.valid)
        {
            std::printf("[trace] free-run invalid\n");
            return 1;
        }
        std::printf("[trace] freerun orbit=%llu\n",
                    static_cast<unsigned long long>(r.orbit_seed));
        ReportFreerunScores("trace", r.vpt_lt, r.duty, r.vpt_x_duty, r.rmse);
        std::printf("%s", r.row.c_str());
        if (!verify_csv(out_path))
            return 1;
        std::printf("[trace] plot:\n"
                    "  python examples/Lorenz/plot_freerun_overlay.py \"%s\"\n",
                    out_path.string().c_str());
        dumped = 1;
    }
    else
    {
        // Hunt/dump mode: cycle free-runs (no per-step print; CSV only).
        config::ENABLE_PRINTF = false;
        for (int i = 0; i < max_freeruns; ++i)
        {
            const fs::path tmp_path = trace_dir / ("_tmp_" + std::to_string(esn_seed) + ".csv");
            FreeRunResult r = lorenz.FreeRun(false, tmp_path.string().c_str(), 0);
            if (!r.valid)
            {
                std::printf("[trace] free-run %d invalid - stop\n", i);
                break;
            }
            std::printf("[trace] freerun %d/%d  orbit=%llu  DUMP\n",
                        i + 1, max_freeruns,
                        static_cast<unsigned long long>(r.orbit_seed));
            ReportFreerunScores("trace", r.vpt_lt, r.duty, r.vpt_x_duty, r.rmse);
            std::fflush(stdout);

            const fs::path out_path = trace_dir /
                ("seed" + std::to_string(esn_seed) + "_orbit" +
                 std::to_string(r.orbit_seed) + ".csv");
            fs::remove(out_path, ec);
            fs::rename(tmp_path, out_path, ec);
            if (ec)
            {
                std::printf("[trace] rename failed (%s) - CSV left at %s\n",
                            ec.message().c_str(), tmp_path.string().c_str());
                if (verify_csv(tmp_path))
                    ++dumped;
            }
            else if (verify_csv(out_path))
            {
                std::printf("%s", r.row.c_str());
                ++dumped;
            }
        }
    }

    const double elapsed =
        std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("[trace] CSV file(s): %d\n", dumped);
    ReportDone("trace", elapsed);
    return dumped > 0 ? 0 : 1;
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
// FreeRun (campaign: load weights + freerun one attractor IC)
// ---------------------------------------------------------------------------
// Load readout from file; free-run once on an explicit attractor IC (no train).
// CSV under RUNS_DIR/traces/. Distinct from Lorenz::FreeRun (member).
int FreeRun(size_t dim, size_t history_depth, uint64_t esn_seed,
            double ic_x, double ic_y, double ic_z,
            const char* weights_stem,
            float spectral_radius, float input_scaling)
{
    if (!ValidateDim(dim, "freerun"))
        return 2;
    if (!ValidateHistoryDepth(history_depth, "freerun"))
        return 2;
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

    const LorenzAttractor::State ic{ic_x, ic_y, ic_z};

    char ic_tag[96];
    std::snprintf(ic_tag, sizeof ic_tag, "ic%.6f_%.6f_%.6f", ic_x, ic_y, ic_z);
    for (char* p = ic_tag; *p; ++p)
        if (*p == '.')
            *p = 'p';

    const fs::path out_path = trace_dir /
        ("seed" + std::to_string(esn_seed) + "_" + ic_tag + ".csv");

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
    std::printf("[freerun] IC=(%.6f, %.6f, %.6f)\n", ic_x, ic_y, ic_z);
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

    std::printf("[freerun] free-run fixed IC (no train)\n");
    std::fflush(stdout);

    FreeRunResult r = lorenz.FreeRun(/*verbose=*/true, out_path.string().c_str(),
                                     0, /*fixed_orbit_seed=*/0, &ic);
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
// FreeRunSurvey (campaign: load weights + many freeruns + rank ICs)
// ---------------------------------------------------------------------------
// Pipeline middle step: Train -> FreeRunSurvey -> FreeRun (cherry-pick IC).
int FreeRunSurvey(size_t dim, size_t history_depth, uint64_t esn_seed,
                  int num_runs, uint64_t orbit_seed, const char* weights_stem,
                  int top_k, FreeRunSurveySummary* out,
                  float spectral_radius, float input_scaling)
{
    if (out)
        *out = FreeRunSurveySummary{};

    if (!ValidateDim(dim, "freerun-survey"))
        return 2;
    if (!ValidateHistoryDepth(history_depth, "freerun-survey"))
        return 2;
    if (num_runs < 1)
    {
        std::fprintf(stderr, "[freerun-survey] refused: num_runs=%d (need >= 1)\n",
                     num_runs);
        return 2;
    }
    if (top_k < 1)
        top_k = 1;
    if (!ValidateDynamicsOverrides(spectral_radius, input_scaling, "freerun-survey"))
        return 2;

    const char* stem = (weights_stem && weights_stem[0] != '\0')
                           ? weights_stem
                           : config::LOAD_WEIGHTS_STEM;
    if (stem == nullptr || stem[0] == '\0')
    {
        std::fprintf(stderr,
                     "[freerun-survey] refused: no weights stem "
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

    const fs::path survey_dir = SurveysDir();
    if (!EnsureDir(survey_dir, "freerun-survey"))
        return 2;

    ReportBanner("FreeRunSurvey (load-only; multi-orbit)");
    std::printf("[freerun-survey] DIM=%zu (N=%zu)  M=%zu  esn_seed=%llu  num_runs=%d  "
                "orbit_seed=%llu  top_k=%d\n",
                config::DIM, size_t{1} << config::DIM, config::HISTORY_DEPTH,
                static_cast<unsigned long long>(esn_seed), num_runs,
                static_cast<unsigned long long>(orbit_seed), top_k);
    std::printf("[freerun-survey] SR=%.4f%s  input_scaling=%.4f%s\n",
                config::SPECTRAL_RADIUS,
                spectral_radius > 0.0f ? " (override)" : " (config)",
                config::INPUT_SCALING,
                input_scaling > 0.0f ? " (override)" : " (config)");
    {
        const std::string ch = FormatDriveGains();
        std::printf("[freerun-survey] drive=[x,y,z,xz]  n_in=%zu  drive_ch=%s\n",
                    n_in, ch.c_str());
    }
    std::printf("[freerun-survey] load stem: %s\n", stem);
    std::printf("[freerun-survey] freerun remix IC each run; "
                "stats use top-10%% pool; metrics=VPT,duty,VPT*duty,RMSE\n");
    std::printf("[freerun-survey] CSV dir: %s\n", survey_dir.string().c_str());
    std::fflush(stdout);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    // Quiet per-step freerun; still print epoch-level survey progress.
    config::ENABLE_PRINTF = false;
    config::ENABLE_PROGRESS = true;

    Lorenz lorenz(esn_seed, orbit_seed);
    std::cout << lorenz.ReadoutArchSummary();
    try
    {
        lorenz.LoadTrainedWeights(stem);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "[freerun-survey] load failed: %s\n", e.what());
        return 1;
    }

    struct Row
    {
        FreeRunResult r;
        LorenzAttractor::State ic{};
    };
    std::vector<Row> rows;
    rows.reserve(static_cast<size_t>(num_runs));

    for (int i = 0; i < num_runs; ++i)
    {
        // Remix orbit_seed_ each call (multi-IC challenge).
        FreeRunResult r = lorenz.FreeRun(/*verbose=*/false, /*csv=*/nullptr, 0);
        if (!r.valid)
        {
            std::fprintf(stderr, "[freerun-survey] freerun %d/%d invalid - skip\n",
                         i + 1, num_runs);
            continue;
        }
        Row row;
        row.r = std::move(r);
        row.ic = Lorenz::IcFromOrbitSeed(row.r.orbit_seed);
        rows.push_back(std::move(row));

        if (config::ENABLE_PROGRESS &&
            ((i + 1) % 10 == 0 || i + 1 == num_runs))
        {
            std::fprintf(stderr, "[freerun-survey] freerun %d/%d  last VPT=%.2f lt  "
                                 "VPT*duty=%.3f\n",
                         i + 1, num_runs, rows.back().r.vpt_lt, rows.back().r.vpt_x_duty);
            std::fflush(stderr);
        }
    }

    if (rows.empty())
    {
        std::printf("[freerun-survey] no valid freeruns\n");
        if (out)
        {
            out->ok = false;
            out->esn_seed = esn_seed;
        }
        return 1;
    }

    std::vector<double> vpt_lts, rmses, duties, vxds;
    vpt_lts.reserve(rows.size());
    rmses.reserve(rows.size());
    duties.reserve(rows.size());
    vxds.reserve(rows.size());
    for (const auto& row : rows)
    {
        vpt_lts.push_back(row.r.vpt_lt);
        rmses.push_back(row.r.rmse);
        duties.push_back(row.r.duty);
        vxds.push_back(row.r.vpt_x_duty);
    }

    const size_t n_full = rows.size();
    const auto vpt_best = Top10HigherIsBetter(vpt_lts);
    const auto duty_best = Top10HigherIsBetter(duties);
    const auto vxd_best = Top10HigherIsBetter(vxds);
    const auto rmse_best = Top10LowerIsBetter(rmses);

    auto report = [](const char* label, std::vector<double> v, int prec, size_t n_pool) {
        if (v.empty())
            return;
        std::sort(v.begin(), v.end());
        double mean = 0, sd = 0;
        MeanStd(v, mean, sd);
        const size_t n = v.size();
        const double median = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
        std::printf("  %-14s n=%zu (top 10%% of %zu)  min=%.*f  max=%.*f  "
                    "mean=%.*f  median=%.*f  std=%.*f\n",
                    label, n, n_pool, prec, v.front(), prec, v.back(),
                    prec, mean, prec, median, prec, sd);
    };

    std::printf("\n=== FreeRunSurvey stats (%zu valid / %d requested) ===\n",
                n_full, num_runs);
    report("VPT (lt)", vpt_best, 2, n_full);
    report("RMSE", rmse_best, 6, n_full);
    report("duty", duty_best, 3, n_full);
    report("VPT*duty", vxd_best, 3, n_full);
    std::printf("  note: top 10%% per metric; primary sort key for leaderboard = VPT*duty\n");

    // Rank all rows by VPT*duty (higher better) for cherry-picks.
    std::vector<size_t> order(rows.size());
    for (size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return rows[a].r.vpt_x_duty > rows[b].r.vpt_x_duty;
    });

    const size_t k = std::min(static_cast<size_t>(top_k), order.size());
    std::printf("\n=== Top %zu freeruns by VPT*duty (paste IC into FreeRun) ===\n", k);
    for (size_t rank = 0; rank < k; ++rank)
    {
        const auto& row = rows[order[rank]];
        const auto& r = row.r;
        const auto& ic = row.ic;
        std::printf("  #%zu  VPT*duty=%.3f  VPT=%.2f lt  duty=%.3f  RMSE=%.6f  "
                    "orbit_seed=%llu\n",
                    rank + 1, r.vpt_x_duty, r.vpt_lt, r.duty, r.rmse,
                    static_cast<unsigned long long>(r.orbit_seed));
        std::printf("       IC=(%.6f, %.6f, %.6f)\n", ic.x, ic.y, ic.z);
        std::printf("       FreeRun(%zu, %zu, %llu, %.6f, %.6f, %.6f, stem);\n",
                    config::DIM, config::HISTORY_DEPTH,
                    static_cast<unsigned long long>(esn_seed),
                    ic.x, ic.y, ic.z);
    }

    // Full ranked table for post-analysis (atomic write).
    const fs::path csv_path = survey_dir /
        ("survey_seed" + std::to_string(esn_seed) +
         "_D" + std::to_string(config::DIM) +
         "_M" + std::to_string(config::HISTORY_DEPTH) +
         "_n" + std::to_string(num_runs) + ".csv");
    {
        std::ostringstream body;
        body << "rank,orbit_seed,ic_x,ic_y,ic_z,vpt_lt,duty,vpt_x_duty,rmse,crossed\n";
        for (size_t rank = 0; rank < order.size(); ++rank)
        {
            const auto& row = rows[order[rank]];
            body << (rank + 1) << ','
                 << row.r.orbit_seed << ','
                 << row.ic.x << ',' << row.ic.y << ',' << row.ic.z << ','
                 << row.r.vpt_lt << ',' << row.r.duty << ','
                 << row.r.vpt_x_duty << ',' << row.r.rmse << ','
                 << (row.r.crossed ? 1 : 0) << '\n';
        }
        if (WriteAtomicFile(csv_path, body.str(), "freerun-survey"))
            ReportWrote("freerun-survey", csv_path);
    }

    if (out)
    {
        out->ok = true;
        out->esn_seed = esn_seed;
        out->n_valid = n_full;
        out->mean_vpt = MeanOf(vpt_best);
        out->mean_duty = MeanOf(duty_best);
        out->mean_vpt_x_duty = MeanOf(vxd_best);
        out->mean_rmse = MeanOf(rmse_best);
        const auto& best = rows[order[0]];
        out->best_vpt_x_duty = best.r.vpt_x_duty;
        out->best_orbit_seed = best.r.orbit_seed;
        out->best_ic_x = best.ic.x;
        out->best_ic_y = best.ic.y;
        out->best_ic_z = best.ic.z;
    }

    ReportDone("freerun-survey",
               std::chrono::duration<double>(clock::now() - t0).count());
    return 0;
}

// ---------------------------------------------------------------------------
// SeedSweep (Train? -> FreeRunSurvey per seed; rank seeds by mean VPT*duty)
// ---------------------------------------------------------------------------
int SeedSweep(size_t dim, size_t history_depth,
              std::initializer_list<uint64_t> esn_seeds,
              size_t epochs, int freerun_runs,
              uint64_t train_orbit, uint64_t freerun_orbit_seed,
              int top_k, bool do_train,
              float spectral_radius, float input_scaling)
{
    if (!ValidateDim(dim, "seed-sweep"))
        return 2;
    if (!ValidateHistoryDepth(history_depth, "seed-sweep"))
        return 2;
    if (esn_seeds.size() == 0)
    {
        std::fprintf(stderr, "[seed-sweep] empty esn_seeds list\n");
        return 2;
    }
    if (freerun_runs < 1)
    {
        std::fprintf(stderr, "[seed-sweep] refused: freerun_runs=%d (need >= 1)\n",
                     freerun_runs);
        return 2;
    }
    if (do_train && epochs < 1)
    {
        std::fprintf(stderr, "[seed-sweep] refused: epochs=%zu (need >= 1 when do_train)\n",
                     epochs);
        return 2;
    }
    if (!ValidateDynamicsOverrides(spectral_radius, input_scaling, "seed-sweep"))
        return 2;

    // RAII: campaigns below also restore; outer restore keeps caller knobs stable.
    ConfigSizeRestore dim_restore(config::DIM, dim);
    ConfigSizeRestore m_restore(config::HISTORY_DEPTH, history_depth);
    ConfigSizeRestore epochs_restore(config::EPOCHS, do_train ? epochs : config::EPOCHS);
    ConfigFloatRestore sr_restore(
        config::SPECTRAL_RADIUS,
        spectral_radius > 0.0f ? spectral_radius : config::SPECTRAL_RADIUS);
    ConfigFloatRestore is_restore(
        config::INPUT_SCALING,
        input_scaling > 0.0f ? input_scaling : config::INPUT_SCALING);

    const size_t n_in = kNumDriveChannels;

    const fs::path model_dir = config::MODEL_SAVE_DIR;
    const fs::path survey_dir = SurveysDir();
    if (!EnsureDir(model_dir, "seed-sweep"))
        std::fprintf(stderr, "[seed-sweep] WARN: model dir may be unwritable\n");
    if (!EnsureDir(survey_dir, "seed-sweep"))
        return 2;

    // Final + partial ranking paths (partial updated after each seed for crash safety).
    const fs::path rank_csv = survey_dir /
        ("seed_sweep_D" + std::to_string(dim) +
         "_M" + std::to_string(history_depth) +
         "_n" + std::to_string(freerun_runs) +
         "_seeds" + std::to_string(esn_seeds.size()) + ".csv");
    const fs::path rank_csv_partial = survey_dir /
        ("seed_sweep_D" + std::to_string(dim) +
         "_M" + std::to_string(history_depth) +
         "_n" + std::to_string(freerun_runs) +
         "_seeds" + std::to_string(esn_seeds.size()) + ".partial.csv");

    ReportBanner("SeedSweep");
    std::printf("[seed-sweep] DIM=%zu  M=%zu  n_seeds=%zu  epochs=%zu  freerun_runs=%d  "
                "do_train=%s  top_k=%d\n",
                dim, history_depth, esn_seeds.size(), epochs, freerun_runs,
                do_train ? "yes" : "no", top_k);
    std::printf("[seed-sweep] SR=%.4f%s  input_scaling=%.4f%s\n",
                static_cast<double>(config::SPECTRAL_RADIUS),
                spectral_radius > 0.0f ? " (override)" : " (config)",
                static_cast<double>(config::INPUT_SCALING),
                input_scaling > 0.0f ? " (override)" : " (config)");
    {
        const std::string ch = FormatDriveGains();
        std::printf("[seed-sweep] drive=[x,y,z,xz]  n_in=%zu  drive_ch=%s\n",
                    n_in, ch.c_str());
    }
    std::printf("[seed-sweep] train_orbit=%llu  freerun_orbit_seed=%llu\n",
                static_cast<unsigned long long>(train_orbit),
                static_cast<unsigned long long>(freerun_orbit_seed));
    std::printf("[seed-sweep] weight stems: %s/lorenz_seed{S}_D%zu_M%zu  "
                "(stems omit SR/IS/drive_ch -- document in ranking banner)\n",
                model_dir.string().c_str(), dim, history_depth);
    std::printf("[seed-sweep] seed ranking metric = mean VPT*duty (top-10%% freeruns)\n");
    std::printf("[seed-sweep] metrics=VPT,duty,VPT*duty,RMSE  CSV dir: %s\n",
                survey_dir.string().c_str());
    std::printf("[seed-sweep] ranking CSV: %s\n", rank_csv.string().c_str());
    std::printf("[seed-sweep] partial CSV (after each seed): %s\n",
                rank_csv_partial.string().c_str());
    std::fflush(stdout);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    struct SeedRow
    {
        uint64_t seed = 0;
        FreeRunSurveySummary sum{};
        int train_rc = 0;
        int survey_rc = 0;
        std::string stem;
    };
    std::vector<SeedRow> seed_rows;
    seed_rows.reserve(esn_seeds.size());

    // Write ranked/partial seed table. Columns (16):
    // rank,esn_seed,ok,mean_vpt_x_duty,mean_vpt,mean_duty,mean_rmse,
    // best_vpt_x_duty,best_orbit_seed,best_ic_x,best_ic_y,best_ic_z,
    // n_valid,train_rc,survey_rc,stem
    auto write_seed_rank_csv = [&](const fs::path& path) -> bool {
        const fs::path tmp = fs::path(path.string() + ".tmp");
        {
            std::ofstream csv(tmp, std::ios::out | std::ios::trunc);
            if (!csv)
            {
                std::fprintf(stderr, "[seed-sweep] failed to open %s\n",
                             tmp.string().c_str());
                return false;
            }
            csv << "rank,esn_seed,ok,mean_vpt_x_duty,mean_vpt,mean_duty,mean_rmse,"
                   "best_vpt_x_duty,best_orbit_seed,best_ic_x,best_ic_y,best_ic_z,"
                   "n_valid,train_rc,survey_rc,stem\n";

            std::vector<size_t> ord;
            ord.reserve(seed_rows.size());
            for (size_t i = 0; i < seed_rows.size(); ++i)
                if (seed_rows[i].sum.ok)
                    ord.push_back(i);
            std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
                return seed_rows[a].sum.mean_vpt_x_duty > seed_rows[b].sum.mean_vpt_x_duty;
            });

            size_t rank = 0;
            for (size_t idx : ord)
            {
                const auto& sr = seed_rows[idx];
                const auto& s = sr.sum;
                ++rank;
                // 16 fields
                csv << rank << ',' << s.esn_seed << ",1,"
                    << s.mean_vpt_x_duty << ',' << s.mean_vpt << ','
                    << s.mean_duty << ',' << s.mean_rmse << ','
                    << s.best_vpt_x_duty << ',' << s.best_orbit_seed << ','
                    << s.best_ic_x << ',' << s.best_ic_y << ',' << s.best_ic_z << ','
                    << s.n_valid << ',' << sr.train_rc << ',' << sr.survey_rc << ','
                    << sr.stem << '\n';
            }
            for (const auto& sr : seed_rows)
            {
                if (sr.sum.ok)
                    continue;
                // 16 fields: empty rank, seed, ok=0, 10 empty metrics, train_rc, survey_rc, stem
                // After "0" use exactly 10 commas then train_rc (no extra empty).
                csv << ',' << sr.seed << ",0,,,,,,,,,,"
                    << sr.train_rc << ',' << sr.survey_rc << ',' << sr.stem << '\n';
            }
            csv.flush();
            if (!csv.good())
            {
                std::fprintf(stderr, "[seed-sweep] write failed for %s\n",
                             tmp.string().c_str());
                return false;
            }
        }
        std::error_code ec;
        fs::remove(path, ec);
        fs::rename(tmp, path, ec);
        if (ec)
        {
            std::fprintf(stderr, "[seed-sweep] rename %s -> %s failed: %s\n",
                         tmp.string().c_str(), path.string().c_str(),
                         ec.message().c_str());
            return false;
        }
        return true;
    };

    size_t step = 0;
    const size_t n_seeds = esn_seeds.size();
    for (uint64_t seed : esn_seeds)
    {
        ++step;
        SeedRow sr;
        sr.seed = seed;
        sr.stem = DefaultWeightStem(seed, dim, history_depth);

        std::printf("\n########## SeedSweep %zu/%zu  esn_seed=%llu ##########\n",
                    step, n_seeds, static_cast<unsigned long long>(seed));
        std::printf("[seed-sweep] stem: %s\n", sr.stem.c_str());
        std::fflush(stdout);

        if (do_train)
        {
            sr.train_rc = Train(dim, history_depth, seed, train_orbit, epochs,
                                sr.stem.c_str());
            if (sr.train_rc != 0)
            {
                std::fprintf(stderr, "[seed-sweep] Train failed for seed %llu (rc=%d) -- skip survey\n",
                             static_cast<unsigned long long>(seed), sr.train_rc);
                sr.survey_rc = -1; // not run
                seed_rows.push_back(std::move(sr));
                write_seed_rank_csv(rank_csv_partial);
                std::fflush(stdout);
                std::fflush(stderr);
                continue;
            }
        }

        sr.survey_rc = FreeRunSurvey(dim, history_depth, seed, freerun_runs,
                                     freerun_orbit_seed, sr.stem.c_str(), top_k,
                                     &sr.sum);
        if (sr.survey_rc != 0 || !sr.sum.ok)
        {
            std::fprintf(stderr, "[seed-sweep] FreeRunSurvey failed for seed %llu (rc=%d)\n",
                         static_cast<unsigned long long>(seed), sr.survey_rc);
            sr.sum.ok = false;
        }
        else
        {
            // Explicit "mean" labels so overnight logs are not confused with one freerun.
            std::printf("[seed-sweep] seed %llu  top-10%% means:  "
                        "VPT=%.2f lt  duty=%.3f  VPT*duty=%.3f  RMSE=%.6f  "
                        "(rank key = VPT*duty)\n",
                        static_cast<unsigned long long>(seed),
                        sr.sum.mean_vpt, sr.sum.mean_duty,
                        sr.sum.mean_vpt_x_duty, sr.sum.mean_rmse);
            std::printf("[seed-sweep]   best freerun: VPT*duty=%.3f  IC=(%.6f, %.6f, %.6f)  "
                        "orbit_seed=%llu\n",
                        sr.sum.best_vpt_x_duty,
                        sr.sum.best_ic_x, sr.sum.best_ic_y, sr.sum.best_ic_z,
                        static_cast<unsigned long long>(sr.sum.best_orbit_seed));
        }
        seed_rows.push_back(std::move(sr));
        write_seed_rank_csv(rank_csv_partial);
        std::fflush(stdout);
        std::fflush(stderr);
    }

    // Rank successful seeds by mean VPT*duty.
    std::vector<size_t> order;
    for (size_t i = 0; i < seed_rows.size(); ++i)
        if (seed_rows[i].sum.ok)
            order.push_back(i);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return seed_rows[a].sum.mean_vpt_x_duty > seed_rows[b].sum.mean_vpt_x_duty;
    });

    std::printf("\n========================================================================\n");
    std::printf("=== SeedSweep ranking by mean VPT*duty (top-10%% freeruns) ===\n");
    std::printf("========================================================================\n");
    if (order.empty())
    {
        std::printf("(no successful seed surveys)\n");
    }
    else
    {
        std::printf("%-4s %14s %10s %8s %8s %10s %10s\n",
                    "rank", "esn_seed", "VxD_mn", "VPT_mn", "duty_mn", "RMSE_mn", "VxD_best");
        for (size_t rank = 0; rank < order.size(); ++rank)
        {
            const auto& s = seed_rows[order[rank]].sum;
            std::printf("%-4zu %14llu %10.3f %8.2f %8.3f %10.6f %10.3f\n",
                        rank + 1,
                        static_cast<unsigned long long>(s.esn_seed),
                        s.mean_vpt_x_duty, s.mean_vpt, s.mean_duty,
                        s.mean_rmse, s.best_vpt_x_duty);
        }

        const auto& best = seed_rows[order[0]];
        std::printf("\nBest seed: %llu  mean VPT*duty=%.3f\n",
                    static_cast<unsigned long long>(best.seed),
                    best.sum.mean_vpt_x_duty);
        std::printf("  stem: %s\n", best.stem.c_str());
        std::printf("  best freerun IC=(%.6f, %.6f, %.6f)  VPT*duty=%.3f\n",
                    best.sum.best_ic_x, best.sum.best_ic_y, best.sum.best_ic_z,
                    best.sum.best_vpt_x_duty);
        std::printf("  FreeRun(%zu, %zu, %llu, %.6f, %.6f, %.6f, \"%s\");\n",
                    dim, history_depth,
                    static_cast<unsigned long long>(best.seed),
                    best.sum.best_ic_x, best.sum.best_ic_y, best.sum.best_ic_z,
                    best.stem.c_str());
    }
    std::fflush(stdout);

    if (write_seed_rank_csv(rank_csv))
    {
        ReportWrote("seed-sweep", rank_csv);
        write_seed_rank_csv(rank_csv_partial); // final snapshot of partial
    }
    else
    {
        std::fprintf(stderr, "[seed-sweep] FAILED to write final ranking CSV\n");
    }

    ReportDone("seed-sweep",
               std::chrono::duration<double>(clock::now() - t0).count());
    return order.empty() ? 1 : 0;
}

// ---------------------------------------------------------------------------
// ParallelSeedSweep (train+freerun in memory; parallel ESN seeds; no weight I/O)
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

int ParallelSeedSweep(size_t dim, size_t history_depth,
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
                  "ParallelSeedSweep requires Lorenz::kReadoutNumThreads == 1");

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

    ReportBanner("ParallelSeedSweep");
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
    auto make_order = [&](auto metric) {
        std::vector<size_t> ord;
        ord.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i)
            if (rows[i].ok)
                ord.push_back(i);
        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
            return metric(rows[a]) > metric(rows[b]);
        });
        return ord;
    };
    const auto by_vxd = make_order([](const ParSeedRow& r) { return r.mean_vpt_x_duty; });
    const auto by_vpt = make_order([](const ParSeedRow& r) { return r.mean_vpt; });
    const auto by_duty = make_order([](const ParSeedRow& r) { return r.mean_duty; });

    auto rank_of = [](const std::vector<size_t>& ord, size_t idx) -> size_t {
        for (size_t r = 0; r < ord.size(); ++r)
            if (ord[r] == idx)
                return r + 1;
        return 0;
    };

    std::ostringstream report;
    auto emit = [&](const char* s) {
        report << s;
        std::fputs(s, stdout);
    };
    auto emitf = [&](const char* fmt, auto... args) {
        char buf[512];
        std::snprintf(buf, sizeof buf, fmt, args...);
        emit(buf);
    };

    emit("\n========================================================================\n");
    emit("=== ParallelSeedSweep final report (top-10% freerun means) ===\n");
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

    auto emit_top = [&](const char* title, const std::vector<size_t>& ord) {
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

    emit("\nCherry-pick: use esn_seed from the tables above with Train / SeedSweep / "
         "FreeRunSurvey (same dim/M/dynamics). Weights were not saved here.\n");
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
                csv << "# ParallelSeedSweep\n"
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
// ParallelOrbitSweep (train once; parallel one-freerun-per-orbit ranking)
// ---------------------------------------------------------------------------
int ParallelOrbitSweep(size_t dim, size_t history_depth,
                       uint64_t base_esn_seed, uint64_t base_orbit_seed,
                       size_t num_orbits, size_t num_threads, size_t epochs,
                       int top_k, float spectral_radius, float input_scaling)
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
    if (epochs < 1)
    {
        std::fprintf(stderr, "[par-orbit-sweep] refused: epochs=%zu (need >= 1)\n",
                     epochs);
        return 2;
    }
    if (top_k < 1)
        top_k = 1;
    if (!ValidateDynamicsOverrides(spectral_radius, input_scaling, "par-orbit-sweep"))
        return 2;
    if (config::LOAD_TRAINED_WEIGHTS)
    {
        std::fprintf(stderr,
                     "[par-orbit-sweep] refused: LOAD_TRAINED_WEIGHTS is on "
                     "(this campaign always trains from scratch)\n");
        return 2;
    }
    if (config::SAVE_TRAINED_WEIGHTS)
    {
        std::fprintf(stderr,
                     "[par-orbit-sweep] refused: SAVE_TRAINED_WEIGHTS is on "
                     "(campaign uses a temp stem only; turn config auto-save off)\n");
        return 2;
    }
    static_assert(Lorenz::kReadoutNumThreads == 1,
                  "ParallelOrbitSweep requires Lorenz::kReadoutNumThreads == 1");

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
    ConfigSizeRestore epochs_restore(config::EPOCHS, epochs);
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
    const fs::path rank_csv = survey_dir /
        ("par_orbit_sweep_D" + std::to_string(dim) +
         "_M" + std::to_string(history_depth) +
         "_orbits" + std::to_string(num_orbits) +
         "_" + ts + ".csv");
    const fs::path rank_txt = survey_dir /
        ("par_orbit_sweep_D" + std::to_string(dim) +
         "_M" + std::to_string(history_depth) +
         "_orbits" + std::to_string(num_orbits) +
         "_" + ts + ".txt");
    // Temp stem so workers can Load after one train (deleted before return).
    const fs::path temp_stem = survey_dir /
        (".par_orbit_tmp_seed" + std::to_string(base_esn_seed) + "_" + ts);

    const bool saved_printf = config::ENABLE_PRINTF;
    const bool saved_progress = config::ENABLE_PROGRESS;
    config::ENABLE_PRINTF = false;
    config::ENABLE_PROGRESS = false;

    ReportBanner("ParallelOrbitSweep");
    std::printf("[par-orbit-sweep] DIM=%zu  M=%zu  num_orbits=%zu  num_threads=%zu  "
                "epochs=%zu  top_k=%d\n",
                dim, history_depth, num_orbits, num_threads, epochs, top_k);
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
    std::printf("[par-orbit-sweep] train once, then one freerun per orbit (parallel)\n");
    std::printf("[par-orbit-sweep] report: %s\n", rank_txt.string().c_str());
    std::printf("[par-orbit-sweep] CSV:    %s\n", rank_csv.string().c_str());
    std::fflush(stdout);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    // Always remove temp readout files (train fail, early return, or success).
    const std::string stem_str = temp_stem.string();
    struct TempStemGuard
    {
        std::string stem;
        explicit TempStemGuard(std::string s) : stem(std::move(s)) {}
        ~TempStemGuard()
        {
            std::error_code ec;
            fs::remove(fs::path(stem + ".hcnw"), ec);
            fs::remove(fs::path(stem + ".arch.json"), ec);
        }
        TempStemGuard(const TempStemGuard&) = delete;
        TempStemGuard& operator=(const TempStemGuard&) = delete;
    };
    TempStemGuard temp_guard(stem_str);

    // --- Train once ---
    try
    {
        Lorenz trainer(base_esn_seed, base_orbit_seed);
        trainer.Train();
        trainer.SaveTrainedWeights(stem_str.c_str());
    }
    catch (const std::exception& e)
    {
        config::ENABLE_PRINTF = saved_printf;
        config::ENABLE_PROGRESS = saved_progress;
        std::fprintf(stderr, "[par-orbit-sweep] train/save failed: %s\n", e.what());
        return 1; // temp_guard removes any partial stem
    }

    std::printf("[par-orbit-sweep] trained; freerunning %zu orbits (quiet per-job load)\n",
                num_orbits);
    std::fflush(stdout);

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
    std::atomic<size_t> done_count{0};
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
                    const size_t d = done_count.fetch_add(1, std::memory_order_relaxed) + 1;
                    {
                        std::lock_guard<std::mutex> lock(stderr_mu);
                        std::fprintf(stderr,
                                     "[par-orbit-sweep] heartbeat %zu/%zu  "
                                     "(idx=%zu orbit=%llu ok=%d)\n",
                                     d, num_orbits, i,
                                     static_cast<unsigned long long>(orbit),
                                     row.ok ? 1 : 0);
                        std::fflush(stderr);
                    }
                }
            });
        }
    }

    {
        std::lock_guard<std::mutex> lock(stderr_mu);
        std::fprintf(stderr, "[par-orbit-sweep] all orbits finished (%zu/%zu)\n",
                     done_count.load(std::memory_order_relaxed), num_orbits);
        std::fflush(stderr);
    }
    std::fflush(stdout);

    // Temp weights no longer needed; drop before ranking so files do not linger
    // if report I/O is slow. Guard dtor also removes (idempotent).
    {
        std::error_code ec;
        fs::remove(fs::path(stem_str + ".hcnw"), ec);
        fs::remove(fs::path(stem_str + ".arch.json"), ec);
    }

    config::ENABLE_PRINTF = saved_printf;
    config::ENABLE_PROGRESS = saved_progress;

    auto make_order = [&](auto metric) {
        std::vector<size_t> ord;
        ord.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i)
            if (rows[i].ok)
                ord.push_back(i);
        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
            return metric(rows[a]) > metric(rows[b]);
        });
        return ord;
    };
    const auto by_vxd = make_order([](const OrbitRow& r) { return r.vpt_x_duty; });
    const auto by_vpt = make_order([](const OrbitRow& r) { return r.vpt; });
    const auto by_duty = make_order([](const OrbitRow& r) { return r.duty; });

    auto rank_of = [](const std::vector<size_t>& ord, size_t idx) -> size_t {
        for (size_t r = 0; r < ord.size(); ++r)
            if (ord[r] == idx)
                return r + 1;
        return 0;
    };

    std::ostringstream report;
    auto emit = [&](const char* s) {
        report << s;
        std::fputs(s, stdout);
    };
    auto emitf = [&](const char* fmt, auto... args) {
        char buf[640];
        std::snprintf(buf, sizeof buf, fmt, args...);
        emit(buf);
    };

    emit("\n========================================================================\n");
    emit("=== ParallelOrbitSweep final report (one freerun per orbit) ===\n");
    emit("========================================================================\n");
    emitf("DIM=%zu  M=%zu  esn_seed=%llu  num_orbits=%zu  threads=%zu  epochs=%zu\n",
          dim, history_depth, static_cast<unsigned long long>(base_esn_seed),
          num_orbits, num_threads, epochs);
    emitf("base_orbit_seed=%llu\n", static_cast<unsigned long long>(base_orbit_seed));
    emitf("ok=%zu / %zu\n\n", by_vxd.size(), num_orbits);

    emit("--- All orbits (sorted by VPT*duty) ---\n");
    emit("note: one freerun per orbit; VxD = VPT*duty for that run.\n");
    emitf("%-4s %6s %20s %10s %10s %10s %10s %6s %6s %6s\n",
          "rank", "idx", "orbit_seed", "VxD", "VPT", "duty", "RMSE",
          "r_VxD", "r_VPT", "r_duty");
    if (by_vxd.empty())
    {
        emit("(no successful freeruns)\n");
    }
    else
    {
        for (size_t r = 0; r < by_vxd.size(); ++r)
        {
            const size_t i = by_vxd[r];
            const auto& s = rows[i];
            emitf("%-4zu %6zu %20llu %10.3f %10.2f %10.3f %10.6f %6zu %6zu %6zu\n",
                  r + 1, s.index,
                  static_cast<unsigned long long>(s.orbit_seed),
                  s.vpt_x_duty, s.vpt, s.duty, s.rmse,
                  rank_of(by_vxd, i), rank_of(by_vpt, i), rank_of(by_duty, i));
        }
    }

    auto emit_top = [&](const char* title, const std::vector<size_t>& ord) {
        emitf("\n--- Top %d by %s ---\n", top_k, title);
        if (ord.empty())
        {
            emit("(none)\n");
            return;
        }
        const size_t n = std::min(static_cast<size_t>(top_k), ord.size());
        emitf("%-4s %6s %20s %10s %10s %10s\n",
              "rank", "idx", "orbit_seed", "VxD", "VPT", "duty");
        for (size_t r = 0; r < n; ++r)
        {
            const auto& s = rows[ord[r]];
            emitf("%-4zu %6zu %20llu %10.4f %10.4f %10.4f\n",
                  r + 1, s.index,
                  static_cast<unsigned long long>(s.orbit_seed),
                  s.vpt_x_duty, s.vpt, s.duty);
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
        emitf("  IC=(%.6f, %.6f, %.6f)\n", best.ic_x, best.ic_y, best.ic_z);
        emitf("  FreeRun(%zu, %zu, %llu, %.6f, %.6f, %.6f, /*stem after Train*/);\n",
              dim, history_depth,
              static_cast<unsigned long long>(base_esn_seed),
              best.ic_x, best.ic_y, best.ic_z);
    }

    emit("\nCherry-pick: orbit_seed / IC above with this esn_seed (Train then FreeRun; "
         "weights were not kept by this campaign).\n");
    std::fflush(stdout);

    {
        const fs::path tmp = fs::path(rank_csv.string() + ".tmp");
        bool wrote_ok = false;
        {
            std::ofstream csv(tmp, std::ios::out | std::ios::trunc);
            if (!csv)
            {
                std::fprintf(stderr, "[par-orbit-sweep] failed to open %s\n",
                             tmp.string().c_str());
            }
            else
            {
                csv << "# ParallelOrbitSweep\n"
                    << "# one freerun per orbit (VxD = VPT*duty that run)\n"
                    << "# dim=" << dim << " history_depth=" << history_depth
                    << " epochs=" << epochs << " num_orbits=" << num_orbits
                    << " num_threads=" << num_threads << "\n"
                    << "# base_esn_seed=" << base_esn_seed
                    << " base_orbit_seed=" << base_orbit_seed << "\n"
                    << "idx,orbit_seed,ok,vpt_x_duty,vpt,duty,rmse,"
                       "ic_x,ic_y,ic_z,rank_vpt_x_duty,rank_vpt,rank_duty\n";
                for (size_t i : by_vxd)
                {
                    const auto& s = rows[i];
                    csv << s.index << ',' << s.orbit_seed << ",1,"
                        << s.vpt_x_duty << ',' << s.vpt << ',' << s.duty << ','
                        << s.rmse << ','
                        << s.ic_x << ',' << s.ic_y << ',' << s.ic_z << ','
                        << rank_of(by_vxd, i) << ','
                        << rank_of(by_vpt, i) << ','
                        << rank_of(by_duty, i) << '\n';
                }
                for (const auto& s : rows)
                {
                    if (s.ok)
                        continue;
                    csv << s.index << ',' << s.orbit_seed << ",0,,,,,,,,,,\n";
                }
                csv.flush();
                wrote_ok = csv.good();
            }
        }
        if (wrote_ok)
        {
            std::error_code ec;
            fs::remove(rank_csv, ec);
            fs::rename(tmp, rank_csv, ec);
            if (ec)
                std::fprintf(stderr,
                             "[par-orbit-sweep] rename CSV failed: %s  (data at %s)\n",
                             ec.message().c_str(), tmp.string().c_str());
            else
                ReportWrote("par-orbit-sweep", rank_csv);
        }
    }

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

