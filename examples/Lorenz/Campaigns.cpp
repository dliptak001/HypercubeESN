#include "Campaigns.h"
#include "Lorenz.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
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
using ConfigDriveRestore = ConfigRestore<DriveLayout>;
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

// Two compact fixed-knob lines for roll-up banners (M is per-row, omitted).
void AppendCompactConfigLines(std::ostream& o)
{
    o << "config: DIM=" << config::DIM
      << " N=" << (size_t{1} << config::DIM)
      << "  epochs=" << config::EPOCHS
      << "  train_win=" << config::TRAINING_WINDOW_SIZE
      << "  warmup=" << config::WARMUP_STEPS
      << "  freerun_win=" << config::FREE_RUN_WINDOW_SIZE
      << "  SR=" << config::SPECTRAL_RADIUS
      << "  in_scale=" << config::INPUT_SCALING
      << "  leak=" << config::LEAK_RATE
      << "\n";
    o << "config: pool=" << (config::USE_POOLING ? "on" : "off")
      << "  slices=" << config::READOUT_SLICES
      << "  ch=" << config::CONV_CHANNELS
      << "  layers=" << config::NUM_LAYERS
      << "  act=" << ActivationName(config::READOUT_ACTIVATION)
      << "  lr=" << config::LEARNING_RATE << ".." << config::LEARNING_RATE_MIN
      << "  theta=" << config::VPT_THRESHOLD
      << "  load_w=" << (config::LOAD_TRAINED_WEIGHTS ? "on" : "off")
      << "  drive=" << Lorenz::DriveLayoutName(config::DRIVE_LAYOUT)
      << "  n_in=" << Lorenz::NumDriveChannels(config::DRIVE_LAYOUT)
      << "\n";
    {
        const size_t n_in = Lorenz::NumDriveChannels(config::DRIVE_LAYOUT);
        o << "config: drive_ch=[";
        for (size_t i = 0; i < n_in; ++i)
        {
            if (i)
                o << ',';
            o << config::INPUT_SCALE_CH[i];
        }
        o << "] (x in_scale)\n";
    }
}

// Full run metadata for post-analysis (comment lines in CSV; prose in .txt).
void WriteMetadataBlock(std::ostream& o, const char* job, const std::string& timestamp,
                        uint64_t base_seed, uint64_t orbit_seed,
                        size_t num_trials, int num_runs,
                        FreeRunProtocol protocol, size_t history_depth)
{
    o << "# HypercubeESN Lorenz results\n"
      << "# job=" << job << "\n"
      << "# timestamp=" << timestamp << "\n"
      << "# protocol=" << Lorenz::ProtocolName(protocol) << "\n"
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
      << "# drive_layout=" << Lorenz::DriveLayoutName(config::DRIVE_LAYOUT)
      << "  num_inputs=" << Lorenz::NumDriveChannels(config::DRIVE_LAYOUT) << "\n";
    {
        const size_t n_in = Lorenz::NumDriveChannels(config::DRIVE_LAYOUT);
        o << "# drive_ch=[";
        for (size_t i = 0; i < n_in; ++i)
        {
            if (i)
                o << ',';
            o << config::INPUT_SCALE_CH[i];
        }
        o << "] (x input_scaling; layout feature order)\n";
    }
    o << "# metrics=mean_of_trial_means (sample std across trials)\n"
      << "# score_vpt_x_duty=VPT_lt*duty\n"
      << "# freerun_metrics=VPT,duty,VPT*duty,RMSE\n"
      << "# freerun_pool=best_half_per_metric (ceil(n/2); weak ICs discarded; "
         "higher: VPT duty VPT*duty; lower: RMSE)\n";
}

const char* SurveyCsvHeader()
{
    return "M,mean_vpt,std_vpt,mean_rmse,std_rmse,mean_duty,std_duty,"
           "mean_vpt_x_duty,std_vpt_x_duty,"
           "n_trials_ok,num_trials,num_runs,wall_seconds,"
           "protocol,ok\n";
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
      << Lorenz::ProtocolName(s.protocol) << ','
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
                           s.num_trials, s.num_runs, s.protocol, s.history_depth);
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
                           s.num_trials, s.num_runs, s.protocol, s.history_depth);
        txt << "\nSurvey aggregate (mean of trial-means)\n";
        txt << "  M=" << s.history_depth
            << "  trials_ok=" << s.n_trials_ok << "/" << s.num_trials
            << "  freeruns/trial=" << s.num_runs << "\n";
        txt << "  VPT   mean=" << s.mean_vpt << "  std=" << s.std_vpt
            << "  (best half freeruns)\n";
        txt << "  RMSE  mean=" << s.mean_rmse << "  std=" << s.std_rmse
            << "  (best half freeruns, lower-is-better)\n";
        txt << "  duty  mean=" << s.mean_duty << "  std=" << s.std_duty
            << "  (best half freeruns)\n";
        txt << "  VPT*duty mean=" << s.mean_vpt_x_duty << "  std=" << s.std_vpt_x_duty
            << "  (best half freeruns)\n";
        txt << "  wall_seconds=" << s.wall_seconds << "  ok=" << (s.ok ? 1 : 0) << "\n";
    }

    // stdout (not stderr): CLion/Windows consoles merge streams asynchronously and will
    // splice stderr mid-line into printf tables if we log "wrote" on a different stream.
    ReportWrote("results", csv_path);
    ReportWrote("results", txt_path);
}

// M-sweep roll-up: one CSV (metadata + all M rows) + matching TXT.
void WriteMsweepResultFiles(const std::vector<SurveySummary>& rows,
                            uint64_t base_seed, uint64_t orbit_seed,
                            size_t num_threads, int num_runs,
                            double total_wall_s,
                            size_t i_best_vpt, size_t i_best_duty,
                            size_t i_best_vxd)
{
    const fs::path dir = EnsureResultsDir();
    if (!fs::exists(dir))
        return;

    const std::string ts = TimestampNow();
    const fs::path csv_path = dir / ("Msweep_" + ts + ".csv");
    const fs::path txt_path = dir / ("Msweep_" + ts + ".txt");

    const FreeRunProtocol proto = rows.empty() ? config::FREE_RUN_PROTOCOL : rows.front().protocol;
    const size_t M_meta = rows.empty() ? config::HISTORY_DEPTH : rows.front().history_depth;

    {
        std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
        if (!csv)
        {
            std::fprintf(stderr, "[results] failed to write %s\n", csv_path.string().c_str());
            return;
        }
        WriteMetadataBlock(csv, "Msweep", ts, base_seed, orbit_seed,
                           num_threads, num_runs, proto, M_meta);
        csv << "# note=history_depth_M column varies per row; metadata M is first successful row\n";
        csv << "# total_wall_seconds=" << total_wall_s << "\n";
        if (!rows.empty())
        {
            csv << "# best_mean_vpt_M=" << rows[i_best_vpt].history_depth
                << "  best_mean_duty_M=" << rows[i_best_duty].history_depth
                << "  best_mean_vpt_x_duty_M=" << rows[i_best_vxd].history_depth << "\n";
        }
        csv << SurveyCsvHeader();
        for (const auto& r : rows)
            WriteSurveyCsvRow(csv, r);
    }
    {
        std::ofstream txt(txt_path, std::ios::out | std::ios::trunc);
        if (!txt)
        {
            std::fprintf(stderr, "[results] failed to write %s\n", txt_path.string().c_str());
            return;
        }
        WriteMetadataBlock(txt, "Msweep", ts, base_seed, orbit_seed,
                           num_threads, num_runs, proto, M_meta);
        txt << "\nM-sweep roll-up (mean of trial-means; code-computed)\n";
        AppendCompactConfigLines(txt);
        if (rows.empty())
        {
            txt << "(no successful M rows)\n";
        }
        else
        {
            txt << "protocol=" << Lorenz::ProtocolName(rows.front().protocol)
                << "  trials/M=" << rows.front().num_trials
                << "  freeruns/trial=" << rows.front().num_runs
                << "  theta=" << config::VPT_THRESHOLD << "\n\n";
            txt << "M,VPT_mn,VPT_sd,RMSE_mn,RMSE_sd,duty_mn,duty_sd,"
                   "VxD_mn,VxD_sd,trials_ok,wall_s\n";
            for (const auto& r : rows)
            {
                txt << r.history_depth << ','
                    << r.mean_vpt << ',' << r.std_vpt << ','
                    << r.mean_rmse << ',' << r.std_rmse << ','
                    << r.mean_duty << ',' << r.std_duty << ','
                    << r.mean_vpt_x_duty << ',' << r.std_vpt_x_duty << ','
                    << r.n_trials_ok << ',' << r.wall_seconds << '\n';
            }
            txt << "\nDeltas vs first row (M=" << rows.front().history_depth << ")\n";
            txt << "M,dVPT,dRMSE,dDuty,dVPT*duty\n";
            const auto& b = rows.front();
            for (const auto& r : rows)
            {
                txt << r.history_depth << ','
                    << (r.mean_vpt - b.mean_vpt) << ','
                    << (r.mean_rmse - b.mean_rmse) << ','
                    << (r.mean_duty - b.mean_duty) << ','
                    << (r.mean_vpt_x_duty - b.mean_vpt_x_duty) << '\n';
            }
            txt << "\nCode picks:\n";
            txt << "  best mean VPT  : M=" << rows[i_best_vpt].history_depth
                << "  VPT=" << rows[i_best_vpt].mean_vpt
                << " +/- " << rows[i_best_vpt].std_vpt << "\n";
            txt << "  best mean duty : M=" << rows[i_best_duty].history_depth
                << "  duty=" << rows[i_best_duty].mean_duty
                << " +/- " << rows[i_best_duty].std_duty << "\n";
            txt << "  best mean VPT*duty : M=" << rows[i_best_vxd].history_depth
                << "  VPT*duty=" << rows[i_best_vxd].mean_vpt_x_duty
                << " +/- " << rows[i_best_vxd].std_vpt_x_duty << "\n";
            txt << "  total_wall_seconds=" << total_wall_s << "\n";
        }
    }

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

/// Upper half of a higher-is-better sample (best 50%). Odd n: keep ceil(n/2).
std::vector<double> BestHalfHigherIsBetter(std::vector<double> v)
{
    if (v.size() <= 1)
        return v;
    std::sort(v.begin(), v.end());
    const size_t keep = (v.size() + 1) / 2; // ceil(n/2)
    return std::vector<double>(v.end() - static_cast<std::ptrdiff_t>(keep), v.end());
}

/// Lower half of a lower-is-better sample (best 50%). Odd n: keep ceil(n/2).
std::vector<double> BestHalfLowerIsBetter(std::vector<double> v)
{
    if (v.size() <= 1)
        return v;
    std::sort(v.begin(), v.end());
    const size_t keep = (v.size() + 1) / 2; // ceil(n/2)
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
    size_t n_valid = 0; ///< valid freeruns before best-half filter
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

    const FreeRunProtocol protocol = lorenz.EffectiveFreeRunProtocol();
    const size_t n_train_orbits = lorenz.NumTrainOrbits();
    if ((protocol == FreeRunProtocol::TrainInSample ||
         protocol == FreeRunProtocol::TrainHoldout) &&
        n_train_orbits > 0 && static_cast<size_t>(num_runs) > n_train_orbits)
    {
        std::fprintf(stderr,
                     "[seed %llu] WARN: %d free-runs > %zu train orbits (protocol=%s) -- "
                     "extra free-runs reuse train ICs (modulo). Not unique coverage.\n",
                     static_cast<unsigned long long>(esn_seed), num_runs, n_train_orbits,
                     Lorenz::ProtocolName(protocol));
        std::fflush(stderr);
    }

    std::vector<FreeRunResult> results;
    results.reserve(num_runs);
    const int prog_every = (num_runs <= 20) ? 1 : std::max(10, num_runs / 20);
    for (int i = 0; i < num_runs; i++)
    {
        results.push_back(lorenz.FreeRun(false, nullptr, 0, protocol));
        if ((i + 1) % prog_every == 0 || i + 1 == num_runs)
            progress("free-run", i + 1, num_runs);
    }

    std::string out;
    char buf[384];
    auto emit = [&](const char* s) { out += s; };

    std::snprintf(buf, sizeof buf,
                  "\n=== ESN seed %llu : %d free-runs (orbit seed %llu) protocol=%s ===\n",
                  static_cast<unsigned long long>(esn_seed), num_runs,
                  static_cast<unsigned long long>(orbit_seed),
                  Lorenz::ProtocolName(protocol));
    emit(buf);
    if ((protocol == FreeRunProtocol::TrainInSample ||
         protocol == FreeRunProtocol::TrainHoldout) &&
        n_train_orbits > 0 && static_cast<size_t>(num_runs) > n_train_orbits)
    {
        std::snprintf(buf, sizeof buf,
                      "  note: free-runs (%d) exceed train orbits (%zu); extras reuse train ICs "
                      "(modulo) -- not unique coverage\n",
                      num_runs, n_train_orbits);
        emit(buf);
    }

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

    // Best half of ICs per metric (independent). Higher: VPT, duty, VPT*duty.
    // Lower: RMSE. Weak ICs discarded on purpose.
    const size_t n_full = vpt_lts.size();
    const std::vector<double> vpt_best = BestHalfHigherIsBetter(vpt_lts);
    const std::vector<double> duty_best = BestHalfHigherIsBetter(duties);
    const std::vector<double> vxd_best = BestHalfHigherIsBetter(vpt_x_duties);
    const std::vector<double> rmse_best = BestHalfLowerIsBetter(rmses);

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
                          "  %-20s n=%2zu (best half of %zu)  min=%.*f  max=%.*f  "
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
    emit("  note: freerun stats = VPT, duty, VPT*duty, RMSE only; best 50% of ICs per "
         "metric (odd n keeps ceil(n/2); higher: VPT duty VPT*duty; lower: RMSE). "
         "Weak ICs discarded on purpose; see examples/Lorenz/README.md\n");
    if (protocol == FreeRunProtocol::TrainInSample)
    {
        emit("  note: TrainInSample free-run scores inside the train window (in-sample generative;"
             " not a held-out free-run claim)\n");
    }
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
    std::printf("[survey] protocol=%s  load_weights=%s  (input-bank free-run; ext-fb off)\n",
                Lorenz::ProtocolName(config::FREE_RUN_PROTOCOL),
                config::LOAD_TRAINED_WEIGHTS ? "on" : "off");
    if (config::LOAD_TRAINED_WEIGHTS)
        std::printf("[survey] load stem: %s\n", config::LOAD_WEIGHTS_STEM);
    if ((config::FREE_RUN_PROTOCOL == FreeRunProtocol::TrainInSample ||
         config::FREE_RUN_PROTOCOL == FreeRunProtocol::TrainHoldout) &&
        !config::LOAD_TRAINED_WEIGHTS &&
        static_cast<size_t>(num_runs) > config::EPOCHS)
    {
        std::printf("[survey] WARN: NUM_RUNS=%d > EPOCHS=%zu for %s -- free-runs will reuse "
                    "train ICs (modulo); not unique coverage (not clamped)\n",
                    num_runs, config::EPOCHS,
                    Lorenz::ProtocolName(config::FREE_RUN_PROTOCOL));
    }
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
    sum.drive_layout = config::DRIVE_LAYOUT;
    sum.num_inputs = Lorenz::NumDriveChannels(config::DRIVE_LAYOUT);
    sum.num_trials = num_threads;
    sum.num_runs = num_runs;
    sum.n_trials_ok = trial_vpt.size();
    sum.protocol = config::FREE_RUN_PROTOCOL;
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
        std::printf("[survey] aggregate (mean of %zu trial-means; best-half ICs):  "
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
    std::printf("[trace] protocol=%s  DIM=%zu (N=%zu)  M=%zu  esn_seed=%llu  max_freeruns=%d  "
                "target_orbit=%llu\n",
                Lorenz::ProtocolName(config::FREE_RUN_PROTOCOL),
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
    const FreeRunProtocol protocol = lorenz.EffectiveFreeRunProtocol();
    const size_t n_train_orbits = lorenz.NumTrainOrbits();
    if (target_orbit == 0 &&
        (protocol == FreeRunProtocol::TrainInSample ||
         protocol == FreeRunProtocol::TrainHoldout) &&
        n_train_orbits > 0 && static_cast<size_t>(max_freeruns) > n_train_orbits)
    {
        std::fprintf(stderr,
                     "[trace] WARN: max_freeruns=%d > %zu train orbits (protocol=%s); "
                     "extras reuse train ICs (modulo)\n",
                     max_freeruns, n_train_orbits, Lorenz::ProtocolName(protocol));
        std::fflush(stderr);
    }

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
                                         0, protocol, static_cast<size_t>(-1), target_orbit);
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
            FreeRunResult r = lorenz.FreeRun(false, tmp_path.string().c_str(), 0, protocol);
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
            const char* weights_stem)
{
    if (!ValidateDim(dim, "freerun"))
        return 2;
    if (!ValidateHistoryDepth(history_depth, "freerun"))
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

    // Seating only: fixed IC builds the stream (no train-orbit list required).
    const FreeRunProtocol protocol = config::FREE_RUN_PROTOCOL;

    ReportBanner("FreeRun (load-only)");
    std::printf("[freerun] protocol=%s  DIM=%zu (N=%zu)  M=%zu  esn_seed=%llu\n",
                Lorenz::ProtocolName(protocol),
                config::DIM, size_t{1} << config::DIM, config::HISTORY_DEPTH,
                static_cast<unsigned long long>(esn_seed));
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

    std::printf("[freerun] free-run fixed IC under protocol=%s (no train)\n",
                Lorenz::ProtocolName(protocol));
    std::fflush(stdout);

    FreeRunResult r = lorenz.FreeRun(/*verbose=*/true, out_path.string().c_str(),
                                     0, protocol, static_cast<size_t>(-1),
                                     /*fixed_orbit_seed=*/0, &ic);
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
// FreeRunSurvey (campaign: load weights + many Unseen freeruns + rank ICs)
// ---------------------------------------------------------------------------
// Pipeline middle step: Train -> FreeRunSurvey -> FreeRun (cherry-pick IC).
int FreeRunSurvey(size_t dim, size_t history_depth, uint64_t esn_seed,
                  int num_runs, uint64_t orbit_seed, const char* weights_stem,
                  int top_k, FreeRunSurveySummary* out)
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

    const fs::path survey_dir = SurveysDir();
    if (!EnsureDir(survey_dir, "freerun-survey"))
        return 2;

    ReportBanner("FreeRunSurvey (load-only; multi-orbit)");
    std::printf("[freerun-survey] DIM=%zu (N=%zu)  M=%zu  esn_seed=%llu  num_runs=%d  "
                "orbit_seed=%llu  top_k=%d\n",
                config::DIM, size_t{1} << config::DIM, config::HISTORY_DEPTH,
                static_cast<unsigned long long>(esn_seed), num_runs,
                static_cast<unsigned long long>(orbit_seed), top_k);
    std::printf("[freerun-survey] load stem: %s\n", stem);
    std::printf("[freerun-survey] freerun protocol=Unseen (remix IC each run); "
                "stats use best-half pool; metrics=VPT,duty,VPT*duty,RMSE\n");
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
        // Unseen: remixes orbit_seed_ each call; no train-orbit list needed.
        FreeRunResult r = lorenz.FreeRun(/*verbose=*/false, /*csv=*/nullptr,
                                         0, FreeRunProtocol::Unseen);
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
    const auto vpt_best = BestHalfHigherIsBetter(vpt_lts);
    const auto duty_best = BestHalfHigherIsBetter(duties);
    const auto vxd_best = BestHalfHigherIsBetter(vxds);
    const auto rmse_best = BestHalfLowerIsBetter(rmses);

    auto report = [](const char* label, std::vector<double> v, int prec, size_t n_pool) {
        if (v.empty())
            return;
        std::sort(v.begin(), v.end());
        double mean = 0, sd = 0;
        MeanStd(v, mean, sd);
        const size_t n = v.size();
        const double median = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
        std::printf("  %-14s n=%zu (best half of %zu)  min=%.*f  max=%.*f  "
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
    std::printf("  note: best-half per metric; primary sort key for leaderboard = VPT*duty\n");

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
// Drive-channel gains (shared: SeedSweep, Campaign_DriveGainAB)
// ---------------------------------------------------------------------------
struct ConfigDriveGainsRestore
{
    float saved[kMaxDriveChannels]{};
    ConfigDriveGainsRestore()
    {
        for (size_t i = 0; i < kMaxDriveChannels; ++i)
            saved[i] = config::INPUT_SCALE_CH[i];
    }
    ~ConfigDriveGainsRestore()
    {
        for (size_t i = 0; i < kMaxDriveChannels; ++i)
            config::INPUT_SCALE_CH[i] = saved[i];
    }
    ConfigDriveGainsRestore(const ConfigDriveGainsRestore&) = delete;
    ConfigDriveGainsRestore& operator=(const ConfigDriveGainsRestore&) = delete;
};

bool ParseDriveGains(std::initializer_list<float> src, float* dst, size_t n_in,
                     const char* campaign)
{
    if (src.size() != n_in)
    {
        std::fprintf(stderr,
                     "[%s] refused: drive_gains has %zu entries; need exactly %zu "
                     "(current drive layout channel count)\n",
                     campaign, src.size(), n_in);
        return false;
    }
    size_t i = 0;
    for (float g : src)
    {
        if (!std::isfinite(g) || g < 0.0f)
        {
            std::fprintf(stderr,
                         "[%s] refused: drive_gains[%zu]=%g (need finite and >= 0)\n",
                         campaign, i, static_cast<double>(g));
            return false;
        }
        dst[i++] = g;
    }
    for (; i < kMaxDriveChannels; ++i)
        dst[i] = 1.0f;
    return true;
}

void ApplyDriveGains(const float* gains)
{
    for (size_t i = 0; i < kMaxDriveChannels; ++i)
        config::INPUT_SCALE_CH[i] = gains[i];
}

std::string FormatDriveGains(const float* gains, size_t n)
{
    std::ostringstream o;
    o << '[';
    for (size_t i = 0; i < n; ++i)
    {
        if (i)
            o << ',';
        o << gains[i];
    }
    o << ']';
    return o.str();
}

// ---------------------------------------------------------------------------
// SeedSweep (Train? -> FreeRunSurvey per seed; rank seeds by mean VPT*duty)
// ---------------------------------------------------------------------------
int SeedSweep(size_t dim, size_t history_depth,
              std::initializer_list<uint64_t> esn_seeds,
              size_t epochs, int freerun_runs,
              uint64_t train_orbit, uint64_t freerun_orbit_seed,
              int top_k, bool do_train,
              float spectral_radius, float input_scaling,
              std::initializer_list<float> drive_gains)
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
    // Optional overrides: 0 keeps config::; > 0 must be finite and positive; < 0 refused.
    if (spectral_radius < 0.0f)
    {
        std::fprintf(stderr,
                     "[seed-sweep] refused: spectral_radius=%g (use > 0 to set, or 0 to keep config)\n",
                     static_cast<double>(spectral_radius));
        return 2;
    }
    if (input_scaling < 0.0f)
    {
        std::fprintf(stderr,
                     "[seed-sweep] refused: input_scaling=%g (use > 0 to set, or 0 to keep config)\n",
                     static_cast<double>(input_scaling));
        return 2;
    }
    if (spectral_radius > 0.0f && !ValidateSpectralRadius(spectral_radius, "seed-sweep"))
        return 2;
    if (input_scaling > 0.0f && !ValidateInputScaling(input_scaling, "seed-sweep"))
        return 2;

    const size_t n_in = Lorenz::NumDriveChannels(config::DRIVE_LAYOUT);
    float gains_buf[kMaxDriveChannels]{};
    const bool override_gains = drive_gains.size() > 0;
    if (override_gains && !ParseDriveGains(drive_gains, gains_buf, n_in, "seed-sweep"))
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
    ConfigDriveGainsRestore gains_restore;
    if (override_gains)
        ApplyDriveGains(gains_buf);

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
        const std::string ch = FormatDriveGains(config::INPUT_SCALE_CH, n_in);
        std::printf("[seed-sweep] drive_ch=%s%s  drive=%s  n_in=%zu\n",
                    ch.c_str(),
                    override_gains ? " (override)" : " (config)",
                    Lorenz::DriveLayoutName(config::DRIVE_LAYOUT), n_in);
    }
    std::printf("[seed-sweep] train_orbit=%llu  freerun_orbit_seed=%llu\n",
                static_cast<unsigned long long>(train_orbit),
                static_cast<unsigned long long>(freerun_orbit_seed));
    std::printf("[seed-sweep] weight stems: %s/lorenz_seed{S}_D%zu_M%zu  "
                "(stems omit SR/IS/drive_ch -- document in ranking banner)\n",
                model_dir.string().c_str(), dim, history_depth);
    std::printf("[seed-sweep] seed ranking metric = mean VPT*duty (best-half freeruns)\n");
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
            std::printf("[seed-sweep] seed %llu  best-half means:  "
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
    std::printf("=== SeedSweep ranking by mean VPT*duty (best-half freeruns) ===\n");
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
// Campaign_HistoryDepthSweep
// ---------------------------------------------------------------------------
int Campaign_HistoryDepthSweep(size_t dim, std::initializer_list<size_t> history_depths,
                               size_t num_threads, int num_runs,
                               uint64_t base_seed, uint64_t orbit_seed)
{
    if (!ValidateDim(dim, "M-sweep"))
        return 2;
    if (history_depths.size() == 0)
    {
        std::fprintf(stderr, "[M-sweep] empty history_depths list -- nothing to do\n");
        return 2;
    }

    // Same HCNW under every M is not a valid comparison (readout trained for one geometry).
    if (config::LOAD_TRAINED_WEIGHTS)
    {
        std::fprintf(stderr,
                     "[M-sweep] refused: LOAD_TRAINED_WEIGHTS is on. "
                     "Loading one readout while sweeping M is meaningless. "
                     "Turn load off and train per M (or load only for fixed-M runs).\n");
        return 2;
    }

    if (config::SAVE_TRAINED_WEIGHTS)
    {
        std::fprintf(stderr,
                     "[M-sweep] NOTE: SAVE_TRAINED_WEIGHTS is on; stems include "
                     "DIM/M/num_inputs (lorenz_seed{{seed}}_D{{DIM}}_M{{M}}_in{{Nin}}) "
                     "so each geometry keeps its own file.\n");
        std::fflush(stderr);
    }

    // RAII: always restore caller's DIM and HISTORY_DEPTH (throw / early return safe).
    ConfigSizeRestore dim_restore(config::DIM, dim);
    ConfigSizeRestore m_restore(config::HISTORY_DEPTH, config::HISTORY_DEPTH);

    std::printf("=== HypercubeESN: Lorenz / history-depth (M) sweep ===\n");
    std::printf("[M-sweep] DIM=%zu (N=%zu)  %zu M value(s):",
                config::DIM, size_t{1} << config::DIM, history_depths.size());
    for (size_t M : history_depths)
        std::printf(" %zu", M);
    std::printf("\n");
    std::printf("[M-sweep] survey threads=%zu  runs=%d  (0 threads => HW)\n",
                num_threads, num_runs);
    std::fflush(stdout);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    int first_err = 0;
    size_t step = 0;
    size_t ran = 0;
    std::vector<SurveySummary> rows;
    rows.reserve(history_depths.size());

    for (size_t M : history_depths)
    {
        ++step;
        if (M < 1 || M > 64)
        {
            std::fprintf(stderr,
                         "[M-sweep] skip M=%zu (reservoir requires 1 <= M <= 64)\n", M);
            std::fflush(stderr);
            if (first_err == 0)
                first_err = 2;
            continue;
        }

        config::HISTORY_DEPTH = M;
        ++ran;
        std::printf("\n########## M-sweep step %zu/%zu (ran %zu): HISTORY_DEPTH (M) = %zu ##########\n",
                    step, history_depths.size(), ran, M);
        std::fflush(stdout);

        SurveySummary sum;
        const int rc = Campaign_SeedSurvey(dim, num_threads, num_runs, base_seed, orbit_seed,
                                           /*completion_beep=*/false, &sum);
        if (sum.ok)
            rows.push_back(sum);
        if (rc != 0 && first_err == 0)
            first_err = rc;
    }

    // ---- Roll-up: all numbers from SurveySummary (code-crunched), not estimated ----
    // Flush any prior survey [results] lines before the table so the console stays clean.
    std::fflush(stdout);

    size_t i_best_vpt = 0;
    size_t i_best_duty = 0;
    size_t i_best_vxd = 0;

    // Build the entire roll-up as one string, then one write + flush: avoids mid-line
    // splice if any other stream noise races the console host.
    {
        std::ostringstream roll;
        roll << "\n========================================================================\n"
             << "=== M-sweep roll-up (mean of trial-means; code-computed) ===\n"
             << "========================================================================\n";
        AppendCompactConfigLines(roll);
        if (rows.empty())
        {
            roll << "(no successful M rows)\n";
        }
        else
        {
            char line[640];
            std::snprintf(line, sizeof line,
                          "protocol=%s  trials/M=%zu  freeruns/trial=%d  theta=%.2f\n",
                          Lorenz::ProtocolName(rows.front().protocol),
                          rows.front().num_trials, rows.front().num_runs,
                          config::VPT_THRESHOLD);
            roll << line;

            std::snprintf(line, sizeof line,
                          "%-6s %8s %8s %10s %8s %8s %8s %8s %8s %10s %10s\n",
                          "M", "VPT_mn", "VPT_sd", "RMSE_mn", "RMSE_sd", "duty_mn", "duty_sd",
                          "VxD_mn", "VxD_sd", "trials_ok", "wall_s");
            roll << line;
            std::snprintf(line, sizeof line,
                          "%-6s %8s %8s %10s %8s %8s %8s %8s %8s %10s %10s\n",
                          "------", "--------", "--------", "----------", "--------", "--------",
                          "--------", "--------", "--------", "----------", "----------");
            roll << line;

            for (size_t i = 0; i < rows.size(); ++i)
            {
                const auto& r = rows[i];
                std::snprintf(line, sizeof line,
                              "%-6zu %8.3f %8.3f %10.6f %8.6f %8.3f %8.3f %8.3f %8.3f %10zu %10.1f\n",
                              r.history_depth,
                              r.mean_vpt, r.std_vpt,
                              r.mean_rmse, r.std_rmse,
                              r.mean_duty, r.std_duty,
                              r.mean_vpt_x_duty, r.std_vpt_x_duty,
                              r.n_trials_ok, r.wall_seconds);
                roll << line;
                if (r.mean_vpt > rows[i_best_vpt].mean_vpt)
                    i_best_vpt = i;
                if (r.mean_duty > rows[i_best_duty].mean_duty)
                    i_best_duty = i;
                if (r.mean_vpt_x_duty > rows[i_best_vxd].mean_vpt_x_duty)
                    i_best_vxd = i;
            }

            // Delta vs first successful row (baseline = first M that ran)
            std::snprintf(line, sizeof line, "\nDeltas vs first row (M=%zu):\n",
                          rows.front().history_depth);
            roll << line;
            std::snprintf(line, sizeof line, "%-6s %10s %12s %10s %12s\n",
                          "M", "dVPT", "dRMSE", "dDuty", "dVPT*duty");
            roll << line;
            const auto& b = rows.front();
            for (const auto& r : rows)
            {
                std::snprintf(line, sizeof line,
                              "%-6zu %+10.3f %+12.6f %+10.3f %+12.3f\n",
                              r.history_depth,
                              r.mean_vpt - b.mean_vpt,
                              r.mean_rmse - b.mean_rmse,
                              r.mean_duty - b.mean_duty,
                              r.mean_vpt_x_duty - b.mean_vpt_x_duty);
                roll << line;
            }

            roll << "\nCode picks (max mean among successful M):\n";
            std::snprintf(line, sizeof line,
                          "  best mean VPT  : M=%zu  VPT=%.3f +/- %.3f\n",
                          rows[i_best_vpt].history_depth,
                          rows[i_best_vpt].mean_vpt, rows[i_best_vpt].std_vpt);
            roll << line;
            std::snprintf(line, sizeof line,
                          "  best mean duty : M=%zu  duty=%.3f +/- %.3f\n",
                          rows[i_best_duty].history_depth,
                          rows[i_best_duty].mean_duty, rows[i_best_duty].std_duty);
            roll << line;
            std::snprintf(line, sizeof line,
                          "  best mean VPT*duty : M=%zu  VPT*duty=%.3f +/- %.3f\n",
                          rows[i_best_vxd].history_depth,
                          rows[i_best_vxd].mean_vpt_x_duty, rows[i_best_vxd].std_vpt_x_duty);
            roll << line;
        }

        const double elapsed =
            std::chrono::duration<double>(clock::now() - t0).count();
        char time_buf[64];
        FormatWallTime(time_buf, sizeof time_buf, elapsed);
        char wall_line[160];
        std::snprintf(wall_line, sizeof wall_line,
                      "\n=== M-sweep wall time: %s (restoring DIM=%zu HISTORY_DEPTH=%zu) ===\n",
                      time_buf, dim_restore.saved, m_restore.saved);
        roll << wall_line;

        const std::string block = roll.str();
        std::fwrite(block.data(), 1, block.size(), stdout);
        std::fflush(stdout);

        // Always persist roll-up under RESULTS_DIR for post-analysis (CSV + TXT).
        WriteMsweepResultFiles(rows, base_seed, orbit_seed, num_threads, num_runs, elapsed,
                               i_best_vpt, i_best_duty, i_best_vxd);
    }

    Beep(2500, 3000);
    return first_err;
}

// Drive-layout A/B: CSV + TXT under RESULTS_DIR.
void WriteDriveAbResultFiles(const SurveySummary& a, const SurveySummary& b,
                             size_t dim, size_t history_depth,
                             uint64_t base_seed, uint64_t orbit_seed,
                             size_t num_threads, int num_runs,
                             double total_wall_s)
{
    const fs::path dir = EnsureResultsDir();
    if (!fs::exists(dir))
        return;

    const std::string ts = TimestampNow();
    const fs::path csv_path = dir / ("DriveAB_" + ts + "_D" + std::to_string(dim) +
                                     "_M" + std::to_string(history_depth) + ".csv");
    const fs::path txt_path = dir / ("DriveAB_" + ts + "_D" + std::to_string(dim) +
                                     "_M" + std::to_string(history_depth) + ".txt");

    auto write_row = [](std::ostream& o, const SurveySummary& s) {
        o << Lorenz::DriveLayoutName(s.drive_layout) << ','
          << s.num_inputs << ','
          << s.history_depth << ','
          << s.mean_vpt << ',' << s.std_vpt << ','
          << s.mean_rmse << ',' << s.std_rmse << ','
          << s.mean_duty << ',' << s.std_duty << ','
          << s.mean_vpt_x_duty << ',' << s.std_vpt_x_duty << ','
          << s.n_trials_ok << ',' << s.num_trials << ',' << s.num_runs << ','
          << s.wall_seconds << ','
          << Lorenz::ProtocolName(s.protocol) << ','
          << (s.ok ? 1 : 0) << '\n';
    };

    {
        std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
        if (!csv)
        {
            std::fprintf(stderr, "[results] failed to write %s\n", csv_path.string().c_str());
            return;
        }
        WriteMetadataBlock(csv, "DriveAB", ts, base_seed, orbit_seed,
                           num_threads, num_runs, config::FREE_RUN_PROTOCOL, history_depth);
        csv << "# dim=" << dim << "  fixed_M=" << history_depth
            << "  total_wall_seconds=" << total_wall_s << "\n";
        csv << "drive,num_inputs,M,mean_vpt,std_vpt,mean_rmse,std_rmse,"
               "mean_duty,std_duty,mean_vpt_x_duty,std_vpt_x_duty,"
               "n_trials_ok,num_trials,num_runs,wall_seconds,protocol,ok\n";
        if (a.ok) write_row(csv, a);
        if (b.ok) write_row(csv, b);
        if (a.ok && b.ok)
        {
            csv << "# deltas (Quadratic8 - XyzXz): dVPT=" << (b.mean_vpt - a.mean_vpt)
                << "  dRMSE=" << (b.mean_rmse - a.mean_rmse)
                << "  dDuty=" << (b.mean_duty - a.mean_duty)
                << "  dVPT*duty=" << (b.mean_vpt_x_duty - a.mean_vpt_x_duty) << "\n";
        }
    }
    {
        std::ofstream txt(txt_path, std::ios::out | std::ios::trunc);
        if (!txt)
        {
            std::fprintf(stderr, "[results] failed to write %s\n", txt_path.string().c_str());
            return;
        }
        WriteMetadataBlock(txt, "DriveAB", ts, base_seed, orbit_seed,
                           num_threads, num_runs, config::FREE_RUN_PROTOCOL, history_depth);
        txt << "\nDrive-layout A/B (fixed M; mean of trial-means)\n";
        txt << "DIM=" << dim << "  N=" << (size_t{1} << dim)
            << "  M=" << history_depth << "\n";
        AppendCompactConfigLines(txt);
        txt << "\narm,drive,n_in,VPT_mn,VPT_sd,RMSE_mn,RMSE_sd,duty_mn,duty_sd,"
               "VxD_mn,VxD_sd,trials_ok,wall_s\n";
        auto arm_line = [&](const char* tag, const SurveySummary& s) {
            if (!s.ok)
            {
                txt << tag << ",(failed)\n";
                return;
            }
            txt << tag << ','
                << Lorenz::DriveLayoutName(s.drive_layout) << ','
                << s.num_inputs << ','
                << s.mean_vpt << ',' << s.std_vpt << ','
                << s.mean_rmse << ',' << s.std_rmse << ','
                << s.mean_duty << ',' << s.std_duty << ','
                << s.mean_vpt_x_duty << ',' << s.std_vpt_x_duty << ','
                << s.n_trials_ok << ',' << s.wall_seconds << '\n';
        };
        arm_line("A", a);
        arm_line("B", b);
        if (a.ok && b.ok)
        {
            txt << "\nDeltas (B - A) = Quadratic8 - XyzXz\n";
            txt << "dVPT=" << (b.mean_vpt - a.mean_vpt)
                << "  dRMSE=" << (b.mean_rmse - a.mean_rmse)
                << "  dDuty=" << (b.mean_duty - a.mean_duty)
                << "  dVPT*duty=" << (b.mean_vpt_x_duty - a.mean_vpt_x_duty) << "\n";
            txt << "\nCode picks among successful arms:\n";
            const SurveySummary& best = (b.mean_vpt >= a.mean_vpt) ? b : a;
            txt << "  best mean VPT : " << Lorenz::DriveLayoutName(best.drive_layout)
                << "  VPT=" << best.mean_vpt << " +/- " << best.std_vpt << "\n";
            const SurveySummary& best_vxd =
                (b.mean_vpt_x_duty >= a.mean_vpt_x_duty) ? b : a;
            txt << "  best mean VPT*duty : " << Lorenz::DriveLayoutName(best_vxd.drive_layout)
                << "  VPT*duty=" << best_vxd.mean_vpt_x_duty
                << " +/- " << best_vxd.std_vpt_x_duty << "\n";
            txt << "  total_wall_seconds=" << total_wall_s << "\n";
        }
    }

    ReportWrote("results", csv_path);
    ReportWrote("results", txt_path);
}

// ---------------------------------------------------------------------------
// Campaign_DriveLayoutAB
// ---------------------------------------------------------------------------
int Campaign_DriveLayoutAB(size_t dim, size_t history_depth,
                           size_t num_threads, int num_runs,
                           uint64_t base_seed, uint64_t orbit_seed)
{
    if (!ValidateDim(dim, "DriveAB"))
        return 2;
    if (!ValidateHistoryDepth(history_depth, "DriveAB"))
        return 2;

    if (config::LOAD_TRAINED_WEIGHTS)
    {
        std::fprintf(stderr,
                     "[DriveAB] refused: LOAD_TRAINED_WEIGHTS is on. "
                     "A/B needs a fresh train per drive layout.\n");
        return 2;
    }

    if (config::SAVE_TRAINED_WEIGHTS)
    {
        std::fprintf(stderr,
                     "[DriveAB] NOTE: SAVE_TRAINED_WEIGHTS is on; stems include "
                     "DIM/M/num_inputs so each arm keeps its own file.\n");
        std::fflush(stderr);
    }

    ConfigSizeRestore dim_restore(config::DIM, dim);
    ConfigSizeRestore m_restore(config::HISTORY_DEPTH, history_depth);
    ConfigDriveRestore drive_restore(config::DRIVE_LAYOUT, config::DRIVE_LAYOUT);

    static const DriveLayout kArms[] = {
        DriveLayout::XyzXz,
        DriveLayout::Quadratic8,
    };

    std::printf("=== HypercubeESN: Lorenz / drive-layout A/B ===\n");
    std::printf("[DriveAB] DIM=%zu (N=%zu)  M=%zu  arms:",
                config::DIM, size_t{1} << config::DIM, history_depth);
    for (DriveLayout L : kArms)
        std::printf(" %s(n_in=%zu)", Lorenz::DriveLayoutName(L), Lorenz::NumDriveChannels(L));
    std::printf("\n");
    std::printf("[DriveAB] survey threads=%zu  runs=%d  (0 threads => HW)\n",
                num_threads, num_runs);
    std::printf("[DriveAB] matched seeds/protocol; train per arm (not load)\n");
    std::fflush(stdout);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    SurveySummary arm_a{};
    SurveySummary arm_b{};
    int first_err = 0;

    for (size_t i = 0; i < 2; ++i)
    {
        const DriveLayout L = kArms[i];
        config::DRIVE_LAYOUT = L;
        std::printf("\n########## DriveAB arm %zu/2: %s  (n_in=%zu)  M=%zu ##########\n",
                    i + 1, Lorenz::DriveLayoutName(L), Lorenz::NumDriveChannels(L),
                    history_depth);
        std::fflush(stdout);

        SurveySummary sum;
        const int rc = Campaign_SeedSurvey(dim, num_threads, num_runs, base_seed, orbit_seed,
                                           /*completion_beep=*/false, &sum);
        if (i == 0)
            arm_a = sum;
        else
            arm_b = sum;
        if (rc != 0 && first_err == 0)
            first_err = rc;
    }

    std::fflush(stdout);

    const double elapsed = std::chrono::duration<double>(clock::now() - t0).count();
    char time_buf[64];
    FormatWallTime(time_buf, sizeof time_buf, elapsed);

    {
        std::ostringstream roll;
        roll << "\n========================================================================\n"
             << "=== Drive A/B roll-up (fixed M; mean of trial-means; code-computed) ===\n"
             << "========================================================================\n";
        AppendCompactConfigLines(roll);

        char line[512];
        std::snprintf(line, sizeof line,
                      "DIM=%zu  N=%zu  M=%zu  protocol=%s  trials/arm=%zu  freeruns/trial=%d  theta=%.2f\n",
                      dim, size_t{1} << dim, history_depth,
                      Lorenz::ProtocolName(config::FREE_RUN_PROTOCOL),
                      arm_a.num_trials ? arm_a.num_trials : arm_b.num_trials,
                      num_runs, config::VPT_THRESHOLD);
        roll << line;

        std::snprintf(line, sizeof line,
                      "%-12s %5s %8s %8s %10s %8s %8s %8s %8s %8s %10s\n",
                      "drive", "n_in", "VPT_mn", "VPT_sd", "RMSE_mn", "RMSE_sd",
                      "duty_mn", "duty_sd", "VxD_mn", "VxD_sd", "trials_ok");
        roll << line;
        std::snprintf(line, sizeof line,
                      "%-12s %5s %8s %8s %10s %8s %8s %8s %8s %8s %10s\n",
                      "------------", "-----", "--------", "--------", "----------", "--------",
                      "--------", "--------", "--------", "--------", "----------");
        roll << line;

        auto row = [&](const SurveySummary& s) {
            if (!s.ok)
            {
                std::snprintf(line, sizeof line, "%-12s  (failed)\n",
                              Lorenz::DriveLayoutName(s.drive_layout));
                roll << line;
                return;
            }
            std::snprintf(line, sizeof line,
                          "%-12s %5zu %8.3f %8.3f %10.6f %8.6f %8.3f %8.3f %8.3f %8.3f %10zu\n",
                          Lorenz::DriveLayoutName(s.drive_layout), s.num_inputs,
                          s.mean_vpt, s.std_vpt,
                          s.mean_rmse, s.std_rmse,
                          s.mean_duty, s.std_duty,
                          s.mean_vpt_x_duty, s.std_vpt_x_duty,
                          s.n_trials_ok);
            roll << line;
        };
        row(arm_a);
        row(arm_b);

        if (arm_a.ok && arm_b.ok)
        {
            roll << "\nDeltas (Quadratic8 - XyzXz):\n";
            std::snprintf(line, sizeof line, "%-12s %10s %12s %10s %12s\n",
                          "", "dVPT", "dRMSE", "dDuty", "dVPT*duty");
            roll << line;
            std::snprintf(line, sizeof line, "%-12s %+10.3f %+12.6f %+10.3f %+12.3f\n",
                          "B - A",
                          arm_b.mean_vpt - arm_a.mean_vpt,
                          arm_b.mean_rmse - arm_a.mean_rmse,
                          arm_b.mean_duty - arm_a.mean_duty,
                          arm_b.mean_vpt_x_duty - arm_a.mean_vpt_x_duty);
            roll << line;

            roll << "\nCode picks among successful arms:\n";
            const SurveySummary& best =
                (arm_b.mean_vpt >= arm_a.mean_vpt) ? arm_b : arm_a;
            std::snprintf(line, sizeof line,
                          "  best mean VPT  : %s  VPT=%.3f +/- %.3f  (n_in=%zu)\n",
                          Lorenz::DriveLayoutName(best.drive_layout),
                          best.mean_vpt, best.std_vpt, best.num_inputs);
            roll << line;
            const SurveySummary& best_d =
                (arm_b.mean_duty >= arm_a.mean_duty) ? arm_b : arm_a;
            std::snprintf(line, sizeof line,
                          "  best mean duty : %s  duty=%.3f +/- %.3f  (n_in=%zu)\n",
                          Lorenz::DriveLayoutName(best_d.drive_layout),
                          best_d.mean_duty, best_d.std_duty, best_d.num_inputs);
            roll << line;
            const SurveySummary& best_vxd =
                (arm_b.mean_vpt_x_duty >= arm_a.mean_vpt_x_duty) ? arm_b : arm_a;
            std::snprintf(line, sizeof line,
                          "  best mean VPT*duty : %s  VPT*duty=%.3f +/- %.3f  (n_in=%zu)\n",
                          Lorenz::DriveLayoutName(best_vxd.drive_layout),
                          best_vxd.mean_vpt_x_duty, best_vxd.std_vpt_x_duty,
                          best_vxd.num_inputs);
            roll << line;
        }
        else if (!arm_a.ok && !arm_b.ok)
        {
            roll << "(no successful arms)\n";
        }

        std::snprintf(line, sizeof line,
                      "\n=== Drive A/B wall time: %s (restoring DIM=%zu M=%zu drive=%s) ===\n",
                      time_buf, dim_restore.saved, m_restore.saved,
                      Lorenz::DriveLayoutName(drive_restore.saved));
        roll << line;

        const std::string block = roll.str();
        std::fwrite(block.data(), 1, block.size(), stdout);
        std::fflush(stdout);
    }

    WriteDriveAbResultFiles(arm_a, arm_b, dim, history_depth, base_seed, orbit_seed,
                            num_threads, num_runs, elapsed);

    Beep(2500, 3000);
    return first_err;
}

// ---------------------------------------------------------------------------
// Campaign_SpectralRadiusAB
// ---------------------------------------------------------------------------
void WriteSrAbResultFiles(const SurveySummary& a, const SurveySummary& b,
                          float sr_a, float sr_b,
                          size_t dim, size_t history_depth,
                          uint64_t base_seed, uint64_t orbit_seed,
                          size_t num_threads, int num_runs,
                          double total_wall_s)
{
    const fs::path dir = EnsureResultsDir();
    if (!fs::exists(dir))
        return;

    const std::string ts = TimestampNow();
    const fs::path csv_path = dir / ("SrAB_" + ts + "_D" + std::to_string(dim) +
                                     "_M" + std::to_string(history_depth) + ".csv");
    const fs::path txt_path = dir / ("SrAB_" + ts + "_D" + std::to_string(dim) +
                                     "_M" + std::to_string(history_depth) + ".txt");

    auto write_row = [](std::ostream& o, float sr, const SurveySummary& s) {
        o << sr << ','
          << s.history_depth << ','
          << s.mean_vpt << ',' << s.std_vpt << ','
          << s.mean_rmse << ',' << s.std_rmse << ','
          << s.mean_duty << ',' << s.std_duty << ','
          << s.mean_vpt_x_duty << ',' << s.std_vpt_x_duty << ','
          << s.n_trials_ok << ',' << s.num_trials << ',' << s.num_runs << ','
          << s.wall_seconds << ','
          << Lorenz::ProtocolName(s.protocol) << ','
          << (s.ok ? 1 : 0) << '\n';
    };

    {
        std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
        if (!csv)
        {
            std::fprintf(stderr, "[results] failed to write %s\n", csv_path.string().c_str());
            return;
        }
        WriteMetadataBlock(csv, "SrAB", ts, base_seed, orbit_seed,
                           num_threads, num_runs, config::FREE_RUN_PROTOCOL, history_depth);
        csv << "# dim=" << dim << "  fixed_M=" << history_depth
            << "  sr_a=" << sr_a << "  sr_b=" << sr_b
            << "  total_wall_seconds=" << total_wall_s << "\n";
        csv << "SR,M,mean_vpt,std_vpt,mean_rmse,std_rmse,"
               "mean_duty,std_duty,mean_vpt_x_duty,std_vpt_x_duty,"
               "n_trials_ok,num_trials,num_runs,wall_seconds,protocol,ok\n";
        if (a.ok) write_row(csv, sr_a, a);
        if (b.ok) write_row(csv, sr_b, b);
        if (a.ok && b.ok)
        {
            csv << "# deltas (B - A): dVPT=" << (b.mean_vpt - a.mean_vpt)
                << "  dRMSE=" << (b.mean_rmse - a.mean_rmse)
                << "  dDuty=" << (b.mean_duty - a.mean_duty)
                << "  dVPT*duty=" << (b.mean_vpt_x_duty - a.mean_vpt_x_duty) << "\n";
        }
    }
    {
        std::ofstream txt(txt_path, std::ios::out | std::ios::trunc);
        if (!txt)
        {
            std::fprintf(stderr, "[results] failed to write %s\n", txt_path.string().c_str());
            return;
        }
        WriteMetadataBlock(txt, "SrAB", ts, base_seed, orbit_seed,
                           num_threads, num_runs, config::FREE_RUN_PROTOCOL, history_depth);
        txt << "\nSpectral-radius A/B (fixed dim/M; mean of trial-means)\n";
        txt << "DIM=" << dim << "  N=" << (size_t{1} << dim)
            << "  M=" << history_depth
            << "  sr_a=" << sr_a << "  sr_b=" << sr_b << "\n";
        AppendCompactConfigLines(txt);
        txt << "\narm,SR,VPT_mn,VPT_sd,RMSE_mn,RMSE_sd,duty_mn,duty_sd,"
               "VxD_mn,VxD_sd,trials_ok,wall_s\n";
        auto arm_line = [&](const char* tag, float sr, const SurveySummary& s) {
            if (!s.ok)
            {
                txt << tag << ',' << sr << ",(failed)\n";
                return;
            }
            txt << tag << ',' << sr << ','
                << s.mean_vpt << ',' << s.std_vpt << ','
                << s.mean_rmse << ',' << s.std_rmse << ','
                << s.mean_duty << ',' << s.std_duty << ','
                << s.mean_vpt_x_duty << ',' << s.std_vpt_x_duty << ','
                << s.n_trials_ok << ',' << s.wall_seconds << '\n';
        };
        arm_line("A", sr_a, a);
        arm_line("B", sr_b, b);
        if (a.ok && b.ok)
        {
            txt << "\nDeltas (B - A) = SR " << sr_b << " - " << sr_a << "\n";
            txt << "dVPT=" << (b.mean_vpt - a.mean_vpt)
                << "  dRMSE=" << (b.mean_rmse - a.mean_rmse)
                << "  dDuty=" << (b.mean_duty - a.mean_duty)
                << "  dVPT*duty=" << (b.mean_vpt_x_duty - a.mean_vpt_x_duty) << "\n";
            txt << "\nCode picks among successful arms:\n";
            const bool b_vpt = b.mean_vpt >= a.mean_vpt;
            txt << "  best mean VPT : SR=" << (b_vpt ? sr_b : sr_a)
                << "  VPT=" << (b_vpt ? b.mean_vpt : a.mean_vpt)
                << " +/- " << (b_vpt ? b.std_vpt : a.std_vpt) << "\n";
            const bool b_vxd = b.mean_vpt_x_duty >= a.mean_vpt_x_duty;
            txt << "  best mean VPT*duty : SR=" << (b_vxd ? sr_b : sr_a)
                << "  VPT*duty=" << (b_vxd ? b.mean_vpt_x_duty : a.mean_vpt_x_duty)
                << " +/- " << (b_vxd ? b.std_vpt_x_duty : a.std_vpt_x_duty) << "\n";
            txt << "  total_wall_seconds=" << total_wall_s << "\n";
        }
    }

    ReportWrote("results", csv_path);
    ReportWrote("results", txt_path);
}

int Campaign_SpectralRadiusAB(size_t dim, size_t history_depth,
                              float sr_a, float sr_b,
                              size_t num_threads, int num_runs,
                              uint64_t base_seed, uint64_t orbit_seed)
{
    if (!ValidateDim(dim, "SrAB"))
        return 2;
    if (!ValidateHistoryDepth(history_depth, "SrAB"))
        return 2;
    if (!ValidateSpectralRadius(sr_a, "SrAB") || !ValidateSpectralRadius(sr_b, "SrAB"))
        return 2;

    if (config::LOAD_TRAINED_WEIGHTS)
    {
        std::fprintf(stderr,
                     "[SrAB] refused: LOAD_TRAINED_WEIGHTS is on. "
                     "A/B needs a fresh train per spectral radius.\n");
        return 2;
    }

    if (config::SAVE_TRAINED_WEIGHTS)
    {
        std::fprintf(stderr,
                     "[SrAB] NOTE: SAVE_TRAINED_WEIGHTS is on; default stems do not "
                     "include SR -- arm B will overwrite arm A weights. Turn save off "
                     "or use distinct stems for durable A/B models.\n");
        std::fflush(stderr);
    }

    ConfigSizeRestore dim_restore(config::DIM, dim);
    ConfigSizeRestore m_restore(config::HISTORY_DEPTH, history_depth);
    ConfigFloatRestore sr_restore(config::SPECTRAL_RADIUS, config::SPECTRAL_RADIUS);

    std::printf("=== HypercubeESN: Lorenz / spectral-radius A/B ===\n");
    std::printf("[SrAB] DIM=%zu (N=%zu)  M=%zu  SR_A=%.4f  SR_B=%.4f\n",
                config::DIM, size_t{1} << config::DIM, history_depth,
                static_cast<double>(sr_a), static_cast<double>(sr_b));
    std::printf("[SrAB] survey threads=%zu  runs=%d  (0 threads => HW)\n",
                num_threads, num_runs);
    std::printf("[SrAB] matched seeds/protocol/drive; train per arm (not load)\n");
    std::fflush(stdout);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const float srs[2] = {sr_a, sr_b};
    SurveySummary arm_a{};
    SurveySummary arm_b{};
    int first_err = 0;

    for (size_t i = 0; i < 2; ++i)
    {
        config::SPECTRAL_RADIUS = srs[i];
        std::printf("\n########## SrAB arm %zu/2: SR=%.4f  M=%zu ##########\n",
                    i + 1, static_cast<double>(srs[i]), history_depth);
        std::fflush(stdout);

        SurveySummary sum;
        const int rc = Campaign_SeedSurvey(dim, num_threads, num_runs, base_seed, orbit_seed,
                                           /*completion_beep=*/false, &sum);
        if (i == 0)
            arm_a = sum;
        else
            arm_b = sum;
        if (rc != 0 && first_err == 0)
            first_err = rc;
    }

    std::fflush(stdout);

    const double elapsed = std::chrono::duration<double>(clock::now() - t0).count();
    char time_buf[64];
    FormatWallTime(time_buf, sizeof time_buf, elapsed);

    {
        std::ostringstream roll;
        roll << "\n========================================================================\n"
             << "=== Spectral-radius A/B roll-up (fixed M; mean of trial-means) ===\n"
             << "========================================================================\n";
        // Print both SRs explicitly; compact config shows the last arm's SR after loop.
        roll << "config: SR_A=" << sr_a << "  SR_B=" << sr_b
             << "  (last-arm SPECTRAL_RADIUS restored on exit)\n";
        AppendCompactConfigLines(roll);

        char line[512];
        std::snprintf(line, sizeof line,
                      "DIM=%zu  N=%zu  M=%zu  protocol=%s  trials/arm=%zu  freeruns/trial=%d  theta=%.2f\n",
                      dim, size_t{1} << dim, history_depth,
                      Lorenz::ProtocolName(config::FREE_RUN_PROTOCOL),
                      arm_a.num_trials ? arm_a.num_trials : arm_b.num_trials,
                      num_runs, config::VPT_THRESHOLD);
        roll << line;

        std::snprintf(line, sizeof line,
                      "%-8s %8s %8s %10s %8s %8s %8s %8s %8s %10s\n",
                      "SR", "VPT_mn", "VPT_sd", "RMSE_mn", "RMSE_sd",
                      "duty_mn", "duty_sd", "VxD_mn", "VxD_sd", "trials_ok");
        roll << line;
        std::snprintf(line, sizeof line,
                      "%-8s %8s %8s %10s %8s %8s %8s %8s %8s %10s\n",
                      "--------", "--------", "--------", "----------", "--------",
                      "--------", "--------", "--------", "--------", "----------");
        roll << line;

        auto row = [&](float sr, const SurveySummary& s) {
            if (!s.ok)
            {
                std::snprintf(line, sizeof line, "%-8.4f  (failed)\n",
                              static_cast<double>(sr));
                roll << line;
                return;
            }
            std::snprintf(line, sizeof line,
                          "%-8.4f %8.3f %8.3f %10.6f %8.6f %8.3f %8.3f %8.3f %8.3f %10zu\n",
                          static_cast<double>(sr),
                          s.mean_vpt, s.std_vpt,
                          s.mean_rmse, s.std_rmse,
                          s.mean_duty, s.std_duty,
                          s.mean_vpt_x_duty, s.std_vpt_x_duty,
                          s.n_trials_ok);
            roll << line;
        };
        row(sr_a, arm_a);
        row(sr_b, arm_b);

        if (arm_a.ok && arm_b.ok)
        {
            roll << "\nDeltas (B - A) = SR " << sr_b << " - " << sr_a << ":\n";
            std::snprintf(line, sizeof line, "%-12s %10s %12s %10s %12s\n",
                          "", "dVPT", "dRMSE", "dDuty", "dVPT*duty");
            roll << line;
            std::snprintf(line, sizeof line, "%-12s %+10.3f %+12.6f %+10.3f %+12.3f\n",
                          "B - A",
                          arm_b.mean_vpt - arm_a.mean_vpt,
                          arm_b.mean_rmse - arm_a.mean_rmse,
                          arm_b.mean_duty - arm_a.mean_duty,
                          arm_b.mean_vpt_x_duty - arm_a.mean_vpt_x_duty);
            roll << line;

            roll << "\nCode picks among successful arms:\n";
            const bool b_vpt = arm_b.mean_vpt >= arm_a.mean_vpt;
            std::snprintf(line, sizeof line,
                          "  best mean VPT  : SR=%.4f  VPT=%.3f +/- %.3f\n",
                          static_cast<double>(b_vpt ? sr_b : sr_a),
                          b_vpt ? arm_b.mean_vpt : arm_a.mean_vpt,
                          b_vpt ? arm_b.std_vpt : arm_a.std_vpt);
            roll << line;
            const bool b_duty = arm_b.mean_duty >= arm_a.mean_duty;
            std::snprintf(line, sizeof line,
                          "  best mean duty : SR=%.4f  duty=%.3f +/- %.3f\n",
                          static_cast<double>(b_duty ? sr_b : sr_a),
                          b_duty ? arm_b.mean_duty : arm_a.mean_duty,
                          b_duty ? arm_b.std_duty : arm_a.std_duty);
            roll << line;
            const bool b_vxd = arm_b.mean_vpt_x_duty >= arm_a.mean_vpt_x_duty;
            std::snprintf(line, sizeof line,
                          "  best mean VPT*duty : SR=%.4f  VPT*duty=%.3f +/- %.3f\n",
                          static_cast<double>(b_vxd ? sr_b : sr_a),
                          b_vxd ? arm_b.mean_vpt_x_duty : arm_a.mean_vpt_x_duty,
                          b_vxd ? arm_b.std_vpt_x_duty : arm_a.std_vpt_x_duty);
            roll << line;
        }
        else if (!arm_a.ok && !arm_b.ok)
        {
            roll << "(no successful arms)\n";
        }

        std::snprintf(line, sizeof line,
                      "\n=== Sr A/B wall time: %s (restoring DIM=%zu M=%zu SR=%.4f) ===\n",
                      time_buf, dim_restore.saved, m_restore.saved,
                      static_cast<double>(sr_restore.saved));
        roll << line;

        const std::string block = roll.str();
        std::fwrite(block.data(), 1, block.size(), stdout);
        std::fflush(stdout);
    }

    WriteSrAbResultFiles(arm_a, arm_b, sr_a, sr_b, dim, history_depth,
                         base_seed, orbit_seed, num_threads, num_runs, elapsed);

    Beep(2500, 3000);
    return first_err;
}

// ---------------------------------------------------------------------------
// Campaign_DriveGainAB
// ---------------------------------------------------------------------------
void WriteGainAbResultFiles(const SurveySummary& a, const SurveySummary& b,
                            const float* gains_a, const float* gains_b, size_t n_in,
                            size_t dim, size_t history_depth,
                            uint64_t base_seed, uint64_t orbit_seed,
                            size_t num_threads, int num_runs,
                            double total_wall_s)
{
    const fs::path dir = EnsureResultsDir();
    if (!fs::exists(dir))
        return;

    const std::string ts = TimestampNow();
    const fs::path csv_path = dir / ("GainAB_" + ts + "_D" + std::to_string(dim) +
                                     "_M" + std::to_string(history_depth) + ".csv");
    const fs::path txt_path = dir / ("GainAB_" + ts + "_D" + std::to_string(dim) +
                                     "_M" + std::to_string(history_depth) + ".txt");

    const std::string ga = FormatDriveGains(gains_a, n_in);
    const std::string gb = FormatDriveGains(gains_b, n_in);

    auto write_row = [&](std::ostream& o, const char* tag, const float* g,
                         const SurveySummary& s) {
        o << tag << ',' << FormatDriveGains(g, n_in) << ','
          << s.history_depth << ','
          << s.mean_vpt << ',' << s.std_vpt << ','
          << s.mean_rmse << ',' << s.std_rmse << ','
          << s.mean_duty << ',' << s.std_duty << ','
          << s.mean_vpt_x_duty << ',' << s.std_vpt_x_duty << ','
          << s.n_trials_ok << ',' << s.num_trials << ',' << s.num_runs << ','
          << s.wall_seconds << ','
          << Lorenz::ProtocolName(s.protocol) << ','
          << (s.ok ? 1 : 0) << '\n';
    };

    {
        std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
        if (!csv)
        {
            std::fprintf(stderr, "[results] failed to write %s\n", csv_path.string().c_str());
            return;
        }
        WriteMetadataBlock(csv, "GainAB", ts, base_seed, orbit_seed,
                           num_threads, num_runs, config::FREE_RUN_PROTOCOL, history_depth);
        csv << "# dim=" << dim << "  fixed_M=" << history_depth
            << "  gains_a=" << ga << "  gains_b=" << gb
            << "  total_wall_seconds=" << total_wall_s << "\n";
        csv << "arm,drive_ch,M,mean_vpt,std_vpt,mean_rmse,std_rmse,"
               "mean_duty,std_duty,mean_vpt_x_duty,std_vpt_x_duty,"
               "n_trials_ok,num_trials,num_runs,wall_seconds,protocol,ok\n";
        if (a.ok) write_row(csv, "A", gains_a, a);
        if (b.ok) write_row(csv, "B", gains_b, b);
        if (a.ok && b.ok)
        {
            csv << "# deltas (B - A): dVPT=" << (b.mean_vpt - a.mean_vpt)
                << "  dRMSE=" << (b.mean_rmse - a.mean_rmse)
                << "  dDuty=" << (b.mean_duty - a.mean_duty)
                << "  dVPT*duty=" << (b.mean_vpt_x_duty - a.mean_vpt_x_duty) << "\n";
        }
    }
    {
        std::ofstream txt(txt_path, std::ios::out | std::ios::trunc);
        if (!txt)
        {
            std::fprintf(stderr, "[results] failed to write %s\n", txt_path.string().c_str());
            return;
        }
        WriteMetadataBlock(txt, "GainAB", ts, base_seed, orbit_seed,
                           num_threads, num_runs, config::FREE_RUN_PROTOCOL, history_depth);
        txt << "\nDrive-gain A/B (fixed dim/M; mean of trial-means)\n";
        txt << "DIM=" << dim << "  N=" << (size_t{1} << dim)
            << "  M=" << history_depth
            << "  drive=" << Lorenz::DriveLayoutName(config::DRIVE_LAYOUT)
            << "  n_in=" << n_in << "\n";
        txt << "gains_a=" << ga << "\ngains_b=" << gb << "\n";
        AppendCompactConfigLines(txt);
        txt << "\narm,drive_ch,VPT_mn,VPT_sd,RMSE_mn,RMSE_sd,duty_mn,duty_sd,"
               "VxD_mn,VxD_sd,trials_ok,wall_s\n";
        auto arm_line = [&](const char* tag, const float* g, const SurveySummary& s) {
            if (!s.ok)
            {
                txt << tag << ',' << FormatDriveGains(g, n_in) << ",(failed)\n";
                return;
            }
            txt << tag << ',' << FormatDriveGains(g, n_in) << ','
                << s.mean_vpt << ',' << s.std_vpt << ','
                << s.mean_rmse << ',' << s.std_rmse << ','
                << s.mean_duty << ',' << s.std_duty << ','
                << s.mean_vpt_x_duty << ',' << s.std_vpt_x_duty << ','
                << s.n_trials_ok << ',' << s.wall_seconds << '\n';
        };
        arm_line("A", gains_a, a);
        arm_line("B", gains_b, b);
        if (a.ok && b.ok)
        {
            txt << "\nDeltas (B - A)\n";
            txt << "dVPT=" << (b.mean_vpt - a.mean_vpt)
                << "  dRMSE=" << (b.mean_rmse - a.mean_rmse)
                << "  dDuty=" << (b.mean_duty - a.mean_duty)
                << "  dVPT*duty=" << (b.mean_vpt_x_duty - a.mean_vpt_x_duty) << "\n";
            txt << "\nCode picks among successful arms:\n";
            const bool b_vpt = b.mean_vpt >= a.mean_vpt;
            txt << "  best mean VPT : " << (b_vpt ? gb : ga)
                << "  VPT=" << (b_vpt ? b.mean_vpt : a.mean_vpt)
                << " +/- " << (b_vpt ? b.std_vpt : a.std_vpt) << "\n";
            const bool b_vxd = b.mean_vpt_x_duty >= a.mean_vpt_x_duty;
            txt << "  best mean VPT*duty : " << (b_vxd ? gb : ga)
                << "  VPT*duty=" << (b_vxd ? b.mean_vpt_x_duty : a.mean_vpt_x_duty)
                << " +/- " << (b_vxd ? b.std_vpt_x_duty : a.std_vpt_x_duty) << "\n";
            txt << "  total_wall_seconds=" << total_wall_s << "\n";
        }
    }

    ReportWrote("results", csv_path);
    ReportWrote("results", txt_path);
}

int Campaign_DriveGainAB(size_t dim, size_t history_depth,
                         std::initializer_list<float> gains_a,
                         std::initializer_list<float> gains_b,
                         size_t num_threads, int num_runs,
                         uint64_t base_seed, uint64_t orbit_seed)
{
    if (!ValidateDim(dim, "GainAB"))
        return 2;
    if (!ValidateHistoryDepth(history_depth, "GainAB"))
        return 2;

    const size_t n_in = Lorenz::NumDriveChannels(config::DRIVE_LAYOUT);
    float ga[kMaxDriveChannels]{};
    float gb[kMaxDriveChannels]{};
    if (!ParseDriveGains(gains_a, ga, n_in, "GainAB") ||
        !ParseDriveGains(gains_b, gb, n_in, "GainAB"))
        return 2;

    if (config::LOAD_TRAINED_WEIGHTS)
    {
        std::fprintf(stderr,
                     "[GainAB] refused: LOAD_TRAINED_WEIGHTS is on. "
                     "A/B needs a fresh train per drive-gain vector.\n");
        return 2;
    }

    if (config::SAVE_TRAINED_WEIGHTS)
    {
        std::fprintf(stderr,
                     "[GainAB] NOTE: SAVE_TRAINED_WEIGHTS is on; default stems do not "
                     "include drive_ch -- arm B will overwrite arm A weights. Turn save "
                     "off or use distinct stems for durable A/B models.\n");
        std::fflush(stderr);
    }

    ConfigSizeRestore dim_restore(config::DIM, dim);
    ConfigSizeRestore m_restore(config::HISTORY_DEPTH, history_depth);
    ConfigDriveGainsRestore gains_restore;

    const std::string ga_s = FormatDriveGains(ga, n_in);
    const std::string gb_s = FormatDriveGains(gb, n_in);

    std::printf("=== HypercubeESN: Lorenz / drive-gain A/B ===\n");
    std::printf("[GainAB] DIM=%zu (N=%zu)  M=%zu  drive=%s  n_in=%zu\n",
                config::DIM, size_t{1} << config::DIM, history_depth,
                Lorenz::DriveLayoutName(config::DRIVE_LAYOUT), n_in);
    std::printf("[GainAB] gains_a=%s\n", ga_s.c_str());
    std::printf("[GainAB] gains_b=%s\n", gb_s.c_str());
    std::printf("[GainAB] survey threads=%zu  runs=%d  (0 threads => HW)\n",
                num_threads, num_runs);
    std::printf("[GainAB] matched seeds/protocol/SR; train per arm (not load)\n");
    std::fflush(stdout);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const float* arms[2] = {ga, gb};
    SurveySummary arm_a{};
    SurveySummary arm_b{};
    int first_err = 0;

    for (size_t i = 0; i < 2; ++i)
    {
        ApplyDriveGains(arms[i]);
        std::printf("\n########## GainAB arm %zu/2: drive_ch=%s  M=%zu ##########\n",
                    i + 1, FormatDriveGains(arms[i], n_in).c_str(), history_depth);
        std::fflush(stdout);

        SurveySummary sum;
        const int rc = Campaign_SeedSurvey(dim, num_threads, num_runs, base_seed, orbit_seed,
                                           /*completion_beep=*/false, &sum);
        if (i == 0)
            arm_a = sum;
        else
            arm_b = sum;
        if (rc != 0 && first_err == 0)
            first_err = rc;
    }

    std::fflush(stdout);

    const double elapsed = std::chrono::duration<double>(clock::now() - t0).count();
    char time_buf[64];
    FormatWallTime(time_buf, sizeof time_buf, elapsed);

    {
        std::ostringstream roll;
        roll << "\n========================================================================\n"
             << "=== Drive-gain A/B roll-up (fixed M; mean of trial-means) ===\n"
             << "========================================================================\n";
        roll << "config: gains_a=" << ga_s << "  gains_b=" << gb_s
             << "  (INPUT_SCALE_CH restored on exit)\n";
        AppendCompactConfigLines(roll);

        char line[640];
        std::snprintf(line, sizeof line,
                      "DIM=%zu  N=%zu  M=%zu  protocol=%s  trials/arm=%zu  freeruns/trial=%d  theta=%.2f\n",
                      dim, size_t{1} << dim, history_depth,
                      Lorenz::ProtocolName(config::FREE_RUN_PROTOCOL),
                      arm_a.num_trials ? arm_a.num_trials : arm_b.num_trials,
                      num_runs, config::VPT_THRESHOLD);
        roll << line;

        std::snprintf(line, sizeof line,
                      "%-4s %-28s %8s %8s %10s %8s %8s %8s %8s %8s %10s\n",
                      "arm", "drive_ch", "VPT_mn", "VPT_sd", "RMSE_mn", "RMSE_sd",
                      "duty_mn", "duty_sd", "VxD_mn", "VxD_sd", "trials_ok");
        roll << line;
        std::snprintf(line, sizeof line,
                      "%-4s %-28s %8s %8s %10s %8s %8s %8s %8s %8s %10s\n",
                      "----", "----------------------------", "--------", "--------",
                      "----------", "--------", "--------", "--------", "--------",
                      "--------", "----------");
        roll << line;

        auto row = [&](const char* tag, const float* g, const SurveySummary& s) {
            const std::string gs = FormatDriveGains(g, n_in);
            if (!s.ok)
            {
                std::snprintf(line, sizeof line, "%-4s %-28s  (failed)\n",
                              tag, gs.c_str());
                roll << line;
                return;
            }
            std::snprintf(line, sizeof line,
                          "%-4s %-28s %8.3f %8.3f %10.6f %8.6f %8.3f %8.3f %8.3f %8.3f %10zu\n",
                          tag, gs.c_str(),
                          s.mean_vpt, s.std_vpt,
                          s.mean_rmse, s.std_rmse,
                          s.mean_duty, s.std_duty,
                          s.mean_vpt_x_duty, s.std_vpt_x_duty,
                          s.n_trials_ok);
            roll << line;
        };
        row("A", ga, arm_a);
        row("B", gb, arm_b);

        if (arm_a.ok && arm_b.ok)
        {
            roll << "\nDeltas (B - A):\n";
            std::snprintf(line, sizeof line, "%-12s %10s %12s %10s %12s\n",
                          "", "dVPT", "dRMSE", "dDuty", "dVPT*duty");
            roll << line;
            std::snprintf(line, sizeof line, "%-12s %+10.3f %+12.6f %+10.3f %+12.3f\n",
                          "B - A",
                          arm_b.mean_vpt - arm_a.mean_vpt,
                          arm_b.mean_rmse - arm_a.mean_rmse,
                          arm_b.mean_duty - arm_a.mean_duty,
                          arm_b.mean_vpt_x_duty - arm_a.mean_vpt_x_duty);
            roll << line;

            roll << "\nCode picks among successful arms:\n";
            const bool b_vpt = arm_b.mean_vpt >= arm_a.mean_vpt;
            std::snprintf(line, sizeof line,
                          "  best mean VPT  : %s  VPT=%.3f +/- %.3f\n",
                          (b_vpt ? gb_s : ga_s).c_str(),
                          b_vpt ? arm_b.mean_vpt : arm_a.mean_vpt,
                          b_vpt ? arm_b.std_vpt : arm_a.std_vpt);
            roll << line;
            const bool b_duty = arm_b.mean_duty >= arm_a.mean_duty;
            std::snprintf(line, sizeof line,
                          "  best mean duty : %s  duty=%.3f +/- %.3f\n",
                          (b_duty ? gb_s : ga_s).c_str(),
                          b_duty ? arm_b.mean_duty : arm_a.mean_duty,
                          b_duty ? arm_b.std_duty : arm_a.std_duty);
            roll << line;
            const bool b_vxd = arm_b.mean_vpt_x_duty >= arm_a.mean_vpt_x_duty;
            std::snprintf(line, sizeof line,
                          "  best mean VPT*duty : %s  VPT*duty=%.3f +/- %.3f\n",
                          (b_vxd ? gb_s : ga_s).c_str(),
                          b_vxd ? arm_b.mean_vpt_x_duty : arm_a.mean_vpt_x_duty,
                          b_vxd ? arm_b.std_vpt_x_duty : arm_a.std_vpt_x_duty);
            roll << line;
        }
        else if (!arm_a.ok && !arm_b.ok)
        {
            roll << "(no successful arms)\n";
        }

        std::snprintf(line, sizeof line,
                      "\n=== Gain A/B wall time: %s (restoring DIM=%zu M=%zu drive_ch) ===\n",
                      time_buf, dim_restore.saved, m_restore.saved);
        roll << line;

        const std::string block = roll.str();
        std::fwrite(block.data(), 1, block.size(), stdout);
        std::fflush(stdout);
    }

    WriteGainAbResultFiles(arm_a, arm_b, ga, gb, n_in, dim, history_depth,
                           base_seed, orbit_seed, num_threads, num_runs, elapsed);

    Beep(2500, 3000);
    return first_err;
}
