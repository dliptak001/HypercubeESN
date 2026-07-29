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

fs::path EnsureResultsDir()
{
    const fs::path dir = config::RESULTS_DIR;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec)
        std::fprintf(stderr, "[results] create_directories(%s) failed: %s\n",
                     dir.string().c_str(), ec.message().c_str());
    return dir;
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
      << "  num_inputs=" << Lorenz::NumDriveChannels(config::DRIVE_LAYOUT) << "\n"
      << "# metrics=mean_of_trial_means (sample std across trials)\n";
}

const char* SurveyCsvHeader()
{
    return "M,mean_vpt,std_vpt,mean_rmse,std_rmse,mean_duty,std_duty,"
           "mean_n_relock,std_n_relock,n_trials_ok,num_trials,num_runs,wall_seconds,"
           "protocol,ok\n";
}

void WriteSurveyCsvRow(std::ostream& o, const SurveySummary& s)
{
    o << s.history_depth << ','
      << s.mean_vpt << ',' << s.std_vpt << ','
      << s.mean_rmse << ',' << s.std_rmse << ','
      << s.mean_duty << ',' << s.std_duty << ','
      << s.mean_n_relock << ',' << s.std_n_relock << ','
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
        txt << "  VPT   mean=" << s.mean_vpt << "  std=" << s.std_vpt << "\n";
        txt << "  RMSE  mean=" << s.mean_rmse << "  std=" << s.std_rmse << "\n";
        txt << "  duty  mean=" << s.mean_duty << "  std=" << s.std_duty << "\n";
        txt << "  relock mean=" << s.mean_n_relock << "  std=" << s.std_n_relock << "\n";
        txt << "  wall_seconds=" << s.wall_seconds << "  ok=" << (s.ok ? 1 : 0) << "\n";
    }

    // stdout (not stderr): CLion/Windows consoles merge streams asynchronously and will
    // splice stderr mid-line into printf tables if we log "wrote" on a different stream.
    std::printf("[results] wrote %s\n", csv_path.string().c_str());
    std::printf("[results] wrote %s\n", txt_path.string().c_str());
    std::fflush(stdout);
}

// M-sweep roll-up: one CSV (metadata + all M rows) + matching TXT.
void WriteMsweepResultFiles(const std::vector<SurveySummary>& rows,
                            uint64_t base_seed, uint64_t orbit_seed,
                            size_t num_threads, int num_runs,
                            double total_wall_s,
                            size_t i_best_vpt, size_t i_best_duty)
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
                << "  best_mean_duty_M=" << rows[i_best_duty].history_depth << "\n";
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
            txt << "M,VPT_mn,VPT_sd,RMSE_mn,RMSE_sd,duty_mn,duty_sd,relk_mn,relk_sd,trials_ok,wall_s\n";
            for (const auto& r : rows)
            {
                txt << r.history_depth << ','
                    << r.mean_vpt << ',' << r.std_vpt << ','
                    << r.mean_rmse << ',' << r.std_rmse << ','
                    << r.mean_duty << ',' << r.std_duty << ','
                    << r.mean_n_relock << ',' << r.std_n_relock << ','
                    << r.n_trials_ok << ',' << r.wall_seconds << '\n';
            }
            txt << "\nDeltas vs first row (M=" << rows.front().history_depth << ")\n";
            txt << "M,dVPT,dRMSE,dDuty,dRelock\n";
            const auto& b = rows.front();
            for (const auto& r : rows)
            {
                txt << r.history_depth << ','
                    << (r.mean_vpt - b.mean_vpt) << ','
                    << (r.mean_rmse - b.mean_rmse) << ','
                    << (r.mean_duty - b.mean_duty) << ','
                    << (r.mean_n_relock - b.mean_n_relock) << '\n';
            }
            txt << "\nCode picks:\n";
            txt << "  best mean VPT  : M=" << rows[i_best_vpt].history_depth
                << "  VPT=" << rows[i_best_vpt].mean_vpt
                << " +/- " << rows[i_best_vpt].std_vpt << "\n";
            txt << "  best mean duty : M=" << rows[i_best_duty].history_depth
                << "  duty=" << rows[i_best_duty].mean_duty
                << " +/- " << rows[i_best_duty].std_duty << "\n";
            txt << "  total_wall_seconds=" << total_wall_s << "\n";
        }
    }

    // Same stream as the roll-up table (see WriteSurveyResultFiles).
    std::printf("[results] wrote %s\n", csv_path.string().c_str());
    std::printf("[results] wrote %s\n", txt_path.string().c_str());
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

/// One trial: report text + per-trial free-run means (valid free-runs only).
struct TrialBundle
{
    std::string report;
    bool ok = false; ///< at least one valid free-run
    double mean_vpt = 0;
    double mean_rmse = 0;
    double mean_duty = 0;
    double mean_n_relock = 0;
    size_t n_valid = 0;
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

    std::vector<double> vpt_lts, rmses, duties, relocks, unlocks, mean_locks;
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
        relocks.push_back(static_cast<double>(r.n_relock));
        unlocks.push_back(static_cast<double>(r.n_unlock));
        mean_locks.push_back(r.mean_locked_sojourn);
        if (!r.crossed) ++censored;
    }

    // Crunched trial-level means (valid free-runs only) -- used by survey / M-sweep.
    tb.n_valid = vpt_lts.size();
    tb.ok = tb.n_valid > 0;
    if (tb.ok)
    {
        tb.mean_vpt = MeanOf(vpt_lts);
        tb.mean_rmse = MeanOf(rmses);
        tb.mean_duty = MeanOf(duties);
        tb.mean_n_relock = MeanOf(relocks);
    }

    auto report = [&](const char* label, std::vector<double> v, int prec)
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
        std::snprintf(buf, sizeof buf, "  %-20s n=%2zu  min=%.*f  max=%.*f  mean=%.*f  median=%.*f  std=%.*f\n",
                      label, n, prec, v.front(), prec, v.back(), prec, mean, prec, median, prec, sd);
        emit(buf);
    };

    std::snprintf(buf, sizeof buf, "\n=== Free-run stats (%d runs) ===\n", num_runs);
    emit(buf);
    report("VPT (lt)", vpt_lts, 2);
    report("free-run RMSE", rmses, 6);
    report("duty (<=theta)", duties, 3);
    report("n_relock", relocks, 1);
    report("n_unlock", unlocks, 1);
    report("meanLock (steps)", mean_locks, 1);
    std::snprintf(buf, sizeof buf,
                  "  note: duty/relock/unlock/meanLock use theta=VPT_THRESHOLD=%.2f "
                  "(re-lock proxies; VPT is first upcrossing only)\n",
                  config::VPT_THRESHOLD);
    emit(buf);
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
                      censored, vpt_lts.size(), config::VPT_THRESHOLD);
        emit(buf);
    }
    if (invalid)
    {
        std::snprintf(buf, sizeof buf, "  note: %zu run(s) scored 0 steps and are excluded from the stats\n", invalid);
        emit(buf);
    }

    // Leaderboards: VPT and duty only (no RMSE top list -- aggregates cover RMSE).
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

    std::cout << "=== HypercubeESN: Lorenz / survey ===\n";

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
    std::vector<double> trial_vpt, trial_rmse, trial_duty, trial_relock;
    trial_vpt.reserve(num_threads);
    trial_rmse.reserve(num_threads);
    trial_duty.reserve(num_threads);
    trial_relock.reserve(num_threads);
    for (const auto& tr : trials)
    {
        if (!tr.ok)
            continue;
        trial_vpt.push_back(tr.mean_vpt);
        trial_rmse.push_back(tr.mean_rmse);
        trial_duty.push_back(tr.mean_duty);
        trial_relock.push_back(tr.mean_n_relock);
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
    MeanStd(trial_relock, sum.mean_n_relock, sum.std_n_relock);
    sum.wall_seconds = std::chrono::duration<double>(clock::now() - survey_t0).count();
    sum.ok = sum.n_trials_ok > 0;

    char time_buf[64];
    FormatWallTime(time_buf, sizeof time_buf, sum.wall_seconds);
    std::printf("\n=== Survey wall time: %s (%zu trial(s), %zu ok) ===\n",
                time_buf, num_threads, sum.n_trials_ok);
    if (sum.ok)
    {
        std::printf("[survey] aggregate (mean of %zu trial-means):  "
                    "VPT mean=%.3f std=%.3f  RMSE mean=%.6f std=%.6f  "
                    "duty mean=%.3f std=%.3f  n_relock mean=%.1f std=%.1f  M=%zu\n",
                    sum.n_trials_ok,
                    sum.mean_vpt, sum.std_vpt,
                    sum.mean_rmse, sum.std_rmse,
                    sum.mean_duty, sum.std_duty,
                    sum.mean_n_relock, sum.std_n_relock,
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
int Campaign_Trace(size_t dim, uint64_t esn_seed, int max_freeruns, uint64_t target_orbit,
                   uint64_t orbit_seed)
{
    if (!ValidateDim(dim, "trace"))
        return 2;
    ConfigSizeRestore dim_restore(config::DIM, dim);

    if (max_freeruns < 1)
        max_freeruns = 1;

    std::printf("=== HypercubeESN: Lorenz / trace ===\n");
    std::printf("[trace] protocol=%s  DIM=%zu (N=%zu)  esn_seed=%llu  max_freeruns=%d  "
                "target_orbit=%llu  history_depth(M)=%zu\n",
                Lorenz::ProtocolName(config::FREE_RUN_PROTOCOL),
                config::DIM, size_t{1} << config::DIM,
                static_cast<unsigned long long>(esn_seed), max_freeruns,
                static_cast<unsigned long long>(target_orbit), config::HISTORY_DEPTH);
    std::printf("[trace] train %zu epochs then free-run; CSV under examples/Lorenz/traces/\n",
                config::EPOCHS);
    std::fflush(stdout);

    config::ENABLE_PRINTF = true;
    Lorenz lorenz(esn_seed, orbit_seed);
    std::cout << lorenz.ReadoutArchSummary();
    if (config::LOAD_TRAINED_WEIGHTS)
        lorenz.LoadTrainedWeights();
    else
        lorenz.Train();
    config::ENABLE_PRINTF = false;
    const FreeRunProtocol protocol = lorenz.EffectiveFreeRunProtocol();
    const size_t n_train_orbits = lorenz.NumTrainOrbits();
    if ((protocol == FreeRunProtocol::TrainInSample ||
         protocol == FreeRunProtocol::TrainHoldout) &&
        n_train_orbits > 0 && static_cast<size_t>(max_freeruns) > n_train_orbits)
    {
        std::fprintf(stderr,
                     "[trace] WARN: max_freeruns=%d > %zu train orbits (protocol=%s); "
                     "extras reuse train ICs (modulo)\n",
                     max_freeruns, n_train_orbits, Lorenz::ProtocolName(protocol));
        std::fflush(stderr);
    }

    namespace fs = std::filesystem;
    const fs::path trace_dir = fs::path("examples") / "Lorenz" / "traces";
    std::error_code ec;
    fs::create_directories(trace_dir, ec);
    if (ec)
        std::fprintf(stderr, "[trace] create_directories(%s): %s\n",
                     trace_dir.string().c_str(), ec.message().c_str());

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    int dumped = 0;
    for (int i = 0; i < max_freeruns; ++i)
    {
        const fs::path tmp_path = trace_dir / ("_tmp_" + std::to_string(esn_seed) + ".csv");
        FreeRunResult r = lorenz.FreeRun(false, tmp_path.string().c_str(), 0, protocol);
        if (!r.valid)
        {
            std::printf("[trace] free-run %d invalid - stop\n", i);
            break;
        }
        const bool want = (target_orbit == 0) || (r.orbit_seed == target_orbit);
        std::printf("[trace] freerun %d/%d  orbit=%llu  VPT=%.2f lt  duty=%.3f  "
                    "relock=%zu unlock=%zu meanLock=%.1f  %s\n",
                    i + 1, max_freeruns,
                    static_cast<unsigned long long>(r.orbit_seed),
                    r.vpt_lt, r.duty, r.n_relock, r.n_unlock, r.mean_locked_sojourn,
                    want ? "DUMP" : "skip");
        std::fflush(stdout);

        if (want)
        {
            const fs::path out_path = trace_dir /
                ("seed" + std::to_string(esn_seed) + "_orbit" +
                 std::to_string(r.orbit_seed) + ".csv");
            fs::remove(out_path, ec);
            fs::rename(tmp_path, out_path, ec);
            if (ec)
            {
                std::printf("[trace] rename failed (%s) - CSV left at %s\n",
                            ec.message().c_str(), tmp_path.string().c_str());
            }
            else
                std::printf("[trace] wrote %s\n", out_path.string().c_str());
            std::printf("%s", r.row.c_str());
            ++dumped;
            if (target_orbit != 0)
                break;
        }
        else
        {
            fs::remove(tmp_path, ec);
        }
    }

    const double elapsed =
        std::chrono::duration<double>(clock::now() - t0).count();
    char time_buf[64];
    FormatWallTime(time_buf, sizeof time_buf, elapsed);
    std::printf("[trace] done - %d CSV file(s)  wall time: %s\n", dumped, time_buf);
    return dumped > 0 ? 0 : 1;
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
            char line[512];
            std::snprintf(line, sizeof line,
                          "protocol=%s  trials/M=%zu  freeruns/trial=%d  theta=%.2f\n",
                          Lorenz::ProtocolName(rows.front().protocol),
                          rows.front().num_trials, rows.front().num_runs,
                          config::VPT_THRESHOLD);
            roll << line;

            std::snprintf(line, sizeof line,
                          "%-6s %8s %8s %10s %8s %8s %10s %8s %8s %10s %10s\n",
                          "M", "VPT_mn", "VPT_sd", "RMSE_mn", "RMSE_sd", "duty_mn", "duty_sd",
                          "relk_mn", "relk_sd", "trials_ok", "wall_s");
            roll << line;
            std::snprintf(line, sizeof line,
                          "%-6s %8s %8s %10s %8s %8s %10s %8s %8s %10s %10s\n",
                          "------", "--------", "--------", "----------", "--------", "--------",
                          "----------", "--------", "--------", "----------", "----------");
            roll << line;

            for (size_t i = 0; i < rows.size(); ++i)
            {
                const auto& r = rows[i];
                std::snprintf(line, sizeof line,
                              "%-6zu %8.3f %8.3f %10.6f %8.6f %8.3f %10.3f %8.1f %8.1f %10zu %10.1f\n",
                              r.history_depth,
                              r.mean_vpt, r.std_vpt,
                              r.mean_rmse, r.std_rmse,
                              r.mean_duty, r.std_duty,
                              r.mean_n_relock, r.std_n_relock,
                              r.n_trials_ok, r.wall_seconds);
                roll << line;
                if (r.mean_vpt > rows[i_best_vpt].mean_vpt)
                    i_best_vpt = i;
                if (r.mean_duty > rows[i_best_duty].mean_duty)
                    i_best_duty = i;
            }

            // Delta vs first successful row (baseline = first M that ran)
            std::snprintf(line, sizeof line, "\nDeltas vs first row (M=%zu):\n",
                          rows.front().history_depth);
            roll << line;
            std::snprintf(line, sizeof line, "%-6s %10s %12s %10s %12s\n",
                          "M", "dVPT", "dRMSE", "dDuty", "dRelock");
            roll << line;
            const auto& b = rows.front();
            for (const auto& r : rows)
            {
                std::snprintf(line, sizeof line,
                              "%-6zu %+10.3f %+12.6f %+10.3f %+12.1f\n",
                              r.history_depth,
                              r.mean_vpt - b.mean_vpt,
                              r.mean_rmse - b.mean_rmse,
                              r.mean_duty - b.mean_duty,
                              r.mean_n_relock - b.mean_n_relock);
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
                               i_best_vpt, i_best_duty);
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
          << s.mean_n_relock << ',' << s.std_n_relock << ','
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
               "mean_duty,std_duty,mean_n_relock,std_n_relock,"
               "n_trials_ok,num_trials,num_runs,wall_seconds,protocol,ok\n";
        if (a.ok) write_row(csv, a);
        if (b.ok) write_row(csv, b);
        if (a.ok && b.ok)
        {
            csv << "# deltas (Quadratic8 - XyzXz): dVPT=" << (b.mean_vpt - a.mean_vpt)
                << "  dRMSE=" << (b.mean_rmse - a.mean_rmse)
                << "  dDuty=" << (b.mean_duty - a.mean_duty)
                << "  dRelock=" << (b.mean_n_relock - a.mean_n_relock) << "\n";
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
               "relk_mn,relk_sd,trials_ok,wall_s\n";
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
                << s.mean_n_relock << ',' << s.std_n_relock << ','
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
                << "  dRelock=" << (b.mean_n_relock - a.mean_n_relock) << "\n";
            txt << "\nCode pick (max mean VPT among successful arms):\n";
            const SurveySummary& best = (b.mean_vpt >= a.mean_vpt) ? b : a;
            txt << "  " << Lorenz::DriveLayoutName(best.drive_layout)
                << "  VPT=" << best.mean_vpt << " +/- " << best.std_vpt << "\n";
            txt << "  total_wall_seconds=" << total_wall_s << "\n";
        }
    }

    std::printf("[results] wrote %s\n", csv_path.string().c_str());
    std::printf("[results] wrote %s\n", txt_path.string().c_str());
    std::fflush(stdout);
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
                      "%-12s %5s %8s %8s %10s %8s %8s %10s %8s %8s %10s %10s\n",
                      "drive", "n_in", "VPT_mn", "VPT_sd", "RMSE_mn", "RMSE_sd",
                      "duty_mn", "duty_sd", "relk_mn", "relk_sd", "trials_ok", "wall_s");
        roll << line;
        std::snprintf(line, sizeof line,
                      "%-12s %5s %8s %8s %10s %8s %8s %10s %8s %8s %10s %10s\n",
                      "------------", "-----", "--------", "--------", "----------", "--------",
                      "--------", "----------", "--------", "--------", "----------", "----------");
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
                          "%-12s %5zu %8.3f %8.3f %10.6f %8.6f %8.3f %10.3f %8.1f %8.1f %10zu %10.1f\n",
                          Lorenz::DriveLayoutName(s.drive_layout), s.num_inputs,
                          s.mean_vpt, s.std_vpt,
                          s.mean_rmse, s.std_rmse,
                          s.mean_duty, s.std_duty,
                          s.mean_n_relock, s.std_n_relock,
                          s.n_trials_ok, s.wall_seconds);
            roll << line;
        };
        row(arm_a);
        row(arm_b);

        if (arm_a.ok && arm_b.ok)
        {
            roll << "\nDeltas (Quadratic8 - XyzXz):\n";
            std::snprintf(line, sizeof line, "%-12s %10s %12s %10s %12s\n",
                          "", "dVPT", "dRMSE", "dDuty", "dRelock");
            roll << line;
            std::snprintf(line, sizeof line, "%-12s %+10.3f %+12.6f %+10.3f %+12.1f\n",
                          "B - A",
                          arm_b.mean_vpt - arm_a.mean_vpt,
                          arm_b.mean_rmse - arm_a.mean_rmse,
                          arm_b.mean_duty - arm_a.mean_duty,
                          arm_b.mean_n_relock - arm_a.mean_n_relock);
            roll << line;

            roll << "\nCode pick (max mean VPT among successful arms):\n";
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
