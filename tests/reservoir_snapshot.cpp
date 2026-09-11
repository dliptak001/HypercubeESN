/// @file reservoir_snapshot.cpp
/// @brief CTest target: reservoir snapshot/restore fidelity (+ Create(GetConfig)).

#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "Reservoir.h"

namespace
{
    /// Deterministic per-step drive values for input + external-feedback channels.
    struct DriveSeries
    {
        std::vector<float> inputs; // steps * num_inputs
        std::vector<float> ext_feedback; // steps * num_external_feedback_channels
    };

    DriveSeries MakeDrive(size_t steps, size_t num_inputs, size_t num_ext_fb, uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        DriveSeries d;
        d.inputs.resize(steps * num_inputs);
        for (float& v : d.inputs) v = dist(rng);
        d.ext_feedback.resize(steps * num_ext_fb);
        for (float& v : d.ext_feedback) v = dist(rng);
        return d;
    }

    void Drive(Reservoir& r, const DriveSeries& d, size_t t0, size_t steps,
               size_t num_inputs, size_t num_ext_fb, std::vector<float>* trace)
    {
        const size_t n = r.Size();
        for (size_t s = 0; s < steps; ++s)
        {
            for (size_t ch = 0; ch < num_inputs; ++ch)
                r.InjectInput(ch, d.inputs[(t0 + s) * num_inputs + ch]);
            for (size_t ch = 0; ch < num_ext_fb; ++ch)
                r.InjectExternalFeedback(ch, d.ext_feedback[(t0 + s) * num_ext_fb + ch]);
            r.Step();
            if (trace)
                trace->insert(trace->end(), r.Outputs(), r.Outputs() + n);
        }
    }

    bool BitIdentical(const std::vector<float>& a, const std::vector<float>& b)
    {
        return a.size() == b.size() &&
               std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    }

    int TestConfig(const char* label, const ReservoirConfig& cfg,
                   size_t warm_steps, size_t replay_steps)
    {
        int failures = 0;
        auto r = Reservoir::Create(cfg);
        const size_t num_ext_fb = cfg.num_external_feedback_channels;
        const DriveSeries d =
            MakeDrive(warm_steps + replay_steps, cfg.num_inputs, num_ext_fb,
                      /*seed=*/0xFEEDBAC + cfg.dim);

        Drive(*r, d, 0, warm_steps, cfg.num_inputs, num_ext_fb, nullptr);
        const Reservoir::Snapshot snap = r->TakeSnapshot();

        std::vector<float> trace_a;
        Drive(*r, d, warm_steps, replay_steps, cfg.num_inputs, num_ext_fb, &trace_a);

        r->RestoreSnapshot(snap);
        std::vector<float> trace_b;
        Drive(*r, d, warm_steps, replay_steps, cfg.num_inputs, num_ext_fb, &trace_b);

        if (!BitIdentical(trace_a, trace_b))
        {
            std::printf("  [%s] FAIL: restore+replay trajectory differs from original\n", label);
            ++failures;
        }

        r->RestoreSnapshot(snap);
        const Reservoir::Snapshot snap2 = r->TakeSnapshot();
        if (!BitIdentical(snap.state, snap2.state) || !BitIdentical(snap.history, snap2.history))
        {
            std::printf("  [%s] FAIL: Take->Restore->Take is not the identity\n", label);
            ++failures;
        }

        r->RestoreSnapshot(snap);
        r->InjectInput(0, 123.456f);
        if (num_ext_fb > 0) r->InjectExternalFeedback(0, -77.7f);
        r->RestoreSnapshot(snap);
        std::vector<float> trace_c;
        Drive(*r, d, warm_steps, replay_steps, cfg.num_inputs, num_ext_fb, &trace_c);
        if (!BitIdentical(trace_a, trace_c))
        {
            std::printf("  [%s] FAIL: staged injections leaked through RestoreSnapshot\n", label);
            ++failures;
        }

        bool threw = false;
        try
        {
            Reservoir::Snapshot bad = snap;
            bad.history.pop_back();
            r->RestoreSnapshot(bad);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }
        if (!threw)
        {
            std::printf("  [%s] FAIL: mismatched snapshot did not throw\n", label);
            ++failures;
        }

        if (failures == 0)
            std::printf("  [%s] PASS (%zu warm + %zu replay steps, N=%zu, M=%zu, ext_fb=%zu)\n",
                        label, warm_steps, replay_steps, r->Size(), cfg.history_depth, num_ext_fb);
        return failures;
    }

    /// Create(GetConfig()) must rebuild matching weights: same drive → bit-identical trajectory.
    int TestCreateGetConfigIdentity()
    {
        int failures = 0;
        constexpr size_t kSteps = 48;

        ReservoirConfig cfg;
        cfg.dim = 6;
        cfg.history_depth = 4;
        cfg.seed = 4242;
        cfg.num_inputs = 2;
        cfg.leak_rate = 0.85f;
        cfg.num_external_feedback_channels = 2;
        cfg.external_feedback_scaling = 0.3f;
        cfg.verbose = false;

        auto a = Reservoir::Create(cfg);
        auto b = Reservoir::Create(a->GetConfig());
        const size_t num_ext_fb = cfg.num_external_feedback_channels;
        const DriveSeries d = MakeDrive(kSteps, cfg.num_inputs, num_ext_fb, 0xA11CE);

        std::vector<float> tr_a, tr_b;
        Drive(*a, d, 0, kSteps, cfg.num_inputs, num_ext_fb, &tr_a);
        Drive(*b, d, 0, kSteps, cfg.num_inputs, num_ext_fb, &tr_b);

        if (!BitIdentical(tr_a, tr_b))
        {
            std::printf("  [Create(GetConfig)] FAIL: rebuild trajectory differs from original\n");
            ++failures;
        }
        else
            std::printf("  [Create(GetConfig)] PASS (bit-identical, %zu steps, N=%zu, M=%zu)\n",
                        kSteps, a->Size(), cfg.history_depth);
        return failures;
    }
} // namespace

int main()
{
    std::printf("=== Reservoir snapshot/restore fidelity ===\n");
    int failures = 0;

    ReservoirConfig minimal;
    minimal.dim = 5;
    minimal.history_depth = 1;
    minimal.verbose = false;
    failures += TestConfig("dim5 M1 open-loop", minimal, 13, 40);

    ReservoirConfig deep;
    deep.dim = 8;
    deep.history_depth = 7;
    deep.leak_rate = 0.8f;
    deep.num_inputs = 2;
    deep.verbose = false;
    failures += TestConfig("dim8 M7 leaky 2-in", deep, 37, 50);

    ReservoirConfig fb = deep;
    fb.num_external_feedback_channels = 3;
    fb.external_feedback_scaling = 0.4f;
    failures += TestConfig("dim8 M7 +ext-feedback", fb, 37, 50);

    std::printf("=== Create(GetConfig) identity ===\n");
    failures += TestCreateGetConfigIdentity();

    if (failures == 0)
    {
        std::printf("All checks passed.\n");
        return 0;
    }
    std::printf("%d check(s) FAILED.\n", failures);
    return 1;
}
