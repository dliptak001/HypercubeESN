/// @file main.cpp
/// @brief Reservoir snapshot/restore fidelity diagnostics.
///
/// Snapshot/restore fidelity: snapshot -> drive N steps -> restore -> replay
/// the same N inputs must reproduce the identical trajectory bit-for-bit. The
/// restore is a branch-point primitive, so equality here is exact (memcmp), not
/// approximate. Exercised across open-loop, leaky multi-input, and
/// feedback-driven configs (the feedback config drives the reservoir's
/// dedicated feedback port directly via InjectFeedback).

#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "Reservoir.h"
#include "examples/Lorenz/LorenzDatastream.h"

namespace
{
    /// Deterministic per-step drive values for input + feedback channels.
    struct DriveSeries
    {
        std::vector<float> inputs; // steps * num_inputs
        std::vector<float> feedback; // steps * num_feedback_channels (empty if none)
    };

    DriveSeries MakeDrive(size_t steps, size_t num_inputs, size_t num_fb, uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        DriveSeries d;
        d.inputs.resize(steps * num_inputs);
        for (float& v : d.inputs) v = dist(rng);
        d.feedback.resize(steps * num_fb);
        for (float& v : d.feedback) v = dist(rng);
        return d;
    }

    /// Drive `steps` steps from the series starting at step `t0`, appending
    /// each post-step state (N floats) to `trace`.
    void Drive(Reservoir& r, const DriveSeries& d, size_t t0, size_t steps,
               size_t num_inputs, size_t num_fb, std::vector<float>* trace)
    {
        const size_t n = r.Size();
        for (size_t s = 0; s < steps; ++s)
        {
            for (size_t ch = 0; ch < num_inputs; ++ch)
                r.InjectInput(ch, d.inputs[(t0 + s) * num_inputs + ch]);
            for (size_t ch = 0; ch < num_fb; ++ch)
                r.InjectFeedback(ch, d.feedback[(t0 + s) * num_fb + ch]);
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

    /// Run the snapshot fidelity suite for one reservoir config.
    /// Returns the number of failed checks (0 = pass).
    int TestConfig(const char* label, const ReservoirConfig& cfg,
                   size_t warm_steps, size_t replay_steps)
    {
        int failures = 0;
        auto r = Reservoir::Create(cfg);
        const size_t num_fb = cfg.num_feedback_channels;
        const DriveSeries d =
            MakeDrive(warm_steps + replay_steps, cfg.num_inputs, num_fb, /*seed=*/0xFEEDBAC + cfg.dim);

        // Warm up to a mid-rotation ring state, then snapshot.
        Drive(*r, d, 0, warm_steps, cfg.num_inputs, num_fb, nullptr);
        const Reservoir::Snapshot snap = r->TakeSnapshot();

        // Branch A: drive forward, record trajectory.
        std::vector<float> trace_a;
        Drive(*r, d, warm_steps, replay_steps, cfg.num_inputs, num_fb, &trace_a);

        // Branch B: restore, replay identical drive, record again.
        r->RestoreSnapshot(snap);
        std::vector<float> trace_b;
        Drive(*r, d, warm_steps, replay_steps, cfg.num_inputs, num_fb, &trace_b);

        if (!BitIdentical(trace_a, trace_b))
        {
            std::printf("  [%s] FAIL: restore+replay trajectory differs from original\n", label);
            ++failures;
        }

        // Canonicality: Take -> Restore -> Take must reproduce the snapshot
        // exactly (restore re-homes the ring; the capture is rotation-free).
        r->RestoreSnapshot(snap);
        const Reservoir::Snapshot snap2 = r->TakeSnapshot();
        if (!BitIdentical(snap.state, snap2.state) || !BitIdentical(snap.history, snap2.history))
        {
            std::printf("  [%s] FAIL: Take->Restore->Take is not the identity\n", label);
            ++failures;
        }

        // Staged-drive isolation: an injection staged before a restore must
        // not leak into the post-restore trajectory.
        r->RestoreSnapshot(snap);
        r->InjectInput(0, 123.456f);
        if (num_fb > 0) r->InjectFeedback(0, -77.7f);
        r->RestoreSnapshot(snap);
        std::vector<float> trace_c;
        Drive(*r, d, warm_steps, replay_steps, cfg.num_inputs, num_fb, &trace_c);
        if (!BitIdentical(trace_a, trace_c))
        {
            std::printf("  [%s] FAIL: staged injections leaked through RestoreSnapshot\n", label);
            ++failures;
        }

        // Size validation: a mismatched snapshot must throw.
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
            std::printf("  [%s] PASS (%zu warm + %zu replay steps, N=%zu, M=%zu, fb=%zu)\n",
                        label, warm_steps, replay_steps, r->Size(), cfg.history_depth, num_fb);
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
    deep.history_depth = 7; // odd depth so warm_steps leaves the ring mid-rotation
    deep.leak_rate = 0.8f; // exercise the leaky-integrator path (old state feeds in)
    deep.num_inputs = 2;
    deep.verbose = false;
    failures += TestConfig("dim8 M7 leaky 2-in", deep, 37, 50);

    ReservoirConfig fb = deep;
    fb.num_feedback_channels = 3; // non-dividing D: drives the feedback port, exercises the D<=N path
    fb.feedback_scaling = 0.4f;
    failures += TestConfig("dim8 M7 +feedback", fb, 37, 50);

    if (failures == 0)
    {
        std::printf("All checks passed.\n");
        return 0;
    }
    std::printf("%d check(s) FAILED.\n", failures);
    return 1;
}
