/// @file main.cpp
/// @brief Reservoir snapshot/restore fidelity + FSF smoke diagnostics.

#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "ESN.h"
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

    /// F0: FSF off. F1: FSF on differs from off. F2: Create(GetConfig) bit-identical.
    /// F3: Clear preserves FSF-closed trajectory. F4: ESN warmup under FSF.
    int TestFsf()
    {
        int failures = 0;
        constexpr size_t kSteps = 64;
        constexpr uint64_t kDriveSeed = 0xA11CE;

        ReservoirConfig base;
        base.dim = 6;
        base.history_depth = 4;
        base.seed = 4242;
        base.verbose = false;

        auto run_trace = [&](Reservoir& r) {
            const DriveSeries d = MakeDrive(kSteps, r.GetConfig().num_inputs, 0, kDriveSeed);
            std::vector<float> tr;
            Drive(r, d, 0, kSteps, r.GetConfig().num_inputs, 0, &tr);
            return tr;
        };

        auto off = Reservoir::Create(base);
        const std::vector<float> tr_off = run_trace(*off);

        ReservoirConfig on_cfg = base;
        on_cfg.full_state_feedback = true;
        on_cfg.fsf_seed = 99;
        on_cfg.fsf_scaling = 0.5f;
        auto on_a = Reservoir::Create(on_cfg);
        if (!on_a->FullStateFeedbackEnabled())
        {
            std::printf("  [FSF] FAIL: FullStateFeedbackEnabled false after enable\n");
            ++failures;
        }
        const std::vector<float> tr_on = run_trace(*on_a);
        if (BitIdentical(tr_off, tr_on))
        {
            std::printf("  [FSF] FAIL: FSF on did not change trajectory vs off (F1)\n");
            ++failures;
        }
        else
            std::printf("  [FSF] PASS F0/F1: off vs on differ\n");

        // F2: seed+fsf_scaling rebuild identical V and B_fsf
        auto on_b = Reservoir::Create(on_a->GetConfig());
        const std::vector<float> tr_rebuild = run_trace(*on_b);
        if (!BitIdentical(tr_on, tr_rebuild))
        {
            std::printf("  [FSF] FAIL: Create(GetConfig) trajectory mismatch (F2)\n");
            ++failures;
        }
        else
            std::printf("  [FSF] PASS F2: Create(GetConfig) bit-identical\n");

        // F3: Clear does not wipe construction-time V — same drive after clear matches
        // a fresh run from zero only if we clear then re-drive from cold; instead check
        // snapshot restore under FSF still works (Clear leaves V, zeros state).
        on_a->Clear();
        const std::vector<float> tr_after_clear = run_trace(*on_a);
        if (!BitIdentical(tr_on, tr_after_clear))
        {
            std::printf("  [FSF] FAIL: Clear changed FSF-closed trajectory (F3)\n");
            ++failures;
        }
        else
            std::printf("  [FSF] PASS F3: Clear preserves FSF params (same re-drive)\n");

        // F4: ESN warmup under FSF
        ESNConfig ec;
        ec.reservoir = on_cfg;
        ec.reservoir.verbose = false;
        ec.readout.num_outputs = 1;
        ESN esn(ec);
        if (!esn.FullStateFeedbackEnabled())
        {
            std::printf("  [FSF] FAIL: ESN FullStateFeedbackEnabled\n");
            ++failures;
        }
        std::vector<float> u(kSteps, 0.3f);
        esn.ReservoirWarmup(u.data(), kSteps);
        std::vector<float> state(esn.ReservoirNeuronCount());
        esn.CopyReservoirState(state.data());
        double nrm = 0.0;
        for (float x : state) nrm += static_cast<double>(x) * x;
        if (nrm <= 0.0)
        {
            std::printf("  [FSF] FAIL: ESN warmup under FSF left zero state\n");
            ++failures;
        }
        else
            std::printf("  [FSF] PASS F4: ESN Warmup under FSF\n");

        // GetConfig round-trip of FSF knobs when disabled
        ReservoirConfig off_seed = base;
        off_seed.full_state_feedback = false;
        off_seed.fsf_seed = 12345;
        off_seed.fsf_scaling = 0.7f;
        auto r_off = Reservoir::Create(off_seed);
        const auto got = r_off->GetConfig();
        if (got.fsf_seed != 12345 || got.fsf_scaling != 0.7f || got.full_state_feedback)
        {
            std::printf("  [FSF] FAIL: GetConfig lost FSF knobs while disabled\n");
            ++failures;
        }
        else
            std::printf("  [FSF] PASS GetConfig round-trip while FSF off\n");

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

    std::printf("=== Full-state feedback smoke ===\n");
    failures += TestFsf();

    if (failures == 0)
    {
        std::printf("All checks passed.\n");
        return 0;
    }
    std::printf("%d check(s) FAILED.\n", failures);
    return 1;
}
