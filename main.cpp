/// @file main.cpp
/// @brief Feedback-training verification diagnostics
/// (docs/FeedbackTrainingMethodology.md §9).
///
/// Item 2 — snapshot/restore fidelity: snapshot -> drive N steps -> restore ->
/// replay the same N inputs must reproduce the identical trajectory
/// bit-for-bit. The restore is the branch-point primitive the Pass-2 feedback
/// probes stand on, so equality here is exact (memcmp), not approximate.
///
/// Item 1 — no-op regression for the closed-loop step driver (ESN::StepLive):
/// with feedback configured but f = 0 forced (each §6.13 kill mechanism in
/// turn), the trajectory must match the open-loop ESN; with the loop live it
/// must not.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "ESN.h"
#include "Reservoir.h"

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

    /// Value equality, not memcmp: the dead-feedback arms add exact-zero terms
    /// to the per-vertex sums, which can flip a zero's sign bit without
    /// changing the value. operator== treats +-0 as equal and NaN as unequal,
    /// both of which are the semantics this comparison wants.
    bool ValueIdentical(const std::vector<float>& a, const std::vector<float>& b)
    {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) return false;
        return true;
    }

    /// §9 item 1: no-op regression + sanity for the closed-loop step driver.
    /// Returns the number of failed checks (0 = pass).
    int TestClosedLoopDriver()
    {
        int failures = 0;
        constexpr size_t kSteps = 60;

        ESNConfig base;
        base.reservoir.dim = 6;
        base.reservoir.history_depth = 4;
        base.reservoir.verbose = false;

        std::mt19937_64 rng(0xC105ED);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> inputs(kSteps * base.reservoir.num_inputs);
        for (float& v : inputs) v = dist(rng);

        auto run_states = [&](const ESNConfig& cfg, bool lesion_at_runtime = false)
        {
            ESN esn(cfg);
            if (esn.HasFeedback() != (cfg.reservoir.num_feedback_channels > 0))
            {
                std::printf("  [driver] FAIL: HasFeedback() disagrees with num_feedback_channels\n");
                ++failures;
            }
            if (lesion_at_runtime) esn.SetForceZeroFeedback(true);
            esn.Run(inputs.data(), kSteps);
            return esn.SelectedStates();
        };

        const std::vector<float> open_loop = run_states(base);

        ESNConfig fb = base;
        fb.reservoir.num_feedback_channels = 1; // feedback_scaling stays at its live 0.5 default

        // §6.13 config arm: weights drawn then scaled to zero. Same RNG draws,
        // every injected f contributes exactly 0 -> must match open-loop.
        ESNConfig dead_weights = fb;
        dead_weights.reservoir.feedback_scaling = 0.0f;
        if (!ValueIdentical(open_loop, run_states(dead_weights)))
        {
            std::printf("  [driver] FAIL: feedback_scaling=0 arm differs from open-loop\n");
            ++failures;
        }

        // §6.13 runtime arm: live weights, value forced to 0 at the clamp seam.
        ESNConfig lesioned = fb;
        lesioned.feedback.force_zero = true;
        if (!ValueIdentical(open_loop, run_states(lesioned)))
        {
            std::printf("  [driver] FAIL: force_zero arm differs from open-loop\n");
            ++failures;
        }

        // Same lesion applied through the runtime setter instead of config.
        if (!ValueIdentical(open_loop, run_states(fb, /*lesion_at_runtime=*/true)))
        {
            std::printf("  [driver] FAIL: SetForceZeroFeedback(true) arm differs from open-loop\n");
            ++failures;
        }

        // Live arm: the loop must actually act — the eagerly built random F
        // (S6.8) injects nonzero tanh(F(x)) from step one.
        if (ValueIdentical(open_loop, run_states(fb)))
        {
            std::printf("  [driver] FAIL: live closed loop had no effect on the trajectory\n");
            ++failures;
        }

        // v1 validation: more than one feedback channel must throw at the ESN
        // seam (the Reservoir itself accepts any channel count dividing N).
        ESNConfig two = fb;
        two.reservoir.num_feedback_channels = 2;
        bool threw = false;
        try
        {
            ESN esn(two);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }
        if (!threw)
        {
            std::printf("  [driver] FAIL: num_feedback_channels=2 did not throw\n");
            ++failures;
        }

        if (failures == 0)
            std::printf("  [driver] PASS (%zu steps, N=%u, both kill arms = open-loop, live arm diverges)\n",
                        kSteps, 1u << base.reservoir.dim);
        return failures;
    }

    /// Two-readout checkpointing (§8): F's photo must carry F's brain — a
    /// fresh ESN with a *different* F seed, restored from another ESN's
    /// feedback state, must reproduce that ESN's closed-loop trajectory.
    /// Returns the number of failed checks (0 = pass).
    int TestTwoReadoutCheckpointing()
    {
        int failures = 0;
        constexpr size_t kSteps = 60;

        ESNConfig fb;
        fb.reservoir.dim = 6;
        fb.reservoir.history_depth = 4;
        fb.reservoir.verbose = false;
        fb.reservoir.num_feedback_channels = 1;

        std::mt19937_64 rng(0x0F0F0F);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> inputs(kSteps * fb.reservoir.num_inputs);
        for (float& v : inputs) v = dist(rng);

        ESN a(fb);
        const ESN::ReadoutState f_photo = a.GetFeedbackState();
        if (!f_photo.is_trained || f_photo.weights.empty())
        {
            std::printf("  [checkpoint] FAIL: eagerly built F is not persist-worthy (S6.15)\n");
            ++failures;
        }
        a.Run(inputs.data(), kSteps);
        const std::vector<float> trace_a = a.SelectedStates();

        // A different F seed must change the closed-loop trajectory...
        ESNConfig other = fb;
        other.feedback.readout.seed = 99;
        ESN b(other);
        b.Run(inputs.data(), kSteps);
        if (ValueIdentical(trace_a, b.SelectedStates()))
        {
            std::printf("  [checkpoint] FAIL: different F seed left the trajectory unchanged\n");
            ++failures;
        }

        // ...and restoring A's photo over it must restore A's trajectory.
        ESN c(other);
        c.SetFeedbackState(f_photo);
        c.Run(inputs.data(), kSteps);
        if (!ValueIdentical(trace_a, c.SelectedStates()))
        {
            std::printf("  [checkpoint] FAIL: restored F does not reproduce the source trajectory\n");
            ++failures;
        }

        // Without feedback configured, both methods must throw.
        ESNConfig open = fb;
        open.reservoir.num_feedback_channels = 0;
        ESN d(open);
        bool threw_get = false, threw_set = false;
        try { (void)d.GetFeedbackState(); } catch (const std::logic_error&) { threw_get = true; }
        try { d.SetFeedbackState(f_photo); } catch (const std::logic_error&) { threw_set = true; }
        if (!threw_get || !threw_set)
        {
            std::printf("  [checkpoint] FAIL: Get/SetFeedbackState did not throw without feedback\n");
            ++failures;
        }

        // Weights() staleness regression (exercised through P's public seam):
        // photos taken before and after further online training must differ —
        // the blob has to re-sync from the live network on every call.
        ESN e(fb);
        e.InitOnline(inputs.data(), kSteps);
        const float target = 0.5f;
        e.TrainLiveStepRegression(&target, /*lr=*/1e-3f, /*weight_decay=*/0.0f);
        const ESN::ReadoutState p1 = e.GetReadoutState();
        e.TrainLiveStepRegression(&target, /*lr=*/1e-3f, /*weight_decay=*/0.0f);
        const ESN::ReadoutState p2 = e.GetReadoutState();
        if (p1.weights == p2.weights)
        {
            std::printf("  [checkpoint] FAIL: checkpoint is stale after further online training\n");
            ++failures;
        }

        if (failures == 0)
            std::printf("  [checkpoint] PASS (F photo round-trips across F seeds; stale-blob regression holds)\n");
        return failures;
    }

    /// Training orchestration (§4): probe sanity (§9.3), kill-switch freeze,
    /// no-op equivalence to a hand-rolled stream loop, liveness, and the
    /// §6.17 validation bracket. Returns the number of failed checks.
    int TestTrainingOrchestration()
    {
        int failures = 0;
        constexpr size_t kWarm = 16;
        constexpr size_t kCycles = 300;

        ESNConfig base;
        base.reservoir.dim = 6;
        base.reservoir.history_depth = 4;
        base.reservoir.verbose = false;
        base.reservoir.num_feedback_channels = 1;
        base.feedback.pretrain_steps = 0; // straight to alternation unless a test overrides

        std::mt19937_64 rng(0x07C4E57A);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> warm(kWarm);
        for (float& v : warm) v = dist(rng);
        std::vector<float> inputs(kCycles), targets(kCycles);
        for (float& v : inputs) v = dist(rng);
        for (float& v : targets) v = dist(rng);
        std::vector<int> labels(kCycles);
        for (int& v : labels) v = static_cast<int>(rng() % 3);

        // --- §9.3 probe sanity, regression: ε = 0 -> all three probes
        // identical, every cycle rejects. ---
        {
            ESNConfig cfg = base;
            cfg.feedback.epsilon = 0.0f;
            ESN esn(cfg);
            esn.InitOnline(warm.data(), kWarm);
            bool ok = true;
            for (size_t i = 0; i < 50 && ok; ++i)
            {
                const auto c = esn.TrainFeedbackCycle(inputs.data() + i, targets.data() + i);
                ok = !c.accepted && c.e0 == c.e_plus && c.e0 == c.e_minus;
            }
            if (!ok)
            {
                std::printf("  [orchestration] FAIL: eps=0 probes not identical / accepted (regression)\n");
                ++failures;
            }
        }

        // --- §9.3 probe sanity, classification: same property under CE. ---
        {
            ESNConfig cfg = base;
            cfg.feedback.epsilon = 0.0f;
            cfg.readout.task = ReadoutTask::Classification;
            cfg.readout.num_outputs = 3;
            ESN esn(cfg);
            esn.InitOnline(warm.data(), kWarm);
            bool ok = true;
            for (size_t i = 0; i < 50 && ok; ++i)
            {
                const auto c = esn.TrainFeedbackCycle(inputs.data() + i, labels[i]);
                ok = !c.accepted && c.e0 == c.e_plus && c.e0 == c.e_minus;
            }
            if (!ok)
            {
                std::printf("  [orchestration] FAIL: eps=0 probes not identical / accepted (classification)\n");
                ++failures;
            }
            // Task-mismatch guard while we have a classification ESN at hand.
            bool threw = false;
            try { (void)esn.TrainFeedbackCycle(inputs.data(), targets.data()); }
            catch (const std::invalid_argument&) { threw = true; }
            if (!threw)
            {
                std::printf("  [orchestration] FAIL: regression overload on classification P did not throw\n");
                ++failures;
            }
        }

        // --- §6.13 kill-switch freeze: dead feedback weights -> every cycle
        // rejects on exact equality and F's weights never move. ---
        {
            ESNConfig cfg = base;
            cfg.reservoir.feedback_scaling = 0.0f; // ε stays at its live 0.05 default
            ESN esn(cfg);
            esn.InitOnline(warm.data(), kWarm);
            const auto f_before = esn.GetFeedbackState();
            bool ok = true;
            for (size_t i = 0; i < 50 && ok; ++i)
                ok = !esn.TrainFeedbackCycle(inputs.data() + i, targets.data() + i).accepted;
            if (!ok || esn.GetFeedbackState().weights != f_before.weights)
            {
                std::printf("  [orchestration] FAIL: feedback_scaling=0 arm accepted or trained F\n");
                ++failures;
            }
        }

        // --- No-op equivalence: a force_zero orchestrated run must leave P
        // bit-identical to a hand-rolled StepLive + per-step-train loop with
        // the same §6.9 lr schedule — the probe bracket perturbs nothing. ---
        {
            ESNConfig cfg = base;
            cfg.feedback.force_zero = true;
            cfg.feedback.pretrain_steps = 10;

            ESN a(cfg);
            a.InitOnline(warm.data(), kWarm);
            std::vector<ESN::FeedbackCycleInfo> infos;
            for (size_t i = 0; i < 30; ++i)
                infos.push_back(a.TrainFeedbackCycle(inputs.data() + i, targets.data() + i));

            // Replay with the lrs A reported. (The cosine cannot be
            // recomputed here bit-exactly: under -ffast-math the inline
            // CosineLR contracts differently per translation unit, up to
            // 1 ulp — what this test pins is the state trajectory and the
            // training calls, not the lr formula.)
            ESN b(cfg);
            b.InitOnline(warm.data(), kWarm);
            for (size_t i = 0; i < 30; ++i)
            {
                b.StepLive(inputs.data() + i);
                b.TrainLiveStepRegression(targets.data() + i, infos[i].p_lr, cfg.readout.weight_decay);
            }

            if (a.GetReadoutState().weights != b.GetReadoutState().weights)
            {
                std::printf("  [orchestration] FAIL: force_zero orchestration != hand-rolled stream loop\n");
                ++failures;
            }

            // §6.9 schedule shape: starts at lr_max, anneals monotonically
            // into the alternation constant p_lr with no discontinuity.
            bool shape_ok = infos[0].p_lr == cfg.readout.lr_max && infos[0].pretrain;
            for (size_t i = 1; i < 10; ++i)
                shape_ok = shape_ok && infos[i].p_lr < infos[i - 1].p_lr && infos[i].pretrain;
            for (size_t i = 10; i < 30; ++i)
                shape_ok = shape_ok && infos[i].p_lr == cfg.feedback.p_lr && !infos[i].pretrain;
            if (!shape_ok)
            {
                std::printf("  [orchestration] FAIL: pretrain lr schedule shape wrong (S6.9)\n");
                ++failures;
            }
        }

        // --- Liveness: the live arm accepts some but not all cycles, and F
        // actually trains. ---
        {
            ESN esn(base);
            esn.InitOnline(warm.data(), kWarm);
            const auto f_before = esn.GetFeedbackState();
            size_t accepts = 0;
            for (size_t i = 0; i < kCycles; ++i)
                accepts += esn.TrainFeedbackCycle(inputs.data() + i, targets.data() + i).accepted;
            if (accepts == 0 || accepts == kCycles)
            {
                std::printf("  [orchestration] FAIL: live accept rate degenerate (%zu/%zu)\n",
                            accepts, kCycles);
                ++failures;
            }
            if (esn.GetFeedbackState().weights == f_before.weights)
            {
                std::printf("  [orchestration] FAIL: live arm accepted but F never trained\n");
                ++failures;
            }
        }

        // --- §6.17 validation bracket: validating mid-training must not
        // change the subsequent trajectory; the score itself is
        // deterministic. ---
        {
            ESN a(base), b(base);
            a.InitOnline(warm.data(), kWarm);
            b.InitOnline(warm.data(), kWarm);

            for (size_t i = 0; i < 20; ++i)
            {
                (void)a.TrainFeedbackCycle(inputs.data() + i, targets.data() + i);
                (void)b.TrainFeedbackCycle(inputs.data() + i, targets.data() + i);
            }
            const double v1 = a.ValidateClosedLoop(inputs.data(), targets.data(), 48);
            const double v2 = a.ValidateClosedLoop(inputs.data(), targets.data(), 48);
            if (!(v1 == v2) || !std::isfinite(v1))
            {
                std::printf("  [orchestration] FAIL: validation score not deterministic/finite\n");
                ++failures;
            }
            for (size_t i = 20; i < 40; ++i)
            {
                (void)a.TrainFeedbackCycle(inputs.data() + i, targets.data() + i);
                (void)b.TrainFeedbackCycle(inputs.data() + i, targets.data() + i);
            }
            if (a.GetReadoutState().weights != b.GetReadoutState().weights ||
                a.GetFeedbackState().weights != b.GetFeedbackState().weights)
            {
                std::printf("  [orchestration] FAIL: validation perturbed the training trajectory\n");
                ++failures;
            }
        }

        if (failures == 0)
            std::printf("  [orchestration] PASS (eps=0 sanity x2 tasks, kill-switch freeze, "
                        "no-op equivalence, liveness, validation bracket)\n");
        return failures;
    }

    /// Bit-level NaN/finite tests: this TU compiles with -ffast-math, under
    /// which gcc folds std::isnan to false and std::isfinite to true — the
    /// telemetry's NaN sentinels are invisible to the standard predicates.
    bool IsNaNBits(double v)
    {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof bits);
        return (bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL &&
               (bits & 0x000FFFFFFFFFFFFFULL) != 0;
    }

    bool IsFiniteBits(double v)
    {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof bits);
        return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
    }

    /// Feedback telemetry (§8): counter exactness, ring wraparound, gauge
    /// ranges, the lesioned-arm introspection property, and guards.
    /// Returns the number of failed checks (0 = pass).
    int TestTelemetry()
    {
        int failures = 0;
        constexpr size_t kWarm = 16;
        constexpr size_t kPre = 10;
        constexpr size_t kAltCycles = 1100; // exceeds the 1000-entry window

        ESNConfig cfg;
        cfg.reservoir.dim = 6;
        cfg.reservoir.history_depth = 4;
        cfg.reservoir.verbose = false;
        cfg.reservoir.num_feedback_channels = 1;
        cfg.feedback.pretrain_steps = kPre;

        std::mt19937_64 rng(0x7E1E0E72);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> warm(kWarm);
        for (float& v : warm) v = dist(rng);
        const size_t total = kPre + kAltCycles;
        std::vector<float> inputs(total), targets(total);
        for (float& v : inputs) v = dist(rng);
        for (float& v : targets) v = dist(rng);

        ESN esn(cfg);
        esn.InitOnline(warm.data(), kWarm);

        // Empty window: NaN gauges, zero counters.
        {
            const auto t = esn.GetFeedbackTelemetry();
            if (t.window != 0 || t.cycles != 0 || !IsNaNBits(t.accept_rate))
            {
                std::printf("  [telemetry] FAIL: empty-window snapshot not NaN/zero\n");
                ++failures;
            }
        }

        size_t accepts = 0, accepts_pos = 0;
        ESN::FeedbackCycleInfo last{};
        for (size_t i = 0; i < kPre + 50; ++i)
        {
            last = esn.TrainFeedbackCycle(inputs.data() + i, targets.data() + i);
            accepts += last.accepted;
            accepts_pos += last.accepted && last.sign > 0.0f;
        }

        // Counter exactness against the FeedbackCycleInfo stream, pre-wrap.
        {
            const auto t = esn.GetFeedbackTelemetry();
            const auto h = esn.GetFeedbackHistory();
            const bool ok =
                t.pretrain_examples == kPre && t.cycles == 50 && t.window == 50 &&
                t.accepts == accepts && t.accepts_pos == accepts_pos &&
                t.accept_rate == static_cast<double>(accepts) / 50.0 &&
                h.size() == 50 &&
                h.back().e0 == last.e0 && h.back().sf == last.sf &&
                h.back().accepted == last.accepted;
            if (!ok)
            {
                std::printf("  [telemetry] FAIL: counters/history disagree with the cycle stream\n");
                ++failures;
            }
        }

        for (size_t i = kPre + 50; i < total; ++i)
        {
            last = esn.TrainFeedbackCycle(inputs.data() + i, targets.data() + i);
            accepts += last.accepted;
        }

        // Wraparound: window pinned at 1000, history chronological, lifetime
        // counters keep counting.
        {
            const auto t = esn.GetFeedbackTelemetry();
            const auto h = esn.GetFeedbackHistory();
            const bool ok =
                t.cycles == kAltCycles && t.window == 1000 && t.accepts == accepts &&
                h.size() == 1000 &&
                h.back().e0 == last.e0 && h.back().sf == last.sf;
            if (!ok)
            {
                std::printf("  [telemetry] FAIL: ring wraparound broke window/history/counters\n");
                ++failures;
            }

            // Gauge ranges on the live arm.
            const bool ranges_ok =
                IsFiniteBits(t.var_f) && t.var_f >= 0.0 &&
                IsFiniteBits(t.var_tanh_f) && t.var_tanh_f >= 0.0 &&
                t.mean_tanh_f >= -1.0 && t.mean_tanh_f <= 1.0 &&
                t.var_tanh_f <= t.var_f + 1e-12 && // tanh is a contraction
                t.saturation_frac >= 0.0 && t.saturation_frac <= 1.0 &&
                t.mean_lever > 0.0 && t.mean_lever <= 1.0 &&
                IsFiniteBits(t.mean_e0) &&
                t.state_rms_mean > 0.0 && t.state_rms_max >= t.state_rms_mean &&
                (t.accepts == 0 ||
                 (IsFiniteBits(t.mean_realization) && t.mean_realization > 0.0 &&
                  t.sign_balance >= -1.0 && t.sign_balance <= 1.0));
            if (!ranges_ok)
            {
                std::printf("  [telemetry] FAIL: live-arm gauge out of range\n");
                ++failures;
            }
        }

        // Lesioned-arm introspection (§6.13): dead weights -> zero accepts,
        // NaN accept-derived gauges, yet F's computed output is still
        // observable (finite mean/var of f_commit).
        {
            ESNConfig dead = cfg;
            dead.reservoir.feedback_scaling = 0.0f;
            dead.feedback.pretrain_steps = 0;
            ESN d(dead);
            d.InitOnline(warm.data(), kWarm);
            for (size_t i = 0; i < 50; ++i)
                (void)d.TrainFeedbackCycle(inputs.data() + i, targets.data() + i);
            const auto t = d.GetFeedbackTelemetry();
            const bool ok =
                t.accept_rate == 0.0 && IsNaNBits(t.mean_realization) &&
                IsNaNBits(t.sign_balance) &&
                IsFiniteBits(t.mean_f) && IsFiniteBits(t.var_f) &&
                IsFiniteBits(t.mean_abs_f);
            if (!ok)
            {
                std::printf("  [telemetry] FAIL: lesioned arm not introspectable as specified\n");
                ++failures;
            }
        }

        // Guard: no feedback configured -> throws.
        {
            ESNConfig open = cfg;
            open.reservoir.num_feedback_channels = 0;
            ESN o(open);
            bool threw = false;
            try { (void)o.GetFeedbackTelemetry(); }
            catch (const std::logic_error&) { threw = true; }
            if (!threw)
            {
                std::printf("  [telemetry] FAIL: GetFeedbackTelemetry without feedback did not throw\n");
                ++failures;
            }
        }

        if (failures == 0)
            std::printf("  [telemetry] PASS (counters exact, 1000-cycle ring wraps, gauges in range, "
                        "lesioned arm introspectable)\n");
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
    fb.num_feedback_channels = 2;
    fb.feedback_scaling = 0.4f;
    failures += TestConfig("dim8 M7 +feedback", fb, 37, 50);

    std::printf("=== ESN closed-loop step driver (no-op regression) ===\n");
    failures += TestClosedLoopDriver();

    std::printf("=== ESN two-readout checkpointing ===\n");
    failures += TestTwoReadoutCheckpointing();

    std::printf("=== ESN feedback training orchestration ===\n");
    failures += TestTrainingOrchestration();

    std::printf("=== ESN feedback telemetry ===\n");
    failures += TestTelemetry();

    if (failures == 0)
    {
        std::printf("All checks passed.\n");
        return 0;
    }
    std::printf("%d check(s) FAILED.\n", failures);
    return 1;
}
