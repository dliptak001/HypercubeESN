#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <memory>
#include "../ESN.h"
#include "../EnsembleESN.h"

namespace py = pybind11;

// Single de-templated ESN binding. The hypercube dimension is a runtime
// constructor argument (cfg.reservoir.dim), so one C++ type and one Python
// class serve every dimension 5-16 — no per-DIM instantiations.
PYBIND11_MODULE(_core, m)
{
    m.doc() = "HypercubeESN: reservoir computing on Boolean hypercube graphs";

    py::class_<ESN>(m, "_ESN")
        // ── Construction ──
        // All reservoir + readout parameters fixed at construction time.
        // The readout config is consumed by train() / warmup() — no
        // per-call config overrides.
        .def(py::init([](size_t dim, uint64_t seed, float spectral_radius, float input_scaling,
                         float leak_rate, size_t num_inputs, size_t history_depth,
                         float history_floor,
                         bool verbose,
                         int readout_num_outputs, const char* readout_task,
                         int readout_num_layers, int readout_conv_channels,
                         int readout_epochs, int readout_batch_size,
                         float readout_lr_max, float readout_lr_min_frac,
                         int readout_lr_decay_epochs, float readout_weight_decay,
                         float readout_momentum, const char* readout_activation,
                         unsigned readout_seed) {
            ESNConfig cfg;
            cfg.reservoir.dim              = dim;
            cfg.reservoir.seed             = seed;
            cfg.reservoir.spectral_radius  = spectral_radius;
            cfg.reservoir.input_scaling    = input_scaling;
            cfg.reservoir.leak_rate        = leak_rate;
            cfg.reservoir.num_inputs       = num_inputs;
            cfg.reservoir.history_depth    = history_depth;
            cfg.reservoir.history_floor    = history_floor;
            cfg.reservoir.verbose          = verbose;
            cfg.readout.num_outputs        = readout_num_outputs;
            cfg.readout.task               = (std::strcmp(readout_task, "classification") == 0)
                                                 ? ReadoutTask::Classification
                                                 : ReadoutTask::Regression;
            cfg.readout.num_layers         = readout_num_layers;
            cfg.readout.conv_channels      = readout_conv_channels;
            cfg.readout.epochs             = readout_epochs;
            cfg.readout.batch_size         = readout_batch_size;
            cfg.readout.lr_max             = readout_lr_max;
            cfg.readout.lr_min_frac        = readout_lr_min_frac;
            cfg.readout.lr_decay_epochs    = readout_lr_decay_epochs;
            cfg.readout.weight_decay       = readout_weight_decay;
            cfg.readout.momentum           = readout_momentum;
            // String -> ReadoutActivation, mirroring the readout_task pattern above.
            if      (std::strcmp(readout_activation, "relu")       == 0) cfg.readout.activation = ReadoutActivation::RELU;
            else if (std::strcmp(readout_activation, "leaky_relu") == 0) cfg.readout.activation = ReadoutActivation::LEAKY_RELU;
            else if (std::strcmp(readout_activation, "none")       == 0) cfg.readout.activation = ReadoutActivation::NONE;
            else if (std::strcmp(readout_activation, "tanh")       == 0) cfg.readout.activation = ReadoutActivation::TANH;
            else throw std::invalid_argument(
                std::string("readout_activation must be one of "
                            "'tanh', 'relu', 'leaky_relu', 'none' (got '") + readout_activation + "')");
            cfg.readout.seed               = readout_seed;
            return std::make_unique<ESN>(cfg);
        }),
            py::arg("dim"),
            py::arg("seed")                     = 73895ULL,
            py::arg("spectral_radius")          = 0.99f,
            py::arg("input_scaling")            = 0.5f,
            py::arg("leak_rate")                = 1.0f,
            py::arg("num_inputs")               = 1ULL,
            py::arg("history_depth")            = 16ULL,
            py::arg("history_floor")            = 1.0f,
            py::arg("verbose")                  = true,
            py::arg("readout_num_outputs")      = 1,
            py::arg("readout_task")             = "regression",
            py::arg("readout_num_layers")       = 0,
            py::arg("readout_conv_channels")    = 16,
            py::arg("readout_epochs")           = 200,
            py::arg("readout_batch_size")       = 32,
            py::arg("readout_lr_max")           = 0.0015f,
            py::arg("readout_lr_min_frac")      = 0.01f,
            py::arg("readout_lr_decay_epochs")  = 0,
            py::arg("readout_weight_decay")     = 0.0f,
            py::arg("readout_momentum")         = 0.0f,
            py::arg("readout_activation")       = "tanh",
            py::arg("readout_seed")             = 42u)

        // ── Reservoir driving ──
        .def("reservoir_warmup", [](ESN& self, py::array_t<float, py::array::c_style | py::array::forcecast> inputs) {
            auto buf = inputs.request();
            size_t total = static_cast<size_t>(buf.size);
            size_t K = self.NumInputs();
            if (total % K != 0)
                throw std::invalid_argument("Input size must be divisible by num_inputs");
            self.ReservoirWarmup(static_cast<const float*>(buf.ptr), total / K);
        }, py::arg("inputs"),
           "Drive the reservoir without recording states (wash out initial transient).")

        .def("reservoir_run", [](ESN& self, py::array_t<float, py::array::c_style | py::array::forcecast> inputs,
                       bool clear_recorded) {
            auto buf = inputs.request();
            size_t total = static_cast<size_t>(buf.size);
            size_t K = self.NumInputs();
            if (total % K != 0)
                throw std::invalid_argument("Input size must be divisible by num_inputs");
            self.ReservoirRun(static_cast<const float*>(buf.ptr), total / K, clear_recorded);
        }, py::arg("inputs"), py::kw_only(), py::arg("clear_recorded") = false,
           "Drive the reservoir and record states for training/evaluation. "
           "Successive calls accumulate; pass clear_recorded=True to start a fresh batch.")

        .def("reservoir_clear", &ESN::ReservoirClear,
             "Clear the reservoir state so a new sequence starts from rest. "
             "Recorded states and trained readout preserved.")

        // ── Batch training ──
        .def("train", [](ESN& self,
                         py::array_t<float, py::array::c_style | py::array::forcecast> targets) {
            auto buf = targets.request();
            size_t total = static_cast<size_t>(buf.size);
            const float* ptr = static_cast<const float*>(buf.ptr);

            // targets is laid out [sample][output] row-major, so the number of
            // training samples is total / num_outputs. For single-output this is
            // a no-op; for multi-output it converts the flattened buffer length
            // back into the sample count ESN::Train expects.
            size_t K = self.NumOutputs();
            if (total % K != 0)
                throw std::invalid_argument(
                    "targets length (" + std::to_string(total) +
                    ") must be a multiple of num_outputs (" + std::to_string(K) + ")");
            size_t train_size = total / K;

            if (train_size > self.NumCollected())
                throw std::invalid_argument(
                    "train_size (" + std::to_string(train_size) +
                    ") exceeds num_collected (" + std::to_string(self.NumCollected()) + ")");

            self.Train(ptr, train_size);
        },
            py::arg("targets"),
            "Train the HCNN readout on collected states.\n"
            "Uses the readout config supplied at ESN construction.")

        // ── Streaming HCNN training (task fixed at construction) ──
        .def("train_step", [](ESN& self,
                              py::array_t<float, py::array::c_style | py::array::forcecast> target,
                              float lr, float weight_decay) {
            auto buf = target.request();
            const bool cls = self.GetConfig().readout.task == ReadoutTask::Classification;
            const size_t expected = cls ? 1u : self.NumOutputs();
            if (static_cast<size_t>(buf.size) != expected)
                throw std::invalid_argument(
                    cls ? "target size (" + std::to_string(buf.size) +
                              ") must equal 1 (class index) for classification"
                        : "target size (" + std::to_string(buf.size) +
                              ") must equal num_outputs (" + std::to_string(expected) + ")");
            self.TrainStep(static_cast<const float*>(buf.ptr), lr, weight_decay);
        },
            py::arg("target"), py::arg("lr"), py::arg("weight_decay") = 0.0f,
            "One streaming gradient step on the reservoir's current state.\n"
            "target: regression -> (num_outputs,) values; classification -> (1,) class index.")

        .def("train_step_batch", [](ESN& self,
                                    py::array_t<float, py::array::c_style | py::array::forcecast> states,
                                    py::array_t<float, py::array::c_style | py::array::forcecast> targets,
                                    float lr, float weight_decay) {
            auto sbuf = states.request();
            auto tbuf = targets.request();
            const size_t M = self.Size();
            const bool cls = self.GetConfig().readout.task == ReadoutTask::Classification;
            const size_t K = self.NumOutputs();
            size_t count;
            if (cls) {
                count = static_cast<size_t>(tbuf.size);
            } else {
                if (K == 0 || static_cast<size_t>(tbuf.size) % K != 0)
                    throw std::invalid_argument(
                        "targets size (" + std::to_string(tbuf.size) +
                        ") must be a multiple of num_outputs (" + std::to_string(K) + ")");
                count = static_cast<size_t>(tbuf.size) / K;
            }
            if (static_cast<size_t>(sbuf.size) != count * M)
                throw std::invalid_argument(
                    "states size (" + std::to_string(sbuf.size) +
                    ") must equal count * num_output_verts (" + std::to_string(count) +
                    " * " + std::to_string(M) + " = " + std::to_string(count * M) + ")");
            self.TrainStepBatch(static_cast<const float*>(sbuf.ptr),
                                static_cast<const float*>(tbuf.ptr),
                                count, lr, weight_decay);
        },
            py::arg("states"), py::arg("targets"),
            py::arg("lr"), py::arg("weight_decay") = 0.0f,
            "One streaming gradient step over a mini-batch of pre-accumulated states.\n"
            "states: (count, num_output_verts) float array from copy_reservoir_state.\n"
            "targets: regression -> (count, num_outputs); classification -> (count,) class indices.")

        .def("copy_reservoir_state", [](const ESN& self) {
            size_t M = self.Size();
            py::array_t<float> arr(M);
            self.CopyReservoirState(arr.mutable_data());
            return arr;
        }, "Copy the current reservoir state for external accumulation.\n"
           "Returns a (num_output_verts,) float array.")

        // ── Prediction & evaluation ──
        .def("predict", [](const ESN& self) {
            auto v = self.Predict();
            py::array_t<float> arr(static_cast<py::ssize_t>(v.size()));
            memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
            return arr;
        }, "Predict from the reservoir's current state.\n"
           "Returns (num_outputs,) float array. For autoregressive / streaming loops.")

        .def("predict_from_recorded", [](const ESN& self, size_t timestep) {
            if (timestep >= self.NumCollected())
                throw std::out_of_range(
                    "timestep (" + std::to_string(timestep) +
                    ") >= num_collected (" + std::to_string(self.NumCollected()) + ")");
            auto v = self.PredictFromRecorded(timestep);
            py::array_t<float> arr(static_cast<py::ssize_t>(v.size()));
            memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
            return arr;
        }, py::arg("timestep"),
           "Predict from a recorded timestep: returns (num_outputs,) float array.")

        .def("predict_from_state", [](const ESN& self,
                                      py::array_t<float, py::array::c_style | py::array::forcecast> state) {
            auto buf = state.request();
            size_t M = self.Size();
            if (static_cast<size_t>(buf.size) != M)
                throw std::invalid_argument(
                    "state size (" + std::to_string(buf.size) +
                    ") must equal num_output_verts (" + std::to_string(M) + ")");
            auto v = self.PredictFromState(static_cast<const float*>(buf.ptr));
            py::array_t<float> arr(static_cast<py::ssize_t>(v.size()));
            memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
            return arr;
        }, py::arg("state"),
           "Run the readout on a caller-supplied (num_output_verts,) state.\n"
           "Returns (num_outputs,) float array.")

        .def("r2", [](const ESN& self,
                      py::array_t<float, py::array::c_style | py::array::forcecast> targets,
                      size_t start, size_t count) {
            if (start + count > self.NumCollected())
                throw std::out_of_range(
                    "start + count (" + std::to_string(start + count) +
                    ") > num_collected (" + std::to_string(self.NumCollected()) + ")");
            return self.R2(static_cast<const float*>(targets.request().ptr), start, count);
        }, py::arg("targets"), py::arg("start"), py::arg("count"),
           "Compute R-squared on a slice of collected states.")

        .def("nrmse", [](const ESN& self,
                         py::array_t<float, py::array::c_style | py::array::forcecast> targets,
                         size_t start, size_t count) {
            if (start + count > self.NumCollected())
                throw std::out_of_range(
                    "start + count (" + std::to_string(start + count) +
                    ") > num_collected (" + std::to_string(self.NumCollected()) + ")");
            return self.NRMSE(static_cast<const float*>(targets.request().ptr), start, count);
        }, py::arg("targets"), py::arg("start"), py::arg("count"),
           "Compute Normalized RMSE on a slice of collected states.")

        .def("accuracy", [](const ESN& self,
                            py::array_t<float, py::array::c_style | py::array::forcecast> labels,
                            size_t start, size_t count) {
            if (start + count > self.NumCollected())
                throw std::out_of_range(
                    "start + count (" + std::to_string(start + count) +
                    ") > num_collected (" + std::to_string(self.NumCollected()) + ")");
            return self.Accuracy(static_cast<const float*>(labels.request().ptr), start, count);
        }, py::arg("labels"), py::arg("start"), py::arg("count"),
           "Compute classification accuracy on a slice of collected states.")

        // ── State access ──
        .def("collected_states", [](const ESN& self) {
            auto vec = self.CollectedStates();
            size_t M = self.Size();
            size_t T = self.NumCollected();
            py::array_t<float> arr({T, M});
            memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(float));
            return arr;
        }, "Return collected states as a (num_collected, num_output_verts) array.")

        .def("predictions", [](const ESN& self) {
            size_t T = self.NumCollected();
            size_t K = self.NumOutputs();
            py::array_t<float> arr({T, K});
            float* ptr = arr.mutable_data();
            for (size_t t = 0; t < T; ++t) {
                auto v = self.PredictFromRecorded(t);
                memcpy(ptr + t * K, v.data(), K * sizeof(float));
            }
            return arr;
        }, "Predictions for all collected timesteps: (num_collected, num_outputs) array.")

        // ── Properties ──
        .def_property_readonly("num_collected", &ESN::NumCollected)
        .def_property_readonly("num_outputs", &ESN::NumOutputs)
        .def_property_readonly("num_output_verts", &ESN::Size)
        .def_property_readonly("dim", &ESN::Dim)
        .def_property_readonly("N", &ESN::Size)
        .def_property_readonly("num_inputs", &ESN::NumInputs)
        .def_property_readonly("history_depth", [](const ESN& self) { return self.GetConfig().reservoir.history_depth; })
        .def_property_readonly("history_floor", [](const ESN& self) { return self.GetConfig().reservoir.history_floor; })
        .def_property_readonly("seed", [](const ESN& self) { return self.GetConfig().reservoir.seed; })
        .def_property_readonly("spectral_radius", [](const ESN& self) { return self.GetConfig().reservoir.spectral_radius; })
        .def_property_readonly("leak_rate", [](const ESN& self) { return self.GetConfig().reservoir.leak_rate; })
        .def_property_readonly("input_scaling", [](const ESN& self) { return self.GetConfig().reservoir.input_scaling; })

        // ── Persistence ──
        .def("_get_readout_state", [](const ESN& self) -> py::dict {
            auto s = self.GetReadoutState();
            py::dict d;
            d["is_trained"] = s.is_trained;
            d["weights"] = py::array_t<double>(
                {static_cast<py::ssize_t>(s.weights.size())}, s.weights.data());
            return d;
        })
        .def("_set_readout_state", [](ESN& self, py::dict d) {
            ESN::ReadoutState s;
            s.is_trained = d["is_trained"].cast<bool>();
            auto w = d["weights"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
            s.weights.assign(w.data(), w.data() + w.size());
            self.SetReadoutState(s);
        })
        ;

    // ── EnsembleESN: consensus feedback coupling of M ESN members ──
    // Online-only, feedback-only (design docs/ensemble_esn_feedback.md). Members
    // share one base config and differ only by a derived reservoir seed. The
    // "One D, three roles" identity is enforced here: num_feedback_channels is
    // set from num_outputs, so the Python caller passes a single D.
    py::class_<EnsembleESN>(m, "_EnsembleESN")
        .def(py::init([](size_t dim, uint64_t ensemble_seed, size_t num_members,
                         const char* combine,
                         float spectral_radius, float input_scaling, float leak_rate,
                         size_t num_inputs, size_t history_depth, float history_floor,
                         size_t num_outputs, float feedback_scaling, float bias_scaling,
                         float lorentz_gamma, float lorentz_inv_sigma2,
                         int readout_num_layers, int readout_conv_channels,
                         const char* readout_activation, unsigned readout_seed,
                         float lr, float weight_decay,
                         size_t washout, size_t resequence_washout,
                         float kappa_start, float kappa_target, float kappa_ramp_rate,
                         float gate_threshold, float gate_err_ema_alpha) {
            EnsembleConfig cfg;
            ReservoirConfig& r = cfg.base.reservoir;
            r.dim                    = dim;
            r.spectral_radius        = spectral_radius;
            r.input_scaling          = input_scaling;
            r.leak_rate              = leak_rate;
            r.num_inputs             = num_inputs;
            r.history_depth          = history_depth;
            r.history_floor          = history_floor;
            r.num_feedback_channels  = num_outputs;   // One D, three roles
            r.feedback_scaling       = feedback_scaling;
            r.bias_scaling           = bias_scaling;
            r.lorentz_gamma          = lorentz_gamma;
            r.lorentz_inv_sigma2     = lorentz_inv_sigma2;
            // r.seed is derived per member by the EnsembleESN ctor; r.verbose is
            // forced false per member there (no per-member banners).

            ReadoutConfig& ro = cfg.base.readout;
            ro.num_outputs   = static_cast<int>(num_outputs);
            ro.task          = ReadoutTask::Regression;   // ensemble couples in output space
            ro.num_layers    = readout_num_layers;
            ro.conv_channels = readout_conv_channels;
            ro.seed          = readout_seed;
            if      (std::strcmp(readout_activation, "relu")       == 0) ro.activation = ReadoutActivation::RELU;
            else if (std::strcmp(readout_activation, "leaky_relu") == 0) ro.activation = ReadoutActivation::LEAKY_RELU;
            else if (std::strcmp(readout_activation, "none")       == 0) ro.activation = ReadoutActivation::NONE;
            else if (std::strcmp(readout_activation, "tanh")       == 0) ro.activation = ReadoutActivation::TANH;
            else throw std::invalid_argument(
                std::string("readout_activation must be one of "
                            "'tanh', 'relu', 'leaky_relu', 'none' (got '") + readout_activation + "')");

            cfg.ensemble_seed = ensemble_seed;
            cfg.num_members   = num_members;
            if      (std::strcmp(combine, "median") == 0) cfg.combine = Combine::Median;
            else if (std::strcmp(combine, "mean")   == 0) cfg.combine = Combine::Mean;
            else throw std::invalid_argument(
                std::string("combine must be 'mean' or 'median' (got '") + combine + "')");
            cfg.lr                 = lr;
            cfg.weight_decay       = weight_decay;
            cfg.washout            = washout;
            cfg.resequence_washout = resequence_washout;
            cfg.kappa_start        = kappa_start;
            cfg.kappa_target       = kappa_target;
            cfg.kappa_ramp_rate    = kappa_ramp_rate;
            cfg.gate_threshold     = gate_threshold;
            cfg.gate_err_ema_alpha = gate_err_ema_alpha;
            return std::make_unique<EnsembleESN>(cfg);
        }),
            py::arg("dim"),
            py::arg("ensemble_seed")        = 73895ULL,
            py::arg("num_members")          = 3ULL,
            py::arg("combine")              = "mean",
            py::arg("spectral_radius")      = 0.99f,
            py::arg("input_scaling")        = 0.5f,
            py::arg("leak_rate")            = 1.0f,
            py::arg("num_inputs")           = 1ULL,
            py::arg("history_depth")        = 16ULL,
            py::arg("history_floor")        = 1.0f,
            py::arg("num_outputs")          = 1ULL,
            py::arg("feedback_scaling")     = 0.5f,
            py::arg("bias_scaling")         = 0.02f,
            py::arg("lorentz_gamma")        = 1.1f,
            py::arg("lorentz_inv_sigma2")   = 250.0f,
            py::arg("readout_num_layers")   = 0,
            py::arg("readout_conv_channels")= 16,
            py::arg("readout_activation")   = "tanh",
            py::arg("readout_seed")         = 42u,
            py::arg("lr")                   = 0.01f,
            py::arg("weight_decay")         = 0.0f,
            py::arg("washout")              = 100ULL,
            py::arg("resequence_washout")   = 16ULL,
            py::arg("kappa_start")          = 0.0f,
            py::arg("kappa_target")         = 0.5f,
            py::arg("kappa_ramp_rate")      = 0.0f,
            py::arg("gate_threshold")       = 0.0f,
            py::arg("gate_err_ema_alpha")   = 0.05f)

        // ── Lockstep online step ──
        .def("step", [](EnsembleESN& self,
                        py::array_t<float, py::array::c_style | py::array::forcecast> input,
                        py::object target) {
            auto ibuf = input.request();
            if (static_cast<size_t>(ibuf.size) != self.NumInputs())
                throw std::invalid_argument(
                    "input size (" + std::to_string(ibuf.size) +
                    ") must equal num_inputs (" + std::to_string(self.NumInputs()) + ")");

            const float* tptr = nullptr;
            py::array_t<float, py::array::c_style | py::array::forcecast> tarr;
            if (!target.is_none()) {
                tarr = target.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                if (static_cast<size_t>(tarr.request().size) != self.NumOutputs())
                    throw std::invalid_argument(
                        "target size (" + std::to_string(tarr.request().size) +
                        ") must equal num_outputs (" + std::to_string(self.NumOutputs()) + ")");
                tptr = static_cast<const float*>(tarr.request().ptr);
            }

            py::array_t<float> c_out(self.NumOutputs());
            self.Step(static_cast<const float*>(ibuf.ptr), tptr, c_out.mutable_data());
            return c_out;
        },
            py::arg("input"), py::arg("target") = py::none(),
            "One lockstep online step across all members. Injects task `input`,\n"
            "(when `target` is given) trains each readout, couples members via the\n"
            "consensus deviation, and steps. Returns the consensus (num_outputs,).\n"
            "Pass target=None for inference (no readout update; kappa holds).")

        .def("begin_sequence", &EnsembleESN::BeginSequence,
             "Start a fresh sequence: reset every member's reservoir state and\n"
             "re-impose the short washout. The kappa schedule and competence\n"
             "already achieved are preserved.")

        // ── Diagnostic surface (read-only) ──
        .def("member_output", [](const EnsembleESN& self, size_t i) {
            py::array_t<float> arr(self.NumOutputs());
            self.MemberOutput(i, arr.mutable_data());
            return arr;
        }, py::arg("i"),
           "Member i's last output (num_outputs,) — the value behind the consensus.")

        .def("all_member_outputs", [](const EnsembleESN& self) {
            size_t M = self.NumMembers(), D = self.NumOutputs();
            py::array_t<float> arr({M, D});
            self.AllMemberOutputs(arr.mutable_data());
            return arr;
        }, "All members' last outputs as an (num_members, num_outputs) array.")

        // ── Properties ──
        .def_property_readonly("kappa", &EnsembleESN::Kappa)
        .def_property_readonly("gate_open", &EnsembleESN::GateOpen)
        .def_property_readonly("current_step", &EnsembleESN::CurrentStep)
        .def_property_readonly("num_members", &EnsembleESN::NumMembers)
        .def_property_readonly("num_outputs", &EnsembleESN::NumOutputs)
        .def_property_readonly("num_inputs", &EnsembleESN::NumInputs)

        // ── Persistence (opaque blob consumed by the Python wrapper) ──
        .def("_get_state", [](const EnsembleESN& self) -> py::dict {
            auto s = self.GetState();
            py::list member_weights;
            for (const auto& w : s.member_weights)
                member_weights.append(py::array_t<double>(
                    {static_cast<py::ssize_t>(w.size())}, w.data()));
            py::dict d;
            d["member_weights"] = member_weights;
            d["kappa"]          = s.kappa;
            d["gate_open"]      = s.gate_open;
            d["consensus_err"]  = s.consensus_err;
            d["err_init"]       = s.err_init;
            d["step"]           = s.step;
            return d;
        })
        .def("_set_state", [](EnsembleESN& self, py::dict d) {
            EnsembleESN::State s;
            for (auto item : d["member_weights"].cast<py::list>()) {
                auto w = item.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                s.member_weights.emplace_back(w.data(), w.data() + w.size());
            }
            s.kappa         = d["kappa"].cast<float>();
            s.gate_open     = d["gate_open"].cast<bool>();
            s.consensus_err = d["consensus_err"].cast<float>();
            s.err_init      = d["err_init"].cast<bool>();
            s.step          = d["step"].cast<size_t>();
            self.SetState(s);
        })
        ;
}
