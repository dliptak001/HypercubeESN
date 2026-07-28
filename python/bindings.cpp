#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <memory>
#include <span>
#include "../ESN.h"

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
        .def(py::init([](size_t reservoir_hypercube_dimension, uint64_t seed, float spectral_radius, float input_scaling,
                         float leak_rate, size_t num_inputs, size_t history_depth,
                         bool verbose,
                         size_t num_external_feedback_channels,
                         float external_feedback_scaling,
                         float bias_scaling,
                         size_t readout_slices,
                         int readout_num_outputs, const char* readout_task,
                         int readout_num_layers, int readout_conv_channels,
                         int readout_epochs, int readout_batch_size,
                         float readout_lr_max, float readout_lr_min_frac,
                         int readout_lr_decay_epochs, float readout_weight_decay,
                         float readout_momentum, const char* readout_activation,
                         unsigned readout_seed,
                         size_t readout_num_threads,
                         bool readout_restore_best_epoch,
                         float readout_best_epoch_holdout_frac) {
            ESNConfig cfg;
            cfg.reservoir.dim              = reservoir_hypercube_dimension;
            cfg.reservoir.seed             = seed;
            cfg.reservoir.spectral_radius  = spectral_radius;
            cfg.reservoir.input_scaling    = input_scaling;
            cfg.reservoir.leak_rate        = leak_rate;
            cfg.reservoir.num_inputs       = num_inputs;
            cfg.reservoir.history_depth    = history_depth;
            cfg.reservoir.verbose          = verbose;
            cfg.reservoir.num_external_feedback_channels = num_external_feedback_channels;
            cfg.reservoir.external_feedback_scaling = external_feedback_scaling;
            cfg.reservoir.bias_scaling     = bias_scaling;
            cfg.readout_slices             = readout_slices;
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
            cfg.readout.num_threads        = readout_num_threads;
            cfg.readout.restore_best_epoch = readout_restore_best_epoch;
            cfg.readout.best_epoch_holdout_frac = readout_best_epoch_holdout_frac;
            return std::make_unique<ESN>(cfg);
        }),
            py::arg("reservoir_hypercube_dimension"),
            py::arg("seed")                     = 73895ULL,
            py::arg("spectral_radius")          = 0.99f,
            py::arg("input_scaling")            = 0.5f,
            py::arg("leak_rate")                = 1.0f,
            py::arg("num_inputs")               = 1ULL,
            py::arg("history_depth")            = 16ULL,
            py::arg("verbose")                  = false,
            py::arg("num_external_feedback_channels") = 0ULL,
            py::arg("external_feedback_scaling") = 0.5f,
            py::arg("bias_scaling")             = 0.02f,
            py::arg("readout_slices")           = 1ULL,
            py::arg("readout_num_outputs")      = 1,
            py::arg("readout_task")             = "regression",
            py::arg("readout_num_layers")       = 1,
            py::arg("readout_conv_channels")    = 16,
            py::arg("readout_epochs")           = 200,
            py::arg("readout_batch_size")       = 32,
            py::arg("readout_lr_max")           = 0.0015f,
            py::arg("readout_lr_min_frac")      = 0.01f,
            py::arg("readout_lr_decay_epochs")  = 0,
            py::arg("readout_weight_decay")     = 0.0f,
            py::arg("readout_momentum")         = 0.0f,
            py::arg("readout_activation")       = "tanh",
            py::arg("readout_seed")             = 42u,
            py::arg("readout_num_threads")      = 0ULL,
            py::arg("readout_restore_best_epoch") = true,
            py::arg("readout_best_epoch_holdout_frac") = 0.0f)

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

        .def("reservoir_step", [](ESN& self,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> inputs,
                                  py::object external_feedback) {
            auto ibuf = inputs.request();
            const size_t K = self.NumInputs();
            if (static_cast<size_t>(ibuf.size) != K)
                throw std::invalid_argument(
                    "inputs size (" + std::to_string(ibuf.size) +
                    ") must equal num_inputs (" + std::to_string(K) + ") for one step");
            if (external_feedback.is_none()) {
                self.ReservoirStep(static_cast<const float*>(ibuf.ptr), nullptr);
                return;
            }
            auto fb = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(external_feedback);
            if (!fb)
                throw std::invalid_argument("external_feedback must be a float array or None");
            auto fbuf = fb.request();
            const size_t D = self.NumExternalFeedbackChannels();
            if (D == 0)
                throw std::invalid_argument(
                    "external_feedback provided but num_external_feedback_channels == 0");
            if (static_cast<size_t>(fbuf.size) != D)
                throw std::invalid_argument(
                    "external_feedback size (" + std::to_string(fbuf.size) +
                    ") must equal num_external_feedback_channels (" + std::to_string(D) + ")");
            self.ReservoirStep(static_cast<const float*>(ibuf.ptr),
                               static_cast<const float*>(fbuf.ptr));
        }, py::arg("inputs"), py::arg("external_feedback") = py::none(),
           "One reservoir step. inputs: (num_inputs,) float array.\n"
           "external_feedback: optional (num_external_feedback_channels,) array for closed loop.")

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

            if (train_size > self.NumCollectedStates())
                throw std::invalid_argument(
                    "train_size (" + std::to_string(train_size) +
                    ") exceeds num_collected_states (" + std::to_string(self.NumCollectedStates()) + ")");

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
            // Rows are full readout inputs, not bare reservoir states (they coincide
            // only for a single-slice readout). Validate the width actually read.
            const size_t M = self.ReadoutInputWidth();
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
                    ") must equal count * readout_input_width (" + std::to_string(count) +
                    " * " + std::to_string(M) + " = " + std::to_string(count * M) + ")");
            self.TrainStepBatch(static_cast<const float*>(sbuf.ptr),
                                static_cast<const float*>(tbuf.ptr),
                                count, lr, weight_decay);
        },
            py::arg("states"), py::arg("targets"),
            py::arg("lr"), py::arg("weight_decay") = 0.0f,
            "One streaming gradient step over a mini-batch of pre-accumulated states.\n"
            "states: (count, reservoir_neuron_count) float array from copy_reservoir_state.\n"
            "targets: regression -> (count, num_outputs); classification -> (count,) class indices.")

        .def("copy_reservoir_state", [](const ESN& self) {
            size_t M = self.ReservoirNeuronCount();
            py::array_t<float> arr(M);
            self.CopyReservoirState(arr.mutable_data());
            return arr;
        }, "Copy the current reservoir state (the newest delay-line slice).\n"
           "Returns a (reservoir_neuron_count,) float array. To accumulate rows for\n"
           "train_step_batch, use copy_readout_input instead.")

        .def("copy_readout_input", [](const ESN& self) {
            size_t W = self.ReadoutInputWidth();
            py::array_t<float> arr(W);
            self.CopyReadoutInput(arr.mutable_data());
            return arr;
        }, "Copy the readout's current input — the block-structured vector it actually\n"
           "consumes. Returns a (readout_input_width,) float array; this is the row\n"
           "shape train_step_batch and predict_from_state expect.")

        // ── Prediction & evaluation ──
        .def("predict", [](const ESN& self) {
            auto v = self.Predict();
            py::array_t<float> arr(static_cast<py::ssize_t>(v.size()));
            memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
            return arr;
        }, "Predict from the reservoir's current state.\n"
           "Returns (num_outputs,) float array. For autoregressive / streaming loops.")

        .def("predict_from_recorded", [](const ESN& self, size_t timestep) {
            if (timestep >= self.NumCollectedStates())
                throw std::out_of_range(
                    "timestep (" + std::to_string(timestep) +
                    ") >= num_collected_states (" + std::to_string(self.NumCollectedStates()) + ")");
            auto v = self.PredictFromRecorded(timestep);
            py::array_t<float> arr(static_cast<py::ssize_t>(v.size()));
            memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
            return arr;
        }, py::arg("timestep"),
           "Predict from a recorded timestep: returns (num_outputs,) float array.")

        .def("predict_from_state", [](const ESN& self,
                                      py::array_t<float, py::array::c_style | py::array::forcecast> state) {
            auto buf = state.request();
            // The readout consumes its full block-structured input, not just one
            // reservoir state — these coincide only when the readout was built with a
            // single-slice readout. Validate against the width actually read.
            size_t W = self.ReadoutInputWidth();
            if (static_cast<size_t>(buf.size) != W)
                throw std::invalid_argument(
                    "state size (" + std::to_string(buf.size) +
                    ") must equal readout_input_width (" + std::to_string(W) + ")");
            auto v = self.PredictFromState(static_cast<const float*>(buf.ptr));
            py::array_t<float> arr(static_cast<py::ssize_t>(v.size()));
            memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
            return arr;
        }, py::arg("state"),
           "Run the readout on a caller-supplied (readout_input_width,) input "
           "(B·N floats; equals reservoir state only when B=1).\n"
           "Returns (num_outputs,) float array. Alias: predict_from_readout_input.")

        .def("predict_from_readout_input", [](const ESN& self,
                                      py::array_t<float, py::array::c_style | py::array::forcecast> state) {
            auto buf = state.request();
            size_t W = self.ReadoutInputWidth();
            if (static_cast<size_t>(buf.size) != W)
                throw std::invalid_argument(
                    "readout_input size (" + std::to_string(buf.size) +
                    ") must equal readout_input_width (" + std::to_string(W) + ")");
            auto v = self.PredictFromState(static_cast<const float*>(buf.ptr));
            py::array_t<float> arr(static_cast<py::ssize_t>(v.size()));
            memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
            return arr;
        }, py::arg("readout_input"),
           "Same as predict_from_state: HCNN forward on a (readout_input_width,) vector.")

        .def("r2", [](const ESN& self,
                      py::array_t<float, py::array::c_style | py::array::forcecast> targets,
                      size_t start, size_t count) {
            auto buf = targets.request();
            std::span<const float> sp(static_cast<const float*>(buf.ptr),
                                      static_cast<size_t>(buf.size));
            return self.R2(sp, start, count);
        }, py::arg("targets"), py::arg("start"), py::arg("count"),
           "R² on recorded [start, start+count). targets must cover [0, start+count) "
           "as (start+count)*num_outputs floats (not a window slice).")

        .def("nrmse", [](const ESN& self,
                         py::array_t<float, py::array::c_style | py::array::forcecast> targets,
                         size_t start, size_t count) {
            auto buf = targets.request();
            std::span<const float> sp(static_cast<const float*>(buf.ptr),
                                      static_cast<size_t>(buf.size));
            return self.NRMSE(sp, start, count);
        }, py::arg("targets"), py::arg("start"), py::arg("count"),
           "NRMSE on recorded [start, start+count). Same full-buffer contract as r2().")

        .def("accuracy", [](const ESN& self,
                            py::array_t<float, py::array::c_style | py::array::forcecast> labels,
                            size_t start, size_t count) {
            auto buf = labels.request();
            std::span<const float> sp(static_cast<const float*>(buf.ptr),
                                      static_cast<size_t>(buf.size));
            return self.Accuracy(sp, start, count);
        }, py::arg("labels"), py::arg("start"), py::arg("count"),
           "Classification accuracy on recorded [start, start+count). "
           "labels must cover [0, start+count).")

        // ── State access ──
        .def("collected_states", [](const ESN& self) {
            auto vec = self.CollectedStates();
            // Rows are readout inputs (readout_input_width wide), which equals the
            // neuron count only for a single-slice readout. Sizing the array
            // by the neuron count would overflow the memcpy below once they diverge.
            size_t M = self.ReadoutInputWidth();
            size_t T = self.NumCollectedStates();
            py::array_t<float> arr({T, M});
            memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(float));
            return arr;
        }, "Return collected readout inputs as a (num_collected_states, readout_input_width) array.")

        .def("predictions", [](const ESN& self) {
            size_t T = self.NumCollectedStates();
            size_t K = self.NumOutputs();
            py::array_t<float> arr({T, K});
            float* ptr = arr.mutable_data();
            for (size_t t = 0; t < T; ++t) {
                auto v = self.PredictFromRecorded(t);
                memcpy(ptr + t * K, v.data(), K * sizeof(float));
            }
            return arr;
        }, "Predictions for all collected timesteps: (num_collected_states, num_outputs) array.")

        // ── Properties ──
        .def_property_readonly("num_collected_states", &ESN::NumCollectedStates)
        .def_property_readonly("num_outputs", &ESN::NumOutputs)
        .def_property_readonly("reservoir_hypercube_dimension", &ESN::ReservoirHypercubeDimension)
        .def_property_readonly("dim", &ESN::Dim)
        .def_property_readonly("reservoir_neuron_count", &ESN::ReservoirNeuronCount)
        .def_property_readonly("readout_input_width", &ESN::ReadoutInputWidth)
        .def_property_readonly("readout_block_count", &ESN::ReadoutBlockCount)
        .def_property_readonly("readout_slices", &ESN::ReadoutBlockCount)
        .def_property_readonly("num_inputs", &ESN::NumInputs)
        .def_property_readonly("num_external_feedback_channels", &ESN::NumExternalFeedbackChannels)
        .def_property_readonly("history_depth", [](const ESN& self) { return self.GetConfig().reservoir.history_depth; })
        .def_property_readonly("seed", [](const ESN& self) { return self.GetConfig().reservoir.seed; })
        .def_property_readonly("spectral_radius", [](const ESN& self) { return self.GetConfig().reservoir.spectral_radius; })
        .def_property_readonly("target_spectral_radius", &ESN::TargetSpectralRadius)
        .def_property_readonly("realized_spectral_radius", &ESN::RealizedSpectralRadius)
        .def_property_readonly("leak_rate", [](const ESN& self) { return self.GetConfig().reservoir.leak_rate; })
        .def_property_readonly("input_scaling", [](const ESN& self) { return self.GetConfig().reservoir.input_scaling; })
        .def_property_readonly("external_feedback_scaling",
            [](const ESN& self) { return self.GetConfig().reservoir.external_feedback_scaling; })
        .def_property_readonly("bias_scaling",
            [](const ESN& self) { return self.GetConfig().reservoir.bias_scaling; })

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
            ReadoutLoadMode mode = ReadoutLoadMode::Eval;
            if (d.contains("mode")) {
                const auto m = d["mode"].cast<std::string>();
                if (m == "resume_train" || m == "ResumeTrain")
                    mode = ReadoutLoadMode::ResumeTrain;
            }
            self.SetReadoutState(s, mode);
        })
        .def("save_readout_hcnn_model",
             &ESN::SaveReadoutHcnnModel,
             py::arg("path_stem"),
             "Write portable stem.hcnw + stem.arch.json for the HCNN readout.")
        .def("load_readout_hcnn_model",
             [](ESN& self, const std::string& path_stem, const std::string& mode) {
                 ReadoutLoadMode m = ReadoutLoadMode::Eval;
                 if (mode == "resume_train" || mode == "ResumeTrain")
                     m = ReadoutLoadMode::ResumeTrain;
                 self.LoadReadoutHcnnModel(path_stem, m);
             },
             py::arg("path_stem"),
             py::arg("mode") = "eval",
             "Load stem.hcnw (+ arch sidecar) into the live readout. mode: 'eval' or 'resume_train'.")
        .def("readout_arch_summary",
             &ESN::ReadoutArchSummary,
             "Human-readable HCNN readout architecture and parameter counts.")
        .def_property_readonly("readout_best_epoch",
             &ESN::ReadoutBestEpoch,
             "1-based best epoch after restore_best_epoch training, else 0.")
        ;
}
