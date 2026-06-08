#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <memory>
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
        // The readout config is consumed by train() / init_online() — no
        // per-call config overrides.
        .def(py::init([](size_t dim, uint64_t seed, float spectral_radius, float input_scaling,
                         float leak_rate, size_t num_inputs, size_t history_depth,
                         bool verbose,
                         float output_fraction,
                         int readout_num_outputs, const char* readout_task,
                         int readout_num_layers, int readout_conv_channels,
                         int readout_epochs, int readout_batch_size,
                         float readout_lr_max, float readout_lr_min_frac,
                         int readout_lr_decay_epochs, float readout_weight_decay,
                         float readout_momentum, const char* readout_activation,
                         unsigned readout_seed, bool readout_verbose,
                         bool readout_verbose_train_acc) {
            ESNConfig cfg;
            cfg.reservoir.dim              = dim;
            cfg.reservoir.seed             = seed;
            cfg.reservoir.spectral_radius  = spectral_radius;
            cfg.reservoir.input_scaling    = input_scaling;
            cfg.reservoir.leak_rate        = leak_rate;
            cfg.reservoir.num_inputs       = num_inputs;
            cfg.reservoir.history_depth    = history_depth;
            cfg.reservoir.verbose          = verbose;
            cfg.output_fraction            = output_fraction;
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
            cfg.readout.verbose            = readout_verbose;
            cfg.readout.verbose_train_acc  = readout_verbose_train_acc;
            return std::make_unique<ESN>(cfg);
        }),
            py::arg("dim"),
            py::arg("seed")                     = 73895ULL,
            py::arg("spectral_radius")          = 0.99f,
            py::arg("input_scaling")            = 0.5f,
            py::arg("leak_rate")                = 1.0f,
            py::arg("num_inputs")               = 1ULL,
            py::arg("history_depth")            = 16ULL,
            py::arg("verbose")                  = true,
            py::arg("output_fraction")          = 1.0f,
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
            py::arg("readout_seed")             = 42u,
            py::arg("readout_verbose")          = false,
            py::arg("readout_verbose_train_acc") = false)

        // ── Reservoir driving ──
        .def("warmup", [](ESN& self, py::array_t<float, py::array::c_style | py::array::forcecast> inputs) {
            auto buf = inputs.request();
            size_t total = static_cast<size_t>(buf.size);
            size_t K = self.NumInputs();
            if (total % K != 0)
                throw std::invalid_argument("Input size must be divisible by num_inputs");
            self.Warmup(static_cast<const float*>(buf.ptr), total / K);
        }, py::arg("inputs"),
           "Drive the reservoir without recording states (wash out initial transient).")

        .def("run", [](ESN& self, py::array_t<float, py::array::c_style | py::array::forcecast> inputs) {
            auto buf = inputs.request();
            size_t total = static_cast<size_t>(buf.size);
            size_t K = self.NumInputs();
            if (total % K != 0)
                throw std::invalid_argument("Input size must be divisible by num_inputs");
            self.Run(static_cast<const float*>(buf.ptr), total / K);
        }, py::arg("inputs"),
           "Drive the reservoir and record states for training/evaluation.")

        .def("clear_states", &ESN::ClearStates,
             "Clear collected states and cached features. Keeps trained readout.")

        .def("reset_reservoir_only", &ESN::ResetReservoirOnly,
             "Zero only the reservoir state; collected states preserved.")

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

        // ── Online (streaming) HCNN training ──
        .def("init_online", [](ESN& self,
                               py::array_t<float, py::array::c_style | py::array::forcecast> warmup_inputs) {
            auto buf = warmup_inputs.request();
            size_t total = static_cast<size_t>(buf.size);
            size_t K = self.NumInputs();
            if (total % K != 0)
                throw std::invalid_argument("warmup_inputs size must be divisible by num_inputs");
            self.InitOnline(static_cast<const float*>(buf.ptr), total / K);
        },
            py::arg("warmup_inputs"),
            "Initialize HCNN for online (streaming) training.\n\n"
            "Runs warmup_inputs through reservoir to reach a representative state,\n"
            "then builds CNN architecture. Uses the readout config supplied at construction.\n"
            "Call before train_live_step/train_live_batch.")

        .def("train_live_step", [](ESN& self, float target_class, float lr, float weight_decay) {
            self.TrainLiveStep(target_class, lr, weight_decay);
        },
            py::arg("target_class"), py::arg("lr"), py::arg("weight_decay") = 0.0f,
            "Single-step online classification training on the live reservoir state.")

        .def("train_live_batch", [](ESN& self,
                                    py::array_t<float, py::array::c_style | py::array::forcecast> states,
                                    py::array_t<int, py::array::c_style | py::array::forcecast> targets,
                                    float lr, float weight_decay) {
            auto sbuf = states.request();
            auto tbuf = targets.request();
            size_t count = static_cast<size_t>(tbuf.size);
            size_t M = self.NumOutputVerts();
            if (static_cast<size_t>(sbuf.size) != count * M)
                throw std::invalid_argument(
                    "states size (" + std::to_string(sbuf.size) +
                    ") must equal count * num_output_verts (" + std::to_string(count) +
                    " * " + std::to_string(M) + " = " + std::to_string(count * M) + ")");
            self.TrainLiveBatch(static_cast<const float*>(sbuf.ptr),
                                static_cast<const int*>(tbuf.ptr),
                                count, lr, weight_decay);
        },
            py::arg("states"), py::arg("targets"),
            py::arg("lr"), py::arg("weight_decay") = 0.0f,
            "Mini-batch online classification training on pre-accumulated states.\n"
            "states: (count, num_output_verts) float array from copy_live_state.\n"
            "targets: (count,) int array of class indices.")

        .def("train_live_step_regression", [](ESN& self,
                                              py::array_t<float, py::array::c_style | py::array::forcecast> target,
                                              float lr, float weight_decay) {
            auto buf = target.request();
            size_t K = self.NumOutputs();
            if (static_cast<size_t>(buf.size) != K)
                throw std::invalid_argument(
                    "target size (" + std::to_string(buf.size) +
                    ") must equal num_outputs (" + std::to_string(K) + ")");
            self.TrainLiveStepRegression(static_cast<const float*>(buf.ptr), lr, weight_decay);
        },
            py::arg("target"), py::arg("lr"), py::arg("weight_decay") = 0.0f,
            "Single-step online regression training on the live reservoir state.\n"
            "target: (num_outputs,) float array.")

        .def("train_live_batch_regression", [](ESN& self,
                                               py::array_t<float, py::array::c_style | py::array::forcecast> states,
                                               py::array_t<float, py::array::c_style | py::array::forcecast> targets,
                                               float lr, float weight_decay) {
            auto sbuf = states.request();
            auto tbuf = targets.request();
            size_t K = self.NumOutputs();
            if (static_cast<size_t>(tbuf.size) % K != 0)
                throw std::invalid_argument(
                    "targets size (" + std::to_string(tbuf.size) +
                    ") must be a multiple of num_outputs (" + std::to_string(K) + ")");
            size_t count = static_cast<size_t>(tbuf.size) / K;
            size_t M = self.NumOutputVerts();
            if (static_cast<size_t>(sbuf.size) != count * M)
                throw std::invalid_argument(
                    "states size (" + std::to_string(sbuf.size) +
                    ") must equal count * num_output_verts (" + std::to_string(count) +
                    " * " + std::to_string(M) + " = " + std::to_string(count * M) + ")");
            self.TrainLiveBatchRegression(static_cast<const float*>(sbuf.ptr),
                                          static_cast<const float*>(tbuf.ptr),
                                          count, lr, weight_decay);
        },
            py::arg("states"), py::arg("targets"),
            py::arg("lr"), py::arg("weight_decay") = 0.0f,
            "Mini-batch online regression training on pre-accumulated states.\n"
            "states: (count, num_output_verts) float array from copy_live_state.\n"
            "targets: (count, num_outputs) float array.")

        .def("copy_live_state", [](const ESN& self) {
            size_t M = self.NumOutputVerts();
            py::array_t<float> arr(M);
            self.CopyLiveState(arr.mutable_data());
            return arr;
        }, "Copy the current subsampled reservoir state for external accumulation.\n"
           "Returns a (num_output_verts,) float array.")

        // ── Prediction & evaluation ──
        .def("predict_raw", [](const ESN& self, size_t timestep) {
            if (timestep >= self.NumCollected())
                throw std::out_of_range(
                    "timestep (" + std::to_string(timestep) +
                    ") >= num_collected (" + std::to_string(self.NumCollected()) + ")");
            return self.PredictRaw(timestep);
        }, py::arg("timestep"),
           "Return the raw continuous prediction for a collected timestep.")

        .def("predict_live_raw", [](const ESN& self) {
            return self.PredictLiveRaw();
        }, "Predict from the reservoir's current live state (no cached states needed).\n"
           "For autoregressive / streaming inference loops.")

        .def("predict_live_raw_multi", [](const ESN& self) {
            size_t K = self.NumOutputs();
            py::array_t<float> arr(K);
            self.PredictLiveRaw(arr.mutable_data());
            return arr;
        }, "Multi-output live predict: returns (num_outputs,) float array.")

        .def("predict_raw_multi", [](const ESN& self, size_t timestep) {
            if (timestep >= self.NumCollected())
                throw std::out_of_range(
                    "timestep (" + std::to_string(timestep) +
                    ") >= num_collected (" + std::to_string(self.NumCollected()) + ")");
            size_t K = self.NumOutputs();
            py::array_t<float> arr(K);
            self.PredictRaw(timestep, arr.mutable_data());
            return arr;
        }, py::arg("timestep"),
           "Multi-output prediction for a collected timestep: returns (num_outputs,) array.")

        .def("predictions_multi", [](const ESN& self) {
            size_t T = self.NumCollected();
            size_t K = self.NumOutputs();
            py::array_t<float> arr({T, K});
            float* ptr = arr.mutable_data();
            for (size_t t = 0; t < T; ++t)
                self.PredictRaw(t, ptr + t * K);
            return arr;
        }, "Multi-output predictions for all collected timesteps: (num_collected, num_outputs) array.")

        .def("predict_from_state", [](const ESN& self,
                                      py::array_t<float, py::array::c_style | py::array::forcecast> state) {
            auto buf = state.request();
            size_t M = self.NumOutputVerts();
            if (static_cast<size_t>(buf.size) != M)
                throw std::invalid_argument(
                    "state size (" + std::to_string(buf.size) +
                    ") must equal num_output_verts (" + std::to_string(M) + ")");
            size_t K = self.NumOutputs();
            py::array_t<float> arr(K);
            self.PredictFromState(static_cast<const float*>(buf.ptr), arr.mutable_data());
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
        .def("selected_states", [](const ESN& self) {
            auto vec = self.SelectedStates();
            size_t M = self.NumOutputVerts();
            size_t T = self.NumCollected();
            py::array_t<float> arr({T, M});
            memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(float));
            return arr;
        }, "Return stride-selected states as a (num_collected, M) array.")

        .def("predictions", [](const ESN& self) {
            size_t T = self.NumCollected();
            py::array_t<float> arr(T);
            float* ptr = arr.mutable_data();
            for (size_t t = 0; t < T; ++t)
                ptr[t] = self.PredictRaw(t);
            return arr;
        }, "Return predictions for all collected timesteps as a 1D array.")

        // ── Properties ──
        .def_property_readonly("num_collected", &ESN::NumCollected)
        .def_property_readonly("num_outputs", &ESN::NumOutputs)
        .def_property_readonly("output_fraction", [](const ESN& self) { return self.GetConfig().output_fraction; })
        .def_property_readonly("num_output_verts", &ESN::NumOutputVerts)
        .def_property_readonly("dim", &ESN::Dim)
        .def_property_readonly("N", &ESN::Size)
        .def_property_readonly("num_inputs", &ESN::NumInputs)
        .def_property_readonly("history_depth", [](const ESN& self) { return self.GetConfig().reservoir.history_depth; })
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
}
