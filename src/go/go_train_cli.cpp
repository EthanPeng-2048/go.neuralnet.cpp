#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include "external/nlohmann_json/json.hpp"
#include "go/feature_cache.hpp"

namespace {

using json = nlohmann::json;
using nn::Scalar;

struct CommonOptions {
    bool use_gpu = false;
    std::string gpu_device;
    std::size_t batch_size = 64;
    std::size_t max_samples = 0;
};

struct TrainOptions {
    CommonOptions common;
    std::string dataset;
    std::string eval_dataset;
    std::string checkpoint = "artifacts/m4/c-validator.nnckpt";
    std::string model = "artifacts/m4/c-validator.nnmodel";
    std::string metrics_out = "artifacts/m4/c-validator.json";
    int epochs = 4;
    Scalar lr = 1.0e-3F;
    Scalar min_lr = 1.0e-5F;
    Scalar weight_decay = 1.0e-4F;
    std::uint64_t seed = 20261024;
};

struct EvalOptions {
    CommonOptions common;
    std::string dataset;
    std::string model;
};

struct EvalMetrics {
    Scalar loss = 0;
    double top1 = 0;
    std::size_t samples = 0;
    double seconds = 0;
};

struct LoadedBatch {
    std::vector<Scalar> planes;
    std::vector<std::size_t> labels;
};

nn::Tensor forward_evaluated(nn::Model& model, const nn::Tensor& input) {
    nn::ComputeEngine& engine = model.engine();
    auto begin = engine.begin_batch();
    if (!begin) {
        throw std::runtime_error("cannot begin evaluation batch: " + begin.error().message);
    }
    auto output = model.forward(input);
    auto end = engine.end_batch();
    if (!output) {
        throw std::runtime_error("evaluation forward failed: " + output.error().message);
    }
    if (!end) {
        throw std::runtime_error("cannot end evaluation batch: " + end.error().message);
    }
    return std::move(*output);
}

std::string option_value(int& index, int argc, char** argv) {
    if (++index >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index - 1]);
    }
    return argv[index];
}

void parse_common_option(
    std::string_view arg,
    int& index,
    int argc,
    char** argv,
    CommonOptions& common
) {
    if (arg == "--gpu") {
        common.use_gpu = true;
    } else if (arg.starts_with("--gpu=")) {
        common.use_gpu = true;
        common.gpu_device = std::string(arg.substr(6));
    } else if (arg == "--cpu") {
        common.use_gpu = false;
        common.gpu_device.clear();
    } else if (arg == "--batch-size") {
        common.batch_size = std::stoull(option_value(index, argc, argv));
    } else if (arg == "--max-samples") {
        common.max_samples = std::stoull(option_value(index, argc, argv));
    } else {
        throw std::invalid_argument("not a common option: " + std::string(arg));
    }
}

bool is_common_option(std::string_view arg) {
    return arg == "--gpu" || arg.starts_with("--gpu=") || arg == "--cpu" ||
           arg == "--batch-size" || arg == "--max-samples";
}

TrainOptions parse_train_options(int argc, char** argv) {
    TrainOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (is_common_option(arg)) {
            parse_common_option(arg, i, argc, argv, options.common);
        } else if (arg == "--dataset") {
            options.dataset = option_value(i, argc, argv);
        } else if (arg == "--eval-dataset") {
            options.eval_dataset = option_value(i, argc, argv);
        } else if (arg == "--checkpoint") {
            options.checkpoint = option_value(i, argc, argv);
        } else if (arg == "--model") {
            options.model = option_value(i, argc, argv);
        } else if (arg == "--metrics-out") {
            options.metrics_out = option_value(i, argc, argv);
        } else if (arg == "--epochs") {
            options.epochs = std::stoi(option_value(i, argc, argv));
        } else if (arg == "--lr") {
            options.lr = std::stof(option_value(i, argc, argv));
        } else if (arg == "--min-lr") {
            options.min_lr = std::stof(option_value(i, argc, argv));
        } else if (arg == "--weight-decay") {
            options.weight_decay = std::stof(option_value(i, argc, argv));
        } else if (arg == "--seed") {
            options.seed = std::stoull(option_value(i, argc, argv));
        } else {
            throw std::invalid_argument("unknown train option: " + std::string(arg));
        }
    }
    if (options.dataset.empty() || options.eval_dataset.empty()) {
        throw std::invalid_argument("train requires --dataset and --eval-dataset");
    }
    if (options.epochs <= 0 || options.common.batch_size == 0) {
        throw std::invalid_argument("epochs and batch-size must be positive");
    }
    if (options.lr <= 0 || options.min_lr < 0 || options.min_lr > options.lr) {
        throw std::invalid_argument("invalid lr/min-lr");
    }
    return options;
}

EvalOptions parse_eval_options(int argc, char** argv) {
    EvalOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (is_common_option(arg)) {
            parse_common_option(arg, i, argc, argv, options.common);
        } else if (arg == "--dataset") {
            options.dataset = option_value(i, argc, argv);
        } else if (arg == "--model") {
            options.model = option_value(i, argc, argv);
        } else {
            throw std::invalid_argument("unknown evaluate option: " + std::string(arg));
        }
    }
    if (options.dataset.empty() || options.model.empty()) {
        throw std::invalid_argument("evaluate requires --dataset and --model");
    }
    if (options.common.batch_size == 0) {
        throw std::invalid_argument("batch-size must be positive");
    }
    return options;
}

std::shared_ptr<nn::ComputeEngine> create_engine(const CommonOptions& options) {
    nn::cli::EngineConfig config;
    config.use_gpu = options.use_gpu;
    config.gpu_device = options.gpu_device;
    auto result = nn::cli::create_engine(config, std::cerr);
    if (!result) {
        throw std::runtime_error(result.error().message);
    }
    return std::move(*result);
}

nn::CnnConfig c_validator_config() {
    nn::CnnConfig config;
    config.in_channels = go::feature_plane_count;
    config.in_size = go::training_board_size;
    // Keep the validator deliberately small: two 19x19 3x3 convolutions,
    // 2x2 pooling after each, and the stock FC head. Normalization remains off
    // because this model is an integration probe, not an architecture search.
    config.pool = 2;
    config.convs = {
        {.out_channels = 32, .kernel = 3, .stride = 1, .padding = 1},
        {.out_channels = 64, .kernel = 3, .stride = 1, .padding = 1},
    };
    config.fc_dims = {256, go::policy_class_count};
    config.norm_type = nn::NormType::BatchNorm;
    config.norm_place = nn::NormPlace::None;
    return config;
}

nn::ModelSpec c_validator_spec(const nn::CnnConfig& config) {
    return nn::make_cnn_spec(
        config.in_channels,
        config.in_size,
        config.pool,
        config.convs,
        config.fc_dims,
        config.norm_type,
        config.norm_place
    );
}

std::vector<std::size_t> sample_indices(
    const go::FeatureCache& cache,
    std::size_t maximum
) {
    const std::size_t count =
        maximum == 0 ? cache.size() : std::min(maximum, cache.size());
    std::vector<std::size_t> indices(count);
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    return indices;
}

LoadedBatch load_batch_planes(
    go::FeatureCache& cache,
    const std::vector<std::size_t>& indices,
    std::size_t first,
    std::size_t count
) {
    LoadedBatch result{
        std::vector<Scalar>(cache.feature_count() * count, Scalar{0}),
        std::vector<std::size_t>(count),
    };
    std::vector<float> sample(cache.feature_count());
    for (std::size_t column = 0; column < count; ++column) {
        go::FeatureSampleMeta metadata;
        cache.read_sample(indices[first + column], sample, &metadata);
        result.labels[column] = metadata.target;
        for (std::size_t feature = 0; feature < sample.size(); ++feature) {
            result.planes[feature * count + column] = sample[feature];
        }
    }
    return result;
}

EvalMetrics evaluate(
    nn::Model& model,
    go::FeatureCache& cache,
    std::size_t batch_size,
    std::size_t maximum
) {
    const auto started = std::chrono::steady_clock::now();
    const std::vector<std::size_t> indices = sample_indices(cache, maximum);
    if (indices.empty()) {
        throw std::runtime_error("evaluation cache has no samples");
    }
    nn::ComputeEngine& engine = model.engine();
    nn::CrossEntropyLoss loss_fn;
    model.set_training(false);

    Scalar loss_sum = 0;
    std::size_t correct = 0;
    std::vector<Scalar> logits;
    for (std::size_t first = 0; first < indices.size(); first += batch_size) {
        const std::size_t count = std::min(batch_size, indices.size() - first);
        LoadedBatch batch = load_batch_planes(cache, indices, first, count);
        nn::Tensor input = engine.create_tensor(cache.feature_count(), count);
        if (!input.valid()) {
            throw std::runtime_error("cannot allocate evaluation input tensor");
        }
        NN_UNWRAP(engine.write(input, std::span<const Scalar>(batch.planes)));
        nn::Tensor output = forward_evaluated(model, input);
        const std::vector<std::size_t>& labels = batch.labels;
        const Scalar loss = NN_UNWRAP(
            loss_fn.forward_sparse(engine, output, labels, {}, go::policy_class_count)
        );
        loss_sum += loss * static_cast<Scalar>(count);

        logits.resize(output.size());
        NN_UNWRAP(engine.read(output, std::span<Scalar>(logits)));
        for (std::size_t column = 0; column < count; ++column) {
            std::size_t prediction = 0;
            Scalar best = logits[column];
            for (std::size_t row = 1; row < go::policy_class_count; ++row) {
                const Scalar value = logits[row * count + column];
                if (value > best) {
                    best = value;
                    prediction = row;
                }
            }
            if (prediction == labels[column]) {
                ++correct;
            }
        }
    }

    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    return {
        loss_sum / static_cast<Scalar>(indices.size()),
        static_cast<double>(correct) / static_cast<double>(indices.size()),
        indices.size(),
        elapsed,
    };
}

json metrics_json(const EvalMetrics& metrics) {
    return {
        {"loss", static_cast<double>(metrics.loss)},
        {"top1", metrics.top1},
        {"samples", metrics.samples},
        {"seconds", metrics.seconds},
    };
}

double max_logit_difference(
    nn::Model& left,
    nn::Model& right,
    go::FeatureCache& cache,
    std::size_t batch_size
) {
    if (left.engine().device() != right.engine().device()) {
        throw std::runtime_error("round-trip models use different devices");
    }
    const std::size_t count = std::min(batch_size, cache.size());
    std::vector<std::size_t> indices = sample_indices(cache, count);
    LoadedBatch batch = load_batch_planes(cache, indices, 0, count);
    nn::ComputeEngine& engine = left.engine();
    nn::Tensor input = engine.create_tensor(cache.feature_count(), count);
    if (!input.valid()) {
        throw std::runtime_error("cannot allocate round-trip input tensor");
    }
    NN_UNWRAP(engine.write(input, std::span<const Scalar>(batch.planes)));
    left.set_training(false);
    right.set_training(false);
    nn::Tensor left_out = forward_evaluated(left, input);
    nn::Tensor right_out = forward_evaluated(right, input);
    std::vector<Scalar> left_values(left_out.size());
    std::vector<Scalar> right_values(right_out.size());
    NN_UNWRAP(engine.read(left_out, std::span<Scalar>(left_values)));
    NN_UNWRAP(engine.read(right_out, std::span<Scalar>(right_values)));
    double max_difference = 0;
    for (std::size_t i = 0; i < left_values.size(); ++i) {
        max_difference = std::max(
            max_difference,
            std::abs(static_cast<double>(left_values[i] - right_values[i]))
        );
    }
    return max_difference;
}

void write_json(const std::filesystem::path& path, const json& value) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open metrics output: " + path.string());
    }
    output << value.dump(2) << '\n';
}

void load_weights(const std::string& path, nn::Model& model) {
    std::cerr << "verifying model file: " << path << '\n';
    auto verified = nn::verify_model_integrity(path);
    if (!verified) {
        throw std::runtime_error(
            "model integrity check failed for '" + path + "': " +
            verified.error().message
        );
    }
    std::cerr << "model file verified\n";
    auto result = nn::load_model(path, model);
    if (!result) {
        throw std::runtime_error(
            "cannot load model '" + path + "': " + result.error().message
        );
    }
}

int inspect_cache(const std::string& path) {
    go::FeatureCache cache = go::FeatureCache::open(path);
    json result{
        {"dataset", cache.path().string()},
        {"format", "go.feature-cache"},
        {"format_version", go::feature_cache_version},
        {"records", cache.size()},
        {"features", cache.feature_count()},
        {"board_size", cache.board_size()},
        {"first", {
            {"sample_id", cache.meta(0).sample_id},
            {"game_index", cache.meta(0).game_index},
            {"move_number", cache.meta(0).move_number},
            {"target", cache.meta(0).target},
            {"value", cache.meta(0).value},
        }},
    };
    std::cout << result.dump(2) << '\n';
    return 0;
}

int train(const TrainOptions& options) {
    go::FeatureCache train_cache = go::FeatureCache::open(options.dataset);
    go::FeatureCache eval_cache = go::FeatureCache::open(options.eval_dataset);
    std::shared_ptr<nn::ComputeEngine> engine_ptr = create_engine(options.common);
    nn::ComputeEngine& engine = *engine_ptr;

    const nn::CnnConfig model_config = c_validator_config();
    const nn::ModelSpec spec = c_validator_spec(model_config);
    nn::Model model = NN_UNWRAP(nn::build_cnn_model(engine, model_config));
    model.set_spec(spec);
    model.set_training(true);

    const EvalMetrics initial_eval = evaluate(
        model, eval_cache, options.common.batch_size, 0
    );
    model.set_training(true);

    nn::TrainingConfig training_config;
    training_config.optimizer = "adamw";
    training_config.lr = options.lr;
    training_config.weight_decay = options.weight_decay;
    training_config.clip_grad_norm = 1.0F;
    training_config.grad_accum_steps = 1;
    nn::TrainingSession session = NN_UNWRAP(
        nn::TrainingSession::create(std::move(model), training_config)
    );
    nn::CrossEntropyLoss loss_fn;

    std::vector<std::size_t> indices = sample_indices(
        train_cache, options.common.max_samples
    );
    if (indices.empty()) {
        throw std::runtime_error("training cache has no samples");
    }
    std::mt19937_64 rng(options.seed);
    nn::LrScheduleConfig lr_config;
    lr_config.base_lr = options.lr;
    lr_config.min_lr = options.min_lr;
    lr_config.total_epochs = options.epochs;
    lr_config.schedule = "cosine";

    const auto training_started = std::chrono::steady_clock::now();
    json history = json::array();
    EvalMetrics final_eval = initial_eval;
    Scalar first_train_loss = 0;
    Scalar final_train_loss = 0;
    for (int epoch = 0; epoch < options.epochs; ++epoch) {
        const Scalar epoch_lr = nn::compute_epoch_lr(lr_config, epoch);
        session.set_lr(epoch_lr);
        std::shuffle(indices.begin(), indices.end(), rng);
        const auto epoch_started = std::chrono::steady_clock::now();
        Scalar loss_sum = 0;
        std::size_t seen = 0;

        for (std::size_t first = 0; first < indices.size();
             first += options.common.batch_size) {
            const std::size_t count =
                std::min(options.common.batch_size, indices.size() - first);
            LoadedBatch batch = load_batch_planes(
                train_cache, indices, first, count
            );
            nn::Tensor input = engine.create_tensor(train_cache.feature_count(), count);
            if (!input.valid()) {
                throw std::runtime_error("cannot allocate training input tensor");
            }
            NN_UNWRAP(engine.write(input, std::span<const Scalar>(batch.planes)));
            const std::vector<std::size_t>& labels = batch.labels;

            Scalar batch_loss = 0;
            nn::TrainingStepOptions step_options;
            step_options.lr = epoch_lr;
            step_options.clip_grad_norm = 1.0F;
            step_options.normalization_units = count;
            auto step_report = NN_UNWRAP(session.train_step(
                input,
                step_options,
                [&](nn::Model& current, nn::ComputeEngine& current_engine,
                    const nn::Tensor& batch_input) -> nn::Result<nn::TrainingStepOutput> {
                    auto output = current.forward(batch_input);
                    NN_TRY_CHECK(output);
                    auto loss = loss_fn.forward_sparse(
                        current_engine,
                        *output,
                        labels,
                        {},
                        go::policy_class_count
                    );
                    NN_TRY_CHECK(loss);
                    batch_loss = *loss;
                    auto gradient = loss_fn.backward();
                    NN_TRY_CHECK(gradient);
                    nn::TrainingStepOutput result;
                    result.grad = std::move(*gradient);
                    result.gradient_units = count;
                    return result;
                }
            ));
            (void)step_report;
            loss_sum += batch_loss * static_cast<Scalar>(count);
            seen += count;
        }

        const Scalar mean_loss = loss_sum / static_cast<Scalar>(seen);
        if (epoch == 0) {
            first_train_loss = mean_loss;
        }
        final_train_loss = mean_loss;
        const EvalMetrics eval = evaluate(
            session.model(), eval_cache, options.common.batch_size, 0
        );
        final_eval = eval;
        session.model().set_training(true);
        const double epoch_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - epoch_started
        ).count();
        history.push_back({
            {"epoch", epoch + 1},
            {"lr", static_cast<double>(epoch_lr)},
            {"train_loss", static_cast<double>(mean_loss)},
            {"eval", metrics_json(eval)},
            {"seconds", epoch_seconds},
        });
        std::cerr << "epoch " << (epoch + 1) << '/' << options.epochs
                  << " train_loss=" << static_cast<double>(mean_loss)
                  << " eval_loss=" << static_cast<double>(eval.loss)
                  << " top1=" << eval.top1
                  << " seconds=" << epoch_seconds << '\n';
    }

    session.model().set_training(false);
    std::filesystem::create_directories(
        std::filesystem::path(options.checkpoint).parent_path()
    );
    std::filesystem::create_directories(
        std::filesystem::path(options.model).parent_path()
    );
    NN_UNWRAP(nn::save_model(
        options.checkpoint,
        session.model(),
        spec,
        {},
        nn::ModelRole::Checkpoint
    ));
    NN_UNWRAP(nn::save_model(options.model, session.model(), spec));

    const nn::ModelSpec loaded_checkpoint_spec =
        NN_UNWRAP(nn::peek_any_model_spec(options.checkpoint));
    nn::Model checkpoint_model =
        NN_UNWRAP(nn::build_cnn_model_from_spec(engine, loaded_checkpoint_spec));
    load_weights(options.checkpoint, checkpoint_model);
    const double checkpoint_diff = max_logit_difference(
        session.model(), checkpoint_model, eval_cache, options.common.batch_size
    );

    const nn::ModelSpec loaded_model_spec =
        NN_UNWRAP(nn::peek_any_model_spec(options.model));
    nn::Model inference_model =
        NN_UNWRAP(nn::build_cnn_model_from_spec(engine, loaded_model_spec));
    load_weights(options.model, inference_model);
    const double model_diff = max_logit_difference(
        session.model(), inference_model, eval_cache, options.common.batch_size
    );

    const bool roundtrip_pass = checkpoint_diff <= 1.0e-5 && model_diff <= 1.0e-5;
    const bool loss_declined =
        std::isfinite(static_cast<double>(first_train_loss)) &&
        std::isfinite(static_cast<double>(final_train_loss)) &&
        std::isfinite(static_cast<double>(initial_eval.loss)) &&
        std::isfinite(static_cast<double>(final_eval.loss)) &&
        final_train_loss < first_train_loss &&
        final_eval.loss < initial_eval.loss;
    const bool pass = roundtrip_pass && loss_declined;
    const double total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - training_started
    ).count();
    json report{
        {"schema", 1},
        {"status", pass ? "pass" : "fail"},
        {"model", "c-validator-v1"},
        {"dataset", train_cache.path().string()},
        {"eval_dataset", eval_cache.path().string()},
        {"engine", options.common.use_gpu ? "vulkan" : "cpu"},
        {"gpu_device", options.common.gpu_device},
        {"seed", options.seed},
        {"epochs", options.epochs},
        {"batch_size", options.common.batch_size},
        {"lr", static_cast<double>(options.lr)},
        {"min_lr", static_cast<double>(options.min_lr)},
        {"weight_decay", static_cast<double>(options.weight_decay)},
        {"train_samples", indices.size()},
        {"eval_samples", eval_cache.size()},
        {"initial_eval", metrics_json(initial_eval)},
        {"history", std::move(history)},
        {"checkpoint", {
            {"path", options.checkpoint},
            {"max_logit_diff", checkpoint_diff},
        }},
        {"inference_model", {
            {"path", options.model},
            {"max_logit_diff", model_diff},
        }},
        {"health", {
            {"loss_declined", loss_declined},
            {"checkpoint_roundtrip", roundtrip_pass},
        }},
        {"seconds", total_seconds},
    };
    write_json(options.metrics_out, report);
    std::cout << report.dump(2) << '\n';
    if (!pass) {
        throw std::runtime_error(
            "M4 acceptance failed: checkpoint round-trip or loss-decline gate"
        );
    }
    return 0;
}

int evaluate_cli(const EvalOptions& options) {
    go::FeatureCache cache = go::FeatureCache::open(options.dataset);
    std::shared_ptr<nn::ComputeEngine> engine_ptr = create_engine(options.common);
    nn::ComputeEngine& engine = *engine_ptr;
    const nn::ModelSpec spec = NN_UNWRAP(nn::peek_any_model_spec(options.model));
    std::cerr << "loaded model spec: " << nn::spec_summary(spec) << '\n';
    nn::Model model = NN_UNWRAP(nn::build_cnn_model_from_spec(engine, spec));
    std::cerr << "built model from spec\n";
    const nn::ModelRole role = NN_UNWRAP(nn::peek_model_role(options.model));
    std::cerr << "model role: " << nn::model_role_name(role) << '\n';
    load_weights(options.model, model);
    std::cerr << "loaded model weights\n";
    const EvalMetrics metrics =
        evaluate(model, cache, options.common.batch_size, options.common.max_samples);
    std::cout << json{
        {"schema", 1},
        {"status", "pass"},
        {"model", options.model},
        {"role", nn::model_role_name(role)},
        {"engine", options.common.use_gpu ? "vulkan" : "cpu"},
        {"metrics", metrics_json(metrics)},
    }.dump(2) << '\n';
    return 0;
}

void usage(const char* program) {
    std::cerr
        << "usage:\n"
        << "  " << program << " inspect --dataset FILE.ngf\n"
        << "  " << program << " train --dataset FILE.ngf --eval-dataset FILE.ngf"
        << " [--gpu=40HX] [--epochs 4] [--batch-size 64] [...] \n"
        << "  " << program << " evaluate --dataset FILE.ngf --model FILE.nnmodel"
        << " [--gpu=40HX] [--batch-size 64]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string_view(argv[1]) == "--help" ||
            std::string_view(argv[1]) == "-h") {
            usage(argv[0]);
            return argc < 2 ? 2 : 0;
        }
        const std::string_view command = argv[1];
        if (command == "inspect") {
            if (argc != 4 || std::string_view(argv[2]) != "--dataset") {
                throw std::invalid_argument("inspect requires --dataset FILE");
            }
            return inspect_cache(argv[3]);
        }
        if (command == "train") {
            return train(parse_train_options(argc, argv));
        }
        if (command == "evaluate") {
            return evaluate_cli(parse_eval_options(argc, argv));
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
