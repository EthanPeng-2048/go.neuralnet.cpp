#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
#include <neuralnet.cpp/core_file.hpp>       // nn::sha256_file_hex（数据集身份）
#include <neuralnet.cpp/train_state.hpp>     // nn::TrainState（M5 断点状态段）
#include <neuralnet.cpp/lr_scheduler.hpp>    // nn::compute_epoch_lr

#include "external/nlohmann_json/json.hpp"
#include "go/feature_cache.hpp"
#include "go/go_model.hpp"

namespace {

using json = nlohmann::json;
using nn::Scalar;

// M5 健康门基线（未训练模型 @ 训练集 loss）只是"下降发生了吗"的参照，
// 不需要全量精度：32k 样本的均值对"下降几 nat"的判定绰绰有余，而全量
// 前向在 60 万+ 样本的数据集上要数小时（曾把开训时间顶到不可用）。
constexpr std::size_t kBaselineSamples = 32768;

struct CommonOptions {
    bool use_gpu = false;
    std::string gpu_device;
    std::size_t batch_size = 64;
    std::size_t max_samples = 0;
};

// M5 监督配方：全部可经 CLI 表达，且逐项随断点落盘（TrainState::recipe），
// 续训时以断点为权威。
struct TrainOptions {
    CommonOptions common;
    std::string dataset;
    std::string eval_dataset;
    std::string checkpoint = "artifacts/m4/c-validator.nnckpt";
    std::string model = "artifacts/m4/c-validator.nnmodel";
    std::string metrics_out = "artifacts/m4/c-validator.json";
    std::string metrics_log;   // 空 = 从 metrics_out 派生（.json → .jsonl）
    std::string resume;        // 非空 = 从 .nnckpt 断点续训
    // M7：架构选择。b = B 双分支编码器（默认，主线）；c = M4 CNN 验证器
    // （保留用于 B vs C 对照与 M5 语义回归）。续训时以断点 spec 为权威。
    std::string arch = "b";
    go::GoModelConfig model_config;  // 仅 arch=b 生效；各字段可经 CLI 覆盖
    int epochs = 4;
    int stop_after_epoch = 0;  // >0 = 训到该 epoch（1-based，不含）即存档退出
    std::string optimizer = "adamw";
    std::string schedule = "cosine";
    Scalar lr = 1.0e-3F;
    Scalar min_lr = 1.0e-5F;
    Scalar weight_decay = 1.0e-4F;
    // M7: 价值头损失权重（GoLoss). 0 = 只训策略头；1 = 与策略 CE 等权。
    Scalar value_weight = 1.0F;
    Scalar beta1 = 0.9F;
    Scalar beta2 = 0.999F;
    Scalar clip_grad_norm = 1.0F;
    int warmup_epochs = 0;
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
    // M7 acceptance metrics (roadmap §9): top-3 policy accuracy and the value
    // head's ranking quality. Both are zero for arch=c, which has no value head.
    double top3 = 0;
    double value_auc = 0;
    std::size_t value_samples = 0;
    std::size_t samples = 0;
    double seconds = 0;
};

struct LoadedBatch {
    std::vector<Scalar> planes;
    // M7/B: recent-move history (GoTokenizer point ids, left <pad>-padded).
    // Empty for arch=c or a v1 cache. Row-major: [slot * batch + column].
    std::vector<std::uint16_t> history;
    std::vector<std::size_t> labels;
    // M7/B: value target z in {-1,0,+1} from the side-to-move perspective.
    std::vector<Scalar> values;
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
        } else if (arg == "--metrics-log") {
            options.metrics_log = option_value(i, argc, argv);
        } else if (arg == "--resume") {
            options.resume = option_value(i, argc, argv);
        } else if (arg == "--epochs") {
            options.epochs = std::stoi(option_value(i, argc, argv));
        } else if (arg == "--stop-after-epoch") {
            options.stop_after_epoch = std::stoi(option_value(i, argc, argv));
        } else if (arg == "--optimizer") {
            options.optimizer = option_value(i, argc, argv);
        } else if (arg == "--schedule") {
            options.schedule = option_value(i, argc, argv);
        } else if (arg == "--lr") {
            options.lr = std::stof(option_value(i, argc, argv));
        } else if (arg == "--min-lr") {
            options.min_lr = std::stof(option_value(i, argc, argv));
        } else if (arg == "--weight-decay") {
            options.weight_decay = std::stof(option_value(i, argc, argv));
        } else if (arg == "--value-weight") {
            options.value_weight = std::stof(option_value(i, argc, argv));
        } else if (arg == "--beta1") {
            options.beta1 = std::stof(option_value(i, argc, argv));
        } else if (arg == "--beta2") {
            options.beta2 = std::stof(option_value(i, argc, argv));
        } else if (arg == "--clip-grad-norm") {
            options.clip_grad_norm = std::stof(option_value(i, argc, argv));
        } else if (arg == "--warmup-epochs") {
            options.warmup_epochs = std::stoi(option_value(i, argc, argv));
        } else if (arg == "--arch") {
            options.arch = option_value(i, argc, argv);
        } else if (arg == "--d-model") {
            options.model_config.d_model = std::stoull(option_value(i, argc, argv));
        } else if (arg == "--num-layers") {
            options.model_config.num_layers = std::stoull(option_value(i, argc, argv));
        } else if (arg == "--num-heads") {
            options.model_config.num_heads = std::stoull(option_value(i, argc, argv));
        } else if (arg == "--d-ff") {
            options.model_config.d_ff = std::stoull(option_value(i, argc, argv));
        } else if (arg == "--conv-channels") {
            options.model_config.conv_channels = std::stoull(option_value(i, argc, argv));
        } else if (arg == "--history-len") {
            options.model_config.history_len = std::stoull(option_value(i, argc, argv));
        } else if (arg == "--cnn-policy") {
            options.model_config.cnn_policy = std::stoi(option_value(i, argc, argv)) != 0;
        } else if (arg == "--seed") {
            options.seed = std::stoull(option_value(i, argc, argv));
        } else {
            throw std::invalid_argument("unknown train option: " + std::string(arg));
        }
    }
    if (options.dataset.empty() || options.eval_dataset.empty()) {
        throw std::invalid_argument("train requires --dataset and --eval-dataset");
    }
    if (options.arch != "b" && options.arch != "c") {
        throw std::invalid_argument("--arch must be b or c");
    }
    {
        const go::GoModelConfig& cfg = options.model_config;
        if (cfg.d_model == 0 || cfg.num_layers == 0 || cfg.d_ff == 0 ||
            cfg.conv_channels == 0 || cfg.history_len == 0 ||
            cfg.history_vocab < go::history_vocab_size) {
            throw std::invalid_argument(
                "B model dimensions must be positive and history-vocab >= 367");
        }
        if (cfg.d_model % cfg.num_heads != 0) {
            throw std::invalid_argument("--d-model must be divisible by --num-heads");
        }
    }
    if (options.epochs <= 0 || options.common.batch_size == 0) {
        throw std::invalid_argument("epochs and batch-size must be positive");
    }
    if (options.stop_after_epoch < 0 || options.stop_after_epoch > options.epochs) {
        throw std::invalid_argument("stop-after-epoch must be in [0, epochs]");
    }
    if (options.lr <= 0 || options.min_lr < 0 || options.min_lr > options.lr) {
        throw std::invalid_argument("invalid lr/min-lr");
    }
    if (options.beta1 < 0 || options.beta1 >= 1 || options.beta2 < 0 || options.beta2 >= 1) {
        throw std::invalid_argument("beta1/beta2 must be in [0,1)");
    }
    if (options.clip_grad_norm < 0) {
        throw std::invalid_argument("clip-grad-norm must be non-negative");
    }
    if (options.value_weight < 0) {
        throw std::invalid_argument("value-weight must be non-negative");
    }
    if (options.warmup_epochs < 0 || options.warmup_epochs >= options.epochs) {
        throw std::invalid_argument("warmup-epochs must be in [0, epochs)");
    }
    if (options.schedule != "cosine" && options.schedule != "constant" &&
        options.schedule != "step_cosine") {
        throw std::invalid_argument("schedule must be cosine, constant or step_cosine");
    }
    if (options.metrics_log.empty()) {
        options.metrics_log = options.metrics_out;
        const std::string suffix = ".json";
        if (options.metrics_log.size() > suffix.size() &&
            options.metrics_log.ends_with(suffix)) {
            options.metrics_log.resize(options.metrics_log.size() - suffix.size());
        }
        options.metrics_log += ".jsonl";
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

// M4 C 验证器：纯现成层（Conv2D + MaxPool + FC + CE），
// 保留用于 B vs C 对照与 M5 语义回归。
nn::CnnConfig c_validator_config() {
    nn::CnnConfig config;
    config.in_channels = go::feature_plane_count;
    config.in_size = go::training_board_size;
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

// Model input rows: planes only, or planes + K history rows for the B model.
std::size_t feature_count_plus_history(
    const go::FeatureCache& cache,
    std::size_t history_len
) {
    return cache.feature_count() + history_len;
}

// Loads one batch. When `history_len` > 0 (B model), also reads the per-sample
// move history and packs it row-major [slot * batch + column] so it can be
// concatenated onto the planes tensor along the row axis. Value targets are
// always read (v1 caches carry them too; arch=c simply ignores them).
LoadedBatch load_batch_planes(
    go::FeatureCache& cache,
    const std::vector<std::size_t>& indices,
    std::size_t first,
    std::size_t count,
    std::size_t history_len = 0
) {
    LoadedBatch result{
        std::vector<Scalar>(cache.feature_count() * count, Scalar{0}),
        history_len > 0
            ? std::vector<std::uint16_t>(history_len * count, std::uint16_t{0})
            : std::vector<std::uint16_t>{},
        std::vector<std::size_t>(count),
        std::vector<Scalar>(count, Scalar{0}),
    };
    std::vector<float> sample(cache.feature_count());
    std::vector<std::uint16_t> sample_history(history_len);
    for (std::size_t column = 0; column < count; ++column) {
        go::FeatureSampleMeta metadata;
        cache.read_sample(
            indices[first + column], sample, &metadata,
            history_len > 0
                ? std::span<std::uint16_t>(sample_history)
                : std::span<std::uint16_t>{}
        );
        result.labels[column] = metadata.target;
        result.values[column] = static_cast<Scalar>(metadata.value);
        for (std::size_t feature = 0; feature < sample.size(); ++feature) {
            result.planes[feature * count + column] = sample[feature];
        }
        for (std::size_t slot = 0; slot < history_len; ++slot) {
            result.history[slot * count + column] = sample_history[slot];
        }
    }
    return result;
}

// Input tensor for the model: planes only (C) or planes + history rows (B).
// B packs history ids as f32 in rows [feature_count, feature_count + K), as a
// single column-major tensor with one batch column per sample.
void write_model_input(
    nn::ComputeEngine& engine,
    nn::Tensor& input,
    const LoadedBatch& batch,
    std::size_t feature_count,
    std::size_t count,
    std::size_t history_len
) {
    if (history_len == 0) {
        NN_UNWRAP(engine.write(input, std::span<const Scalar>(batch.planes)));
        return;
    }
    std::vector<Scalar> packed(feature_count * count + history_len * count);
    std::copy(batch.planes.begin(), batch.planes.end(), packed.begin());
    for (std::size_t i = 0; i < history_len * count; ++i) {
        packed[feature_count * count + i] = static_cast<Scalar>(batch.history[i]);
    }
    NN_UNWRAP(engine.write(input, std::span<const Scalar>(packed)));
}

EvalMetrics evaluate(
    nn::Model& model,
    go::FeatureCache& cache,
    std::size_t batch_size,
    std::size_t maximum,
    std::size_t history_len = 0,
    Scalar value_weight = Scalar{1}
) {
    const auto started = std::chrono::steady_clock::now();
    const std::vector<std::size_t> indices = sample_indices(cache, maximum);
    if (indices.empty()) {
        throw std::runtime_error("evaluation cache has no samples");
    }
    nn::ComputeEngine& engine = model.engine();
    nn::CrossEntropyLoss loss_fn;
    go::GoLoss go_loss(value_weight);
    model.set_training(false);

    Scalar loss_sum = 0;
    std::size_t correct = 0;
    std::size_t correct3 = 0;
    // Value AUC: collect (prediction, target) pairs with a nonzero target, then
    // compute the rank statistic offline. z=0 (draw/unknown) carries no signal.
    std::vector<std::pair<Scalar, Scalar>> value_pairs;
    std::vector<Scalar> logits;
    const std::size_t input_rows = feature_count_plus_history(cache, history_len);
    for (std::size_t first = 0; first < indices.size(); first += batch_size) {
        const std::size_t count = std::min(batch_size, indices.size() - first);
        LoadedBatch batch = load_batch_planes(cache, indices, first, count, history_len);
        nn::Tensor input = engine.create_tensor(input_rows, count);
        if (!input.valid()) {
            throw std::runtime_error("cannot allocate evaluation input tensor");
        }
        write_model_input(engine, input, batch, cache.feature_count(), count, history_len);
        nn::Tensor output = forward_evaluated(model, input);
        const std::vector<std::size_t>& labels = batch.labels;
        Scalar loss = 0;
        if (history_len > 0) {
            // B: dual-head loss = policy CE (row 0..361) + value MSE (row 362).
            loss = NN_UNWRAP(go_loss.forward(
                engine, output, std::span<const std::size_t>(labels),
                std::span<const Scalar>(batch.values)
            ));
        } else {
            loss = NN_UNWRAP(loss_fn.forward_sparse(
                engine, output, labels, {}, go::policy_class_count));
        }
        loss_sum += loss * static_cast<Scalar>(count);

        logits.resize(output.size());
        NN_UNWRAP(engine.read(output, std::span<Scalar>(logits)));
        for (std::size_t column = 0; column < count; ++column) {
            std::size_t prediction = 0;
            Scalar best = logits[column];
            // Policy head only: row 362 is the value logit, never an argmax
            // candidate. (C outputs exactly 362 rows, so this loop is a no-op
            // there and the M4/M5 metrics stay identical.)
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
            // Top-3: exact running top-3 over the policy rows, then test
            // membership. Row 362 (value) is excluded, matching the argmax.
            {
                std::array<std::pair<Scalar, std::size_t>, 3> top{};
                for (auto& entry : top) {
                    entry = {-std::numeric_limits<Scalar>::infinity(), 0};
                }
                for (std::size_t row = 0; row < go::policy_class_count; ++row) {
                    const Scalar value = logits[row * count + column];
                    if (value > top[0].first) {
                        top[2] = top[1];
                        top[1] = top[0];
                        top[0] = {value, row};
                    } else if (value > top[1].first) {
                        top[2] = top[1];
                        top[1] = {value, row};
                    } else if (value > top[2].first) {
                        top[2] = {value, row};
                    }
                }
                for (const auto& entry : top) {
                    if (entry.second == labels[column]) {
                        ++correct3;
                        break;
                    }
                }
            }
            // Row 362 holds tanh(v) in [-1, 1]: the value prediction. AUC is
            // computed only over decisive games (z != 0).
            if (history_len > 0 && batch.values[column] != Scalar{0}) {
                value_pairs.emplace_back(
                    logits[go::value_row_index * count + column],
                    batch.values[column]
                );
            }
        }
    }

    // Value AUC (M7): the Mann-Whitney U statistic over decisive games. AUC is
    // the fraction of (win, loss) prediction pairs ranked correctly; ties count
    // as a half. 0.5 = chance, so a healthy value head sits well above it.
    double value_auc = 0;
    std::size_t value_samples = value_pairs.size();
    if (value_samples >= 2) {
        long double wins = 0;
        long double positives = 0;
        long double negatives = 0;
        for (const auto& [prediction, target] : value_pairs) {
            if (target > 0) {
                ++positives;
            } else {
                ++negatives;
            }
        }
        // Rank predictions ascending, awarding ties the average rank.
        std::vector<std::size_t> order(value_samples);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return value_pairs[a].first < value_pairs[b].first;
        });
        std::size_t i = 0;
        std::size_t rank = 1;
        while (i < order.size()) {
            std::size_t j = i;
            while (j + 1 < order.size() &&
                   value_pairs[order[j + 1]].first == value_pairs[order[i]].first) {
                ++j;
            }
            // Ties share the mean of the ranks they span: ranks [rank, rank+len-1]
            // ⇒ rank + (len-1)/2. (rank+len)/2 is off by half a rank for every
            // group of size 1 — it biases the statistic negative.)
            const std::size_t span = j - i + 1;
            const long double average_rank =
                static_cast<long double>(rank) +
                static_cast<long double>(span - 1) / 2.0L;
            for (std::size_t m = i; m <= j; ++m) {
                if (value_pairs[order[m]].second > 0) {
                    wins += average_rank;
                }
            }
            rank += j - i + 1;
            i = j + 1;
        }
        if (positives > 0 && negatives > 0) {
            value_auc = static_cast<double>(
                (wins - positives * (positives + 1) / 2.0L) /
                (positives * negatives)
            );
        }
    }

    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    return {
        loss_sum / static_cast<Scalar>(indices.size()),
        static_cast<double>(correct) / static_cast<double>(indices.size()),
        static_cast<double>(correct3) / static_cast<double>(indices.size()),
        value_auc,
        value_samples,
        indices.size(),
        elapsed,
    };
}

json metrics_json(const EvalMetrics& metrics) {
    return {
        {"loss", static_cast<double>(metrics.loss)},
        {"top1", metrics.top1},
        {"top3", metrics.top3},
        {"value_auc", metrics.value_auc},
        {"value_samples", metrics.value_samples},
        {"samples", metrics.samples},
        {"seconds", metrics.seconds},
    };
}

double max_logit_difference(
    nn::Model& left,
    nn::Model& right,
    go::FeatureCache& cache,
    std::size_t batch_size,
    std::size_t history_len = 0
) {
    if (left.engine().device() != right.engine().device()) {
        throw std::runtime_error("round-trip models use different devices");
    }
    const std::size_t count = std::min(batch_size, cache.size());
    std::vector<std::size_t> indices = sample_indices(cache, count);
    LoadedBatch batch = load_batch_planes(cache, indices, 0, count, history_len);
    nn::ComputeEngine& engine = left.engine();
    nn::Tensor input = engine.create_tensor(
        feature_count_plus_history(cache, history_len), count);
    if (!input.valid()) {
        throw std::runtime_error("cannot allocate round-trip input tensor");
    }
    write_model_input(
        engine, input, batch, cache.feature_count(), count, history_len);
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

// M5 指标落盘：逐 epoch 追加 JSONL（崩溃不丢已完成 epoch 的指标；断点续训
// 追加到同一文件，run 记录区分进程边界）。
void append_metrics_log(const std::filesystem::path& path, const json& record) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot open metrics log: " + path.string());
    }
    output << record.dump() << '\n';
    output.flush();
}

void load_weights(
    const std::string& path,
    nn::Model& model,
    nn::WeightLoadIntent intent
) {
    std::cerr << "verifying model file: " << path << '\n';
    auto verified = nn::verify_model_integrity(path);
    if (!verified) {
        throw std::runtime_error(
            "model integrity check failed for '" + path + "': " +
            verified.error().message
        );
    }
    std::cerr << "model file verified\n";
    auto result = go::load_go_model(path, model, nullptr, nullptr, intent);
    if (!result) {
        throw std::runtime_error(
            "cannot load model '" + path + "': " + result.error().message
        );
    }
}

// ── M5 断点状态装配 ─────────────────────────────────────────────────────
// 计数器 = 续训起点；配方 = 曲线形状与优化器身份（续训以断点为权威）；
// 数据集身份 = .ngf 文件 sha256 + 样本数（内容变了就不存在正确的续训语义）。
// Go 训练器复用上游 TextTrainState 承载域字段（state_kind="text"）：旧 Go
// 断点（v1.13 无 state_kind 键）读回即 text kind，且上游工厂不接受自定义
// kind —— 数据集身份（.ngf sha256 + 样本数）正好落在 DatasetIdentity。
nn::TextTrainState make_train_state(
    const TrainOptions& options,
    const nn::TrainingSession& session,
    std::uint64_t completed_epochs,
    std::uint64_t global_step_base,
    const std::string& dataset_sha256,
    std::size_t train_samples
) {
    nn::TextTrainState state;
    state.counters.global_step = global_step_base + session.step_count();
    state.counters.epoch = completed_epochs;
    state.counters.step_in_epoch = 0;  // 断点只写在 epoch 边界
    state.counters.lr = static_cast<double>(session.current_lr());

    state.recipe.optimizer = options.optimizer;
    state.recipe.lr_schedule = options.schedule;
    state.recipe.lr = static_cast<double>(options.lr);
    state.recipe.min_lr = static_cast<double>(options.min_lr);
    state.recipe.weight_decay = static_cast<double>(options.weight_decay);
    state.recipe.beta1 = static_cast<double>(options.beta1);
    state.recipe.beta2 = static_cast<double>(options.beta2);
    state.recipe.max_norm = static_cast<double>(options.clip_grad_norm);
    state.recipe.epochs = static_cast<std::uint64_t>(options.epochs);
    state.recipe.warmup_epochs = static_cast<std::uint64_t>(options.warmup_epochs);
    state.recipe.batch_size = options.common.batch_size;
    state.recipe.micro_batch_size = options.common.batch_size;
    state.recipe.batch_size_is_global = true;
    state.recipe.accum_steps = 1;

    state.dataset.path = options.dataset;
    state.dataset.sha256 = dataset_sha256;
    state.dataset.num_samples = train_samples;
    state.dataset.steps_per_epoch =
        (train_samples + options.common.batch_size - 1) / options.common.batch_size;

    state.optimizer_step = session.state().optimizer_step;
    return state;
}

// 断点保存：模型参数（Checkpoint 角色）+ 训练状态段 + 优化器状态张量。
void save_checkpoint_file(
    const std::string& path,
    nn::Model& model,
    const nn::ModelSpec& spec,
    const nn::TrainState& state,
    nn::Optimizer& optimizer
) {
    std::vector<nn::Tensor> blobs;
    NN_UNWRAP(optimizer.save_state(blobs));
    // save_model 内部 clone 状态并回写 optimizer_blob_count，调用方只给只读引用。
    const nn::CheckpointState ckpt{&state, std::span<const nn::Tensor>(blobs)};
    NN_UNWRAP(nn::save_model(
        path, model, spec, {}, nn::ModelRole::Checkpoint, ckpt
    ));
}

int inspect_cache(const std::string& path) {
    go::FeatureCache cache = go::FeatureCache::open(path);
    json result{
        {"dataset", cache.path().string()},
        {"format", "go.feature-cache"},
        {"format_version", cache.format_version()},
        {"records", cache.size()},
        {"features", cache.feature_count()},
        {"board_size", cache.board_size()},
        {"history_len", cache.history_len()},
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

    const std::string dataset_sha256 = NN_UNWRAP(nn::sha256_file_hex(options.dataset));

    // B needs history rows: both caches must be v2 with a K matching the model.
    // A v1 cache (history_len()==0) is rejected loudly rather than silently fed
    // zeros — that would look like "the history branch buys nothing". This is a
    // data-contract gate, so it runs before any model construction. On resume
    // the authoritative K comes from the checkpoint spec, so that path checks
    // after deriving it (below) instead of trusting --history-len.
    auto require_history_k = [&](std::size_t requested_k) {
        for (const go::FeatureCache* cache : {&train_cache, &eval_cache}) {
            if (cache->history_len() != requested_k) {
                throw std::runtime_error(
                    std::string("B model needs move history but cache '") +
                    cache->path().string() + "' has history_len=" +
                    std::to_string(cache->history_len()) + " (model K=" +
                    std::to_string(requested_k) +
                    "). Regenerate with --history-len " +
                    std::to_string(requested_k) + "."
                );
            }
        }
    };
    if (options.resume.empty() && options.arch == "b") {
        require_history_k(options.model_config.history_len);
    }

    // Effective architecture. On resume the checkpoint spec is authoritative
    // (same rule as the recipe); on a fresh run it comes from --arch.
    std::string arch = options.arch;
    std::size_t history_len = 0;  // 0 = C (planes only); >0 = B (planes+history)

    nn::ModelSpec spec;
    std::unique_ptr<nn::Model> model;
    std::optional<nn::LoadedTrainState> resumed;
    std::uint64_t global_step_base = 0;
    int start_epoch = 0;

    if (!options.resume.empty()) {
        // ── 断点续训：状态段为唯一权威 ────────────────────────────────
        nn::LoadedTrainState loaded = NN_UNWRAP(nn::load_train_state(engine, options.resume));
        const auto* saved = dynamic_cast<const nn::TextTrainState*>(loaded.state.get());
        if (saved == nullptr) {
            throw std::runtime_error(
                std::string("--resume 的训练状态段不是 Go 训练器可识别的类型（kind=") +
                loaded.state->state_kind() + "）: " + options.resume
            );
        }

        // 数据集身份门禁（硬错）：sha256 或样本数变了 ⇒ step 号 / lr 曲线
        // 语义全变，续训不存在正确出口。
        if (saved->dataset.sha256 != dataset_sha256 ||
            saved->dataset.num_samples != train_cache.size()) {
            throw std::runtime_error(
                std::string("--resume 的数据集与断点不一致（内容身份变了 ⇒ 续训不存在正确语义）:\n") +
                "  断点记录 sha256=" + saved->dataset.sha256 +
                " samples=" + std::to_string(saved->dataset.num_samples) + "\n" +
                "  本次     sha256=" + dataset_sha256 +
                " samples=" + std::to_string(train_cache.size()) + "\n" +
                "  断点文件: " + options.resume + "\n" +
                "  换数据集 = 新任务 ⇒ 请从推理模型（.nnmodel）重开一轮"
            );
        }
        if (saved->recipe.epochs != static_cast<std::uint64_t>(options.epochs)) {
            throw std::runtime_error(
                "--epochs 与断点配方不一致（lr 曲线形状必须跨续训不变）: 断点 epochs=" +
                std::to_string(saved->recipe.epochs) + "，本次 --epochs " +
                std::to_string(options.epochs)
            );
        }
        if (saved->recipe.accum_steps != 1) {
            throw std::runtime_error("本训练器只支持 accum_steps=1 的断点");
        }
        if (saved->counters.epoch > static_cast<std::uint64_t>(options.epochs)) {
            throw std::runtime_error(
                "断点已训完 " + std::to_string(saved->counters.epoch) +
                " 个 epoch，超过 --epochs " + std::to_string(options.epochs) +
                "，没有可续的训练"
            );
        }

        const nn::ModelSpec file_spec = NN_UNWRAP(go::peek_go_model_spec(options.resume));
        model = std::make_unique<nn::Model>(
            NN_UNWRAP(go::build_project_model_from_spec(engine, file_spec))
        );
        load_weights(options.resume, *model, nn::WeightLoadIntent::Trainable);
        spec = file_spec;
        // Derive the effective architecture from the checkpoint spec so the
        // data path (history rows, loss) matches what the weights expect.
        if (go::is_go_spec(spec)) {
            arch = "b";
            history_len = NN_UNWRAP(go::go_config_from_spec(spec)).history_len;
        } else {
            arch = "c";
            history_len = 0;
        }
        if (history_len > 0) {
            // 续训的 K 由断点 spec 决定（--history-len 不参与），缓存必须匹配。
            require_history_k(history_len);
        }

        global_step_base = saved->counters.global_step;
        start_epoch = static_cast<int>(saved->counters.epoch);
        const std::size_t optimizer_tensors = loaded.optimizer_blobs.size();
        const std::uint64_t optimizer_step = saved->optimizer_step;
        const std::string optimizer_name = saved->recipe.optimizer;
        resumed = std::move(loaded);
        std::cerr << "resumed from " << options.resume
                  << ": epoch=" << start_epoch
                  << " global_step=" << global_step_base
                  << " optimizer=" << optimizer_name
                  << " optimizer_step=" << optimizer_step
                  << " optimizer_tensors=" << optimizer_tensors
                  << '\n';
    } else if (options.arch == "b") {
        spec = go::make_go_spec(options.model_config);
        model = std::make_unique<nn::Model>(
            NN_UNWRAP(go::build_project_model(engine, options.model_config))
        );
        model->set_spec(spec);
        history_len = options.model_config.history_len;
    } else {
        const nn::CnnConfig cnn_config = c_validator_config();
        spec = c_validator_spec(cnn_config);
        model = std::make_unique<nn::Model>(
            NN_UNWRAP(nn::build_cnn_model(engine, cnn_config))
        );
        model->set_spec(spec);
        history_len = 0;
    }
    model->set_training(true);

    const EvalMetrics initial_eval = evaluate(
        *model, eval_cache, options.common.batch_size, 0, history_len,
        options.value_weight
    );
    model->set_training(true);

    // Baseline for the loss-decline gate: the untrained model's loss on the
    // *training* set. It must be measured before the model is moved into the
    // TrainingSession below, and it must share a dataset with the final
    // train_loss — using initial_eval (eval set) as the baseline misjudges a
    // legitimate single-epoch convergence as a failure.
    Scalar first_train_loss = initial_eval.loss;
    const int planned_final_epoch = options.stop_after_epoch > 0
        ? options.stop_after_epoch
        : options.epochs;
    if (planned_final_epoch > start_epoch) {
        first_train_loss = evaluate(
            *model, train_cache, options.common.batch_size, kBaselineSamples,
            history_len, options.value_weight
        ).loss;
        model->set_training(true);
    }

    nn::TrainingConfig training_config;
    training_config.optimizer = options.optimizer;
    training_config.lr = options.lr;
    training_config.weight_decay = options.weight_decay;
    training_config.beta1 = options.beta1;
    training_config.beta2 = options.beta2;
    training_config.clip_grad_norm = options.clip_grad_norm;
    training_config.grad_accum_steps = 1;
    nn::TrainingSession session = NN_UNWRAP(
        nn::TrainingSession::create(std::move(*model), training_config)
    );
    nn::CrossEntropyLoss loss_fn;
    go::GoLoss go_loss(options.value_weight);

    if (resumed) {
        // 优化器身份已由状态段决定；张量数量/形状对不上由 load_state 硬错。
        session.optimizer().set_state_step(resumed->state->optimizer_step);
        NN_UNWRAP(session.optimizer().load_state(resumed->optimizer_blobs));
        std::cerr << "restored optimizer state: "
                  << resumed->optimizer_blobs.size() << " tensors\n";
    }

    std::vector<std::size_t> indices = sample_indices(
        train_cache, options.common.max_samples
    );
    if (indices.empty()) {
        throw std::runtime_error("training cache has no samples");
    }
    nn::LrScheduleConfig lr_config;
    lr_config.base_lr = options.lr;
    lr_config.min_lr = options.min_lr;
    lr_config.total_epochs = options.epochs;
    lr_config.warmup_epochs = options.warmup_epochs;
    lr_config.schedule = options.schedule;
    if (options.schedule == "step_cosine") {
        // 库只提供 constant/cosine，这里用 lr_per_epoch 表达"三段余弦 + 重启"
        // （SGDR 口径）：每段内余弦从 base_lr 退到 min_lr，段边界重启；
        // warmup 作用在第一段。段数 = 3（可被 --epochs 截断）。
        const int segments = 3;
        const int per_segment = std::max(1, options.epochs / segments);
        std::vector<Scalar> table(static_cast<std::size_t>(options.epochs));
        for (int epoch = 0; epoch < options.epochs; ++epoch) {
            const int local = epoch % per_segment;
            const Scalar progress =
                static_cast<Scalar>(local) / static_cast<Scalar>(per_segment);
            const Scalar cosine =
                Scalar{0.5} * (Scalar{1} + std::cos(static_cast<Scalar>(3.14159265358979323846) * progress));
            table[static_cast<std::size_t>(epoch)] =
                options.min_lr + (options.lr - options.min_lr) * cosine;
        }
        if (options.warmup_epochs > 0) {
            for (int epoch = 0; epoch < options.warmup_epochs && epoch < options.epochs; ++epoch) {
                table[static_cast<std::size_t>(epoch)] =
                    options.lr * static_cast<Scalar>(epoch + 1) /
                    static_cast<Scalar>(options.warmup_epochs);
            }
        }
        lr_config.lr_per_epoch = std::move(table);
        lr_config.schedule = "constant";  // 表优先；显式写明不留歧义
    }

    const int final_epoch = options.stop_after_epoch > 0
        ? options.stop_after_epoch
        : options.epochs;

    append_metrics_log(options.metrics_log, {
        {"record", "run"},
        {"mode", resumed ? "resume" : "fresh"},
        {"resume_from", options.resume},
        {"dataset", train_cache.path().string()},
        {"dataset_sha256", dataset_sha256},
        {"eval_dataset", eval_cache.path().string()},
        {"engine", options.common.use_gpu ? "vulkan" : "cpu"},
        {"gpu_device", options.common.gpu_device},
        {"seed", options.seed},
        {"epochs", options.epochs},
        {"start_epoch", start_epoch},
        {"stop_after_epoch", final_epoch},
        {"batch_size", options.common.batch_size},
        {"optimizer", options.optimizer},
        {"schedule", options.schedule},
        {"lr", static_cast<double>(options.lr)},
        {"min_lr", static_cast<double>(options.min_lr)},
        {"weight_decay", static_cast<double>(options.weight_decay)},
        {"beta1", static_cast<double>(options.beta1)},
        {"beta2", static_cast<double>(options.beta2)},
        {"clip_grad_norm", static_cast<double>(options.clip_grad_norm)},
        {"warmup_epochs", options.warmup_epochs},
        {"train_samples", indices.size()},
        {"initial_eval", metrics_json(initial_eval)},
    });

    const auto training_started = std::chrono::steady_clock::now();
    json history = json::array();
    EvalMetrics final_eval = initial_eval;
    Scalar final_train_loss = 0;

    for (int epoch = start_epoch; epoch < final_epoch; ++epoch) {
        const Scalar epoch_lr = nn::compute_epoch_lr(lr_config, epoch);
        session.set_lr(epoch_lr);
        // 洗牌序是 (seed, epoch) 的纯函数：每 epoch 先还原成自然序再用该
        // epoch 派生的 RNG 洗牌（std::shuffle 的输出依赖输入序，若沿用上一
        // epoch 的排列，续训进程与一口气训练在同一 epoch 会得到不同数据序）。
        // 这是"续训无损"的前提之一。
        std::iota(indices.begin(), indices.end(), std::size_t{0});
        std::mt19937_64 epoch_rng(
            options.seed ^ (0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>(epoch + 1))
        );
        std::shuffle(indices.begin(), indices.end(), epoch_rng);
        const auto epoch_started = std::chrono::steady_clock::now();
        Scalar loss_sum = 0;
        std::size_t seen = 0;

        for (std::size_t first = 0; first < indices.size();
             first += options.common.batch_size) {
            const std::size_t count =
                std::min(options.common.batch_size, indices.size() - first);
            LoadedBatch batch = load_batch_planes(
                train_cache, indices, first, count, history_len
            );
            nn::Tensor input = engine.create_tensor(
                feature_count_plus_history(train_cache, history_len), count);
            if (!input.valid()) {
                throw std::runtime_error("cannot allocate training input tensor");
            }
            write_model_input(
                engine, input, batch, train_cache.feature_count(), count, history_len);
            const std::vector<std::size_t>& labels = batch.labels;
            const std::vector<Scalar>& values = batch.values;

            Scalar batch_loss = 0;
            nn::TrainingStepOptions step_options;
            step_options.lr = epoch_lr;
            step_options.clip_grad_norm = options.clip_grad_norm;
            step_options.normalization_units = count;
            auto step_report = NN_UNWRAP(session.train_step(
                input,
                step_options,
                [&](nn::Model& current, nn::ComputeEngine& current_engine,
                    const nn::Tensor& batch_input) -> nn::Result<nn::TrainingStepOutput> {
                    auto output = current.forward(batch_input);
                    NN_TRY_CHECK(output);
                    Scalar loss_value = 0;
                    nn::Result<nn::Tensor> gradient{};
                    if (history_len > 0) {
                        // B: dual-head loss. Policy CE over rows 0..361 +
                        // value MSE over row 362, combined into one grad.
                        auto loss = go_loss.forward(
                            current_engine, *output,
                            std::span<const std::size_t>(labels),
                            std::span<const Scalar>(values)
                        );
                        NN_TRY_CHECK(loss);
                        loss_value = *loss;
                        gradient = go_loss.backward();
                    } else {
                        // C: policy-only cross-entropy (M4/M5 semantics).
                        auto loss = loss_fn.forward_sparse(
                            current_engine, *output, labels, {},
                            go::policy_class_count
                        );
                        NN_TRY_CHECK(loss);
                        loss_value = *loss;
                        gradient = loss_fn.backward();
                    }
                    NN_TRY_CHECK(gradient);
                    batch_loss = loss_value;
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
        final_train_loss = mean_loss;
        const EvalMetrics eval = evaluate(
            session.model(), eval_cache, options.common.batch_size, 0, history_len,
            options.value_weight
        );
        final_eval = eval;
        session.model().set_training(true);
        const double epoch_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - epoch_started
        ).count();
        const std::uint64_t completed = static_cast<std::uint64_t>(epoch + 1);
        history.push_back({
            {"epoch", epoch + 1},
            {"lr", static_cast<double>(epoch_lr)},
            {"train_loss", static_cast<double>(mean_loss)},
            {"eval", metrics_json(eval)},
            {"seconds", epoch_seconds},
        });
        append_metrics_log(options.metrics_log, {
            {"record", "epoch"},
            {"epoch", epoch + 1},
            {"lr", static_cast<double>(epoch_lr)},
            {"train_loss", static_cast<double>(mean_loss)},
            {"eval", metrics_json(eval)},
            {"seconds", epoch_seconds},
            {"global_step", global_step_base + session.step_count()},
        });
        std::cerr << "epoch " << (epoch + 1) << '/' << options.epochs
                  << " train_loss=" << static_cast<double>(mean_loss)
                  << " eval_loss=" << static_cast<double>(eval.loss)
                  << " top1=" << eval.top1
                  << " top3=" << eval.top3
                  << " value_auc=" << eval.value_auc
                  << " seconds=" << epoch_seconds << '\n';

        // epoch 边界存档：断点永远是"可续点"，崩溃最多丢一个 epoch。
        std::filesystem::create_directories(
            std::filesystem::path(options.checkpoint).parent_path()
        );
        save_checkpoint_file(
            options.checkpoint,
            session.model(),
            spec,
            make_train_state(
                options, session, completed, global_step_base,
                dataset_sha256, indices.size()
            ),
            session.optimizer()
        );
    }

    session.model().set_training(false);
    std::filesystem::create_directories(
        std::filesystem::path(options.checkpoint).parent_path()
    );
    std::filesystem::create_directories(
        std::filesystem::path(options.model).parent_path()
    );
    save_checkpoint_file(
        options.checkpoint,
        session.model(),
        spec,
        make_train_state(
            options, session, static_cast<std::uint64_t>(final_epoch),
            global_step_base, dataset_sha256, indices.size()
        ),
        session.optimizer()
    );
    NN_UNWRAP(nn::save_model(options.model, session.model(), spec));

    const nn::ModelSpec loaded_checkpoint_spec =
        NN_UNWRAP(go::peek_go_model_spec(options.checkpoint));
    nn::Model checkpoint_model =
        NN_UNWRAP(go::build_project_model_from_spec(engine, loaded_checkpoint_spec));
    load_weights(options.checkpoint, checkpoint_model, nn::WeightLoadIntent::Inference);
    const double checkpoint_diff = max_logit_difference(
        session.model(), checkpoint_model, eval_cache,
        options.common.batch_size, history_len
    );

    const nn::ModelSpec loaded_model_spec =
        NN_UNWRAP(go::peek_go_model_spec(options.model));
    nn::Model inference_model =
        NN_UNWRAP(go::build_project_model_from_spec(engine, loaded_model_spec));
    load_weights(options.model, inference_model, nn::WeightLoadIntent::Inference);
    const double model_diff = max_logit_difference(
        session.model(), inference_model, eval_cache,
        options.common.batch_size, history_len
    );

    const bool roundtrip_pass = checkpoint_diff <= 1.0e-5 && model_diff <= 1.0e-5;
    const bool finite_losses =
        std::isfinite(static_cast<double>(first_train_loss)) &&
        std::isfinite(static_cast<double>(final_train_loss)) &&
        std::isfinite(static_cast<double>(initial_eval.loss)) &&
        std::isfinite(static_cast<double>(final_eval.loss));
    // 损失下降门槛只对"从头训练的完整轮"生效：续训尾巴的 cosine 后段本来
    // 就趋于平坦，把它当失败会误杀合法续训。续训轮的门槛 = 有限 + 往返无损。
    const bool loss_declined = resumed
        ? finite_losses
        : finite_losses &&
          final_train_loss < first_train_loss &&
          final_eval.loss < initial_eval.loss;
    const bool pass = roundtrip_pass && loss_declined;
    const double total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - training_started
    ).count();
    json report{
        {"schema", 2},
        {"status", pass ? "pass" : "fail"},
        {"model", arch == "b" ? "b-encoder-v1" : "c-validator-v1"},
        {"arch", arch},
        {"history_len", history_len},
        {"mode", resumed ? "resume" : "fresh"},
        {"resume_from", options.resume},
        {"dataset", train_cache.path().string()},
        {"dataset_sha256", dataset_sha256},
        {"eval_dataset", eval_cache.path().string()},
        {"engine", options.common.use_gpu ? "vulkan" : "cpu"},
        {"gpu_device", options.common.gpu_device},
        {"seed", options.seed},
        {"epochs", options.epochs},
        {"start_epoch", start_epoch},
        {"stop_after_epoch", final_epoch},
        {"batch_size", options.common.batch_size},
        {"optimizer", options.optimizer},
        {"schedule", options.schedule},
        {"lr", static_cast<double>(options.lr)},
        {"min_lr", static_cast<double>(options.min_lr)},
        {"weight_decay", static_cast<double>(options.weight_decay)},
        {"beta1", static_cast<double>(options.beta1)},
        {"beta2", static_cast<double>(options.beta2)},
        {"clip_grad_norm", static_cast<double>(options.clip_grad_norm)},
        {"warmup_epochs", options.warmup_epochs},
        {"train_samples", indices.size()},
        {"eval_samples", eval_cache.size()},
        {"global_step", global_step_base + session.step_count()},
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
        {"metrics_log", options.metrics_log},
        {"health", {
            {"loss_declined", loss_declined},
            {"checkpoint_roundtrip", roundtrip_pass},
        }},
        {"seconds", total_seconds},
    };
    write_json(options.metrics_out, report);
    append_metrics_log(options.metrics_log, {
        {"record", "final"},
        {"status", pass ? "pass" : "fail"},
        {"mode", resumed ? "resume" : "fresh"},
        {"final_epoch", final_epoch},
        {"train_loss", static_cast<double>(final_train_loss)},
        {"eval", metrics_json(final_eval)},
        {"checkpoint_max_logit_diff", checkpoint_diff},
        {"model_max_logit_diff", model_diff},
        {"global_step", global_step_base + session.step_count()},
        {"seconds", total_seconds},
    });
    std::cout << report.dump(2) << '\n';
    if (!pass) {
        throw std::runtime_error(
            "M5 acceptance failed: checkpoint round-trip or loss gate"
        );
    }
    return 0;
}

struct MakeModelOptions {
    CommonOptions common;
    std::string output;
    std::string arch = "b";  // b = B encoder (default), c = M4 CNN validator
    go::GoModelConfig model_config;
};

MakeModelOptions parse_make_model_options(int argc, char** argv) {
    MakeModelOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (is_common_option(arg)) {
            parse_common_option(arg, i, argc, argv, options.common);
        } else if (arg == "--output") {
            options.output = option_value(i, argc, argv);
        } else if (arg == "--arch") {
            options.arch = option_value(i, argc, argv);
        } else if (arg == "--d-model") {
            options.model_config.d_model = std::stoull(option_value(i, argc, argv));
        } else if (arg == "--num-layers") {
            options.model_config.num_layers = std::stoull(option_value(i, argc, argv));
        } else if (arg == "--num-heads") {
            options.model_config.num_heads = std::stoull(option_value(i, argc, argv));
        } else if (arg == "--d-ff") {
            options.model_config.d_ff = std::stoull(option_value(i, argc, argv));
        } else if (arg == "--conv-channels") {
            options.model_config.conv_channels = std::stoull(option_value(i, argc, argv));
        } else if (arg == "--history-len") {
            options.model_config.history_len = std::stoull(option_value(i, argc, argv));
        } else if (arg == "--cnn-policy") {
            options.model_config.cnn_policy = std::stoi(option_value(i, argc, argv)) != 0;
        } else {
            throw std::invalid_argument("unknown make-model option: " + std::string(arg));
        }
    }
    if (options.output.empty()) {
        throw std::invalid_argument("make-model requires --output");
    }
    if (options.arch != "b" && options.arch != "c") {
        throw std::invalid_argument("--arch must be b or c");
    }
    return options;
}

// 确定性地构建一个未训练的推理模型（GTP/Arena 测试需要一个与训练侧同一
// spec 的模型文件；模型初始化本身是 seed 确定的）。默认产 B；--arch c
// 保留以复现 M4 C 验证器。
int make_model(const MakeModelOptions& options) {
    std::shared_ptr<nn::ComputeEngine> engine_ptr = create_engine(options.common);
    nn::ComputeEngine& engine = *engine_ptr;
    nn::ModelSpec spec;
    nn::Model model =
        options.arch == "b"
            ? NN_UNWRAP(go::build_project_model(engine, options.model_config))
            : NN_UNWRAP(nn::build_cnn_model(engine, c_validator_config()));
    if (options.arch == "b") {
        spec = go::make_go_spec(options.model_config);
    } else {
        spec = c_validator_spec(c_validator_config());
    }
    model.set_spec(spec);
    model.set_training(false);
    std::filesystem::create_directories(
        std::filesystem::path(options.output).parent_path()
    );
    NN_UNWRAP(nn::save_model(options.output, model, spec));
    std::cout << json{
        {"schema", 1},
        {"status", "pass"},
        {"model", options.output},
        {"arch", options.arch},
        {"engine", options.common.use_gpu ? "vulkan" : "cpu"},
    }.dump(2) << '\n';
    return 0;
}

int evaluate_cli(const EvalOptions& options) {
    go::FeatureCache cache = go::FeatureCache::open(options.dataset);
    std::shared_ptr<nn::ComputeEngine> engine_ptr = create_engine(options.common);
    nn::ComputeEngine& engine = *engine_ptr;
    const nn::ModelSpec spec = NN_UNWRAP(go::peek_go_model_spec(options.model));
    std::cerr << "loaded model spec: " << nn::spec_summary(spec) << '\n';
    nn::Model model = NN_UNWRAP(go::build_project_model_from_spec(engine, spec));
    const std::size_t history_len =
        go::is_go_spec(spec) ? NN_UNWRAP(go::go_config_from_spec(spec)).history_len : 0;
    if (history_len > 0 && cache.history_len() != history_len) {
        throw std::runtime_error(
            "B model needs move history but cache '" + options.dataset +
            "' has history_len=" + std::to_string(cache.history_len()) +
            " (model K=" + std::to_string(history_len) +
            "). Regenerate with --history-len " + std::to_string(history_len) + "."
        );
    }
    std::cerr << "built model from spec\n";
    const nn::ModelRole role = NN_UNWRAP(nn::peek_model_role(options.model));
    std::cerr << "model role: " << nn::model_role_name(role) << '\n';
    load_weights(options.model, model, nn::WeightLoadIntent::Inference);
    std::cerr << "loaded model weights\n";
    const EvalMetrics metrics = evaluate(
        model, cache, options.common.batch_size, options.common.max_samples,
        history_len
    );
    std::cout << json{
        {"schema", 1},
        {"status", "pass"},
        {"model", options.model},
        {"role", nn::model_role_name(role)},
        {"arch", go::is_go_spec(spec) ? "b" : "c"},
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
        << " [--gpu=40HX] [--epochs 4] [--batch-size 64] [--arch b|c]\n"
        << "       [--d-model 192] [--num-layers 8] [--num-heads 8] [--d-ff 768]"
        << " [--conv-channels 64] [--history-len 64] [--value-weight 1.0]"
        << " [--cnn-policy 0|1]\n"
        << "       [--optimizer adamw] [--schedule cosine] [--lr 0.001]"
        << " [--min-lr 0.00001] [--weight-decay 0.0001]\n"
        << "       [--schedule cosine|constant|step_cosine]"
        << "       [--beta1 0.9] [--beta2 0.999] [--clip-grad-norm 1.0]"
        << " [--warmup-epochs 0]\n"
        << "       [--resume FILE.nnckpt] [--stop-after-epoch N] [--metrics-log FILE.jsonl]\n"
        << "  " << program << " evaluate --dataset FILE.ngf --model FILE.nnmodel"
        << " [--gpu=40HX] [--batch-size 64]\n"
        << "  " << program << " make-model --output FILE.nnmodel [--arch b|c]"
        << " [--history-len 64] [--gpu=40HX]\n";
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
        if (command == "make-model") {
            MakeModelOptions options = parse_make_model_options(argc, argv);
            return make_model(options);
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
