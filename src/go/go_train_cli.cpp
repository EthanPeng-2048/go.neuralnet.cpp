#include <algorithm>
#include <any>
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
#include <neuralnet.cpp/batch_source.hpp>    // IndexedBatchSource（RFC 32 §4）
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>
#include <neuralnet.cpp/run_driver.hpp>      // RunDriver（RFC 32 §5）
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
    // M10b 诊断：>0 = 每 N 个 optimizer step 把分组梯度 L2 范数 + 分项损失
    // 追加写进 <metrics_log>.grads.jsonl（价值头梯度埋点，判"梯度断/信号弱"）。
    int grad_debug = 0;
    std::string grad_debug_log;  // 空 = 从 metrics_log 派生（.jsonl → .grads.jsonl）
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
    // M10b 诊断：未加权分项损失。价值 AUC 停滞时单看 combined loss 无法
    // 区分"策略在降、价值平"——分项直接暴露价值 MSE 是否真的在学。
    double policy_loss = 0;
    double value_loss = 0;
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
        } else if (arg == "--grad-debug") {
            options.grad_debug = std::stoi(option_value(i, argc, argv));
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
        } else if (arg == "--value-spatial") {
            options.model_config.value_from_spatial =
                std::stoi(option_value(i, argc, argv)) != 0;
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
    if (options.grad_debug < 0) {
        throw std::invalid_argument("grad-debug must be non-negative");
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
    if (options.grad_debug > 0 && options.grad_debug_log.empty()) {
        options.grad_debug_log = options.metrics_log;
        const std::string jsonl_suffix = ".jsonl";
        if (options.grad_debug_log.size() > jsonl_suffix.size() &&
            options.grad_debug_log.ends_with(jsonl_suffix)) {
            options.grad_debug_log.resize(
                options.grad_debug_log.size() - jsonl_suffix.size());
        }
        options.grad_debug_log += ".grads.jsonl";
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
LoadedBatch load_batch_span(
    go::FeatureCache& cache,
    std::span<const std::size_t> indices,
    std::size_t history_len = 0
) {
    const std::size_t count = indices.size();
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
            indices[column], sample, &metadata,
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

LoadedBatch load_batch_planes(
    go::FeatureCache& cache,
    const std::vector<std::size_t>& indices,
    std::size_t first,
    std::size_t count,
    std::size_t history_len = 0
) {
    return load_batch_span(
        cache,
        std::span<const std::size_t>(indices.data() + first, count),
        history_len
    );
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
    Scalar policy_loss_sum = 0;
    Scalar value_loss_sum = 0;
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
            // 未加权分项：诊断用（价值 MSE 是否在降）。
            policy_loss_sum += go_loss.last_policy_loss() * static_cast<Scalar>(count);
            value_loss_sum += go_loss.last_value_loss() * static_cast<Scalar>(count);
        } else {
            loss = NN_UNWRAP(loss_fn.forward_sparse(
                engine, output, labels, {}, go::policy_class_count));
            policy_loss_sum += loss * static_cast<Scalar>(count);
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
        static_cast<double>(policy_loss_sum / static_cast<Scalar>(indices.size())),
        static_cast<double>(value_loss_sum / static_cast<Scalar>(indices.size())),
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
        {"policy_loss", metrics.policy_loss},
        {"value_loss", metrics.value_loss},
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

// ── 断点状态装配（v1.14.1：域只填域字段，库负责其余）─────────────────────
// 计数器 / 优化器 t / recipe 的 TrainingConfig 段（optimizer/lr/decay/beta/
// 裁剪/累积/精度）全部由 TrainingSession::save_checkpoint 内部同步与派生
// （fill_recipe）；RunDriver 在断点时机把 epoch/step_in_epoch/global_step/lr
// 写进状态。域侧只拥有：曲线形状（epochs/调度/warmup/batch_size）与数据集
// 身份（.ngf sha256 + 样本数 —— 内容变了就不存在正确的续训语义）。
//
// Go 训练器复用上游 TextTrainState 承载域字段（state_kind="text"）：上游
// train_state_from_kv 工厂硬编码只认 text kind，自定义 kind 的断点写了读不
// 回来（自定义域注册表缺失，已报上游 issue neuralnet.cpp#44）。DatasetIdentity
// 正好覆盖 Go 的域字段；待上游补注册表后切换 GoTrainState。
nn::TextTrainState make_go_state(
    const TrainOptions& options,
    const std::string& dataset_sha256,
    std::size_t train_samples
) {
    nn::TextTrainState state;
    state.recipe.lr_schedule = options.schedule;
    state.recipe.epochs = static_cast<std::uint64_t>(options.epochs);
    state.recipe.warmup_epochs = static_cast<std::uint64_t>(options.warmup_epochs);
    state.recipe.batch_size = options.common.batch_size;
    state.recipe.micro_batch_size = options.common.batch_size;
    state.recipe.batch_size_is_global = true;
    state.dataset.path = options.dataset;
    state.dataset.sha256 = dataset_sha256;
    state.dataset.num_samples = train_samples;
    state.dataset.steps_per_epoch =
        (train_samples + options.common.batch_size - 1) / options.common.batch_size;
    return state;
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
    const bool resumed = !options.resume.empty();
    int start_epoch = 0;

    if (resumed) {
        // ── 断点续训：状态段为唯一权威（v1.14.1：权重载入交给
        // session.resume，校验先行、无半提交；优化器 t 取权威值）──────
        const nn::ModelSpec file_spec = NN_UNWRAP(go::peek_go_model_spec(options.resume));
        model = std::make_unique<nn::Model>(
            NN_UNWRAP(go::build_project_model_from_spec(engine, file_spec))
        );
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

    const std::size_t train_samples = sample_indices(
        train_cache, options.common.max_samples
    ).size();
    if (train_samples == 0) {
        throw std::runtime_error("training cache has no samples");
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

    // 域状态：曲线形状 + 数据集身份。优化器/精度/TrainingConfig 段由
    // session.save_checkpoint 的 fill_recipe 派生；计数器由 RunDriver 同步。
    nn::TextTrainState domain_state = make_go_state(
        options, dataset_sha256, train_samples
    );

    if (resumed) {
        // 库口径续训（Strict）：kind + 配方关键维度（optimizer/调度/epochs/
        // warmup/batch/accum/精度）+ 计数器不越界 + 域级 verify_extra
        //（TextTrainState = 数据集 sha256/样本数）。校验通过后才载入权重
        //（v1.14.1 无半提交），优化器 t 取权威值。
        session.fill_recipe(domain_state.recipe);
        std::unique_ptr<nn::TrainState> loaded = NN_UNWRAP(
            session.resume(options.resume, &domain_state, nn::ResumePolicy::Strict)
        );
        const auto* saved = dynamic_cast<const nn::TextTrainState*>(loaded.get());
        if (saved == nullptr) {
            throw std::runtime_error(
                std::string("--resume 的训练状态段不是 Go 训练器可识别的类型（kind=") +
                loaded->state_kind() + "）: " + options.resume
            );
        }
        start_epoch = static_cast<int>(saved->counters.epoch);
        std::cerr << "resumed from " << options.resume
                  << ": epoch=" << start_epoch
                  << " global_step=" << loaded->counters.global_step
                  << " optimizer=" << saved->recipe.optimizer
                  << " optimizer_step=" << saved->optimizer_step
                  << '\n';
        // RunDriver 从断点计数器继续（epoch/step_in_epoch）；域字段随状态走。
        domain_state = *static_cast<nn::TextTrainState*>(loaded.get());
    }

    // 初始评估（续训 = 断点权重建模）+ 损失下降门槛的未训练基线。
    EvalMetrics initial_eval = evaluate(
        session.model(), eval_cache, options.common.batch_size, 0, history_len,
        options.value_weight
    );
    session.model().set_training(true);

    Scalar first_train_loss = initial_eval.loss;
    const int planned_final_epoch = options.stop_after_epoch > 0
        ? options.stop_after_epoch
        : options.epochs;
    if (planned_final_epoch > start_epoch) {
        // 基线必须与最终 train_loss 同数据集（用 eval 集当基线会把合法的
        // 单轮收敛误判成失败）。
        first_train_loss = evaluate(
            session.model(), train_cache, options.common.batch_size, kBaselineSamples,
            history_len, options.value_weight
        ).loss;
        session.model().set_training(true);
    }

    // ── 数据供给（IndexedBatchSource：epoch 网格 + 单流重放 shuffle）─────
    // shuffle 语义 = 库口径（恒等序起、单 RNG 依次重放 epoch+1 次），与
    // 旧 CLI 的 (seed,epoch) 独立播种不同：旧 checkpoint 不承诺数据序兼容
    //（本项目决定不回续旧断点，M10b 停在 epoch 2 的断点只当只读权重）。
    struct GoBatchPayload {
        nn::Tensor input;
        std::vector<std::size_t> labels;
        std::vector<Scalar> values;
    };
    const std::size_t input_rows = feature_count_plus_history(train_cache, history_len);
    auto assemble = [&](std::span<const std::size_t> indices)
        -> nn::Result<std::pair<std::any, std::size_t>> {
        const std::size_t count = indices.size();
        LoadedBatch batch = load_batch_span(train_cache, indices, history_len);
        nn::Tensor input = engine.create_tensor(input_rows, count);
        if (!input.valid()) {
            NN_FAIL("cannot allocate training input tensor");
        }
        write_model_input(
            engine, input, batch, train_cache.feature_count(), count, history_len
        );
        GoBatchPayload payload{
            std::move(input), std::move(batch.labels), std::move(batch.values)
        };
        return std::make_pair(std::any(std::move(payload)), count);
    };
    auto source_result = nn::IndexedBatchSource::create(
        train_samples, options.common.batch_size, options.seed, std::move(assemble)
    );
    if (!source_result) {
        throw std::runtime_error(
            "cannot create batch source: " + source_result.error().message
        );
    }
    nn::IndexedBatchSource source = std::move(*source_result);

    // ── lr 调度（v1.14.1：库原生；step_cosine 用 epoch 级分段余弦表）────
    nn::RunDriverConfig driver_config;
    driver_config.epochs = options.epochs;
    driver_config.stop_after_epoch = options.stop_after_epoch;
    driver_config.lr = options.lr;
    driver_config.min_lr = options.min_lr;
    driver_config.warmup_epochs = options.warmup_epochs;
    if (options.schedule == "step_cosine") {
        // 3 段 SGDR：段内余弦退火 + 段末重启。上游 epoch 级 step_cosine
        // （issue #31）与旧手拼表同语义，经 lr_per_epoch 逃生舱喂给 driver
        //（RunDriver 把字符串 "step_cosine" 解释成 step 级，不能直接传）。
        nn::LrScheduleConfig segments;
        segments.base_lr = options.lr;
        segments.min_lr = options.min_lr;
        segments.total_epochs = options.epochs;
        segments.warmup_epochs = options.warmup_epochs;
        segments.schedule = "step_cosine";
        segments.n_segments = 3;
        for (int epoch = 0; epoch < options.epochs; ++epoch) {
            driver_config.lr_per_epoch.push_back(
                nn::compute_epoch_lr(segments, epoch));
        }
        driver_config.lr_schedule = "fixed";  // 表优先；显式写明不留歧义
    } else {
        driver_config.lr_schedule =
            options.schedule == "constant" ? "fixed" : options.schedule;
    }
    driver_config.save_interval = 0;  // epoch 边界断点（崩溃最多丢一个 epoch）

    // ── 钩子：do_update / evaluate / on_epoch_end / on_checkpoint ────────
    go::GoLoss go_loss(options.value_weight);
    nn::CrossEntropyLoss loss_fn;

    EvalMetrics final_eval = initial_eval;
    Scalar final_train_loss = 0;
    json history = json::array();
    Scalar epoch_loss_sum = 0;
    Scalar epoch_policy_sum = 0;
    Scalar epoch_value_sum = 0;
    std::size_t epoch_seen = 0;
    std::size_t epoch_index = 0;
    std::uint64_t last_global_step = 0;
    auto epoch_started = std::chrono::steady_clock::now();

    nn::RunHooks hooks;
    hooks.do_update = [&](nn::TrainingSession& s, const nn::UpdateRequest& req)
        -> nn::Result<nn::UpdateOutcome> {
        if (req.epoch != epoch_index || req.step == 0) {
            epoch_index = req.epoch;
            epoch_loss_sum = 0;
            epoch_policy_sum = 0;
            epoch_value_sum = 0;
            epoch_seen = 0;
            epoch_started = std::chrono::steady_clock::now();
        }
        last_global_step = req.global_step;
        NN_TRY(at_r, source.next(req.step));
        const std::size_t count = at_r->samples;
        auto* payload = std::any_cast<GoBatchPayload>(&at_r->payload);
        if (payload == nullptr) {
            NN_FAIL("batch payload 类型错误（GoBatchPayload）");
        }

        Scalar batch_loss = 0;
        nn::TrainingStepOptions step_options;
        step_options.lr = req.lr;
        step_options.clip_grad_norm = options.clip_grad_norm;
        step_options.normalization_units = at_r->units;
        // M10b 价值梯度埋点：clip 前的 hook——每 N 个 optimizer step 把
        // 分组梯度 L2 + 本批分项损失写进 .grads.jsonl（低频：逐张量
        // 读回宿主有同步开销，grad-debug=0 时零开销）。
        if (options.grad_debug > 0 &&
            (s.step_count() % static_cast<std::uint64_t>(options.grad_debug)) == 0) {
            step_options.before_update = [&]() -> nn::Result<bool> {
                json record = {
                    {"record", "grad_debug"},
                    {"epoch", req.epoch + 1},
                    {"step", static_cast<std::uint64_t>(s.step_count())},
                    {"policy_loss", static_cast<double>(go_loss.last_policy_loss())},
                    {"value_loss", static_cast<double>(go_loss.last_value_loss())},
                };
                nn::Layer& root = s.model().layer_at(0);
                if (std::string_view(root.layer_name()) == "GoModel") {
                    auto* go_model = static_cast<go::GoModel*>(&root);
                    auto norms = go_model->debug_grad_norms();
                    if (norms) {
                        json grads = json::object();
                        for (const auto& [name, value] : *norms) {
                            grads[name] = value;
                        }
                        record["grad_l2"] = grads;
                    } else {
                        record["grad_error"] = norms.error().message;
                    }
                }
                append_metrics_log(options.grad_debug_log, record);
                return nn::Result<bool>(false);
            };
        }
        auto step_report = s.train_step(
            payload->input,
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
                        std::span<const std::size_t>(payload->labels),
                        std::span<const Scalar>(payload->values)
                    );
                    NN_TRY_CHECK(loss);
                    loss_value = *loss;
                    gradient = go_loss.backward();
                } else {
                    // C: policy-only cross-entropy (M4/M5 semantics).
                    auto loss = loss_fn.forward_sparse(
                        current_engine, *output, payload->labels, {},
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
        );
        NN_TRY_CHECK(step_report);
        epoch_loss_sum += batch_loss * static_cast<Scalar>(count);
        if (history_len > 0) {
            epoch_policy_sum += go_loss.last_policy_loss() * static_cast<Scalar>(count);
            epoch_value_sum += go_loss.last_value_loss() * static_cast<Scalar>(count);
        } else {
            epoch_policy_sum += batch_loss * static_cast<Scalar>(count);
        }
        epoch_seen += count;

        nn::UpdateOutcome outcome;
        outcome.parameters_updated = step_report->parameters_updated;
        outcome.samples = count;
        outcome.loss = batch_loss;
        outcome.has_loss = true;
        return outcome;
    };
    hooks.evaluate = [&](nn::TrainingSession& s, std::size_t epoch)
        -> nn::Result<Scalar> {
        (void)epoch;
        final_eval = evaluate(
            s.model(), eval_cache, options.common.batch_size, 0, history_len,
            options.value_weight
        );
        s.model().set_training(true);
        return final_eval.loss;
    };
    hooks.on_epoch_end = [&](const nn::RunReport& report, std::size_t epoch)
        -> nn::Result<void> {
        const Scalar mean_loss = epoch_seen > 0
            ? epoch_loss_sum / static_cast<Scalar>(epoch_seen)
            : Scalar{0};
        final_train_loss = mean_loss;
        const Scalar seen = epoch_seen > 0 ? static_cast<Scalar>(epoch_seen) : Scalar{1};
        const double train_policy_loss =
            static_cast<double>(epoch_policy_sum / seen);
        const double train_value_loss =
            static_cast<double>(epoch_value_sum / seen);
        const double epoch_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - epoch_started
        ).count();
        history.push_back({
            {"epoch", epoch + 1},
            {"lr", static_cast<double>(report.final_lr)},
            {"train_loss", static_cast<double>(mean_loss)},
            {"train_policy_loss", train_policy_loss},
            {"train_value_loss", train_value_loss},
            {"eval", metrics_json(final_eval)},
            {"seconds", epoch_seconds},
        });
        append_metrics_log(options.metrics_log, {
            {"record", "epoch"},
            {"epoch", epoch + 1},
            {"lr", static_cast<double>(report.final_lr)},
            {"train_loss", static_cast<double>(mean_loss)},
            {"train_policy_loss", train_policy_loss},
            {"train_value_loss", train_value_loss},
            {"eval", metrics_json(final_eval)},
            {"seconds", epoch_seconds},
            {"global_step", last_global_step},
        });
        std::cerr << "epoch " << (epoch + 1) << '/' << options.epochs
                  << " train_loss=" << static_cast<double>(mean_loss)
                  << " policy_loss=" << train_policy_loss
                  << " value_loss=" << train_value_loss
                  << " eval_loss=" << static_cast<double>(final_eval.loss)
                  << " top1=" << final_eval.top1
                  << " top3=" << final_eval.top3
                  << " value_auc=" << final_eval.value_auc
                  << " seconds=" << epoch_seconds << '\n';
        return {};
    };
    hooks.on_checkpoint = [&](nn::TrainingSession& s, const nn::TrainState& state)
        -> nn::Result<void> {
        // 驱动已同步 epoch/step_in_epoch/global_step/lr；域字段随状态走；
        // save_checkpoint 再派生 TrainingConfig 段并导出优化器状态。
        std::unique_ptr<nn::TrainState> snapshot = state.clone();
        std::filesystem::create_directories(
            std::filesystem::path(options.checkpoint).parent_path()
        );
        return s.save_checkpoint(options.checkpoint, spec, *snapshot);
    };

    auto driver = nn::RunDriver::create(
        session, source,
        std::make_unique<nn::TextTrainState>(domain_state),
        std::move(driver_config)
    );
    if (!driver) {
        throw std::runtime_error("cannot create run driver: " + driver.error().message);
    }
    driver->set_hooks(std::move(hooks));

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
        {"stop_after_epoch", planned_final_epoch},
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
        {"train_samples", train_samples},
        {"initial_eval", metrics_json(initial_eval)},
    });

    const auto training_started = std::chrono::steady_clock::now();
    nn::RunReport run_report = NN_UNWRAP(driver->fit());
    const int final_epoch = static_cast<int>(run_report.final_epoch);

    // ── 收尾：最终断点 + 推理模型 + 往返验证 ────────────────────────────
    session.model().set_training(false);
    std::filesystem::create_directories(
        std::filesystem::path(options.checkpoint).parent_path()
    );
    std::filesystem::create_directories(
        std::filesystem::path(options.model).parent_path()
    );
    {
        nn::TextTrainState final_state = make_go_state(
            options, dataset_sha256, train_samples
        );
        // fit 的停止位置（"下一步"语义，epoch 边界已归一化）。
        final_state.counters.epoch = run_report.final_epoch;
        final_state.counters.step_in_epoch = run_report.final_step_in_epoch;
        final_state.counters.global_step = last_global_step;
        final_state.counters.lr = static_cast<double>(run_report.final_lr);
        NN_UNWRAP(session.save_checkpoint(options.checkpoint, spec, final_state));
    }
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
        {"stop_after_epoch", planned_final_epoch},
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
        {"train_samples", train_samples},
        {"eval_samples", eval_cache.size()},
        {"global_step", last_global_step},
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
        {"global_step", last_global_step},
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
        } else if (arg == "--value-spatial") {
            options.model_config.value_from_spatial =
                std::stoi(option_value(i, argc, argv)) != 0;
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
