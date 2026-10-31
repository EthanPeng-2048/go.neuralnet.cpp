#include "go/policy.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include "go/features.hpp"
#include "go/go_model.hpp"

namespace go {
namespace {

// 与 go_train_cli 相同的前向包裹：显式 batch 作用域（多 Conv 模型在
// Vulkan 下不包作用域会在第二次 Conv 分配时报 VK_ERROR_OUT_OF_DEVICE_MEMORY，
// 见 docs/benchmarks/m4-c-validator.md §7）。
nn::Tensor forward_scoped(nn::Model& model, const nn::Tensor& input) {
    nn::ComputeEngine& engine = model.engine();
    auto begin = engine.begin_batch();
    if (!begin) {
        throw std::runtime_error("cannot begin inference batch: " + begin.error().message);
    }
    auto output = model.forward(input);
    auto end = engine.end_batch();
    if (!output) {
        throw std::runtime_error("inference forward failed: " + output.error().message);
    }
    if (!end) {
        throw std::runtime_error("cannot end inference batch: " + end.error().message);
    }
    return std::move(*output);
}

// Point → GoTokenizer id（y * 19 + x）。与 .ngf 的目标坐标、Python 训练器
// 同一约定，训练与推理看到的历史 id 完全一致。
std::uint16_t history_token(const std::optional<Point>& point) {
    if (!point.has_value()) {
        return history_pass_token;  // 361
    }
    return static_cast<std::uint16_t>(
        static_cast<std::size_t>(point->y) * training_board_size +
        static_cast<std::size_t>(point->x)
    );
}

// 从当前对局重建最近 K 手历史：时间顺序、左侧 <pad> 补齐、右对齐，
// 与 .ngf v2 导出口径一致。
void fill_history(const Game& game, std::span<std::uint16_t> history) {
    const std::vector<PlayedMove> moves = game.history();
    const std::size_t k = history.size();
    std::fill(history.begin(), history.end(), history_pad_token);
    const std::size_t window = std::min<std::size_t>(k, moves.size());
    for (std::size_t offset = 0; offset < window; ++offset) {
        history[k - window + offset] =
            history_token(moves[moves.size() - window + offset].point);
    }
}

}  // namespace

struct PolicyPlayer::Impl {
    std::string model_path;
    std::shared_ptr<nn::ComputeEngine> engine;
    nn::Model model;
    std::size_t history_len;  // 0 = C 验证器（仅平面），>0 = B 编码器
    std::vector<float> features;
    std::vector<std::uint16_t> history;  // B: K 手历史，左侧 <pad> 补齐
    std::vector<nn::Scalar> packed;      // B: 平面行 + 历史行，单列
    std::vector<nn::Scalar> logits;

    explicit Impl(std::shared_ptr<nn::ComputeEngine> engine_value,
                  nn::Model model_value, std::size_t history_len_value)
        : engine(std::move(engine_value)),
          model(std::move(model_value)),
          history_len(history_len_value) {}
};

PolicyPlayer PolicyPlayer::load(
    const std::string& model_path,
    const std::string& device
) {
    nn::cli::EngineConfig config;
    config.use_gpu = !device.empty();
    config.gpu_device = device;
    auto engine_result = nn::cli::create_engine(config, std::cerr);
    if (!engine_result) {
        throw std::runtime_error(engine_result.error().message);
    }
    std::shared_ptr<nn::ComputeEngine> engine = std::move(*engine_result);

    const nn::ModelSpec spec = NN_UNWRAP(go::peek_go_model_spec(model_path));
    nn::Model model = NN_UNWRAP(go::build_project_model_from_spec(*engine, spec));
    const std::size_t history_len =
        go::is_go_spec(spec) ? NN_UNWRAP(go::go_config_from_spec(spec)).history_len : 0;
    auto verified = nn::verify_model_integrity(model_path);
    if (!verified) {
        throw std::runtime_error(
            "model integrity check failed for '" + model_path + "': " +
            verified.error().message
        );
    }
    // load_model 返回内嵌 tokenizer 字节（本模型无分词器，忽略）。
    const std::string tokenizer_bytes = NN_UNWRAP(go::load_go_model(model_path, model));
    (void)tokenizer_bytes;
    model.set_training(false);

    auto impl = std::make_unique<Impl>(std::move(engine), std::move(model), history_len);
    impl->model_path = model_path;
    impl->features.resize(feature_element_count);
    if (history_len > 0) {
        impl->history.assign(history_len, history_pad_token);
        impl->packed.resize(feature_element_count + history_len);
    }
    return PolicyPlayer(std::move(impl));
}

PolicyPlayer::PolicyPlayer(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

PolicyPlayer::PolicyPlayer(PolicyPlayer&&) noexcept = default;
PolicyPlayer& PolicyPlayer::operator=(PolicyPlayer&&) noexcept = default;
PolicyPlayer::~PolicyPlayer() = default;

const std::string& PolicyPlayer::model_path() const noexcept {
    return impl_->model_path;
}

void PolicyPlayer::forward_logits(const Game& game) {
    if (game.width() != static_cast<int>(training_board_size) ||
        game.height() != static_cast<int>(training_board_size)) {
        throw std::invalid_argument(
            "policy inference requires a 19x19 board (M3 v1 feature contract)"
        );
    }
    Impl& state = *impl_;
    const Stone color = game.to_move();

    extract_features(game, color, state.features);

    // 输入布局：列主（feature, batch），batch=1。B 模型把历史 id 作为额外的
    // K 行追加到平面之后（f32 存放的整数 id），契约见 go_model.hpp。
    nn::ComputeEngine& engine = *state.engine;
    const std::size_t input_rows = state.history_len > 0
        ? go::go_input_rows(state.history_len)
        : feature_element_count;
    nn::Tensor input = engine.create_tensor(input_rows, 1);
    if (!input.valid()) {
        throw std::runtime_error("cannot allocate policy input tensor");
    }
    if (state.history_len > 0) {
        fill_history(game, state.history);
        for (std::size_t i = 0; i < feature_element_count; ++i) {
            state.packed[i] = static_cast<nn::Scalar>(state.features[i]);
        }
        for (std::size_t i = 0; i < state.history_len; ++i) {
            state.packed[feature_element_count + i] =
                static_cast<nn::Scalar>(state.history[i]);
        }
        NN_UNWRAP(engine.write(input, std::span<const nn::Scalar>(state.packed)));
    } else {
        NN_UNWRAP(engine.write(input, std::span<const nn::Scalar>(state.features)));
    }

    nn::Tensor output = forward_scoped(state.model, input);
    state.logits.resize(output.size());
    NN_UNWRAP(engine.read(output, std::span<nn::Scalar>(state.logits)));
}

std::optional<Point> PolicyPlayer::select_move(const Game& game) {
    forward_logits(game);
    Impl& state = *impl_;
    const Stone color = game.to_move();

    // 合法性掩码 + argmax（host 侧，roadmap §4）。argmax 只扫策略行
    // [0, policy_class_count)；B 的第 362 行是价值输出，不参与选点。
    constexpr std::size_t kPass = feature_plane_stride;  // 361
    nn::Scalar best = -std::numeric_limits<nn::Scalar>::infinity();
    std::optional<Point> best_point;  // nullopt = pass
    bool have_move = false;

    if (game.is_legal(color, std::nullopt)) {
        best = state.logits[kPass];
        have_move = true;
    }
    for (int y = 0; y < game.height(); ++y) {
        for (int x = 0; x < game.width(); ++x) {
            const Point point{x, y};
            if (!game.is_legal(color, point)) {
                continue;
            }
            const std::size_t index =
                static_cast<std::size_t>(y) * game.width() + x;
            if (state.logits[index] > best) {
                best = state.logits[index];
                best_point = point;
                have_move = true;
            }
        }
    }
    if (!have_move) {
        // 全部非法（理论上只有终局后才会发生）：按 pass 处理。
        return std::nullopt;
    }
    return best_point;
}

PolicyEvaluation PolicyPlayer::evaluate(const Game& game) {
    forward_logits(game);
    Impl& state = *impl_;
    const Stone color = game.to_move();

    PolicyEvaluation result;
    result.policy.assign(policy_class_count, 0.0F);

    // 合法着集合上的 softmax：非法着保持 0，这份分布可以直接画落子热力图。
    std::vector<std::size_t> legal;
    legal.reserve(40);
    constexpr std::size_t kPass = feature_plane_stride;  // 361
    if (game.is_legal(color, std::nullopt)) {
        legal.push_back(kPass);
    }
    for (int y = 0; y < game.height(); ++y) {
        for (int x = 0; x < game.width(); ++x) {
            if (game.is_legal(color, Point{x, y})) {
                legal.push_back(static_cast<std::size_t>(y) * game.width() + x);
            }
        }
    }
    if (!legal.empty()) {
        nn::Scalar max_logit = -std::numeric_limits<nn::Scalar>::infinity();
        for (const std::size_t index : legal) {
            max_logit = std::max(max_logit, state.logits[index]);
        }
        double total = 0.0;
        for (const std::size_t index : legal) {
            const double weight =
                std::exp(static_cast<double>(state.logits[index] - max_logit));
            total += weight;
            result.policy[index] = static_cast<float>(weight);
        }
        for (const std::size_t index : legal) {
            result.policy[index] =
                static_cast<float>(static_cast<double>(result.policy[index]) / total);
        }
    }

    // 价值行只存在于 B 编码器（363 行打包输出）；C 验证器 362 行全是策略。
    if (state.history_len > 0 && state.logits.size() > value_row_index) {
        result.value = static_cast<float>(state.logits[value_row_index]);
    }
    return result;
}

}  // namespace go
