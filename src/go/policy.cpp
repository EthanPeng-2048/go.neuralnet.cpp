#include "go/policy.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include "go/features.hpp"

namespace go {

struct PolicyPlayer::Impl {
    std::string model_path;
    std::shared_ptr<nn::ComputeEngine> engine;
};

PolicyPlayer PolicyPlayer::load(
    const std::string& model_path,
    const std::string& device
) {
    // 引擎仍然在这里建立：GTP/Arena 的 --gpu/--cpu 语义与设备可用性检查
    // 不依赖模型层，先于模型重写保留下来。
    nn::cli::EngineConfig config;
    config.use_gpu = !device.empty();
    config.gpu_device = device;
    auto engine_result = nn::cli::create_engine(config, std::cerr);
    if (!engine_result) {
        throw std::runtime_error(engine_result.error().message);
    }
    auto impl = std::make_unique<Impl>();
    impl->engine = std::move(*engine_result);
    impl->model_path = model_path;
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

std::optional<Point> PolicyPlayer::select_move(const Game& game) {
    if (game.width() != static_cast<int>(training_board_size) ||
        game.height() != static_cast<int>(training_board_size)) {
        throw std::invalid_argument(
            "policy inference requires a 19x19 board (M3 v1 feature contract)"
        );
    }
    // 模型层正在重写（B 架构 + dsl::compute + AOT 融合）。在它落地前，
    // genmove 明确报错，而不是退回已删除的手拼实现继续出招——一个"能出招
    // 但输入契约与训练侧不一致"的模型比报错更危险。
    throw std::runtime_error(
        "policy model is not loadable: M7 model rewrite in progress"
    );
}

}  // namespace go
