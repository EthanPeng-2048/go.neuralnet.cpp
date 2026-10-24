// ── M7 GoModel 契约测试 ───────────────────────────────────────────────────
//
// 三层断言，从便宜到昂贵：
//   1. 输入/输出契约常量（行数、token 数、打包行、历史 id 语义）；
//   2. ModelSpec 往返（D6 零补丁：Transformer + 魔数标记，外来 spec 必须拒绝）；
//   3. 一次真实的 forward/backward（tiny 配置，CPU），锁住 (363,batch) 打包
//      输出与输入梯度形状——布局错误在这里就炸，而不是等到训练 loss 异常。

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <neuralnet.cpp/nn.hpp>

#include "go/go_model.hpp"

namespace {

int failures = 0;

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void require_eq(std::size_t actual, std::size_t expected, const char* message) {
    if (actual != expected) {
        std::fprintf(stderr, "FAIL: %s (expected %zu, got %zu)\n",
                     message, expected, actual);
        ++failures;
    }
}

// ── 1. 契约常量 ─────────────────────────────────────────────────────────
void test_contract_constants() {
    require_eq(go::go_input_rows(64), 3313, "go_input_rows(64) = 3249+64");
    require_eq(go::go_total_tokens(64), 90, "go_total_tokens(64) = 25+64+1");
    require_eq(go::packed_output_rows, 363, "packed_output_rows = 362+1");
    require_eq(go::value_row_index, 362, "value_row_index");
    require_eq(go::region_token_count, 25, "region_token_count");
    require_eq(go::history_pass_token, 361, "history_pass_token");
    require_eq(go::history_pad_token, 365, "history_pad_token");
    require_eq(go::history_vocab_size, 367, "history_vocab_size");
    require(go::history_pad_token < go::history_vocab_size,
            "pad token must be inside the embedding table");
    require(go::history_pass_token < go::policy_class_count,
            "pass token must be inside the policy classes");
    require_eq(go::segment_count, 3, "segment_count");
}

// ── 2. ModelSpec 往返 ───────────────────────────────────────────────────
void test_spec_roundtrip() {
    go::GoModelConfig config;
    config.d_model = 48;
    config.num_layers = 3;
    config.num_heads = 4;
    config.d_ff = 96;
    config.conv_channels = 16;
    config.history_len = 12;
    config.history_vocab = go::history_vocab_size;
    config.region_grid = go::region_grid;

    const nn::ModelSpec spec = go::make_go_spec(config);
    require(go::is_go_spec(spec), "make_go_spec output must be recognized");
    require_eq(spec.seq_len, go::go_total_tokens(12), "spec.seq_len = T");
    require_eq(spec.patch_size, go::region_grid, "spec.patch_size = region grid");

    const go::GoModelConfig restored = NN_UNWRAP(go::go_config_from_spec(spec));
    require_eq(restored.d_model, config.d_model, "roundtrip d_model");
    require_eq(restored.num_layers, config.num_layers, "roundtrip num_layers");
    require_eq(restored.num_heads, config.num_heads, "roundtrip num_heads");
    require_eq(restored.d_ff, config.d_ff, "roundtrip d_ff");
    require_eq(restored.conv_channels, config.conv_channels, "roundtrip conv_channels");
    require_eq(restored.history_len, config.history_len, "roundtrip history_len");

    // 外来 Transformer spec（无 Go 标记）必须被拒绝，否则会拿 Go 维度去
    // 构造一个语义完全不同的模型。
    nn::ModelSpec foreign;
    foreign.type = nn::ModelType::Transformer;
    foreign.d_model = 48;
    foreign.num_heads = 4;
    foreign.d_ff = 96;
    foreign.num_layers = 3;
    foreign.patch_size = 4;
    foreign.vocab_size = 10;
    foreign.seq_len = 16;
    require(!go::is_go_spec(foreign), "foreign Transformer spec must not pass");
}

// ── 3. forward / backward 形状（tiny 配置，CPU）─────────────────────────
void test_forward_backward() {
    auto engine = nn::CpuEngine::create();
    require(static_cast<bool>(engine), "cpu engine");

    go::GoModelConfig config;
    config.d_model = 16;
    config.num_layers = 1;
    config.num_heads = 2;
    config.d_ff = 32;
    config.conv_channels = 8;
    config.history_len = 4;
    config.history_vocab = go::history_vocab_size;
    config.region_grid = go::region_grid;

    nn::Model model = NN_UNWRAP(go::build_project_model(*engine, config));
    model.set_training(true);

    const std::size_t batch = 2;
    const std::size_t rows = go::go_input_rows(config.history_len);
    std::vector<nn::Scalar> input_data(rows * batch, nn::Scalar{0});
    for (std::size_t column = 0; column < batch; ++column) {
        // 平面行：小随机值（确定性）
        for (std::size_t r = 0; r < go::feature_element_count; ++r) {
            input_data[r * batch + column] =
                static_cast<nn::Scalar>(((r * 7 + column * 3) % 11) - 5) * nn::Scalar{0.01};
        }
        // 历史行：合法 point id（0..360）+ pass/pad
        for (std::size_t k = 0; k < config.history_len; ++k) {
            input_data[(go::feature_element_count + k) * batch + column] =
                static_cast<nn::Scalar>((k * 37 + column) % go::history_vocab_size);
        }
    }
    nn::Tensor input = NN_UNWRAP(engine->try_create_tensor(rows, batch));
    NN_UNWRAP(engine->write(input, std::span<const nn::Scalar>(input_data)));

    nn::Tensor packed = NN_UNWRAP(model.forward(input));
    require_eq(packed.rows(), go::packed_output_rows, "packed rows = 363");
    require_eq(packed.cols(), batch, "packed cols = batch");

    std::vector<nn::Scalar> logits(packed.size(), nn::Scalar{0});
    NN_UNWRAP(engine->read(packed, std::span<nn::Scalar>(logits)));
    for (std::size_t column = 0; column < batch; ++column) {
        const nn::Scalar value = logits[go::value_row_index * batch + column];
        require(std::isfinite(static_cast<double>(value)), "value row is finite");
        require(value >= nn::Scalar{-1} && value <= nn::Scalar{1},
                "value row must be tanh-bounded");
    }

    // 打包输出的梯度必须原样回传到 (rows, batch)。
    std::vector<nn::Scalar> grad_data(go::packed_output_rows * batch, nn::Scalar{0});
    for (std::size_t i = 0; i < go::policy_class_count * batch; ++i) {
        grad_data[i] = static_cast<nn::Scalar>((i % 5) - 2) * nn::Scalar{0.01};
    }
    for (std::size_t column = 0; column < batch; ++column) {
        grad_data[go::value_row_index * batch + column] = nn::Scalar{0.5};
    }
    nn::Tensor grad_output = NN_UNWRAP(engine->try_create_tensor(
        go::packed_output_rows, batch));
    NN_UNWRAP(engine->write(grad_output, std::span<const nn::Scalar>(grad_data)));
    nn::Tensor grad_input = NN_UNWRAP(model.backward(grad_output));
    require_eq(grad_input.rows(), rows, "grad_input rows");
    require_eq(grad_input.cols(), batch, "grad_input cols");

    // 参数梯度必须全部有限（NaN 会静默毁掉整个训练）。
    auto gradients = model.param_gradients();
    require(!gradients.empty(), "model has parameters");
    for (nn::TensorRef reference : gradients) {
        nn::Tensor& tensor = reference.get();
        std::vector<nn::Scalar> buffer(tensor.size(), nn::Scalar{0});
        NN_UNWRAP(engine->read(tensor, std::span<nn::Scalar>(buffer)));
        for (nn::Scalar g : buffer) {
            if (!std::isfinite(static_cast<double>(g))) {
                require(false, "parameter gradient is not finite");
                return;
            }
        }
    }
}

// ── 4. GoLoss：双头损失 + 打包梯度 ──────────────────────────────────────
void test_go_loss() {
    auto engine = nn::CpuEngine::create();
    require(static_cast<bool>(engine), "cpu engine");

    const std::size_t batch = 3;
    nn::Tensor packed = NN_UNWRAP(engine->try_create_tensor(
        go::packed_output_rows, batch));
    std::vector<nn::Scalar> logits(go::packed_output_rows * batch, nn::Scalar{0});
    for (std::size_t i = 0; i < logits.size(); ++i) {
        logits[i] = static_cast<nn::Scalar>((i % 7) - 3) * nn::Scalar{0.1};
    }
    for (std::size_t column = 0; column < batch; ++column) {
        logits[go::value_row_index * batch + column] = nn::Scalar{0.25};
    }
    NN_UNWRAP(engine->write(packed, std::span<const nn::Scalar>(logits)));

    const std::vector<std::size_t> labels{5, 100, 361};
    const std::vector<nn::Scalar> values{nn::Scalar{1}, nn::Scalar{-1}, nn::Scalar{0}};
    go::GoLoss loss;
    const nn::Scalar value = NN_UNWRAP(loss.forward(
        *engine, packed, std::span<const std::size_t>(labels),
        std::span<const nn::Scalar>(values)));
    require(std::isfinite(static_cast<double>(value)), "GoLoss is finite");
    require(value > nn::Scalar{0}, "GoLoss is positive");

    nn::Tensor grad = NN_UNWRAP(loss.backward());
    require_eq(grad.rows(), go::packed_output_rows, "GoLoss grad rows = 363");
    require_eq(grad.cols(), batch, "GoLoss grad cols = batch");
}

}  // namespace

int main() {
    test_contract_constants();
    test_spec_roundtrip();
    test_forward_backward();
    test_go_loss();
    if (failures != 0) {
        std::fprintf(stderr, "go_model_tests: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("go_model_tests: all checks passed");
    return 0;
}
