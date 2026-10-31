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
    config.value_from_spatial = true;

    const nn::ModelSpec spec = go::make_go_spec(config);
    require(go::is_go_spec(spec), "make_go_spec output must be recognized");
    const auto* arch = spec.as<go::GoTransformerSpec>();
    require(arch != nullptr, "Go spec uses the v1.14 open specification type");
    require(std::string(arch->type_name()) == "GoTransformer",
            "spec.type_name = GoTransformer");
    require_eq(arch->config.region_grid, go::region_grid, "spec region grid");

    const go::GoModelConfig restored = NN_UNWRAP(go::go_config_from_spec(spec));
    require_eq(restored.d_model, config.d_model, "roundtrip d_model");
    require_eq(restored.num_layers, config.num_layers, "roundtrip num_layers");
    require_eq(restored.num_heads, config.num_heads, "roundtrip num_heads");
    require_eq(restored.d_ff, config.d_ff, "roundtrip d_ff");
    require_eq(restored.conv_channels, config.conv_channels, "roundtrip conv_channels");
    require_eq(restored.history_len, config.history_len, "roundtrip history_len");
    require(restored.value_from_spatial, "roundtrip value_from_spatial");

    // 旧规格文件（a01790e 之前写出、无 value_from_spatial 具名键）必须回退
    // 旧接线（价值头读 query）——否则老 .nnmodel 的价值头权重形状 (1, d)
    // 会与新构造的 (1, d·25) 错位。这里手工重放旧键集（= write_kv 减去
    // 新键）来模拟。
    {
        nn::KeyValueRecord legacy_kv;
        legacy_kv.set("d_model", std::uint64_t{config.d_model});
        legacy_kv.set("num_heads", std::uint64_t{config.num_heads});
        legacy_kv.set("d_ff", std::uint64_t{config.d_ff});
        legacy_kv.set("num_layers", std::uint64_t{config.num_layers});
        legacy_kv.set("patch_size", std::uint64_t{config.region_grid});
        legacy_kv.set("vocab_size", std::uint64_t{go::policy_class_count});
        legacy_kv.set("seq_len", std::uint64_t{go::go_total_tokens(config.history_len)});
        legacy_kv.set("pos_encoding",
                      std::uint64_t{static_cast<int>(nn::PosEncodingType::Learned)});
        legacy_kv.set("activation",
                      std::uint64_t{static_cast<int>(nn::ActivationType::GeLU)});
        legacy_kv.set("norm_type",
                      std::uint64_t{static_cast<int>(nn::NormType::LayerNorm)});
        legacy_kv.set("layer_dims", std::vector<std::uint64_t>{
            go::go_spec_tag, config.history_len, config.conv_channels,
            config.region_grid, config.history_vocab, 0,
        });
        go::GoTransformerSpec legacy_reader;
        NN_UNWRAP(legacy_reader.read_kv(legacy_kv));
        require(!legacy_reader.config.value_from_spatial,
                "legacy spec without the named key must fall back to the "
                "query-wired value head");
    }

    // 外来 Transformer spec（无 Go 标记）必须被拒绝，否则会拿 Go 维度去
    // 构造一个语义完全不同的模型。
    nn::ModelSpec foreign;
    auto& transformer = foreign.emplace<nn::TransformerSpec>();
    transformer.d_model = 48;
    transformer.num_heads = 4;
    transformer.d_ff = 96;
    transformer.num_layers = 3;
    transformer.patch_size = 4;
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
                // ⚠ 先转有符号再做减法：r/column 是 size_t，直接 `% 11 - 5`
                // 在余数<5 时下溢成 2^64 级垃圾 float（P2 的真根因）。
                static_cast<nn::Scalar>(
                    static_cast<int>((r * 7 + column * 3) % 11) - 5) * nn::Scalar{0.01};
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
        grad_data[i] = static_cast<nn::Scalar>(static_cast<int>(i % 5) - 2) *
                       nn::Scalar{0.01};
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
    std::vector<bool> touches(gradients.size(), false);
    for (std::size_t index = 0; index < gradients.size(); ++index) {
        nn::Tensor& tensor = gradients[index].get();
        std::vector<nn::Scalar> buffer(tensor.size(), nn::Scalar{0});
        NN_UNWRAP(engine->read(tensor, std::span<nn::Scalar>(buffer)));
        for (nn::Scalar g : buffer) {
            if (!std::isfinite(static_cast<double>(g))) {
                require(false, "parameter gradient is not finite");
                return;
            }
            if (g != nn::Scalar{0}) {
                touches[index] = true;
            }
        }
    }
    // parameters() 的前四个是自身参数：history / position / segment / query
    // 四张表。它们必须真的拿到梯度，否则"历史分支"只是形状上的装饰
    // （例如历史 id 被全 pad 覆盖、或查询行取错位置时，前向仍然跑得通）。
    require(gradients.size() >= 4, "embedding tables are parameters");
    const char* names[] = {"history", "position", "segment", "query"};
    for (std::size_t i = 0; i < 4; ++i) {
        if (!touches[i]) {
            std::fprintf(stderr, "FAIL: %s embedding has zero gradient\n", names[i]);
            ++failures;
        }
    }
}

// ── 3b. 价值头接线路由（M10 修复的回归门）────────────────────────────────
//
// 只给价值行（362）回传梯度、策略行全零：
//   · value_from_spatial=true  → 梯度必须到达价值头自己的权重与 value_norm
//     的 γ/β，但**不回流**盘面 trunk（conv1）与历史嵌入（探针式）；
//   · value_from_spatial=false（旧接线）→ query 嵌入有梯度，盘面分支只
//     是注意力泄漏的小通道。
// 这把"接线焊死在正确一侧"变成可执行断言——上一轮价值 AUC 塌缩到 0.5
// 的事故（价值头悬空读 query）在这里必然 FAIL。
void test_value_head_routing() {
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

    for (const bool spatial : {true, false}) {
        config.value_from_spatial = spatial;
        nn::Model model = NN_UNWRAP(go::build_project_model(*engine, config));
        model.set_training(true);

        const std::size_t batch = 2;
        const std::size_t rows = go::go_input_rows(config.history_len);
        std::vector<nn::Scalar> input_data(rows * batch, nn::Scalar{0});
        for (std::size_t column = 0; column < batch; ++column) {
            // 空间上丰富的确定性输入（LCG）：周期性模式（如 r%9）会在
            // 卷积后让 5×5 区域网格全同值——LayerNorm 把常数列归一成 β=0，
            // 价值行精确为 0，把正确的接线测挂（第一版 ×0.01 全负同理）。
            for (std::size_t r = 0; r < go::feature_element_count; ++r) {
                const std::uint32_t h =
                    (static_cast<std::uint32_t>(r) * 1103515245u +
                     static_cast<std::uint32_t>(column) * 12345u + 1u) >> 8;
                input_data[r * batch + column] =
                    static_cast<nn::Scalar>(static_cast<int>(h % 17u) - 8);
            }
            for (std::size_t k = 0; k < config.history_len; ++k) {
                input_data[(go::feature_element_count + k) * batch + column] =
                    static_cast<nn::Scalar>((k * 31 + column) % go::history_vocab_size);
            }
        }
        nn::Tensor input = NN_UNWRAP(engine->try_create_tensor(rows, batch));
        NN_UNWRAP(engine->write(input, std::span<const nn::Scalar>(input_data)));

        nn::Tensor packed = NN_UNWRAP(model.forward(input));

        // 策略行零梯度，价值行 1.0——只测价值路径的去向。
        std::vector<nn::Scalar> grad_data(
            go::packed_output_rows * batch, nn::Scalar{0});
        for (std::size_t column = 0; column < batch; ++column) {
            grad_data[go::value_row_index * batch + column] = nn::Scalar{1};
        }
        nn::Tensor grad_output = NN_UNWRAP(engine->try_create_tensor(
            go::packed_output_rows, batch));
        NN_UNWRAP(engine->write(grad_output, std::span<const nn::Scalar>(grad_data)));

        nn::Tensor grad_input = NN_UNWRAP(model.backward(grad_output));
        require_eq(grad_input.rows(), rows, "routing grad_input rows");

        // parameters() 顺序契约：hist_emb_ 是第 0 张；conv1_.parameters()
        // 紧随四张嵌入表之后（第 4 张起）。
        auto gradients = model.param_gradients();
        require(gradients.size() > 4, "routing: gradients cover embeddings + conv");

        const auto max_abs = [&](std::size_t index) -> double {
            nn::Tensor& tensor = gradients[index].get();
            std::vector<nn::Scalar> buffer(tensor.size(), nn::Scalar{0});
            NN_UNWRAP(engine->read(tensor, std::span<nn::Scalar>(buffer)));
            double mx = 0;
            for (nn::Scalar g : buffer) {
                mx = std::max(mx, std::abs(static_cast<double>(g)));
            }
            return mx;
        };
        // 阈值口径：真实梯度链量级 ~1e-2 以上；浮点残渣 <=1e-12。
        const double conv_grad = max_abs(4);   // conv1 weight
        const double emb_grad = max_abs(0);    // hist_emb_ = 嵌入表代表
        // 价值头自身权重。参数表尾部（collect_refs 拼接序）：
        //   spatial: … policy(w,b) value_head(w,b) value_norm(γ,β)
        //   legacy: … policy(w,b) value_head(w,b)
        // → 价值头 weight 在倒数第 4（spatial）/ 倒数第 2（legacy）。
        const std::size_t vhead_index =
            spatial ? gradients.size() - 4 : gradients.size() - 2;
        const double vhead_grad = max_abs(vhead_index);

        // 诊断埋点自检（M10b）：debug_grad_norms 的分组范数必须与逐张量
        // 结论一致——value_head > 0；spatial 下 conv/trunk == 0。
        auto* go_root = static_cast<go::GoModel*>(&model.layer_at(0));
        const auto group_norms = NN_UNWRAP(go_root->debug_grad_norms());
        const auto find_norm = [&](const char* name) -> double {
            for (const auto& [key, val] : group_norms) {
                if (key == name) return val;
            }
            return -1.0;
        };
        const double vg = find_norm("value_head");
        require(std::isfinite(vg) && vg > 1e-6,
                "debug_grad_norms: value_head group must be present and nonzero");
        if (spatial) {
            require(find_norm("conv1") == 0.0,
                    "debug_grad_norms: probe must not reach conv trunk");
            require(find_norm("value_norm") > 1e-6,
                    "debug_grad_norms: value_norm gamma/beta must train");
        }

        if (spatial) {
            // M10b 探针式：价值头读空间特征、学自己的权重，但梯度不得
            // 回流共享 trunk（回流曾把策略摧毁——放大的价值梯度经全局
            // grad clip 压死策略头的更新）。
            require(vhead_grad > 1e-6,
                    "spatial value head: value head weights must learn");
            require(conv_grad == 0.0,
                    "spatial value head: probe must not backprop into conv trunk");
            require(emb_grad == 0.0,
                    "spatial value head: probe must not backprop into embeddings");
        } else {
            require(vhead_grad > 1e-6,
                    "legacy value head: value head weights must learn");
            require(emb_grad > 1e-6,
                    "legacy value head: gradient must reach the query path");
            // 注意力会把 query 的价值梯度混进盘面 token（实测 ~0.07 vs 嵌入
            // ~3.2）——legacy 下盘面泄漏非零但远小于 query 主通道，只断言
            // 它不是主通道即可。
            require(conv_grad < emb_grad / 10.0,
                    "legacy value head: board branch must not be the main channel");
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
        // ⚠ 先转有符号（P2 根因修复，见 test_forward_backward 注释）。
        logits[i] = static_cast<nn::Scalar>(static_cast<int>(i % 7) - 3) *
                    nn::Scalar{0.1};
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

    // 分项损失（M10b 诊断）：都为正，且加权和等于 combined loss。
    const double policy_part = static_cast<double>(loss.last_policy_loss());
    const double value_part = static_cast<double>(loss.last_value_loss());
    require(std::isfinite(policy_part) && policy_part > 0,
            "GoLoss policy split is positive");
    require(std::isfinite(value_part) && value_part > 0,
            "GoLoss value split is positive");
    require(std::abs(policy_part + 1.0 * value_part -
                      static_cast<double>(value)) < 1e-5,
            "GoLoss combined = policy + value_weight * value");

    nn::Tensor grad = NN_UNWRAP(loss.backward());
    require_eq(grad.rows(), go::packed_output_rows, "GoLoss grad rows = 363");
    require_eq(grad.cols(), batch, "GoLoss grad cols = batch");
}

}  // namespace

int main() {
    test_contract_constants();
    test_spec_roundtrip();
    test_forward_backward();
    test_value_head_routing();
    test_go_loss();
    if (failures != 0) {
        std::fprintf(stderr, "go_model_tests: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("go_model_tests: all checks passed");
    return 0;
}
