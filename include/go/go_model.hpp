#pragma once

// ── B 架构：GoModel 复合层 + GoLoss ─────────────────────────────────────────
//
// 架构（设计见 docs/design/02-scheme1-encoder-transformer.md §3）：
//
//   盘面分支: (9, 19×19) → Conv(3×3) → ReLU → Conv(4×4, s4) → ReLU
//             → (d, 5×5) 区域 token（25 个）
//   历史分支: 最近 K 手 point id → gather_rows 查表（可学习）
//   决策:     <DECISION> query token（可学习）
//             ↓ 单序列 [25 区域 | K 历史 | 1 query]，Plain 双向掩码
//             ↓ TransformerEncoderLayer × N（Pre-Norm，现成层）
//             ↓ query 行 → Linear(d,362) 策略头 / Linear(d,1) 价值头(tanh)
//             ↓ 打包 (363, batch) 单进单出
//   GoLoss:   前 362 行稀疏 CE + 末行 MSE
//
// 书写约定与 neuralnet.cpp 的 GPTModel 一致：
//   · 复合层派生 `nn::Layer`，子层经 `param_children()` 转发（参数协议）；
//   · 逐元素/归约数学一律写 `dsl::compute`（GPU 侧由 AOT 融合注册表承接）；
//   · 布局重排（rearrange_3d / transpose / gather_rows / insert_rows）是
//     op-level 数据搬运原语——它们没有逐元素语义，不进 DSL；
//   · 参数张量在 `init_impl` 里用 `create_tensor` 建（M2 声明式）。
//
// 张量布局（全局 batch-major，铁律 5）：
//   输入  (9·19·19 + K, batch)      平面行在前，历史 id（f32 存的整数）在后
//   序列  (d, batch·T)              T = 25 + K + 1，列序 b·T + t
//   输出  (363, batch)              0..361 策略 logits，362 = tanh(v)
//
// ModelSpec：neuralnet.cpp v1.14.1 起使用开放规格扩展点。项目定义
// GoTransformerSpec（类型名 GoTransformer），并保留对 v1.13 扁平
// Transformer 规格的只读兼容层。

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <neuralnet.cpp/nn.hpp>

#include "go/features.hpp"  // feature_element_count / policy_class_count / training_board_size

namespace go {

// ── 输入/输出契约 ─────────────────────────────────────────────────────────
inline constexpr std::size_t region_grid = 5;                     // 5×5 区域 token
inline constexpr std::size_t region_token_count = region_grid * region_grid;  // 25
inline constexpr std::size_t history_vocab_size = 367;            // GoTokenizer 词表
inline constexpr std::uint16_t history_pass_token = 361;          // <pass>
inline constexpr std::uint16_t history_pad_token = 365;           // <pad>
inline constexpr std::size_t segment_count = 3;                   // 盘面 / 历史 / query
inline constexpr std::size_t packed_output_rows = policy_class_count + 1;  // 363
inline constexpr std::size_t value_row_index = policy_class_count;          // 362
// v1.13 及更早文件的 "GoM1" 魔数，位于 layer_dims[0]；仅用于只读迁移识别。
inline constexpr std::uint32_t go_spec_tag = 0x476F4D31u;

// 模型输入行数：9 平面 + K 行历史 id。
[[nodiscard]] constexpr std::size_t go_input_rows(std::size_t history_len) {
    return feature_element_count + history_len;
}

// 序列长度：25 区域 + K 历史 + 1 query。
[[nodiscard]] constexpr std::size_t go_total_tokens(std::size_t history_len) {
    return region_token_count + history_len + 1;
}

struct GoModelConfig {
    std::size_t d_model = 192;
    std::size_t num_layers = 8;
    std::size_t num_heads = 8;
    std::size_t d_ff = 768;
    std::size_t conv_channels = 64;    // 盘面 stem 的中间通道数
    std::size_t history_len = 64;      // K
    std::size_t history_vocab = history_vocab_size;
    std::size_t region_grid = go::region_grid;
    // 设计 §3.1 兜底：策略头 + CNN 直通 logits。盘面细节不再只能经 25 个区域
    // token 挤进 query——特征图展平后直接线性出 362 类，与 query 的 logits 相加。
    bool cnn_policy = false;
};

// neuralnet.cpp v1.14.1 的开放模型规格：类型参数与判别符随 ModelSpecBase
// 多态对象走。字段顺序沿用 v1.13 的扁平规格布局，便于旧文件只读迁移。
class GoTransformerSpec final : public nn::ModelSpecBase {
public:
    GoModelConfig config;

    [[nodiscard]] const char* type_name() const noexcept override {
        return "GoTransformer";
    }
    [[nodiscard]] std::unique_ptr<nn::ModelSpecBase> clone() const override;
    [[nodiscard]] bool arch_compatible(
        const nn::ModelSpecBase& other
    ) const override;
    [[nodiscard]] std::string summary(const nn::ModelSpec& shell) const override;
    void write_kv(nn::KeyValueRecord& kv) const override;
    [[nodiscard]] nn::Result<void> read_kv(const nn::KeyValueRecord& kv) override;
};

// ── ModelSpec 往返（开放规格 + 旧格式只读兼容）───────────────────────────
[[nodiscard]] nn::ModelSpec make_go_spec(const GoModelConfig& config);
[[nodiscard]] bool is_go_spec(const nn::ModelSpec& spec) noexcept;
[[nodiscard]] nn::Result<GoModelConfig> go_config_from_spec(const nn::ModelSpec& spec);
[[nodiscard]] nn::Result<nn::ModelSpec> peek_go_model_spec(const std::string& path);
[[nodiscard]] nn::Result<std::string> load_go_model(
    const std::string& path,
    nn::Model& model,
    std::uint32_t* version_out = nullptr,
    nn::ModelRole* role_out = nullptr,
    nn::WeightLoadIntent intent = nn::WeightLoadIntent::Inference
);

// ── 工厂 ───────────────────────────────────────────────────────────────────
[[nodiscard]] nn::Result<nn::Model> build_project_model(
    nn::ComputeEngine& engine, const GoModelConfig& config);

// 按 spec 重建（续训 / 评估 / 推理共用）：Go spec → GoModel，
// CNN spec → 库的 build_cnn_model_from_spec（M4 C 验证器），其余报错。
[[nodiscard]] nn::Result<nn::Model> build_project_model_from_spec(
    nn::ComputeEngine& engine, const nn::ModelSpec& spec);

// ══════════════════════════════════════════════════════════════════════════
// GoModel — 双分支编码器 + decision query + 双头（单进单出复合层）
// ══════════════════════════════════════════════════════════════════════════
class GoModel final : public nn::Layer {
public:
    explicit GoModel(const GoModelConfig& config);

    [[nodiscard]] const char* layer_name() const noexcept override { return "GoModel"; }

    [[nodiscard]] const GoModelConfig& config() const noexcept { return cfg_; }
    [[nodiscard]] std::size_t seq_len() const noexcept { return seq_len_; }
    [[nodiscard]] std::size_t input_rows() const noexcept {
        return go_input_rows(cfg_.history_len);
    }

    [[nodiscard]] nn::Result<void> init_impl(nn::ComputeEngine& engine) override;
    [[nodiscard]] nn::Result<nn::Tensor> forward(const nn::Tensor& input) override;
    [[nodiscard]] nn::Result<nn::Tensor> backward(const nn::Tensor& grad_output) override;

    [[nodiscard]] std::vector<nn::TensorRef> parameters() override;
    [[nodiscard]] std::vector<nn::TensorRef> param_gradients() override;

    void set_precision_profile(const nn::PrecisionProfile& profile) override;
    void set_training(bool training) override;
    void clear_cache() override;

protected:
    [[nodiscard]] std::vector<nn::Layer*> param_children() noexcept override;

private:
    // 索引张量一次性上传（形状相关、与数据无关）：区域重排、序列重排、
    // 位置/段/query 下标。按 batch 缓存，训练期间不重建。
    [[nodiscard]] nn::Result<void> ensure_indices_(
        nn::ComputeEngine& engine, std::size_t batch);

    GoModelConfig cfg_;
    std::size_t seq_len_ = 0;
    std::size_t batch_ = 0;

    // 子层（参数顺序 = parameters() / param_children() 的拼接顺序）
    nn::Conv2D conv1_;
    nn::ReLU relu1_;
    nn::Conv2D conv2_;
    nn::ReLU relu2_;
    std::vector<nn::TransformerEncoderLayer> blocks_;
    std::unique_ptr<nn::Layer> ln_f_;
    nn::Linear policy_head_;
    nn::Linear value_head_;
    // 可选：CNN 直通策略头（Linear(d·25 → 362)，见 GoModelConfig::cnn_policy）。
    // 始终构造，只在启用时进入 parameters()/param_children() 与前反向。
    nn::Linear cnn_policy_head_;

    // 自身参数张量（行主序：(rows, d_model)，与 GPTModel 的 token_emb_ 同形）
    nn::Tensor hist_emb_;    // (history_vocab, d)
    nn::Tensor pos_emb_;     // (T, d)  位置编码（可学习）
    nn::Tensor seg_emb_;     // (3, d)  段嵌入（可学习）
    nn::Tensor query_tok_;   // (1, d)  <DECISION>

    nn::Tensor grad_hist_emb_;
    nn::Tensor grad_pos_emb_;
    nn::Tensor grad_seg_emb_;
    nn::Tensor grad_query_tok_;

    // 索引缓存
    nn::Tensor region_perm_;      // (d·25, 1)  c·25+p → p·d+c
    nn::Tensor region_perm_inv_;  // (d·25, 1)  p·d+c → c·25+p
    nn::Tensor seq_perm_;         // (batch·T, 1)
    nn::Tensor pos_idx_;          // (batch·T, 1) = t
    nn::Tensor seg_idx_;          // (batch·T, 1) = 段号
    nn::Tensor query_idx_;        // (batch, 1)  全 0
    nn::Tensor query_row_idx_;    // (batch, 1)  b·T + T-1

    // 反向缓存
    nn::Tensor ids_flat_;   // (batch, K)  历史 id（gather 的平铺序，
                            // scatter_add_rows 复用同一张量）
    nn::Tensor value_t_;    // (1, batch)  tanh 输出（值头反传的 1-t² 需要）
};

// ══════════════════════════════════════════════════════════════════════════
// GoLoss — 打包输出 (363, batch) 的分行损失
//
//   policy = CE（稀疏标签，行 [0,362)）
//   value  = MSE（行 362，tanh 后的预测 vs z∈{−1,0,+1}，当前手方视角）
//   loss   = policy + value_weight · value
//
// backward 把两段梯度用 insert_rows 拼回 (363, batch)。
// ══════════════════════════════════════════════════════════════════════════
class GoLoss final : public nn::Loss {
public:
    explicit GoLoss(nn::Scalar value_weight = nn::Scalar{1});

    // `packed`: (363, batch)；`labels`: 每列的策略目标类；`values`: 价值目标。
    [[nodiscard]] nn::Result<nn::Scalar> forward(
        nn::ComputeEngine& engine,
        const nn::Tensor& packed,
        std::span<const std::size_t> labels,
        std::span<const nn::Scalar> values);

    // Loss 基类的双入参纯虚接口：GoLoss 的第二个目标不是张量（价值目标走
    // forward 的 span 重载），这里显式拒绝，避免误用。
    [[nodiscard]] nn::Result<nn::Scalar> forward(
        nn::ComputeEngine& engine,
        const nn::Tensor& pred,
        const nn::Tensor& target) override;

    [[nodiscard]] nn::Result<nn::Tensor> backward() override;

private:
    nn::Scalar value_weight_;
    nn::CrossEntropyLoss policy_loss_;
    nn::MSELoss value_loss_;
    nn::Tensor grad_packed_;
};

}  // namespace go
