// ── B 架构实现：GoModel 复合层 + GoLoss ────────────────────────────────────
//
// 布局说明（全部 batch-major，铁律 5）：
//   · 卷积 IO         (C·H·W, batch)         行序 c·H·W + y·W + x
//   · 序列            (d, batch·T)           列序 b·T + t
//   · 嵌入表/行空间    (N, d)                 行主序
//
// 盘面分支的两次重排（c·25+p ↔ p·d+c ↔ 列 b·25+p）是本层唯一的"手工"布局
// 工作：Conv2D 的输出行按通道分组，而区域 token 需要按区分组。这两步用
// op-level 搬运原语表达（gather_rows + rearrange_3d + transpose），与
// compute_layer_conv.hpp 的 conv_engine 助手同一性质。
//
// AOT 闭合世界：本文件里每个 `nn::dsl::compute` 都由 FusedAnchor 在静态初始化期
// 自登记（编译进 nn_enable_gpu_fusion 的收集器即可，不需要 dry-run 条目）。

#include "go/go_model.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

namespace go {
namespace {

using nn::Scalar;
using nn::Tensor;

[[nodiscard]] nn::Result<Tensor> upload_index(
    nn::ComputeEngine& engine, const std::vector<Scalar>& values
) {
    return nn::detail::upload_span(
        engine, values.size(), 1, nn::Precision::F32,
        std::span<const Scalar>(values)
    );
}

void validate_config(const GoModelConfig& cfg) {
    if (cfg.d_model == 0 || cfg.num_layers == 0 || cfg.num_heads == 0 ||
        cfg.d_ff == 0 || cfg.conv_channels == 0 || cfg.history_len == 0) {
        NN_FATAL("GoModel: 维度必须为正");
    }
    if (cfg.d_model % cfg.num_heads != 0) {
        NN_FATAL("GoModel: d_model 必须能被 num_heads 整除");
    }
    if (cfg.region_grid != go::region_grid) {
        NN_FATAL("GoModel: region_grid 固定为 5（19×19 特化，D5）");
    }
    if (cfg.history_vocab < history_vocab_size) {
        NN_FATAL("GoModel: history_vocab 必须 >= GoTokenizer 词表 367");
    }
}

}  // namespace

// ── ModelSpec 往返（neuralnet.cpp v1.14.1 开放规格）───────────────────────
std::unique_ptr<nn::ModelSpecBase> GoTransformerSpec::clone() const {
    return std::make_unique<GoTransformerSpec>(*this);
}

bool GoTransformerSpec::arch_compatible(
    const nn::ModelSpecBase& other
) const {
    const auto* peer = dynamic_cast<const GoTransformerSpec*>(&other);
    if (!peer) {
        return false;
    }
    return config.d_model == peer->config.d_model &&
           config.num_layers == peer->config.num_layers &&
           config.num_heads == peer->config.num_heads &&
           config.d_ff == peer->config.d_ff &&
           config.conv_channels == peer->config.conv_channels &&
           config.history_len == peer->config.history_len &&
           config.history_vocab == peer->config.history_vocab &&
           config.region_grid == peer->config.region_grid &&
           config.cnn_policy == peer->config.cnn_policy &&
           config.value_from_spatial == peer->config.value_from_spatial;
}

std::string GoTransformerSpec::summary(const nn::ModelSpec&) const {
    return "GoTransformer(d_model=" + std::to_string(config.d_model) +
           ",layers=" + std::to_string(config.num_layers) +
           ",heads=" + std::to_string(config.num_heads) +
           ",d_ff=" + std::to_string(config.d_ff) +
           ",conv=" + std::to_string(config.conv_channels) +
           ",history=" + std::to_string(config.history_len) +
           (config.cnn_policy ? ",cnn_policy=1" : "") +
           (config.value_from_spatial ? ",value_spatial=1" : ",value_spatial=0") + ")";
}

void GoTransformerSpec::write_kv(nn::KeyValueRecord& kv) const {
    // 键名与 v1.13 的扁平规格保持一致，旧文件的 extra keys 可原样复用。
    kv.set("d_model", static_cast<std::uint64_t>(config.d_model));
    kv.set("num_heads", static_cast<std::uint64_t>(config.num_heads));
    kv.set("d_ff", static_cast<std::uint64_t>(config.d_ff));
    kv.set("num_layers", static_cast<std::uint64_t>(config.num_layers));
    kv.set("patch_size", static_cast<std::uint64_t>(config.region_grid));
    kv.set("vocab_size", static_cast<std::uint64_t>(policy_class_count));
    kv.set("seq_len", static_cast<std::uint64_t>(go_total_tokens(config.history_len)));
    kv.set("pos_encoding", static_cast<std::uint64_t>(nn::PosEncodingType::Learned));
    kv.set("activation", static_cast<std::uint64_t>(nn::ActivationType::GeLU));
    kv.set("norm_type", static_cast<std::uint64_t>(nn::NormType::LayerNorm));
    kv.set("layer_dims", std::vector<std::uint64_t>{
        go_spec_tag,
        config.history_len,
        config.conv_channels,
        config.region_grid,
        config.history_vocab,
        config.cnn_policy ? std::uint64_t{1} : std::uint64_t{0},
    });
    // M10 价值头接线（开放规格的具名键，不进 layer_dims 魔数下标）。
    // 总是写出；旧文件缺键 ⇒ read_kv 回退 query 接线（旧权重形状 (1,d)）。
    kv.set("value_from_spatial", static_cast<std::uint64_t>(
        config.value_from_spatial ? 1 : 0
    ));
}

nn::Result<void> GoTransformerSpec::read_kv(const nn::KeyValueRecord& kv) {
    auto require_u64 = [&kv](const char* key) -> nn::Result<std::uint64_t> {
        std::uint64_t value = 0;
        if (!kv.get(key, value)) {
            NN_FAIL(std::string("GoTransformerSpec: 规格缺少键 '") + key + "'");
        }
        return value;
    };

    NN_TRY(d_model_r, require_u64("d_model"));
    NN_TRY(num_heads_r, require_u64("num_heads"));
    NN_TRY(d_ff_r, require_u64("d_ff"));
    NN_TRY(num_layers_r, require_u64("num_layers"));
    NN_TRY(patch_size_r, require_u64("patch_size"));
    NN_TRY(vocab_size_r, require_u64("vocab_size"));
    NN_TRY(seq_len_r, require_u64("seq_len"));
    const std::uint64_t d_model = *d_model_r;
    const std::uint64_t num_heads = *num_heads_r;
    const std::uint64_t d_ff = *d_ff_r;
    const std::uint64_t num_layers = *num_layers_r;
    const std::uint64_t patch_size = *patch_size_r;
    const std::uint64_t vocab_size = *vocab_size_r;
    const std::uint64_t seq_len = *seq_len_r;

    std::vector<std::uint64_t> dims;
    if (!kv.get("layer_dims", dims) || dims.size() < 5 ||
        dims[0] != go_spec_tag) {
        NN_FAIL("GoTransformerSpec: layer_dims 缺少 GoM1 标记或 Go 维度");
    }
    if (vocab_size != policy_class_count) {
        NN_FAIL("GoTransformerSpec: vocab_size 必须是 362");
    }
    if (patch_size != region_grid || dims[3] != region_grid) {
        NN_FAIL("GoTransformerSpec: region_grid 固定为 5（19×19 特化）");
    }
    if (d_model == 0 || num_heads == 0 || d_ff == 0 || num_layers == 0 ||
        d_model % num_heads != 0) {
        NN_FAIL("GoTransformerSpec: Transformer 维度非法");
    }

    config.d_model = static_cast<std::size_t>(d_model);
    config.num_heads = static_cast<std::size_t>(num_heads);
    config.d_ff = static_cast<std::size_t>(d_ff);
    config.num_layers = static_cast<std::size_t>(num_layers);
    config.conv_channels = static_cast<std::size_t>(dims[2]);
    config.history_len = static_cast<std::size_t>(dims[1]);
    config.history_vocab = static_cast<std::size_t>(dims[4]);
    config.region_grid = static_cast<std::size_t>(dims[3]);
    config.cnn_policy = dims.size() >= 6 && dims[5] != 0;
    // M10 价值头接线：缺键（v1.13 及更早 / a01790e 之前的 GoTransformer）
    // 回退旧接线（价值头读 query），老 .nnmodel/.nnckpt 保持可加载且
    // 行为与训练时一致——权重形状 (1, d) vs (1, d·25) 由该键区分。
    {
        std::uint64_t value_spatial = 0;
        config.value_from_spatial =
            kv.get("value_from_spatial", value_spatial) && value_spatial != 0;
    }

    if (config.conv_channels == 0 || config.history_len == 0 ||
        config.history_vocab < history_vocab_size) {
        NN_FAIL("GoTransformerSpec: Go 模型维度非法");
    }
    if (seq_len != go_total_tokens(config.history_len)) {
        NN_FAIL("GoTransformerSpec: seq_len 与 history_len 不自洽");
    }
    return {};
}

[[maybe_unused]] const bool register_go_transformer_spec = [] {
    nn::ModelRegistry::instance().register_type(
        "GoTransformer", [] { return std::make_unique<GoTransformerSpec>(); }
    );
    return true;
}();

nn::ModelSpec make_go_spec(const GoModelConfig& config) {
    nn::ModelSpec spec;
    auto& arch = spec.emplace<GoTransformerSpec>();
    arch.config = config;
    return spec;
}

bool is_go_spec(const nn::ModelSpec& spec) noexcept {
    return spec.as<GoTransformerSpec>() != nullptr;
}

nn::Result<GoModelConfig> go_config_from_spec(const nn::ModelSpec& spec) {
    const auto* arch = spec.as<GoTransformerSpec>();
    if (!arch) {
        NN_FAIL("go_config_from_spec: 该 spec 不描述 Go 模型");
    }
    return arch->config;
}

// ── v1.13 及更早规格的只读兼容层 ──────────────────────────────────────────
namespace {

struct LegacyGoSpecHeader {
    bool is_go = false;
    std::uint32_t version = 0;
    std::size_t spec_end = 0;
    nn::ModelSpec spec;
};

nn::Result<LegacyGoSpecHeader> read_legacy_go_spec_header(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        NN_FAIL("Cannot open file for reading: " + path);
    }
    nn::ModelRole role = nn::ModelRole::Model;
    NN_TRY(version, nn::detail::read_and_validate_header(input, &role));
    NN_TRY(spec_len, nn::detail::read_bytes<std::uint64_t>(input));
    if (*spec_len > 64u * 1024u * 1024u) {
        NN_FAIL("规格头长度越界（超过 64 MiB）: " + path);
    }
    std::string bytes(static_cast<std::size_t>(*spec_len), '\0');
    if (!bytes.empty()) {
        input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!input) {
            NN_FAIL("Unexpected end while reading spec header: " + path);
        }
    }
    const auto pos = input.tellg();
    if (pos < 0) {
        NN_FAIL("Cannot determine spec header end: " + path);
    }

    NN_TRY(kv, nn::KeyValueRecord::parse(bytes));
    LegacyGoSpecHeader out;
    out.version = *version;
    out.spec_end = static_cast<std::size_t>(pos);

    std::string type_name;
    std::uint64_t type_number = 0;
    const bool legacy_transformer =
        (!kv->get("type", type_name) && kv->get("type", type_number) &&
         type_number == static_cast<std::uint64_t>(nn::ModelType::Transformer)) ||
        type_name == "Transformer";
    std::vector<std::uint64_t> dims;
    if (!legacy_transformer || !kv->get("layer_dims", dims) || dims.empty() ||
        dims[0] != go_spec_tag) {
        return out;
    }

    out.spec.emplace<GoTransformerSpec>();
    NN_TRY_CHECK(out.spec.arch->read_kv(*kv));
    NN_TRY_CHECK(go_config_from_spec(out.spec));
    out.is_go = true;
    return out;
}

nn::Result<std::string> make_legacy_upgrade_path(const std::string& source) {
    std::error_code error;
    const auto directory = std::filesystem::temp_directory_path(error);
    if (error) {
        NN_FAIL("无法确定临时目录以兼容旧 Go 模型文件: " + error.message());
    }
    static std::atomic<std::uint64_t> counter{0};
    const auto suffix = std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
    return (directory / ("go_neuralnet_v1_14_1_" + suffix +
                         std::filesystem::path(source).extension().string()))
        .string();
}

nn::Result<std::string> write_legacy_upgrade(
    const std::string& source, const LegacyGoSpecHeader& header
) {
    std::ifstream input(source, std::ios::binary | std::ios::ate);
    if (!input) {
        NN_FAIL("Cannot open file for reading: " + source);
    }
    const auto end = input.tellg();
    if (end < 0) {
        NN_FAIL("Cannot determine file size: " + source);
    }
    const auto tail_size = header.version >= 8 ? nn::MODEL_TAIL_BYTES : 0;
    const auto total = static_cast<std::size_t>(end);
    if (total < header.spec_end + tail_size) {
        NN_FAIL("模型文件过小或规格头截断: " + source);
    }
    std::string content(total, '\0');
    input.seekg(0, std::ios::beg);
    input.read(content.data(), static_cast<std::streamsize>(content.size()));
    if (!input) {
        NN_FAIL("Read error while reading model file: " + source);
    }

    const auto new_spec = nn::detail::spec_to_kv(header.spec).serialize();
    const auto new_len = static_cast<std::uint64_t>(new_spec.size());
    std::ostringstream output(std::ios::binary);
    output.write(content.data(), static_cast<std::streamsize>(18));
    output.write(reinterpret_cast<const char*>(&new_len), sizeof(new_len));
    output.write(new_spec.data(), static_cast<std::streamsize>(new_spec.size()));
    const auto payload_size = total - header.spec_end - tail_size;
    output.write(
        content.data() + header.spec_end, static_cast<std::streamsize>(payload_size)
    );

    std::string upgraded = output.str();
    if (header.version >= 8) {
        nn::detail::Sha256State hasher;
        nn::detail::sha256_update(hasher, upgraded);
        const auto digest = nn::detail::sha256_finalize(hasher);
        upgraded.append(reinterpret_cast<const char*>(digest.data()), digest.size());
        upgraded.append(nn::MODEL_TAIL_MAGIC.data(), nn::MODEL_TAIL_MAGIC.size());
    }

    NN_TRY(destination, make_legacy_upgrade_path(source));
    std::ofstream output_file(*destination, std::ios::binary | std::ios::trunc);
    if (!output_file) {
        NN_FAIL("Cannot open temporary compatibility file for writing: " + *destination);
    }
    output_file.write(
        upgraded.data(), static_cast<std::streamsize>(upgraded.size())
    );
    if (!output_file) {
        NN_FAIL("Write error while creating compatibility file: " + *destination);
    }
    return *std::move(destination);
}

}  // namespace

nn::Result<nn::ModelSpec> peek_go_model_spec(const std::string& path) {
    NN_TRY(legacy, read_legacy_go_spec_header(path));
    if (legacy->is_go) {
        return std::move(legacy->spec);
    }
    return nn::peek_any_model_spec(path);
}

nn::Result<std::string> load_go_model(
    const std::string& path, nn::Model& model, std::uint32_t* version_out,
    nn::ModelRole* role_out, nn::WeightLoadIntent intent
) {
    NN_TRY(legacy, read_legacy_go_spec_header(path));
    if (!legacy->is_go) {
        return nn::load_model(path, model, version_out, role_out, intent);
    }

    NN_TRY(upgraded, write_legacy_upgrade(path, *legacy));
    auto loaded = nn::load_model(*upgraded, model, version_out, role_out, intent);
    std::error_code error;
    std::filesystem::remove(*upgraded, error);
    NN_TRY_CHECK(loaded);
    return *std::move(loaded);
}

// ── 工厂 ───────────────────────────────────────────────────────────────────
nn::Result<nn::Model> build_project_model(
    nn::ComputeEngine& engine, const GoModelConfig& config
) {
    validate_config(config);
    nn::Model model(engine);
    NN_TRY_CHECK(model.add<GoModel>(config));
    model.set_spec(make_go_spec(config));
    return model;
}

nn::Result<nn::Model> build_project_model_from_spec(
    nn::ComputeEngine& engine, const nn::ModelSpec& spec
) {
    if (is_go_spec(spec)) {
        NN_TRY(config, go_config_from_spec(spec));
        return build_project_model(engine, *config);
    }
    if (spec.is_cnn()) {
        return nn::build_cnn_model_from_spec(engine, spec);
    }
    NN_FAIL("build_project_model_from_spec: 不认识该 ModelSpec（既非 Go 也非 CNN）");
}

// ══════════════════════════════════════════════════════════════════════════
// GoModel
// ══════════════════════════════════════════════════════════════════════════
GoModel::GoModel(const GoModelConfig& config)
    : cfg_(config),
      seq_len_(go_total_tokens(config.history_len)),
      conv1_(feature_plane_count, config.conv_channels, 3, 1, 1,
             training_board_size, training_board_size),
      relu1_(),
      conv2_(config.conv_channels, config.d_model, 4, 4, 1,
             training_board_size, training_board_size),
      relu2_(),
      ln_f_(nn::make_norm_layer(config.d_model, nn::NormType::LayerNorm)),
      policy_head_(config.d_model, policy_class_count),
      // M10：价值头读盘面特征图 r2（d·25）时输入维随之放大；旧接线读
      // query（d）。权重形状由该维决定，spec 具名键 value_from_spatial
      // 保证两者互不误载。
      value_head_(config.value_from_spatial
                      ? config.d_model * region_token_count
                      : config.d_model, 1),
      cnn_policy_head_(config.d_model * region_token_count, policy_class_count) {
    validate_config(config);
    // 4×4 stride 4 padding 1 把 19×19 压到 5×5：区域 token 数必须与常量一致。
    const std::size_t out_hw =
        (training_board_size + 2 * 1 - 4) / 4 + 1;
    NN_ASSERT_ALWAYS(
        out_hw * out_hw == region_token_count,
        "GoModel: 卷积 stem 的输出网格必须等于 5×5 区域 token"
    );
    blocks_.reserve(config.num_layers);
    for (std::size_t i = 0; i < config.num_layers; ++i) {
        blocks_.emplace_back(
            config.d_model, config.num_heads, config.d_ff, seq_len_,
            nn::NormType::LayerNorm
        );
    }
    // 价值头输入归一化：r2 未归一化直接进 Linear+tanh 会饱和
    //（float32 下 tanh(|x|≳9) 精确 =±1 → 1-t²=0 → 价值头零梯度）。
    if (config.value_from_spatial) {
        value_norm_ = nn::make_norm_layer(
            config.d_model * region_token_count, nn::NormType::LayerNorm);
    }
}

nn::Result<void> GoModel::init_impl(nn::ComputeEngine& engine) {
    const std::size_t d = cfg_.d_model;
    constexpr Scalar kEmbStd = Scalar{0.02};
    // M2 声明式：分布参数由层给出、引擎填数；seed 显式（U1）。
    auto make_table = [&](std::size_t rows) {
        return engine.create_tensor(
            rows, d, p_.param, nn::InitSpec::normal(0, kEmbStd, nn::kInitSeed)
        );
    };
    auto make_grad = [&](std::size_t rows) {
        return engine.create_tensor(rows, d, p_.param, nn::InitSpec::zero());
    };

    hist_emb_ = make_table(cfg_.history_vocab);
    pos_emb_ = make_table(seq_len_);
    seg_emb_ = make_table(segment_count);
    query_tok_ = make_table(1);
    if (!hist_emb_.valid() || !pos_emb_.valid() || !seg_emb_.valid() ||
        !query_tok_.valid()) {
        NN_FAIL("GoModel: 嵌入参数张量初始化失败");
    }
    grad_hist_emb_ = make_grad(cfg_.history_vocab);
    grad_pos_emb_ = make_grad(seq_len_);
    grad_seg_emb_ = make_grad(segment_count);
    grad_query_tok_ = make_grad(1);
    if (!grad_hist_emb_.valid() || !grad_pos_emb_.valid() ||
        !grad_seg_emb_.valid() || !grad_query_tok_.valid()) {
        NN_FAIL("GoModel: 嵌入参数梯度张量初始化失败");
    }

    NN_TRY(r0, conv1_.init(engine));
    NN_TRY(r1, relu1_.init(engine));
    NN_TRY(r2, conv2_.init(engine));
    NN_TRY(r3, relu2_.init(engine));
    for (auto& block : blocks_) {
        NN_TRY(rb, block.init(engine));
    }
    NN_TRY(r4, ln_f_->init(engine));
    NN_TRY(r5, policy_head_.init(engine));
    NN_TRY(r6, value_head_.init(engine));
    if (value_norm_) {
        NN_TRY(r8, value_norm_->init(engine));
    }
    if (cfg_.cnn_policy) {
        NN_TRY(r7, cnn_policy_head_.init(engine));
    }
    return {};
}

std::vector<nn::TensorRef> GoModel::parameters() {
    // 顺序契约：自身参数在前，随后与 param_children() 同序拼接。
    return collect_refs(
        hist_emb_, pos_emb_, seg_emb_, query_tok_,
        conv1_.parameters(), conv2_.parameters(),
        collect_block_refs_(blocks_, &nn::TransformerEncoderLayer::parameters),
        ln_f_->parameters(),
        policy_head_.parameters(), value_head_.parameters(),
        value_norm_ ? value_norm_->parameters()
                    : std::vector<nn::TensorRef>{},
        cfg_.cnn_policy ? cnn_policy_head_.parameters()
                        : std::vector<nn::TensorRef>{}
    );
}

std::vector<nn::TensorRef> GoModel::param_gradients() {
    return collect_refs(
        grad_hist_emb_, grad_pos_emb_, grad_seg_emb_, grad_query_tok_,
        conv1_.param_gradients(), conv2_.param_gradients(),
        collect_block_refs_(blocks_, &nn::TransformerEncoderLayer::param_gradients),
        ln_f_->param_gradients(),
        policy_head_.param_gradients(), value_head_.param_gradients(),
        value_norm_ ? value_norm_->param_gradients()
                    : std::vector<nn::TensorRef>{},
        cfg_.cnn_policy ? cnn_policy_head_.param_gradients()
                        : std::vector<nn::TensorRef>{}
    );
}

std::vector<nn::Layer*> GoModel::param_children() noexcept {
    std::vector<nn::Layer*> children;
    children.reserve(blocks_.size() + 6);
    children.push_back(&conv1_);
    children.push_back(&conv2_);
    for (auto& block : blocks_) {
        children.push_back(&block);
    }
    children.push_back(ln_f_.get());
    children.push_back(&policy_head_);
    children.push_back(&value_head_);
    if (value_norm_) {
        children.push_back(value_norm_.get());
    }
    if (cfg_.cnn_policy) {
        children.push_back(&cnn_policy_head_);
    }
    return children;
}

nn::Result<std::vector<std::pair<std::string, double>>>
GoModel::debug_grad_norms() {
    nn::ComputeEngine& engine = engine_ref();
    // 逐张量读回算 L2（诊断专用：GPU 上每张量一次同步，不进热路径）。
    const auto group_norm = [&engine](const std::vector<nn::TensorRef>& refs)
        -> nn::Result<double> {
        double sum_sq = 0;
        for (const nn::TensorRef& ref : refs) {
            nn::Tensor& tensor = ref.get();
            if (!tensor.valid()) {
                continue;
            }
            std::vector<nn::Scalar> buffer(tensor.size(), nn::Scalar{0});
            NN_TRY_CHECK(engine.read(tensor, std::span<nn::Scalar>(buffer)));
            for (nn::Scalar value : buffer) {
                const double v = static_cast<double>(value);
                sum_sq += v * v;
            }
        }
        return std::sqrt(sum_sq);
    };

    std::vector<std::pair<std::string, double>> out;
    const auto emit = [&out](const char* name, double norm) {
        out.emplace_back(name, norm);
    };

    NN_TRY(embed, group_norm(
        std::vector<nn::TensorRef>{
            grad_hist_emb_, grad_pos_emb_, grad_seg_emb_, grad_query_tok_}));
    emit("embeddings", *embed);
    NN_TRY(conv1, group_norm(conv1_.param_gradients()));
    emit("conv1", *conv1);
    NN_TRY(conv2, group_norm(conv2_.param_gradients()));
    emit("conv2", *conv2);
    double blocks_sq = 0;
    for (auto& block : blocks_) {
        NN_TRY(bn, group_norm(block.param_gradients()));
        blocks_sq += (*bn) * (*bn);
    }
    emit("blocks", std::sqrt(blocks_sq));
    NN_TRY(ln_f, group_norm(ln_f_->param_gradients()));
    emit("ln_f", *ln_f);
    NN_TRY(policy, group_norm(policy_head_.param_gradients()));
    emit("policy_head", *policy);
    NN_TRY(value, group_norm(value_head_.param_gradients()));
    emit("value_head", *value);
    if (value_norm_) {
        NN_TRY(vn, group_norm(value_norm_->param_gradients()));
        emit("value_norm", *vn);
    }
    if (cfg_.cnn_policy) {
        NN_TRY(cnn, group_norm(cnn_policy_head_.param_gradients()));
        emit("cnn_policy", *cnn);
    }
    return out;
}

void GoModel::set_precision_profile(const nn::PrecisionProfile& profile) {
    nn::Layer::set_precision_profile(profile);
    conv1_.set_precision_profile(profile);
    relu1_.set_precision_profile(profile);
    conv2_.set_precision_profile(profile);
    relu2_.set_precision_profile(profile);
    for (auto& block : blocks_) {
        block.set_precision_profile(profile);
    }
    if (ln_f_) ln_f_->set_precision_profile(profile);
    policy_head_.set_precision_profile(profile);
    value_head_.set_precision_profile(profile);
    if (value_norm_) value_norm_->set_precision_profile(profile);
    if (cfg_.cnn_policy) cnn_policy_head_.set_precision_profile(profile);
}

void GoModel::set_training(bool training) {
    nn::Layer::set_training(training);
    conv1_.set_training(training);
    relu1_.set_training(training);
    conv2_.set_training(training);
    relu2_.set_training(training);
    for (auto& block : blocks_) {
        block.set_training(training);
    }
    if (ln_f_) ln_f_->set_training(training);
    policy_head_.set_training(training);
    value_head_.set_training(training);
    if (value_norm_) value_norm_->set_training(training);
    if (cfg_.cnn_policy) cnn_policy_head_.set_training(training);
}

void GoModel::clear_cache() {
    conv1_.clear_cache();
    relu1_.clear_cache();
    conv2_.clear_cache();
    relu2_.clear_cache();
    for (auto& block : blocks_) {
        block.clear_cache();
    }
    if (ln_f_) ln_f_->clear_cache();
    policy_head_.clear_cache();
    value_head_.clear_cache();
    if (value_norm_) value_norm_->clear_cache();
    if (cfg_.cnn_policy) cnn_policy_head_.clear_cache();
}

nn::Result<void> GoModel::ensure_indices_(
    nn::ComputeEngine& engine, std::size_t batch
) {
    const std::size_t d = cfg_.d_model;
    const std::size_t k = cfg_.history_len;
    const std::size_t t_count = seq_len_;

    if (!region_perm_.valid()) {
        // c·25+p → p·d+c（把通道分组改成区域分组），反向为逆置换。
        std::vector<Scalar> perm(d * region_token_count);
        std::vector<Scalar> inverse(d * region_token_count);
        for (std::size_t c = 0; c < d; ++c) {
            for (std::size_t p = 0; p < region_token_count; ++p) {
                perm[p * d + c] = static_cast<Scalar>(c * region_token_count + p);
                inverse[c * region_token_count + p] = static_cast<Scalar>(p * d + c);
            }
        }
        NN_TRY(a, upload_index(engine, perm));
        region_perm_ = std::move(*a);
        NN_TRY(b, upload_index(engine, inverse));
        region_perm_inv_ = std::move(*b);
    }

    if (seq_perm_.valid() && batch == batch_) {
        return {};
    }

    std::vector<Scalar> seq(batch * t_count);
    std::vector<Scalar> pos(batch * t_count);
    std::vector<Scalar> seg(batch * t_count);
    std::vector<Scalar> query_index(batch, Scalar{0});
    std::vector<Scalar> query_row(batch);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t t = 0; t < t_count; ++t) {
            const std::size_t col = b * t_count + t;
            pos[col] = static_cast<Scalar>(t);
            if (t < region_token_count) {
                // 盘面块行序 p·batch+b
                seq[col] = static_cast<Scalar>(t * batch + b);
                seg[col] = Scalar{0};
            } else if (t < region_token_count + k) {
                // 历史块行序 b·K+k
                seq[col] = static_cast<Scalar>(
                    region_token_count * batch + b * k + (t - region_token_count)
                );
                seg[col] = Scalar{1};
            } else {
                // query 块行序 b
                seq[col] = static_cast<Scalar>(
                    (region_token_count + k) * batch + b
                );
                seg[col] = Scalar{2};
            }
        }
        query_row[b] = static_cast<Scalar>(b * t_count + (t_count - 1));
    }

    NN_TRY(c, upload_index(engine, seq));
    seq_perm_ = std::move(*c);
    NN_TRY(d1, upload_index(engine, pos));
    pos_idx_ = std::move(*d1);
    NN_TRY(e, upload_index(engine, seg));
    seg_idx_ = std::move(*e);
    NN_TRY(f, upload_index(engine, query_index));
    query_idx_ = std::move(*f);
    NN_TRY(g, upload_index(engine, query_row));
    query_row_idx_ = std::move(*g);
    return {};
}

nn::Result<Tensor> GoModel::forward(const Tensor& input) {
    nn::ComputeEngine& engine = engine_ref();
    const std::size_t k = cfg_.history_len;
    const std::size_t d = cfg_.d_model;
    const std::size_t tokens = seq_len_;
    if (input.rows() != go_input_rows(k)) {
        NN_FAIL(
            "GoModel::forward: 输入行数应为 " + std::to_string(go_input_rows(k)) +
            "，实际 " + std::to_string(input.rows())
        );
    }
    if (input.cols() == 0) {
        NN_FAIL("GoModel::forward: batch 不能为 0");
    }
    const std::size_t batch = input.cols();
    NN_TRY(ri, ensure_indices_(engine, batch));
    batch_ = batch;

    // ── 1. 盘面分支：9 平面 → (d, 5×5) 区域 token ──────────────────────
    NN_TRY(planes, engine.slice_rows(input, 0, feature_element_count));
    NN_TRY(c1, conv1_.forward(*planes));
    NN_TRY(r1, relu1_.forward(*c1));
    NN_TRY(c2, conv2_.forward(*r1));
    NN_TRY(r2, relu2_.forward(*c2));
    // (d·25, batch) 行序 c·25+p；先换成区域分组行序 p·d+c，
    // 再 rearrange 成 (d, 25·batch) 的列主序（列 p·batch+b）。
    NN_TRY(board_cp, engine.gather_rows(*r2, region_perm_));
    NN_TRY(
        board_cols,
        engine.rearrange_3d(*board_cp, d, region_token_count, batch, /*inverse=*/true)
    );
    NN_TRY(board_rows, engine.transpose(*board_cols));  // (25·batch, d)

    // ── 2. 历史分支：K 手 id 查表 ───────────────────────────────────────
    NN_TRY(ids, engine.slice_rows(input, feature_element_count, k));  // (K, batch)
    NN_TRY(ids_flat, engine.transpose(*ids));                         // (batch, K) 行序 b·K+kk
    ids_flat_ = std::move(*ids_flat);
    NN_TRY(hist_rows, engine.gather_rows(hist_emb_, ids_flat_));      // (batch·K, d)

    // ── 3. decision query ──────────────────────────────────────────────
    NN_TRY(query_rows, engine.gather_rows(query_tok_, query_idx_));  // (batch, d)

    // ── 4. token 袋：行序 [盘面 p·batch+b][历史 b·K+kk][query b] ────────
    Tensor bag = engine.create_tensor(batch * tokens, d, p_.compute);
    if (!bag.valid()) {
        NN_FAIL("GoModel::forward: 序列暂存张量分配失败");
    }
    NN_TRY(i1, engine.insert_rows(bag, 0, *board_rows));
    NN_TRY(i2, engine.insert_rows(bag, region_token_count * batch, *hist_rows));
    NN_TRY(i3, engine.insert_rows(
        bag, (region_token_count + k) * batch, *query_rows));

    // ── 5. 重排成 batch-major 列序（行 b·T+t）→ 转列主序 ────────────────
    NN_TRY(seq_rows, engine.gather_rows(bag, seq_perm_));
    NN_TRY(seq_cols, engine.transpose(*seq_rows));  // (d, batch·T)

    // ── 6. 位置编码 + 段嵌入：一次融合加法 ──────────────────────────────
    NN_TRY(pos_rows, engine.gather_rows(pos_emb_, pos_idx_));
    NN_TRY(pos_cols, engine.transpose(*pos_rows));
    NN_TRY(seg_rows, engine.gather_rows(seg_emb_, seg_idx_));
    NN_TRY(seg_cols, engine.transpose(*seg_rows));
    NN_TRY(x0, nn::dsl::compute(engine,
        nn::dsl::leaf(*seq_cols) + nn::dsl::leaf(*pos_cols) + nn::dsl::leaf(*seg_cols),
        d, batch * tokens, p_.compute));

    // ── 7. 编码器（Plain 双向掩码，Pre-Norm）───────────────────────────
    Tensor x = std::move(*x0);
    for (auto& block : blocks_) {
        NN_TRY(bx, block.forward(x));
        x = std::move(*bx);
    }
    NN_TRY(lf, ln_f_->forward(x));

    // ── 8. 取 query 行（列 b·T+T-1）────────────────────────────────────
    NN_TRY(lf_rows, engine.transpose(*lf));                          // (batch·T, d)
    NN_TRY(q_row, engine.gather_rows(*lf_rows, query_row_idx_));     // (batch, d)
    NN_TRY(q_col, engine.transpose(*q_row));                         // (d, batch)

    // ── 9. 双头 + tanh + 打包 ──────────────────────────────────────────
    NN_TRY(logits, policy_head_.forward(*q_col));      // (362, batch)
    if (cfg_.cnn_policy) {
        // 直通路径：特征图展平 (d·25, batch) → 362，与 query logits 相加。
        NN_TRY(cnn_logits, cnn_policy_head_.forward(*r2));
        NN_TRY(merged, nn::dsl::compute(engine,
            nn::dsl::leaf(*logits) + nn::dsl::leaf(*cnn_logits),
            policy_class_count, batch, p_.compute));
        logits = std::move(merged);
    }
    // M10：价值头读盘面特征图 r2（KataGo 同款接线）——query token 被 362
    // 类策略损失主导，价值信号流不进去（实测三代模型 value AUC 塌缩 0.5）。
    // 读 r2 前过专用 LayerNorm：未归一化的卷积输出直进 Linear+tanh 会饱和
    //（float32 下 tanh(|x|≳9) 精确 =±1 → 1-t²=0 → 价值头零梯度）。
    nn::Result<Tensor> value_in = cfg_.value_from_spatial
        ? value_norm_->forward(*r2)
        : nn::Result<Tensor>(std::move(*q_col));
    NN_TRY(value_pre, value_head_.forward(*value_in));  // (1, batch)
    NN_TRY(value, nn::dsl::compute(engine,
        nn::dsl::tanh(nn::dsl::leaf(*value_pre)), 1, batch, p_.compute));
    value_t_ = std::move(*value);

    Tensor packed = engine.create_tensor(packed_output_rows, batch, p_.compute);
    if (!packed.valid()) {
        NN_FAIL("GoModel::forward: 打包输出张量分配失败");
    }
    NN_TRY(p1, engine.insert_rows(packed, 0, *logits));
    NN_TRY(p2, engine.insert_rows(packed, value_row_index, value_t_));
    return packed;
}

nn::Result<Tensor> GoModel::backward(const Tensor& grad_output) {
    nn::ComputeEngine& engine = engine_ref();
    const std::size_t k = cfg_.history_len;
    const std::size_t d = cfg_.d_model;
    const std::size_t tokens = seq_len_;
    const std::size_t batch = batch_;
    if (batch == 0) {
        NN_FAIL("GoModel::backward: 必须先调用 forward");
    }
    if (grad_output.rows() != packed_output_rows || grad_output.cols() != batch) {
        NN_FAIL("GoModel::backward: grad_output 形状必须是 (363, batch)");
    }

    // ── 1. 拆包 ────────────────────────────────────────────────────────
    NN_TRY(g_policy, engine.slice_rows(grad_output, 0, policy_class_count));
    NN_TRY(g_value, engine.slice_rows(grad_output, value_row_index, 1));

    // ── 2. tanh 反传：dL/dv = dL/dy · (1 - t²) ─────────────────────────
    // 同一张量出现两次会让输入槽去重把运行期结构折叠成与锚点不同的 key
    //（AGENTS §7 "输入槽去重"），故先 clone 出第二份 t 再写表达式。
    NN_TRY(t_copy, engine.clone(value_t_));
    NN_TRY(g_pre, nn::dsl::compute(engine,
        nn::dsl::leaf(*g_value) *
            (nn::dsl::rparam(Scalar{1}) - nn::dsl::leaf(value_t_) * nn::dsl::leaf(*t_copy)),
        1, batch, p_.compute));

    // ── 3. 双头反传 ─────────────────────────────────────────────────────
    // 接线路由（M10/M10b）：
    // · legacy：价值头读 query，输入梯度与策略头合并回传 query；
    // · spatial（探针式）：价值头读 LayerNorm(r2)，**输入梯度不回流 trunk**。
    //   第一版让它并入盘面分支，实测把策略摧毁（epoch1 top1 27.8%→0.8%）：
    //   LayerNorm 反向的 1/std 放大让价值梯度比策略梯度大 2-3 个量级，而
    //   clip_grad_norm 是全局裁剪 → 裁剪系数被价值分支压到极小 → 策略头
    //   拿不到有效更新。trunk 在拿到自身归一化（BN 类）之前，价值头只做
    //   "探针"：读空间特征、学自己的权重（价值头/γ/β 梯度照常），不塑形
    //   共享 trunk——KataGo 的联合塑形等 trunk 归一化落地后再开。
    NN_TRY(b_policy, policy_head_.backward(*g_policy));   // (d, batch)
    nn::Result<Tensor> g_query = std::move(*b_policy);
    if (cfg_.value_from_spatial) {
        // 反向照走（价值头与 norm 的权重/γ/β 梯度由此产生），输入梯度丢弃。
        NN_TRY(b_value_pre, value_head_.backward(*g_pre));   // (d·25, batch)
        NN_TRY_CHECK(value_norm_->backward(*b_value_pre));   // γ/β 梯度
    } else {
        NN_TRY(b_value, value_head_.backward(*g_pre));       // (d, batch)
        NN_TRY(merged_q, nn::dsl::compute(engine,
            nn::dsl::leaf(*g_query) + nn::dsl::leaf(*b_value), d, batch,
            p_.compute));
        g_query = std::move(*merged_q);
    }

    // ── 4. query 列 → 序列行空间（其余位置梯度为 0）────────────────────
    NN_TRY(g_query_rows, engine.transpose(*g_query));   // (batch, d)
    Tensor seq_grad = engine.create_tensor(batch * tokens, d, p_.compute);
    if (!seq_grad.valid()) {
        NN_FAIL("GoModel::backward: 序列梯度张量分配失败");
    }
    NN_TRY(z0, engine.zero(seq_grad));
    NN_TRY(s0, engine.scatter_add_rows(seq_grad, query_row_idx_, *g_query_rows));

    // ── 5. 位置 / 段嵌入梯度（行序 b·T+t，与 *_idx 对齐）────────────────
    NN_TRY(s1, engine.scatter_add_rows(grad_pos_emb_, pos_idx_, seq_grad));
    NN_TRY(s2, engine.scatter_add_rows(grad_seg_emb_, seg_idx_, seq_grad));

    // ── 6. 编码器反向（逐块逆序 + 释放激活）────────────────────────────
    NN_TRY(g_seq, engine.transpose(seq_grad));   // (d, batch·T)
    Tensor grad_x = std::move(*g_seq);
    for (std::size_t i = blocks_.size(); i-- > 0;) {
        NN_TRY(bb, blocks_[i].backward(grad_x));
        grad_x = std::move(*bb);
        blocks_[i].clear_cache();
    }
    NN_TRY(b_ln, ln_f_->backward(grad_x));
    ln_f_->clear_cache();

    // ── 7. 序列 → token 袋的反向（gather 的伴随 = scatter_add）──────────
    NN_TRY(g_seq_rows, engine.transpose(*b_ln));  // (batch·T, d)
    Tensor bag_grad = engine.create_tensor(batch * tokens, d, p_.compute);
    if (!bag_grad.valid()) {
        NN_FAIL("GoModel::backward: token 袋梯度张量分配失败");
    }
    NN_TRY(z1, engine.zero(bag_grad));
    NN_TRY(s3, engine.scatter_add_rows(bag_grad, seq_perm_, *g_seq_rows));

    // ── 8. 盘面块反向 ──────────────────────────────────────────────────
    NN_TRY(g_board_rows,
        engine.slice_rows(bag_grad, 0, region_token_count * batch));   // (25·batch, d)
    NN_TRY(g_board_cols, engine.transpose(*g_board_rows));            // (d, 25·batch)
    NN_TRY(g_board_cp,
        engine.rearrange_3d(*g_board_cols, d, region_token_count, batch,
                            /*inverse=*/false));                      // (25·d, batch) 行序 p·d+c
    NN_TRY(g_board, engine.gather_rows(*g_board_cp, region_perm_inv_)); // (d·25, batch)
    Tensor board_grad = std::move(*g_board);
    if (cfg_.cnn_policy) {
        NN_TRY(cnn_grad, cnn_policy_head_.backward(*g_policy));  // (d·25, batch)
        NN_TRY(merged_grad, nn::dsl::compute(engine,
            nn::dsl::leaf(board_grad) + nn::dsl::leaf(*cnn_grad),
            cfg_.d_model * region_token_count, batch, p_.compute));
        board_grad = std::move(*merged_grad);
    }
    // （spatial 接线的价值梯度在此前已丢弃——探针式，见第 3 步注释。）
    NN_TRY(b_r2, relu2_.backward(board_grad));
    NN_TRY(b_c2, conv2_.backward(*b_r2));
    NN_TRY(b_r1, relu1_.backward(*b_c2));
    NN_TRY(b_planes, conv1_.backward(*b_r1));                          // (3249, batch)

    // ── 9. 历史块 / query 块反向（查表的伴随）──────────────────────────
    NN_TRY(g_hist_rows,
        engine.slice_rows(bag_grad, region_token_count * batch, k * batch));
    NN_TRY(s4, engine.scatter_add_rows(grad_hist_emb_, ids_flat_, *g_hist_rows));
    NN_TRY(g_query_rows2, engine.slice_rows(
        bag_grad, (region_token_count + k) * batch, batch));
    NN_TRY(s5, engine.scatter_add_rows(grad_query_tok_, query_idx_, *g_query_rows2));

    // ── 10. 输入梯度：平面块有梯度，历史 id 行无梯度 ────────────────────
    Tensor grad_input = engine.create_tensor(go_input_rows(k), batch, p_.compute);
    if (!grad_input.valid()) {
        NN_FAIL("GoModel::backward: 输入梯度张量分配失败");
    }
    NN_TRY(z2, engine.zero(grad_input));
    NN_TRY(i4, engine.insert_rows(grad_input, 0, *b_planes));
    return grad_input;
}

// ══════════════════════════════════════════════════════════════════════════
// GoLoss
// ══════════════════════════════════════════════════════════════════════════
GoLoss::GoLoss(Scalar value_weight) : value_weight_(value_weight) {}

nn::Result<Scalar> GoLoss::forward(
    nn::ComputeEngine& engine,
    const Tensor& packed,
    std::span<const std::size_t> labels,
    std::span<const Scalar> values
) {
    if (packed.rows() != packed_output_rows) {
        NN_FAIL(
            "GoLoss::forward: 输入行数必须为 363，实际 " +
            std::to_string(packed.rows())
        );
    }
    const std::size_t batch = packed.cols();
    if (labels.size() != batch || values.size() != batch) {
        NN_FAIL("GoLoss::forward: 标签/价值目标数量与 batch 不一致");
    }

    // 策略：行 [0,362) 的稀疏 CE
    NN_TRY(policy_logits, engine.slice_rows(packed, 0, policy_class_count));
    NN_TRY(policy_loss, policy_loss_.forward_sparse(
        engine, *policy_logits, labels, {}, policy_class_count));

    // 价值：行 362 的 MSE（目标 z 经宿主桥上传，每步一次小批量）
    NN_TRY(value_pred, engine.slice_rows(packed, value_row_index, 1));
    NN_TRY(value_target, nn::detail::upload_span(
        engine, 1, batch, nn::Precision::F32, values));
    NN_TRY(value_loss, value_loss_.forward(engine, *value_pred, *value_target));
    // 诊断：留档未加权分项（价值 AUC 停滞时看价值 MSE 是否真在降）。
    last_policy_loss_ = *policy_loss;
    last_value_loss_ = *value_loss;

    // 两段梯度拼回 (363, batch)：策略段直接拷入，价值段按权重缩放后拷入。
    NN_TRY(g_policy, policy_loss_.backward());        // (362, batch)
    NN_TRY(g_value, value_loss_.backward());          // (1, batch)
    NN_TRY(scaled, nn::dsl::compute_into(engine,
        nn::dsl::leaf(*g_value) * nn::dsl::rparam(value_weight_), *g_value));

    Tensor grads = engine.create_tensor(packed_output_rows, batch, p_.stable);
    if (!grads.valid()) {
        NN_FAIL("GoLoss::forward: 梯度打包张量分配失败");
    }
    NN_TRY(z, engine.zero(grads));
    NN_TRY(i1, engine.insert_rows(grads, 0, *g_policy));
    NN_TRY(i2, engine.insert_rows(grads, value_row_index, *g_value));
    grad_packed_ = std::move(grads);

    return *policy_loss + value_weight_ * *value_loss;
}

nn::Result<Scalar> GoLoss::forward(
    nn::ComputeEngine& /*engine*/,
    const Tensor& /*pred*/,
    const Tensor& /*target*/
) {
    NN_FAIL(
        "GoLoss: 双张量 forward 不适用——请用 (packed, labels, values) 重载"
    );
}

nn::Result<Tensor> GoLoss::backward() {
    if (!grad_packed_.valid()) {
        NN_FAIL("GoLoss::backward: 先调用 forward");
    }
    return grad_packed_;
}

}  // namespace go
