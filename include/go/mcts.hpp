#pragma once

// ── M8：MCTS 推理引擎（只搜索不训练，D9）──────────────────────────────────
//
// PUCT 树搜索 + 网络先验/叶子评估，全部经 LeafEvaluator 抽象取数：
//   · 先验 = 评估器的合法着 softmax（362 类契约，非法着恒 0，展开时再归一）；
//   · 叶值 = B 的价值头（tanh，节点行棋方视角）；C 没有价值头，走
//     ValueFallback 的静态计分回退，保证 C/B 两条模型线都能驱动同一棵树。
//
// 结构（roadmap §6 M8 DoD 的四件套）：
//   · PUCT 选择：q + c_puct · P · sqrt(N)/(1+n)，未访问子着用 FPU 折扣；
//   · 先验归一化：只在合法着（含 pass）上重归一，全零先验退化为均匀；
//   · 合法性：子着集合完全来自 KataGo 的 legal_moves()，网络先验永远不能
//     把非法点带进树（与 PolicyPlayer 的 host 侧掩码同源）；
//   · 转置表：评估缓存。key 覆盖"评估器看到的一切"——盘面 + 行棋方 +
//     简单劫 + 最近 tt_history_window 手历史（C 的 recent-moves 平面与 B 的
//     K 手历史分支都在窗口内）。合法性始终按当前对局历史由 KataGo 判定，
//     所以 key 碰撞/合并只影响评估复用，不影响规则正确性。
//
// 代价口径：节点不存盘面，每次模拟从根重放走子序列（KataGo makeBoardMove
// 的成本远低于一次网络前向），内存只随节点数线性增长。
//
// 盘面尺寸：引擎本身与尺寸无关（测试用 5×5/9×9 mock）；真模型线受 M3 v1
// 特征契约约束，只跑 19×19（PolicyPlayer 会拒绝其它尺寸）。

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "go/game.hpp"
#include "go/policy.hpp"

namespace go {

// 搜索配置。simulations 就是赌约条款里的 visits。
struct MctsConfig {
    std::size_t simulations = 250;       // 模拟次数（visits）
    float c_puct = 1.5F;                 // PUCT 探索常数
    float fpu_reduction = 0.25F;         // 未访问子着的先验紧迫度折扣（父 Q - 该值）
    float temperature = 0.0F;            // >0：按访问数^(1/T) 采样行棋；0：取访问数最大
    bool root_noise = false;             // 根 Dirichlet 噪声（自对局多样性；D9 默认关）
    float root_noise_alpha = 0.3F;
    float root_noise_eps = 0.25F;
    std::uint64_t seed = 0;              // 0 = 用 random_device 播种
    // 转置表条目上限（0 = 关闭缓存）。每条缓存一个 362 类先验（≈1.5 KiB），
    // 8192 条 ≈ 12 MiB 常驻；单棵树的节点数 ≤ simulations，跨局面命中主要
    // 来自 pass / 劫循环，容量满即整体淘汰。
    std::size_t tt_capacity = 8192;
    std::size_t tt_history_window = 64;  // key 覆盖的近手窗口（须 ≥ 模型 history_len，≥3）
};

// 无价值头模型的叶子价值回退。
enum class ValueFallback {
    none,        // 恒 0（只调树形时用）
    area_score,  // KataGo 面积计分静态评估（C 验证器默认）：tanh(目差/15)
};

// 叶子评估器：单局面 → 合法着先验 + 价值。测试注入 mock，产品线用模型。
class LeafEvaluator {
public:
    virtual ~LeafEvaluator() = default;

    // 返回的 policy 按「y * width + x」索引，pass = width*height；
    // 19×19 时与 PolicyEvaluation 的 362 类布局完全重合（pass=361）。
    // value = 当前行棋方视角；nullopt 视为无价值头，由 MctsSearch 按
    // ValueFallback 处理（make_policy_evaluator 已内置该处理）。
    [[nodiscard]] virtual PolicyEvaluation evaluate(const Game& game) = 0;
};

// PolicyPlayer → LeafEvaluator 适配（C/B 通用）。
[[nodiscard]] std::unique_ptr<LeafEvaluator> make_policy_evaluator(
    PolicyPlayer& player,
    ValueFallback fallback = ValueFallback::area_score);

// 根子着统计（value 已换算回根行棋方视角）。
struct MctsMove {
    std::optional<Point> move;  // nullopt = pass
    std::size_t visits = 0;
    float prior = 0.0F;
    float value = 0.0F;
};

struct MctsResult {
    std::optional<Point> best_move;  // nullopt = pass（根无合法着时同样返回 pass）
    float root_value = 0.0F;         // 根 Q（根行棋方视角）
    std::vector<MctsMove> moves;     // 根子着，按 visits 降序
    std::size_t simulations = 0;
    std::size_t nn_evaluations = 0;  // 本次搜索实际调用评估器次数
    std::size_t tt_hits = 0;         // 本次搜索转置表命中次数
};

class MctsSearch {
public:
    MctsSearch(LeafEvaluator& evaluator, MctsConfig config = {});
    MctsSearch(MctsSearch&&) noexcept;
    MctsSearch& operator=(MctsSearch&&) noexcept;
    ~MctsSearch();

    MctsSearch(const MctsSearch&) = delete;
    MctsSearch& operator=(const MctsSearch&) = delete;

    // 为 root 的当前行棋方搜索最优着。可对同一实例连续调用（换一局面
    // 重新搜索）；转置表跨调用保留，节点树每次重建。
    [[nodiscard]] MctsResult search(const Game& root);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace go
