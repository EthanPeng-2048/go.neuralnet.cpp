#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "go/game.hpp"

namespace go {

// 单次前向的完整评估输出。
struct PolicyEvaluation {
    // 362 类概率（前 361 类 = 棋盘点 y*19+x，最后一类 = pass）：在**合法着**
    // 集合上做 softmax，非法着恒为 0，因此可以直接当落子分布可视化。
    std::vector<float> policy;
    // 价值头输出（当前行棋方视角，tanh 后 ∈ [-1,1]）。C 验证器没有价值头
    // （362 行全是策略），此时为 nullopt。
    std::optional<float> value;
    // true = 这不是模型分布，而是引擎的"将下点预告"（单点 100%）之类的合成
    // 分布，仅供可视化；false = 模型的真实 softmax。
    bool synthetic = false;
};

// 策略玩家（M6）：把一个已加载的策略模型接到棋规上——提取当前盘面的
// M3 v1 特征、前向一次、在**合法着点**（含 pass）上取 argmax。
// GTP 适配器（人类下棋）与 Arena（自动评估）共用这一个类，保证两条
// 路径的选点语义完全一致。
//
// 合法性掩码在 host 侧做（roadmap §4 扩展点）：先验再高也不能下在
// 禁入点/劫争点上；若模型对所有合法点都给出 -inf（不应发生），回退到
// pass，保证 genmove 永远有合法应答。
class PolicyPlayer {
public:
    // 从 .nnmodel / .nnckpt 载入。`device` 为空 = CPU，否则为 GPU 设备名。
    static PolicyPlayer load(
        const std::string& model_path,
        const std::string& device = {}
    );

    PolicyPlayer(PolicyPlayer&&) noexcept;
    PolicyPlayer& operator=(PolicyPlayer&&) noexcept;
    ~PolicyPlayer();

    PolicyPlayer(const PolicyPlayer&) = delete;
    PolicyPlayer& operator=(const PolicyPlayer&) = delete;

    // 为 `game` 的当前行棋方选一步棋。返回 nullopt = pass。
    // 非 19×19 盘面抛异常（M3 v1 特征契约是 19×19 专用）。
    [[nodiscard]] std::optional<Point> select_move(const Game& game);

    // 单次前向的完整评估（GUI 的候选着/形势判断，M8 的 MCTS 先验也复用它）。
    // 语义与 select_move 同源：同一特征、同一前向，只是不取 argmax。
    [[nodiscard]] PolicyEvaluation evaluate(const Game& game);

    // 模型文件路径（诊断用）。
    [[nodiscard]] const std::string& model_path() const noexcept;

private:
    struct Impl;
    explicit PolicyPlayer(std::unique_ptr<Impl> impl) noexcept;

    // 特征 → 前向 → logits（策略行 + 可选价值行）。select_move 与 evaluate
    // 共用同一次前向，保证选点与可视化永远同源。
    void forward_logits(const Game& game);

    std::unique_ptr<Impl> impl_;
};

}  // namespace go
