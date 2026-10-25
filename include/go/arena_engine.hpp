#pragma once

// ── 对局引擎抽象（M6 抽出）────────────────────────────────────────────────
//
// go_arena_cli（批量评估 + SPRT）与 go_gui_cli（竞技台观战 / 对弈）共用的
// 引擎接口。genmove 返回 nullopt = pass；notify_move 把对手刚下的棋同步给
// 引擎（进程内引擎靠 go::Game 自行追踪，外部 GTP 引擎必须收到 play 指令）。
//
// 引擎 spec 语法（make_engine）：
//   random | model:FILE[=DEVICE] | gtp:COMMAND

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "go/game.hpp"
#include "go/policy.hpp"

namespace go {

class ArenaEngine {
public:
    virtual ~ArenaEngine() = default;

    // 为 game 的当前行棋方选一步棋。返回 nullopt = pass。
    virtual std::optional<Point> genmove(Game& game) = 0;

    // 新一局开始：外部 GTP 子进程需要 boardsize / clear_board / komi 同步。
    virtual void clear_board(int size, float komi) = 0;

    // 对手刚下的棋（含 pass）。
    virtual void notify_move(Stone, const std::optional<Point>&) {}

    // 当前局面的策略/价值评估（GUI 候选着热力图）。默认实现不直接拒绝，
    // 而是把 preview_move() 的"将下点"合成一个单点 100% 的分布（synthetic
    // = true）——AI 视角对任何引擎都有内容可看。连预告都给不出时才 nullopt。
    [[nodiscard]] virtual std::optional<PolicyEvaluation> evaluate(const Game& game);

    // 下一手预告：不落子、不改变引擎内部盘面。外层 nullopt = 无法预告；
    // 内层 nullopt = 这一手是 pass。Random 引擎会把预告的一步记住，下一次
    // genmove 就下它（预告与实战一致）；外部 GTP 引擎走 reg_genmove。
    [[nodiscard]] virtual std::optional<std::optional<Point>> preview_move(
        const Game& game);

    [[nodiscard]] virtual std::string name() const = 0;
};

// 解析引擎 spec；seed 只作用于 random 引擎。非法 spec 抛 std::invalid_argument。
[[nodiscard]] std::unique_ptr<ArenaEngine> make_engine(
    const std::string& spec, std::uint64_t seed);

// 展示名：random → "随机引擎"，model:x/y.nnmodel → "y"，gtp:cmd → cmd 首词。
[[nodiscard]] std::string engine_display_name(const std::string& spec);

}  // namespace go
