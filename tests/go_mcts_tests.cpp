// ── M8 MCTS 契约测试 ─────────────────────────────────────────────────────
//
// mock 评估器（无模型、无 NN 依赖），从便宜到深刻住四件套语义：
//   1. PUCT 选择：高先验着拿走绝大多数 visits，且先验在合法着上归一；
//   2. 合法性：网络给非法点再高的先验，树里也永远不出现；
//   3. negamax 符号：一致的"白优"评估在根子边上全部换算成根行棋方视角；
//   4. 终局计分：pass-pass 终局按 KataGo 计分备份；
//   5. 转置表：同一局面第二次搜索命中缓存，评估器调用减少。

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <vector>

#include "go/game.hpp"
#include "go/mcts.hpp"

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

// mock：按"点位偏置 + 白优价值"给先验/价值，同时统计调用次数。
class MockEvaluator final : public go::LeafEvaluator {
public:
    std::optional<go::Point> bias_point;   // 该点先验取 bias_weight，其余均分
    float bias_weight = 0.0F;
    float white_advantage = 0.0F;          // 白棋优势；与行棋方无关
    bool allow_pass = true;                // false = pass 先验恒 0
    std::size_t calls = 0;

    go::PolicyEvaluation evaluate(const go::Game& game) override {
        ++calls;
        go::PolicyEvaluation result;
        const std::size_t points =
            static_cast<std::size_t>(game.width()) * game.height();
        result.policy.assign(points + 1, 0.0F);

        std::vector<std::size_t> indices;
        for (int y = 0; y < game.height(); ++y) {
            for (int x = 0; x < game.width(); ++x) {
                indices.push_back(
                    static_cast<std::size_t>(y) * game.width() + x);
            }
        }
        if (allow_pass) {
            indices.push_back(points);
        }

        const float uniform =
            (1.0F - bias_weight) / static_cast<float>(indices.size());
        for (const std::size_t index : indices) {
            result.policy[index] = uniform;
        }
        if (bias_point.has_value()) {
            const std::size_t index =
                static_cast<std::size_t>(bias_point->y) * game.width() +
                bias_point->x;
            result.policy[index] = bias_weight;
        }

        // 一致评估：白优 white_advantage → 换算到当前行棋方视角。
        result.value = game.to_move() == go::Stone::white ? white_advantage
                                                          : -white_advantage;
        return result;
    }
};

// pass-only mock：先验 100% 给 pass（终局计分测试用）。
class PassOnlyEvaluator final : public go::LeafEvaluator {
public:
    float white_advantage = 0.0F;

    go::PolicyEvaluation evaluate(const go::Game& game) override {
        go::PolicyEvaluation result;
        const std::size_t points =
            static_cast<std::size_t>(game.width()) * game.height();
        result.policy.assign(points + 1, 0.0F);
        result.policy[points] = 1.0F;
        result.value = game.to_move() == go::Stone::white ? white_advantage
                                                          : -white_advantage;
        return result;
    }
};

go::Game empty_board(int size, go::Stone to_move) {
    return go::Game(size, size, to_move, go::RuleSet::chinese());
}

// ── 1. PUCT 选择 + 先验归一化 ────────────────────────────────────────────
void test_prior_driven_choice() {
    go::Game game = empty_board(5, go::Stone::black);
    MockEvaluator mock;
    mock.bias_point = go::Point{2, 2};
    mock.bias_weight = 0.9F;

    go::MctsConfig config;
    config.simulations = 200;
    config.seed = 20261025;
    go::MctsSearch search(mock, config);
    const go::MctsResult result = search.search(game);

    require(result.best_move.has_value(), "best move must be a board point");
    if (result.best_move.has_value()) {
        require(result.best_move->x == 2 && result.best_move->y == 2,
                "PUCT must return the high-prior point");
    }
    require_eq(result.moves.size(), 26, "5x5 legal moves = 25 points + pass");
    require_eq(result.simulations, 200, "simulations are reported");

    double prior_total = 0.0;
    std::size_t visit_total = 0;
    for (const go::MctsMove& move : result.moves) {
        prior_total += static_cast<double>(move.prior);
        visit_total += move.visits;
    }
    require(std::abs(prior_total - 1.0) < 1e-4,
            "priors must be renormalized over legal moves");
    // 第 1 次模拟展开根自身，之后每次模拟走一条根子边。
    require_eq(visit_total + 1, result.simulations,
               "root child visits = simulations - root expansion");

    require(result.moves.front().visits > result.moves.back().visits,
            "high-prior move must dominate visit counts");
}

// ── 2. 合法性：非法点永远进不了树 ────────────────────────────────────────
void test_legality_mask() {
    const go::Game game = go::Game::from_ascii(
        ".x...\n"
        ".....\n"
        "..o..\n"
        ".....\n"
        ".....\n",
        go::Stone::black, go::RuleSet::chinese());

    // 把 0.9 的先验堆到占用点（非法着）上。
    MockEvaluator mock;
    mock.bias_point = go::Point{1, 0};
    mock.bias_weight = 0.9F;

    go::MctsConfig config;
    config.simulations = 120;
    config.seed = 7;
    go::MctsSearch search(mock, config);
    const go::MctsResult result = search.search(game);

    double prior_total = 0.0;
    for (const go::MctsMove& move : result.moves) {
        require(game.is_legal(go::Stone::black, move.move),
                "every root child must be legal");
        require(move.prior > 0.0F,
                "illegal priors must not leak into the tree");
        if (move.move.has_value()) {
            require(!(move.move->x == 1 && move.move->y == 0),
                    "occupied point must never appear as a root child");
            require(!(move.move->x == 2 && move.move->y == 2),
                    "occupied point must never appear as a root child");
        }
        prior_total += static_cast<double>(move.prior);
    }
    require(std::abs(prior_total - 1.0) < 1e-4,
            "priors must be renormalized on the legal set only");
    require(result.moves.size() < 26, "occupied points are excluded");
}

// ── 3. negamax 符号：一致评估换算到根行棋方视角 ──────────────────────────
void test_value_sign_convention() {
    go::Game game = empty_board(5, go::Stone::black);
    MockEvaluator mock;
    mock.white_advantage = 0.9F;  // 白优 0.9，与行棋方无关
    mock.allow_pass = false;      // 不让对局走到终局，避免终局 ±1 混入

    go::MctsConfig config;
    config.simulations = 150;
    config.seed = 11;
    go::MctsSearch search(mock, config);
    const go::MctsResult result = search.search(game);

    // 根是黑棋行棋：白优 0.9 → 根视角必须为负，且每条根子边同号。
    require(result.root_value < -0.5F && result.root_value > -1.05F,
            "root value must be the black view of a white advantage");
    for (const go::MctsMove& move : result.moves) {
        if (move.visits > 0) {
            require(move.value < 0.0F,
                    "child edge value must be the root player's view");
        }
    }
}

// ── 4. 终局计分：pass-pass 后按 KataGo 目差备份 ──────────────────────────
void test_terminal_scoring() {
    go::Game game = empty_board(5, go::Stone::black);
    // 先验 100% 给 pass：树只会走 pass → pass → 终局（中国规则，黑贴 7.5，
    // 空盘 = 白胜 7.5 → 黑视角 -1）。
    PassOnlyEvaluator pass_only;

    go::MctsConfig config;
    config.simulations = 120;
    config.seed = 13;
    go::MctsSearch search(pass_only, config);
    const go::MctsResult result = search.search(game);

    require(!result.best_move.has_value(), "pass-only policy must choose pass");
    require(result.root_value < -0.5F,
            "terminal value must back up -1 for black");
    require_eq(result.simulations, 120, "simulations are reported");
}

// ── 5. 转置表：同一局面重搜命中缓存 ──────────────────────────────────────
void test_transposition_table() {
    go::Game game = empty_board(5, go::Stone::black);
    MockEvaluator mock;

    go::MctsConfig config;
    config.simulations = 80;
    config.seed = 17;
    go::MctsSearch search(mock, config);

    const go::MctsResult first = search.search(game);
    const std::size_t calls_after_first = mock.calls;
    const go::MctsResult second = search.search(game);

    require(second.tt_hits > 0, "second search must hit the transposition table");
    require(second.nn_evaluations < first.nn_evaluations,
            "cached evaluations must reduce evaluator calls");
    require(mock.calls == calls_after_first + second.nn_evaluations,
            "evaluator calls must equal reported nn_evaluations");
}

}  // namespace

int main() {
    test_prior_driven_choice();
    test_legality_mask();
    test_value_sign_convention();
    test_terminal_scoring();
    test_transposition_table();

    if (failures == 0) {
        std::printf("go_mcts_tests: all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
