#include "go/mcts.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace go {
namespace {

// ── 转置表 key ────────────────────────────────────────────────────────────
// 覆盖"评估器输入的全部信息"：盘面石、行棋方、简单劫点、最近 W 手走子 id。
// 双 64 位独立混合；碰撞只会造成一次评估误复用，不影响合法性（合法性永远
// 由 KataGo 按当前对局历史判定）。
struct PositionKey {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;

    friend bool operator==(const PositionKey&, const PositionKey&) = default;
};

constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;

std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

struct KeyHash {
    std::size_t operator()(const PositionKey& key) const noexcept {
        return static_cast<std::size_t>(splitmix64(key.lo ^ (key.hi << 1)));
    }
};

PositionKey make_key(const Game& game, std::size_t history_window) {
    PositionKey key{splitmix64(kFnvOffset), splitmix64(kFnvOffset ^ 0x5bf03635)};
    const auto mix = [&key](std::uint64_t salt, std::uint64_t value) {
        key.lo = splitmix64(key.lo ^ (value + salt));
        key.hi = splitmix64(key.hi ^ (value * 0x9E3779B97F4A7C15ULL + salt));
    };

    const int width = game.width();
    for (int y = 0; y < game.height(); ++y) {
        for (int x = 0; x < width; ++x) {
            const Stone stone = game.at(Point{x, y});
            if (stone != Stone::empty) {
                mix(static_cast<std::uint64_t>(y) * width + x,
                    static_cast<std::uint64_t>(stone));
            }
        }
    }
    mix(0x10001ULL, static_cast<std::uint64_t>(game.to_move()));

    const std::optional<Point> ko = game.ko_point();
    if (ko.has_value()) {
        mix(0x10002ULL,
            static_cast<std::uint64_t>(ko->y) * width + ko->x);
    }

    if (history_window > 0) {
        const std::vector<PlayedMove> moves = game.history();
        const std::size_t window = std::min(history_window, moves.size());
        for (std::size_t offset = 0; offset < window; ++offset) {
            const PlayedMove& move = moves[moves.size() - window + offset];
            const std::uint64_t token = move.point.has_value()
                ? static_cast<std::uint64_t>(move.point->y) * width + move.point->x
                : static_cast<std::uint64_t>(width) * game.height();
            mix(0x20000ULL + offset, token);
        }
    }
    return key;
}

struct CachedEval {
    std::vector<float> policy;
    double value = 0.0;
};

struct Edge {
    std::optional<Point> move;  // nullopt = pass
    float prior = 0.0F;
    std::uint32_t child = 0;    // 子节点下标；0 = 尚未创建（根永远占 0）
    std::size_t visits = 0;
    double w = 0.0;             // 累计价值（父节点行棋方视角）
};

struct Node {
    std::vector<Edge> edges;
    bool expanded = false;
    bool terminal = false;
    double static_value = 0.0;  // 节点评估（节点行棋方视角）
    std::size_t visits = 0;
    double w = 0.0;             // 节点行棋方视角
};

constexpr std::uint32_t kNoChild = std::numeric_limits<std::uint32_t>::max();

struct PathEntry {
    std::uint32_t node = 0;
    std::size_t edge = 0;
};

// 终局价值：当前行棋方视角的 ±1/0。
double terminal_value(const Game& game) {
    const Score score = game.score();
    if (!score.winner.has_value()) {
        return 0.0;
    }
    return *score.winner == game.to_move() ? 1.0 : -1.0;
}

// C 验证器的静态叶子价值：面积目差 → tanh。
float area_score_value(const Game& game) {
    const Score score = game.score();
    const float margin = game.to_move() == Stone::white
        ? score.white_minus_black
        : -score.white_minus_black;
    return std::tanh(margin / 15.0F);
}

class PolicyLeafEvaluator final : public LeafEvaluator {
public:
    PolicyLeafEvaluator(PolicyPlayer& player, ValueFallback fallback)
        : player_(player), fallback_(fallback) {}

    PolicyEvaluation evaluate(const Game& game) override {
        PolicyEvaluation result = player_.evaluate(game);
        if (!result.value.has_value()) {
            result.value = fallback_ == ValueFallback::area_score
                ? area_score_value(game)
                : 0.0F;
        }
        return result;
    }

private:
    PolicyPlayer& player_;
    ValueFallback fallback_;
};

}  // namespace

std::unique_ptr<LeafEvaluator> make_policy_evaluator(
    PolicyPlayer& player, ValueFallback fallback) {
    return std::make_unique<PolicyLeafEvaluator>(player, fallback);
}

struct MctsSearch::Impl {
    Impl(LeafEvaluator& evaluator_value, MctsConfig config_value)
        : evaluator(evaluator_value),
          config(config_value),
          rng(config_value.seed != 0
                  ? config_value.seed
                  : std::random_device{}()) {
        if (config.simulations == 0) {
            throw std::invalid_argument("MctsConfig.simulations must be >= 1");
        }
    }

    LeafEvaluator& evaluator;
    MctsConfig config;
    std::mt19937_64 rng;

    std::vector<Node> nodes;
    std::unordered_map<PositionKey, CachedEval, KeyHash> tt;
    std::size_t nn_evaluations = 0;
    std::size_t tt_hits = 0;

    CachedEval evaluate_cached(const Game& game) {
        if (config.tt_capacity == 0) {
            PolicyEvaluation eval = evaluator.evaluate(game);
            ++nn_evaluations;
            return CachedEval{std::move(eval.policy), eval.value.value_or(0.0)};
        }

        const PositionKey key = make_key(game, config.tt_history_window);
        if (const auto it = tt.find(key); it != tt.end()) {
            ++tt_hits;
            return it->second;
        }

        PolicyEvaluation eval = evaluator.evaluate(game);
        ++nn_evaluations;
        CachedEval cached{std::move(eval.policy), eval.value.value_or(0.0)};
        if (tt.size() >= config.tt_capacity) {
            tt.clear();  // 容量满即整体淘汰：简单、无锁、命中率损失可忽略
        }
        tt.emplace(key, cached);
        return cached;
    }

    // 展开节点：合法着生成子边 + 先验归一化（+ 根噪声）。
    void expand(std::uint32_t node_index, const Game& game, const CachedEval& eval) {
        Node& node = nodes[node_index];
        const std::vector<std::optional<Point>> legal = game.legal_moves();
        if (legal.empty()) {
            node.terminal = true;
            node.static_value = terminal_value(game);
            return;
        }

        const int width = game.width();
        const std::size_t pass_index =
            static_cast<std::size_t>(width) * game.height();
        std::vector<double> priors;
        priors.reserve(legal.size());
        double total = 0.0;
        for (const std::optional<Point>& move : legal) {
            const std::size_t index = move.has_value()
                ? static_cast<std::size_t>(move->y) * width + move->x
                : pass_index;
            double prior = index < eval.policy.size()
                ? static_cast<double>(eval.policy[index])
                : 0.0;
            prior = prior > 0.0 ? prior : 0.0;
            priors.push_back(prior);
            total += prior;
        }
        if (total > 0.0) {
            for (double& prior : priors) {
                prior /= total;
            }
        } else {
            const double uniform = 1.0 / static_cast<double>(legal.size());
            std::fill(priors.begin(), priors.end(), uniform);
        }

        if (node_index == 0 && config.root_noise) {
            std::gamma_distribution<double> gamma(config.root_noise_alpha, 1.0);
            double noise_total = 0.0;
            std::vector<double> noise;
            noise.reserve(legal.size());
            for (std::size_t i = 0; i < legal.size(); ++i) {
                const double sample = gamma(rng);
                noise.push_back(sample);
                noise_total += sample;
            }
            // alpha=0 时 gamma 全为 0：跳过噪声，避免 0/0 污染先验。
            if (noise_total > 0.0) {
                for (std::size_t i = 0; i < legal.size(); ++i) {
                    priors[i] = (1.0 - config.root_noise_eps) * priors[i] +
                                config.root_noise_eps * noise[i] / noise_total;
                }
            }
        }

        node.edges.reserve(legal.size());
        for (std::size_t i = 0; i < legal.size(); ++i) {
            node.edges.push_back(Edge{
                .move = legal[i],
                .prior = static_cast<float>(priors[i]),
                .child = kNoChild,
                .visits = 0,
                .w = 0.0,
            });
        }
        node.expanded = true;
        node.static_value = eval.value;
    }

    [[nodiscard]] std::size_t select_edge(const Node& node) const {
        const double sqrt_total =
            std::sqrt(static_cast<double>(node.visits) + 1.0);
        const double parent_q = node.visits > 0
            ? node.w / static_cast<double>(node.visits)
            : 0.0;
        const double fpu = parent_q - static_cast<double>(config.fpu_reduction);

        std::size_t best_index = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < node.edges.size(); ++i) {
            const Edge& edge = node.edges[i];
            const double q = edge.visits > 0
                ? edge.w / static_cast<double>(edge.visits)
                : fpu;
            const double u = static_cast<double>(config.c_puct) *
                             static_cast<double>(edge.prior) * sqrt_total /
                             (1.0 + static_cast<double>(edge.visits));
            const double score = q + u;
            if (score > best_score) {
                best_score = score;
                best_index = i;
            }
        }
        return best_index;
    }

    void backup(
        std::uint32_t leaf, const std::vector<PathEntry>& path, double value) {
        Node& leaf_node = nodes[leaf];
        leaf_node.visits += 1;
        leaf_node.w += value;

        double v = value;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            Node& parent = nodes[it->node];
            Edge& edge = parent.edges[it->edge];
            v = -v;  // 子节点行棋方视角 → 父节点行棋方视角
            edge.visits += 1;
            edge.w += v;
            parent.visits += 1;
            parent.w += v;
        }
    }
};

MctsSearch::MctsSearch(LeafEvaluator& evaluator, MctsConfig config)
    : impl_(std::make_unique<Impl>(evaluator, config)) {}

MctsSearch::MctsSearch(MctsSearch&&) noexcept = default;
MctsSearch& MctsSearch::operator=(MctsSearch&&) noexcept = default;
MctsSearch::~MctsSearch() = default;

MctsResult MctsSearch::search(const Game& root) {
    Impl& impl = *impl_;
    impl.nodes.clear();
    impl.nn_evaluations = 0;  // 计数按本次搜索口径；转置表跨搜索保留
    impl.tt_hits = 0;
    impl.nodes.emplace_back();  // 根节点占下标 0

    // 根终局：直接给终局值，best_move = pass。
    if (root.game_over()) {
        MctsResult result;
        result.best_move = std::nullopt;
        result.root_value = static_cast<float>(terminal_value(root));
        return result;
    }

    std::vector<PathEntry> path;
    for (std::size_t sim = 0; sim < impl.config.simulations; ++sim) {
        // 每次模拟从根重放走子序列：节点不存盘面，内存只随树增长。
        Game game = root;
        path.clear();
        std::uint32_t node_index = 0;

        while (true) {
            const Node& node = impl.nodes[node_index];
            if (node.terminal || !node.expanded || node.edges.empty()) {
                break;
            }
            const std::size_t edge_index = impl.select_edge(node);
            const Edge edge = impl.nodes[node_index].edges[edge_index];
            if (!game.play(game.to_move(), edge.move)) {
                throw std::logic_error("MCTS selected an illegal move");
            }
            std::uint32_t child = edge.child;
            if (child == kNoChild) {
                child = static_cast<std::uint32_t>(impl.nodes.size());
                impl.nodes[node_index].edges[edge_index].child = child;
                impl.nodes.emplace_back();
            }
            path.push_back(PathEntry{node_index, edge_index});
            node_index = child;
        }

        Node& leaf = impl.nodes[node_index];
        if (leaf.terminal) {
            impl.backup(node_index, path, leaf.static_value);
            continue;
        }
        if (!leaf.expanded) {
            if (game.game_over()) {
                leaf.terminal = true;
                leaf.static_value = terminal_value(game);
            } else {
                const CachedEval eval = impl.evaluate_cached(game);
                leaf.static_value = eval.value;
                impl.expand(node_index, game, eval);
            }
            impl.backup(node_index, path, leaf.static_value);
        }
    }

    const Node& root_node = impl.nodes[0];
    MctsResult result;
    result.simulations = impl.config.simulations;
    result.nn_evaluations = impl.nn_evaluations;
    result.tt_hits = impl.tt_hits;
    result.root_value = root_node.visits > 0
        ? static_cast<float>(root_node.w / static_cast<double>(root_node.visits))
        : 0.0F;

    for (const Edge& edge : root_node.edges) {
        result.moves.push_back(MctsMove{
            .move = edge.move,
            .visits = edge.visits,
            .prior = edge.prior,
            .value = edge.visits > 0
                ? static_cast<float>(edge.w / static_cast<double>(edge.visits))
                : 0.0F,
        });
    }
    std::sort(result.moves.begin(), result.moves.end(),
              [](const MctsMove& a, const MctsMove& b) {
                  return a.visits > b.visits;
              });

    if (result.moves.empty()) {
        result.best_move = std::nullopt;
        return result;
    }

    if (impl.config.temperature > 0.0F && root_node.visits > 0) {
        const double exponent = 1.0 / static_cast<double>(impl.config.temperature);
        double total = 0.0;
        std::vector<double> weights;
        weights.reserve(result.moves.size());
        for (const MctsMove& move : result.moves) {
            const double weight =
                std::pow(static_cast<double>(move.visits), exponent);
            weights.push_back(weight);
            total += weight;
        }
        std::uniform_real_distribution<double> uniform(0.0, total);
        double draw = uniform(impl.rng);
        for (std::size_t i = 0; i < result.moves.size(); ++i) {
            draw -= weights[i];
            if (draw <= 0.0) {
                result.best_move = result.moves[i].move;
                return result;
            }
        }
        result.best_move = result.moves.front().move;
        return result;
    }

    // visits 全 0（只可能发生在根无合法着）→ 按先验兜底。
    if (result.moves.front().visits == 0) {
        const auto best = std::max_element(
            result.moves.begin(), result.moves.end(),
            [](const MctsMove& a, const MctsMove& b) {
                return a.prior < b.prior;
            });
        result.best_move = best->move;
        return result;
    }
    result.best_move = result.moves.front().move;
    return result;
}

}  // namespace go
