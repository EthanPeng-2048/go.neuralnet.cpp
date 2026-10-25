#pragma once

// ── GUI 会话状态机（go_gui_cli 的后端）────────────────────────────────────
//
// 一个 GuiSession 管一盘棋 + 一个竞技台对局序列，被 HTTP 层（go_gui_cli）
// 与单测共同驱动。规则一律中国规则（go::RuleSet::chinese：位置超级劫 +
// 面积/数子计分 + 禁止自杀，默认贴 7.5 目）——与挑战对手约定的口径一致。
//
// 两种模式：
//   * play  —— 人机对弈：人类落子 → 引擎应手；也支持"自由摆棋"
//               （human_color = none，双方都由人下）。
//   * arena —— 引擎自动对局（竞技台观战）：后台线程按 speed_ms 节奏落子，
//               终局后自动换先开下一局并累计比分。
//
// 线程模型：一把互斥锁保护全部状态；竞技台的后台线程、HTTP 工作线程都
// 在锁内读写（引擎前向也在锁内执行——模型推理是毫秒级，简单正确优先）。

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "external/nlohmann_json/json.hpp"

#include "go/arena_engine.hpp"
#include "go/game.hpp"

namespace go::gui {

struct SessionConfig {
    std::string engine = "random";        // play 模式对手引擎 spec
    std::string black_engine = "random";  // 竞技台第 1 局执黑的引擎 spec
    std::string white_engine = "random";  // 竞技台第 1 局执白的引擎 spec
    float komi = 7.5F;                    // 中国规则默认贴 7.5 目
    int speed_ms = 1200;                  // 竞技台每手间隔
    bool auto_next = true;                // 终局后自动开下一局
    std::uint64_t seed = 20261024;
};

class GuiSession {
public:
    explicit GuiSession(SessionConfig config);
    ~GuiSession();

    GuiSession(const GuiSession&) = delete;
    GuiSession& operator=(const GuiSession&) = delete;

    // ── 只读 ────────────────────────────────────────────────────────────
    // 完整局面快照（HTTP GET /api/state 的响应体）。
    [[nodiscard]] nlohmann::json state() const;
    // 候选着分布 + 价值头（GET /api/evaluate）。模型引擎给真实 softmax；
    // 不支持评估的引擎给"将下点 100%"的合成分布（synthetic = true）。
    [[nodiscard]] nlohmann::json evaluate() const;
    // 当前对局的 SGF（GET /api/sgf）。
    [[nodiscard]] std::string sgf() const;

    // ── 动作 ────────────────────────────────────────────────────────────
    // 都返回 {"ok":true, ...state} 或 {"ok":false,"error":...}（失败时附
    // "status" 给 HTTP 层做状态码）。
    nlohmann::json set_mode(const std::string& mode);
    nlohmann::json human_move(int x, int y);
    nlohmann::json human_pass();
    nlohmann::json engine_move();
    nlohmann::json undo();
    nlohmann::json resign();
    // human_color: "black" | "white" | "none"（自由摆棋）；komi <= 0 表示不变。
    nlohmann::json new_game(const std::string& human_color, float komi);
    nlohmann::json configure(int speed_ms, bool auto_next, float komi);
    nlohmann::json arena_start();
    nlohmann::json arena_pause();
    nlohmann::json arena_step();
    nlohmann::json arena_next();
    nlohmann::json arena_reset();

private:
    enum class Mode { play, arena };
    struct GameResult {
        std::optional<Stone> winner;  // nullopt = 和棋（中国规则 7.5 目不会出现）
        float margin = 0.0F;
        std::string reason;           // "two-passes" | "resign"
    };

    // 全部 *_locked 假定已持有 mutex_。
    [[nodiscard]] nlohmann::json state_locked() const;
    [[nodiscard]] nlohmann::json ok_locked() const;
    [[nodiscard]] nlohmann::json error(const std::string& message, int status) const;
    void reset_game_locked(float komi);
    void replay_locked(const std::vector<PlayedMove>& moves);
    bool play_locked(Stone color, const std::optional<Point>& point);
    void finish_by_score_locked();
    void start_next_arena_game_locked();
    void arena_play_one_locked();
    [[nodiscard]] std::string board_string_locked() const;
    [[nodiscard]] ArenaEngine* side_engine_locked(Stone color) const;

    void worker_loop();

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    bool stop_worker_ = false;

    SessionConfig config_;
    Mode mode_ = Mode::play;
    Game game_;
    float komi_ = 7.5F;
    float pending_komi_ = 7.5F;
    bool game_over_ = false;
    GameResult result_;
    int consecutive_passes_ = 0;

    // play 模式
    std::optional<Stone> human_color_ = Stone::black;  // nullopt = 自由摆棋
    std::unique_ptr<ArenaEngine> play_engine_;

    // arena 模式
    std::unique_ptr<ArenaEngine> black_engine_;
    std::unique_ptr<ArenaEngine> white_engine_;
    bool running_ = false;
    int speed_ms_ = 1200;
    bool auto_next_ = true;
    int game_index_ = 1;
    // 胜局记在当前执黑/执白的**引擎**名下（席位，不是引擎身份）：每次换先
    // 与引擎一起 swap，界面上"引擎名 + 胜局"永远对应同一引擎。
    int wins_black_ = 0;
    int wins_white_ = 0;
    int draws_ = 0;
    std::chrono::steady_clock::time_point next_game_at_{};
};

}  // namespace go::gui
