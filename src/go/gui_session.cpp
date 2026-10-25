#include "go/gui_session.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace go::gui {
namespace {

using json = nlohmann::json;

constexpr int kBoardSize = 19;      // M3 v1 特征契约：模型专用 19×19
constexpr int kMaxMoves = 4 * 19 * 19;  // 防御性上限（与 go_arena_cli 同口径）
constexpr std::size_t kPassIndex = 361;  // 策略类 361 = pass

std::string stone_name(Stone color) {
    return color == Stone::black ? "black" : "white";
}

// 小数格式化：komi / 目数只在需要时保留小数（7.5、3、0.5）。
std::string format_float(float value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

// SGF 属性值转义（只有 \ 与 ] 需要转义）。
std::string sgf_escape(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    for (const char c : text) {
        if (c == '\\' || c == ']') {
            result.push_back('\\');
        }
        result.push_back(c);
    }
    return result;
}

// SGF 坐标：'a'+x, 'a'+y；pass 为空括号。
std::string sgf_point(const Point& point) {
    std::string result;
    result.push_back(static_cast<char>('a' + point.x));
    result.push_back(static_cast<char>('a' + point.y));
    return result;
}

// 重放 N 手之后的行棋方（黑先，交替）。
Stone mover_after(std::size_t moves) {
    return moves % 2 == 0 ? Stone::black : Stone::white;
}

}  // namespace

GuiSession::GuiSession(SessionConfig config)
    : config_(std::move(config)),
      komi_(config_.komi),
      pending_komi_(config_.komi),
      speed_ms_(config_.speed_ms),
      auto_next_(config_.auto_next) {
    game_ = Game(kBoardSize, kBoardSize, Stone::black, RuleSet::chinese(komi_));
    play_engine_ = make_engine(config_.engine, config_.seed);
    black_engine_ = make_engine(config_.black_engine, config_.seed + 1);
    white_engine_ = make_engine(config_.white_engine, config_.seed + 2);
    if (mode_ == Mode::play) {
        play_engine_->clear_board(kBoardSize, komi_);
    }
    worker_ = std::thread([this] { worker_loop(); });
}

GuiSession::~GuiSession() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_worker_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

// ── 内部工具（都需持有 mutex_）──────────────────────────────────────────

std::string GuiSession::board_string_locked() const {
    std::string board;
    board.reserve(static_cast<std::size_t>(game_.width() * game_.height()));
    for (int y = 0; y < game_.height(); ++y) {
        for (int x = 0; x < game_.width(); ++x) {
            switch (game_.at(Point{x, y})) {
                case Stone::black: board.push_back('B'); break;
                case Stone::white: board.push_back('W'); break;
                default: board.push_back('.'); break;
            }
        }
    }
    return board;
}

ArenaEngine* GuiSession::side_engine_locked(Stone color) const {
    return color == Stone::black ? black_engine_.get() : white_engine_.get();
}

void GuiSession::reset_game_locked(float komi) {
    komi_ = komi;
    game_ = Game(kBoardSize, kBoardSize, Stone::black, RuleSet::chinese(komi_));
    game_over_ = false;
    result_ = GameResult{};
    consecutive_passes_ = 0;
    // 引擎的内部盘面同步（外部 GTP 引擎尤其需要）。
    if (mode_ == Mode::play) {
        play_engine_->clear_board(kBoardSize, komi_);
    } else {
        black_engine_->clear_board(kBoardSize, komi_);
        white_engine_->clear_board(kBoardSize, komi_);
    }
}

void GuiSession::replay_locked(const std::vector<PlayedMove>& moves) {
    game_ = Game(kBoardSize, kBoardSize, Stone::black, RuleSet::chinese(komi_));
    consecutive_passes_ = 0;
    for (const PlayedMove& move : moves) {
        if (!game_.play(move.color, move.point)) {
            // 历史来自本对局的合法着，重放失败只可能是实现错误；保守截断。
            break;
        }
        consecutive_passes_ = move.point.has_value() ? 0 : consecutive_passes_ + 1;
    }
    game_over_ = false;
    result_ = GameResult{};
    // 引擎盘面重置后重放（GTP 引擎需要逐步 play 同步）。
    if (mode_ == Mode::play) {
        play_engine_->clear_board(kBoardSize, komi_);
        for (const PlayedMove& move : moves) {
            play_engine_->notify_move(move.color, move.point);
        }
    }
}

bool GuiSession::play_locked(Stone color, const std::optional<Point>& point) {
    if (!game_.play(color, point)) {
        return false;
    }
    consecutive_passes_ = point.has_value() ? 0 : consecutive_passes_ + 1;
    return true;
}

void GuiSession::finish_by_score_locked() {
    const Score score = game_.score();
    game_over_ = true;
    result_ = GameResult{};
    result_.winner = score.winner;
    result_.margin = std::fabs(score.white_minus_black);
    result_.reason = "two-passes";
    if (mode_ == Mode::arena) {
        // 记在"当前执该色的引擎"名下；开下一局换先时随席位一起交换。
        if (!score.winner.has_value()) {
            ++draws_;
        } else if (*score.winner == Stone::black) {
            ++wins_black_;
        } else {
            ++wins_white_;
        }
        // 终局画面停留约 3 手时间，再自动开下一局。
        next_game_at_ = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(3 * std::max(50, speed_ms_));
    }
}

void GuiSession::start_next_arena_game_locked() {
    std::swap(black_engine_, white_engine_);  // 换先
    // 胜局跟着引擎走：换边的同时把比分换到新席位。黑/白只是当前席位，
    // 引擎才是身份——否则换先后名字换了、分数留在原地，记分板就张冠李戴。
    std::swap(wins_black_, wins_white_);
    ++game_index_;
    reset_game_locked(pending_komi_);
}

void GuiSession::arena_play_one_locked() {
    if (game_over_ || game_.move_count() >= kMaxMoves) {
        if (!game_over_) {
            finish_by_score_locked();
        }
        return;
    }
    const Stone mover = game_.to_move();
    ArenaEngine* engine = side_engine_locked(mover);
    ArenaEngine* spectator =
        side_engine_locked(mover == Stone::black ? Stone::white : Stone::black);
    try {
        const std::optional<Point> move = engine->genmove(game_);
        if (!play_locked(mover, move)) {
            throw std::runtime_error(engine->name() + " produced an illegal move");
        }
        spectator->notify_move(mover, move);
    } catch (const std::exception& error) {
        // 引擎故障：暂停竞技台并留日志，局面保持可查。
        running_ = false;
        std::cerr << "[go.gui] arena engine failed: " << error.what() << '\n';
        return;
    }
    if (consecutive_passes_ >= 2) {
        finish_by_score_locked();
    }
}

// ── 状态输出 ────────────────────────────────────────────────────────────

nlohmann::json GuiSession::state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_locked();
}

nlohmann::json GuiSession::state_locked() const {
    json state;
    state["ok"] = true;
    state["mode"] = mode_ == Mode::play ? "play" : "arena";
    state["size"] = game_.width();
    state["komi"] = komi_;
    state["to_move"] =
        game_over_ ? json(nullptr) : json(stone_name(game_.to_move()));
    state["move_number"] = game_.move_count();
    state["board"] = board_string_locked();

    const std::vector<PlayedMove> history = game_.history();
    if (history.empty() || !history.back().point.has_value()) {
        state["last_move"] = nullptr;
    } else {
        state["last_move"] = {
            {"x", history.back().point->x}, {"y", history.back().point->y}};
    }
    if (const std::optional<Point> ko = game_.ko_point(); ko.has_value()) {
        state["ko"] = {{"x", ko->x}, {"y", ko->y}};
    } else {
        state["ko"] = nullptr;
    }
    // captured_count(黑) = 被提掉的黑子数 = 白方提子数，界面按"谁提的"显示。
    state["captures"] = {
        {"black", game_.captured_count(Stone::white)},
        {"white", game_.captured_count(Stone::black)}};

    json moves = json::array();
    int number = 1;
    for (const PlayedMove& move : history) {
        json entry;
        entry["n"] = number++;
        entry["color"] = stone_name(move.color);
        if (move.point.has_value()) {
            entry["x"] = move.point->x;
            entry["y"] = move.point->y;
        } else {
            entry["pass"] = true;
        }
        moves.push_back(entry);
    }
    state["history"] = moves;
    state["game_over"] = game_over_;
    if (game_over_) {
        json result;
        result["winner"] = result_.winner.has_value()
            ? json(stone_name(*result_.winner)) : json("draw");
        result["margin"] = result_.margin;
        result["reason"] = result_.reason;
        state["result"] = result;
    } else {
        state["result"] = nullptr;
    }

    const bool human_turn = !game_over_ && mode_ == Mode::play &&
        (!human_color_.has_value() || game_.to_move() == *human_color_);
    state["clickable"] = human_turn;

    if (mode_ == Mode::play) {
        json play;
        play["human_color"] = human_color_.has_value()
            ? json(stone_name(*human_color_)) : json("none");
        play["engine"] = play_engine_->name();
        play["engine_turn"] = !game_over_ && human_color_.has_value() &&
            game_.to_move() != *human_color_;
        play["busy"] = false;
        state["play"] = play;
    } else {
        json arena;
        arena["running"] = running_;
        arena["speed_ms"] = speed_ms_;
        arena["auto_next"] = auto_next_;
        arena["game_index"] = game_index_;  // 1 基：当前是第几局
        arena["black_engine"] = black_engine_->name();
        arena["white_engine"] = white_engine_->name();
        arena["black_name"] = engine_display_name(black_engine_->name());
        arena["white_name"] = engine_display_name(white_engine_->name());
        arena["wins"] = {
            {"black", wins_black_}, {"white", wins_white_}, {"draw", draws_}};
        arena["games_played"] = wins_black_ + wins_white_ + draws_;
        state["arena"] = arena;
    }
    return state;
}

nlohmann::json GuiSession::ok_locked() const {
    return state_locked();
}

nlohmann::json GuiSession::error(const std::string& message, int status) const {
    return json{{"ok", false}, {"error", message}, {"status", status}};
}

nlohmann::json GuiSession::evaluate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    ArenaEngine* engine = mode_ == Mode::play
        ? play_engine_.get()
        : side_engine_locked(game_.to_move());
    if (engine == nullptr) {
        throw std::runtime_error("没有可用引擎");
    }
    const std::optional<PolicyEvaluation> evaluation = engine->evaluate(game_);
    if (!evaluation.has_value()) {
        // 不该走到：所有引擎都有"将下点 100%"兜底，只有连预告都给不出才到这里。
        throw std::runtime_error("该引擎无法给出评估或下一手预告");
    }

    struct Candidate {
        std::size_t index;
        float probability;
    };
    std::vector<Candidate> candidates;
    for (std::size_t index = 0; index < evaluation->policy.size(); ++index) {
        if (evaluation->policy[index] > 0.0F) {
            candidates.push_back({index, evaluation->policy[index]});
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& left, const Candidate& right) {
                  return left.probability > right.probability;
              });

    json top = json::array();
    constexpr std::size_t kTopCount = 10;
    for (std::size_t i = 0; i < std::min(kTopCount, candidates.size()); ++i) {
        json entry;
        if (candidates[i].index == kPassIndex) {
            entry["pass"] = true;
        } else {
            entry["x"] = static_cast<int>(candidates[i].index % kBoardSize);
            entry["y"] = static_cast<int>(candidates[i].index / kBoardSize);
        }
        entry["p"] = candidates[i].probability;
        top.push_back(entry);
    }

    json result;
    result["ok"] = true;
    result["to_move"] = stone_name(game_.to_move());
    result["engine"] = engine->name();
    // synthetic = true：单点 100% 的"将下点预告"，不是模型分布。
    result["synthetic"] = evaluation->synthetic;
    result["value"] = evaluation->value.has_value()
        ? json(*evaluation->value) : json(nullptr);
    result["top"] = top;
    return result;
}

std::string GuiSession::sgf() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream out;
    out << "(;GM[1]FF[4]CA[UTF-8]AP[go.neuralnet.cpp]SZ[" << game_.width()
        << "]KM[" << format_float(komi_) << "]RU[Chinese]";

    const std::string engine_name =
        engine_display_name(play_engine_ ? play_engine_->name() : "random");
    std::string black_name = "黑方";
    std::string white_name = "白方";
    if (mode_ == Mode::play) {
        if (!human_color_.has_value()) {
            // 自由摆棋：双方都记为执棋方。
        } else if (*human_color_ == Stone::black) {
            black_name = "人类";
            white_name = engine_name;
        } else {
            black_name = engine_name;
            white_name = "人类";
        }
    } else {
        black_name = engine_display_name(black_engine_->name());
        white_name = engine_display_name(white_engine_->name());
    }
    out << "PB[" << sgf_escape(black_name) << "]PW[" << sgf_escape(white_name) << "]";

    if (game_over_) {
        out << "RE[";
        if (!result_.winner.has_value()) {
            out << "0";
        } else {
            out << (*result_.winner == Stone::black ? "B" : "W");
            out << (result_.reason == "resign" ? "+R" : "+" + format_float(result_.margin));
        }
        out << "]";
    }
    for (const PlayedMove& move : game_.history()) {
        out << ";" << (move.color == Stone::black ? "B" : "W") << "[";
        if (move.point.has_value()) {
            out << sgf_point(*move.point);
        }
        out << "]";
    }
    out << ")\n";
    return out.str();
}

// ── 对弈动作 ────────────────────────────────────────────────────────────

nlohmann::json GuiSession::set_mode(const std::string& mode) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode == "play") {
        mode_ = Mode::play;
        human_color_ = Stone::black;
        reset_game_locked(pending_komi_);
    } else if (mode == "arena") {
        mode_ = Mode::arena;
        running_ = false;
        wins_black_ = wins_white_ = draws_ = 0;
        game_index_ = 1;
        reset_game_locked(pending_komi_);
    } else {
        return error("mode 必须是 play 或 arena", 400);
    }
    return ok_locked();
}

nlohmann::json GuiSession::human_move(int x, int y) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::play) {
        return error("竞技台模式不接受人类落子", 409);
    }
    if (game_over_) {
        return error("对局已结束", 409);
    }
    if (human_color_.has_value() && game_.to_move() != *human_color_) {
        return error("现在轮到引擎走棋", 409);
    }
    if (x < 0 || x >= game_.width() || y < 0 || y >= game_.height()) {
        return error("坐标越界", 400);
    }
    const Stone color = game_.to_move();
    const Point point{x, y};
    if (!game_.is_legal(color, point)) {
        return error("非法落点", 409);
    }
    if (!play_locked(color, point)) {
        return error("落子失败", 409);
    }
    play_engine_->notify_move(color, point);
    return ok_locked();
}

nlohmann::json GuiSession::human_pass() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::play) {
        return error("竞技台模式不接受人类落子", 409);
    }
    if (game_over_) {
        return error("对局已结束", 409);
    }
    if (human_color_.has_value() && game_.to_move() != *human_color_) {
        return error("现在轮到引擎走棋", 409);
    }
    const Stone color = game_.to_move();
    if (!play_locked(color, std::nullopt)) {
        return error("停一手失败", 409);
    }
    play_engine_->notify_move(color, std::nullopt);
    if (consecutive_passes_ >= 2) {
        finish_by_score_locked();
    }
    return ok_locked();
}

nlohmann::json GuiSession::engine_move() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::play) {
        return error("竞技台模式由后台线程走棋", 409);
    }
    if (game_over_) {
        return error("对局已结束", 409);
    }
    if (!human_color_.has_value()) {
        return error("自由摆棋模式没有引擎手", 409);
    }
    if (game_.to_move() == *human_color_) {
        return error("现在轮到你走棋", 409);
    }
    const Stone color = game_.to_move();
    try {
        const std::optional<Point> move = play_engine_->genmove(game_);
        if (!play_locked(color, move)) {
            return error("引擎产出非法着", 500);
        }
    } catch (const std::exception& exception) {
        return error(std::string("引擎走棋失败：") + exception.what(), 500);
    }
    if (consecutive_passes_ >= 2) {
        finish_by_score_locked();
    }
    return ok_locked();
}

nlohmann::json GuiSession::undo() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::play) {
        return error("竞技台模式不支持悔棋", 409);
    }
    std::vector<PlayedMove> kept = game_.history();
    if (kept.empty()) {
        return error("已经没有可退的棋", 409);
    }
    kept.pop_back();
    if (human_color_.has_value()) {
        // 退回到轮到人类：引擎已应手时连退两手。
        while (!kept.empty() && mover_after(kept.size()) != *human_color_) {
            kept.pop_back();
        }
    }
    replay_locked(kept);
    return ok_locked();
}

nlohmann::json GuiSession::resign() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::play) {
        return error("竞技台模式不支持认输", 409);
    }
    if (game_over_) {
        return error("对局已结束", 409);
    }
    if (!human_color_.has_value()) {
        return error("自由摆棋模式没有胜负", 409);
    }
    game_over_ = true;
    result_ = GameResult{};
    result_.winner = *human_color_ == Stone::black ? Stone::white : Stone::black;
    result_.margin = 0.0F;
    result_.reason = "resign";
    return ok_locked();
}

nlohmann::json GuiSession::new_game(const std::string& human_color, float komi) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::play) {
        return error("竞技台模式请用 arena 接口开新局", 409);
    }
    if (human_color == "black") {
        human_color_ = Stone::black;
    } else if (human_color == "white") {
        human_color_ = Stone::white;
    } else if (human_color == "none") {
        human_color_ = std::nullopt;
    } else {
        return error("human_color 必须是 black / white / none", 400);
    }
    if (komi > 0.0F) {
        pending_komi_ = komi;
    }
    reset_game_locked(pending_komi_);
    return ok_locked();
}

nlohmann::json GuiSession::configure(int speed_ms, bool auto_next, float komi) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (speed_ms > 0) {
        speed_ms_ = std::clamp(speed_ms, 50, 10000);
    }
    auto_next_ = auto_next;
    if (komi > 0.0F) {
        pending_komi_ = komi;
    }
    cv_.notify_all();
    return ok_locked();
}

// ── 竞技台动作 ──────────────────────────────────────────────────────────

nlohmann::json GuiSession::arena_start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::arena) {
        return error("当前不在竞技台模式", 409);
    }
    running_ = true;
    cv_.notify_all();
    return ok_locked();
}

nlohmann::json GuiSession::arena_pause() {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    return ok_locked();
}

nlohmann::json GuiSession::arena_step() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::arena) {
        return error("当前不在竞技台模式", 409);
    }
    if (game_over_) {
        if (!auto_next_) {
            return error("本局已结束", 409);
        }
        start_next_arena_game_locked();
        return ok_locked();
    }
    arena_play_one_locked();
    return ok_locked();
}

nlohmann::json GuiSession::arena_next() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::arena) {
        return error("当前不在竞技台模式", 409);
    }
    start_next_arena_game_locked();
    return ok_locked();
}

nlohmann::json GuiSession::arena_reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != Mode::arena) {
        return error("当前不在竞技台模式", 409);
    }
    wins_black_ = wins_white_ = draws_ = 0;
    game_index_ = 1;
    reset_game_locked(pending_komi_);
    return ok_locked();
}

// ── 竞技台后台线程 ──────────────────────────────────────────────────────

void GuiSession::worker_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_worker_) {
        if (mode_ != Mode::arena || !running_) {
            cv_.wait_for(lock, std::chrono::milliseconds(100));
            continue;
        }
        if (game_over_) {
            if (auto_next_ && std::chrono::steady_clock::now() >= next_game_at_) {
                start_next_arena_game_locked();
            }
            cv_.wait_for(lock, std::chrono::milliseconds(50));
            continue;
        }
        arena_play_one_locked();
        cv_.wait_for(lock, std::chrono::milliseconds(std::max(50, speed_ms_)));
    }
}

}  // namespace go::gui
