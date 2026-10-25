// go_arena_cli — 自动对局评估（M6）：让策略模型与一个对手引擎批量对局，
// 用 SPRT（sequential probability ratio test）给出"更强/更弱/不确定"的
// 统计判定。roadmap M6 DoD「vs GnuGo 自动评估」由 `--opponent gtp:...`
// 满足（GnuGo 未安装时由测试层跳过，与 M1 的 crosscheck 同款语义）。
//
// 引擎种类（--player / --opponent）：
//   * random  —— 均匀随机合法着（不主动 pass；无子可下才 pass）。用作
//                 冒烟对手与下限基线，不需要模型文件。
//   * model:FILE[=DEVICE] —— 本项目的策略模型（go::PolicyPlayer）。
//   * gtp:COMMAND —— 外部 GTP 引擎子进程（如 `gnugo --mode gtp --quiet`），
//                 通过管道说 GTP；只依赖 GTP v2 的
//                 boardsize/clear_board/komi/genmove/quit。
//
// 判定口径：每局胜者由 KataGo 面积计分（中国规则）+ komi 决定；SPRT 的
// 观测是 player 的每局结果（胜=1 负=0 和=0.5，和局不计入二项似然），
// H0: p = sprt-p0，H1: p = sprt-p1，LLR 按二项分布似然比累计，越过
// ±log((1-β)/α) 边界即停（或达到 --games）。输出 JSON 报告；逐局进度走
// stderr。

#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "external/nlohmann_json/json.hpp"
#include "go/game.hpp"
#include "go/policy.hpp"

#if defined(_WIN32)
#include <fcntl.h>   // _O_TEXT（_fdopen 的翻译模式）
#include <io.h>
#include <process.h>
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using json = nlohmann::json;

// ── SPRT（二项似然比）────────────────────────────────────────────────────
// LLR = w·ln(p₁/p₀) + l·ln((1-p₁)/(1-p₀))；边界取标准 α/β 错误率。
struct Sprt {
    double p0 = 0.5;   // H0 胜率（无优势假设）
    double p1 = 0.55;  // H1 胜率（有意义优势假设）
    double alpha = 0.05;
    double beta = 0.05;
    int wins = 0;
    int losses = 0;
    int jigos = 0;

    void observe(double result) {
        if (result == 0.5) {
            ++jigos;
        } else if (result > 0.5) {
            ++wins;
        } else {
            ++losses;
        }
    }

    [[nodiscard]] double llr() const noexcept {
        if (!(p0 > 0 && p0 < p1 && p1 < 1)) {
            return 0.0;
        }
        const double w = static_cast<double>(wins);
        const double l = static_cast<double>(losses);
        return w * std::log(p1 / p0) + l * std::log((1.0 - p1) / (1.0 - p0));
    }

    [[nodiscard]] double lower_bound() const noexcept {
        return std::log(beta / (1.0 - alpha));
    }

    [[nodiscard]] double upper_bound() const noexcept {
        return std::log((1.0 - beta) / alpha);
    }

    // "continue" / "accept"（player 显著更强）/ "reject"（证据不足以判定更强）
    [[nodiscard]] std::string verdict() const noexcept {
        const double value = llr();
        if (value >= upper_bound()) {
            return "accept";
        }
        if (value <= lower_bound()) {
            return "reject";
        }
        return "continue";
    }

    [[nodiscard]] bool decided() const noexcept { return verdict() != "continue"; }
};

// ── 引擎抽象 ────────────────────────────────────────────────────────────
// genmove 返回 nullopt = pass；notify_move 把对手刚下的棋同步给引擎
// （进程内引擎靠 go::Game 自行追踪，外部 GTP 引擎必须收到 play 指令）。
class ArenaEngine {
public:
    virtual ~ArenaEngine() = default;
    virtual std::optional<go::Point> genmove(go::Game& game) = 0;
    virtual void clear_board(int size, float komi) = 0;
    virtual void notify_move(go::Stone, const std::optional<go::Point>&) {}
    [[nodiscard]] virtual std::string name() const = 0;
};

class RandomEngine final : public ArenaEngine {
public:
    explicit RandomEngine(std::uint64_t seed) : rng_(seed) {}

    std::optional<go::Point> genmove(go::Game& game) override {
        const std::vector<std::optional<go::Point>> moves = game.legal_moves();
        std::vector<go::Point> stones;
        stones.reserve(moves.size());
        for (const auto& move : moves) {
            if (move.has_value()) {
                stones.push_back(*move);
            }
        }
        if (stones.empty()) {
            return std::nullopt;  // 无子可下 → pass
        }
        std::uniform_int_distribution<std::size_t> pick(0, stones.size() - 1);
        return stones[pick(rng_)];
    }

    void clear_board(int, float) override {}

    [[nodiscard]] std::string name() const override { return "random"; }

private:
    std::mt19937_64 rng_;
};

class ModelEngine final : public ArenaEngine {
public:
    ModelEngine(std::string path, std::string device)
        : path_(std::move(path)), device_(std::move(device)) {}

    std::optional<go::Point> genmove(go::Game& game) override {
        if (!player_) {
            player_ = go::PolicyPlayer::load(path_, device_);
        }
        return player_->select_move(game);
    }

    void clear_board(int, float) override {}

    [[nodiscard]] std::string name() const override { return "model:" + path_; }

private:
    std::string path_;
    std::string device_;
    std::optional<go::PolicyPlayer> player_;
};

// ── 外部 GTP 引擎子进程 ────────────────────────────────────────────────
// 最小 GTP v2 客户端：写命令行，读到空行为止。
class GtpProcessEngine final : public ArenaEngine {
public:
    explicit GtpProcessEngine(std::string command) : command_(std::move(command)) {}

    ~GtpProcessEngine() override {
        if (spawned_) {
            try {
                send("quit");
            } catch (const std::exception&) {
                // 子进程可能已退出；析构不抛。
            }
        }
        close_pipes();
        reap();
    }

    std::optional<go::Point> genmove(go::Game& game) override {
        const std::string color = game.to_move() == go::Stone::black ? "b" : "w";
        const std::string response = send("genmove " + color);
        const std::string vertex = first_word(response);
        if (vertex.empty() || vertex == "pass") {
            return std::nullopt;
        }
        return parse_vertex(vertex, game.width());
    }

    void notify_move(go::Stone color, const std::optional<go::Point>& point) override {
        const std::string vertex = point.has_value()
            ? point_to_gtp(*point, last_size_)
            : "pass";
        const std::string text = color == go::Stone::black ? "b" : "w";
        send("play " + text + " " + vertex);
    }

    void clear_board(int size, float komi) override {
        last_size_ = size;
        send("boardsize " + std::to_string(size));
        send("clear_board");
        send("komi " + std::to_string(komi));
        spawned_ = true;
    }

    [[nodiscard]] std::string name() const override { return "gtp:" + command_; }

private:
    static std::string first_word(const std::string& text) {
        const std::size_t start = text.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            return {};
        }
        const std::size_t end = text.find_first_of(" \t\r\n", start);
        return text.substr(start, end == std::string::npos ? end : end - start);
    }

    static std::string point_to_gtp(const go::Point& point, int size) {
        std::string result;
        char column = static_cast<char>('A' + point.x);
        if (column >= 'I') {
            column = static_cast<char>(column + 1);  // 跳过 I
        }
        result.push_back(column);
        result += std::to_string(size - point.y);
        return result;
    }

    static go::Point parse_vertex(const std::string& vertex, int size) {
        if (vertex.size() < 2) {
            throw std::runtime_error("gtp engine returned invalid vertex: " + vertex);
        }
        const char column =
            static_cast<char>(std::toupper(static_cast<unsigned char>(vertex[0])));
        int x = column - 'A';
        if (column > 'I') {
            --x;  // GTP 跳过 I
        }
        int row = 0;
        const auto [end, error] = std::from_chars(
            vertex.data() + 1, vertex.data() + vertex.size(), row
        );
        if (error != std::errc{} || x < 0 || x >= size || row < 1 || row > size) {
            throw std::runtime_error("gtp engine returned invalid vertex: " + vertex);
        }
        return go::Point{x, size - row};
    }

    void spawn() {
#if defined(_WIN32)
        SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        HANDLE to_child_read = nullptr;
        HANDLE from_child_write = nullptr;
        if (!CreatePipe(&to_child_read, &from_child_write, &inherit, 0)) {
            throw std::runtime_error("cannot create gtp pipes");
        }
        HANDLE to_child_write = nullptr;
        HANDLE from_child_read = nullptr;
        if (!CreatePipe(&to_child_write, &from_child_read, &inherit, 0)) {
            throw std::runtime_error("cannot create gtp pipes");
        }
        SetHandleInformation(to_child_write, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(from_child_read, HANDLE_FLAG_INHERIT, 0);
        STARTUPINFOA info{};
        info.cb = sizeof(info);
        info.dwFlags = STARTF_USESTDHANDLES;
        info.hStdInput = to_child_read;
        info.hStdOutput = from_child_write;
        info.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        PROCESS_INFORMATION process{};
        std::string mutable_command = command_;
        const BOOL ok = CreateProcessA(
            nullptr, mutable_command.data(), nullptr, nullptr, TRUE,
            0, nullptr, nullptr, &info, &process
        );
        CloseHandle(to_child_read);
        CloseHandle(from_child_write);
        if (!ok) {
            CloseHandle(to_child_write);
            CloseHandle(from_child_read);
            throw std::runtime_error("cannot start gtp engine: " + command_);
        }
        CloseHandle(process.hThread);
        process_ = process.hProcess;
        write_handle_ = to_child_write;
        read_handle_ = from_child_read;
        input_ = _fdopen(_open_osfhandle(reinterpret_cast<intptr_t>(to_child_write), _O_TEXT), "w");
        output_ = _fdopen(_open_osfhandle(reinterpret_cast<intptr_t>(from_child_read), _O_TEXT), "r");
        // _open_osfhandle 接管了 handle 的所有权，交给 FILE* 关闭。
        write_handle_ = nullptr;
        read_handle_ = nullptr;
#else
        int to_child[2];
        int from_child[2];
        if (pipe(to_child) != 0 || pipe(from_child) != 0) {
            throw std::runtime_error("cannot create gtp pipes");
        }
        const pid_t pid = fork();
        if (pid < 0) {
            throw std::runtime_error("cannot fork gtp engine");
        }
        if (pid == 0) {
            dup2(to_child[0], STDIN_FILENO);
            dup2(from_child[1], STDOUT_FILENO);
            close(to_child[0]);
            close(to_child[1]);
            close(from_child[0]);
            close(from_child[1]);
            execl("/bin/sh", "/bin/sh", "-c", command_.c_str(),
                  static_cast<char*>(nullptr));
            _exit(127);
        }
        pid_ = pid;
        close(to_child[0]);
        close(from_child[1]);
        input_ = fdopen(to_child[1], "w");
        output_ = fdopen(from_child[0], "r");
#endif
        if (input_ == nullptr || output_ == nullptr) {
            throw std::runtime_error("cannot open gtp pipes: " + command_);
        }
    }

    // 从 FILE* 读一行（去掉行尾 \n/\r）；EOF 且无内容返回 false。
    static bool read_line(std::FILE* stream, std::string& line) {
        line.clear();
        bool any = false;
        int character = 0;
        while ((character = std::fgetc(stream)) != EOF) {
            any = true;
            if (character == '\n') {
                break;
            }
            if (character != '\r') {
                line.push_back(static_cast<char>(character));
            }
        }
        return any;
    }

    std::string send(const std::string& command) {
        if (input_ == nullptr) {
            spawn();
        }
        std::fputs((command + "\n").c_str(), input_);
        std::fflush(input_);
        // 响应：成功 "= ..." 或失败 "? ..."，空行结束。
        std::string line;
        std::string payload;
        bool have_prefix = false;
        while (read_line(output_, line)) {
            if (line.empty()) {
                if (have_prefix) {
                    break;
                }
                continue;  // 命令前的空行
            }
            if (!have_prefix && (line.front() == '=' || line.front() == '?')) {
                if (line.front() == '?') {
                    throw std::runtime_error(
                        "gtp engine rejected '" + command + "': " + line
                    );
                }
                have_prefix = true;
                std::string rest = line.substr(1);
                const std::size_t start = rest.find_first_not_of(" \t");
                payload = start == std::string::npos ? std::string{} : rest.substr(start);
                continue;
            }
            payload += payload.empty() ? line : "\n" + line;
        }
        if (!have_prefix) {
            throw std::runtime_error(
                "gtp engine closed without responding to '" + command + "'"
            );
        }
        return payload;
    }

    void close_pipes() {
        if (input_ != nullptr) {
            std::fclose(input_);
            input_ = nullptr;
        }
        if (output_ != nullptr) {
            std::fclose(output_);
            output_ = nullptr;
        }
    }

    void reap() {
#if defined(_WIN32)
        if (process_ != nullptr) {
            WaitForSingleObject(process_, 5000);
            CloseHandle(process_);
            process_ = nullptr;
        }
#else
        if (pid_ > 0) {
            int status = 0;
            for (int i = 0; i < 50; ++i) {  // 最多等 5 秒收尾
                const pid_t done = waitpid(pid_, &status, WNOHANG);
                if (done == pid_) {
                    pid_ = -1;
                    return;
                }
                usleep(100 * 1000);
            }
            kill(pid_, SIGKILL);
            waitpid(pid_, &status, 0);
            pid_ = -1;
        }
#endif
    }

    std::string command_;
    bool spawned_ = false;
    int last_size_ = 19;
    std::FILE* input_ = nullptr;
    std::FILE* output_ = nullptr;
#if defined(_WIN32)
    HANDLE process_ = nullptr;
    HANDLE write_handle_ = nullptr;
    HANDLE read_handle_ = nullptr;
#else
    pid_t pid_ = -1;
#endif
};

std::unique_ptr<ArenaEngine> make_engine(const std::string& spec, std::uint64_t seed) {
    if (spec == "random") {
        return std::make_unique<RandomEngine>(seed);
    }
    if (spec.starts_with("model:")) {
        std::string rest = spec.substr(6);
        std::string device;
        if (const std::size_t equals = rest.find('='); equals != std::string::npos) {
            device = rest.substr(equals + 1);
            rest.resize(equals);
        }
        if (device == "cpu") {
            device.clear();
        }
        return std::make_unique<ModelEngine>(rest, device);
    }
    if (spec.starts_with("gtp:")) {
        return std::make_unique<GtpProcessEngine>(spec.substr(4));
    }
    throw std::invalid_argument(
        "unknown engine spec: " + spec +
        " (expected random | model:FILE[=DEVICE] | gtp:COMMAND)"
    );
}

struct ArenaOptions {
    std::string player;
    std::string opponent = "random";
    int games = 20;
    int size = 19;
    float komi = 7.5F;
    std::uint64_t seed = 20261024;
    std::string report;
    Sprt sprt;
};

float parse_float(std::string_view text) {
    std::size_t consumed = 0;
    const std::string copy(text);
    const float value = std::stof(copy, &consumed);
    if (consumed != copy.size()) {
        throw std::invalid_argument("expected a number: " + copy);
    }
    return value;
}

ArenaOptions parse_options(int argc, char** argv) {
    ArenaOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument(
                    std::string("missing value for ") + argv[i - 1]
                );
            }
            return argv[i];
        };
        if (arg == "--player") {
            options.player = value();
        } else if (arg == "--opponent") {
            options.opponent = value();
        } else if (arg == "--games") {
            options.games = std::stoi(value());
        } else if (arg == "--size") {
            options.size = std::stoi(value());
        } else if (arg == "--komi") {
            options.komi = parse_float(value());
        } else if (arg == "--seed") {
            options.seed = std::stoull(value());
        } else if (arg == "--report") {
            options.report = value();
        } else if (arg == "--sprt-p0") {
            options.sprt.p0 = std::stod(value());
        } else if (arg == "--sprt-p1") {
            options.sprt.p1 = std::stod(value());
        } else if (arg == "--sprt-alpha") {
            options.sprt.alpha = std::stod(value());
        } else if (arg == "--sprt-beta") {
            options.sprt.beta = std::stod(value());
        } else if (arg == "--help" || arg == "-h") {
            std::cerr
                << "usage: go_arena_cli --player ENGINE [--opponent ENGINE]"
                << " [--games 20] [--size 19] [--komi 7.5]\n"
                << "       [--sprt-p0 0.5] [--sprt-p1 0.55]"
                << " [--sprt-alpha 0.05] [--sprt-beta 0.05]\n"
                << "       [--seed N] [--report FILE.json]\n"
                << "  ENGINE = random | model:FILE[=DEVICE] | gtp:COMMAND\n"
                << "  SPRT 观测 --player 的每局结果（和局不计入似然）\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(arg));
        }
    }
    if (options.player.empty()) {
        throw std::invalid_argument("--player requires an engine spec");
    }
    if (options.games <= 0) {
        throw std::invalid_argument("--games must be positive");
    }
    if (options.size < 2 || options.size > 19) {
        throw std::invalid_argument("--size must be in [2,19]");
    }
    if (!(options.sprt.p0 > 0 && options.sprt.p0 < options.sprt.p1 &&
          options.sprt.p1 < 1)) {
        throw std::invalid_argument("require 0 < sprt-p0 < sprt-p1 < 1");
    }
    return options;
}

// 一局：双方交替 genmove，连续两次 pass 终局；KataGo 面积计分定胜负。
// 返回 player（黑色 = 偶数局，白色 = 奇数局，颜色轮换消除先手偏差）的
// 结果：1 胜 / 0 负 / 0.5 和。
double play_game(
    ArenaEngine& player,
    ArenaEngine& opponent,
    bool player_is_black,
    const ArenaOptions& options,
    int& moves_played
) {
    go::Game game(options.size, options.size, go::Stone::black, go::RuleSet::chinese(options.komi));
    ArenaEngine& black = player_is_black ? player : opponent;
    ArenaEngine& white = player_is_black ? opponent : player;
    black.clear_board(options.size, options.komi);
    white.clear_board(options.size, options.komi);

    moves_played = 0;
    int consecutive_passes = 0;
    constexpr int kMaxMoves = 4 * 19 * 19;  // 防御性上限
    while (consecutive_passes < 2 && moves_played < kMaxMoves) {
        const go::Stone mover = game.to_move();
        ArenaEngine& engine = mover == go::Stone::black ? black : white;
        ArenaEngine& spectator = mover == go::Stone::black ? white : black;
        const std::optional<go::Point> move = engine.genmove(game);
        if (!move.has_value()) {
            if (!game.play(mover, std::nullopt)) {
                throw std::runtime_error(engine.name() + " passed into a finished game");
            }
        } else {
            if (!game.play(mover, *move)) {
                throw std::runtime_error(
                    engine.name() + " produced an illegal move at " +
                    std::to_string(move->x) + "," + std::to_string(move->y)
                );
            }
        }
        // 对手引擎（尤其外部 GTP 子进程）必须知道这一步，否则它的内部
        // 盘面与 arena 脱节，下一步 genmove 会报"颜色不对"。
        spectator.notify_move(mover, move);
        if (move.has_value()) {
            consecutive_passes = 0;
        } else {
            ++consecutive_passes;
        }
        ++moves_played;
    }

    const go::Score score = game.score();
    if (!score.winner.has_value()) {
        return 0.5;
    }
    const bool black_won = *score.winner == go::Stone::black;
    const bool player_won = black_won == player_is_black;
    return player_won ? 1.0 : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ArenaOptions options = parse_options(argc, argv);
        std::unique_ptr<ArenaEngine> player =
            make_engine(options.player, options.seed);
        std::unique_ptr<ArenaEngine> opponent =
            make_engine(options.opponent, options.seed + 1);

        json games = json::array();
        Sprt sprt = options.sprt;
        const auto started = std::chrono::steady_clock::now();
        int index = 0;
        for (; index < options.games; ++index) {
            if (sprt.decided()) {
                break;
            }
            const bool player_is_black = index % 2 == 0;  // 颜色轮换
            int moves_played = 0;
            const double result = play_game(
                *player, *opponent, player_is_black, options, moves_played
            );
            sprt.observe(result);
            games.push_back({
                {"game", index + 1},
                {"player_color", player_is_black ? "b" : "w"},
                {"moves", moves_played},
                {"player_result", result},
                {"llr", sprt.llr()},
                {"verdict", sprt.verdict()},
            });
            std::cerr << "game " << (index + 1)
                      << " player=" << (player_is_black ? "b" : "w")
                      << " result=" << result
                      << " W/L/J=" << sprt.wins << "/" << sprt.losses << "/" << sprt.jigos
                      << " llr=" << sprt.llr()
                      << " verdict=" << sprt.verdict() << '\n';
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();

        const int decided_games = sprt.wins + sprt.losses;
        const double win_rate = decided_games > 0
            ? static_cast<double>(sprt.wins) / decided_games
            : 0.0;
        json report{
            {"schema", 1},
            {"player", player->name()},
            {"opponent", opponent->name()},
            {"games_played", index},
            {"games_requested", options.games},
            {"size", options.size},
            {"komi", options.komi},
            {"seed", options.seed},
            {"sprt", {
                {"p0", sprt.p0},
                {"p1", sprt.p1},
                {"alpha", sprt.alpha},
                {"beta", sprt.beta},
                {"wins", sprt.wins},
                {"losses", sprt.losses},
                {"jigos", sprt.jigos},
                {"win_rate", win_rate},
                {"llr", sprt.llr()},
                {"lower_bound", sprt.lower_bound()},
                {"upper_bound", sprt.upper_bound()},
                {"verdict", sprt.verdict()},
            }},
            {"games", std::move(games)},
            {"seconds", seconds},
        };
        if (!options.report.empty()) {
            std::ofstream output(options.report);
            if (!output) {
                throw std::runtime_error("cannot open report: " + options.report);
            }
            output << report.dump(2) << '\n';
        }
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
