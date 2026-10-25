// arena_engine — 对局引擎抽象（M6 抽出，GUI 竞技台复用）：random /
// model / 外部 GTP 子进程三种引擎的统一接口。go_arena_cli 的批量评估与
// go_gui_cli 的竞技台共用这里，保证两条路径的选点与同步语义完全一致。
//
// 引擎 spec（make_engine）：
//   * random  —— 均匀随机合法着（不主动 pass；无子可下才 pass）。
//   * model:FILE[=DEVICE] —— 本项目的策略模型（go::PolicyPlayer）。
//   * gtp:COMMAND —— 外部 GTP 引擎子进程（如 `gnugo --mode gtp --quiet`）。

#include <charconv>
#include <cctype>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "go/arena_engine.hpp"
#include "go/features.hpp"  // feature_plane_stride / policy_class_count

#if defined(_WIN32)
#include <fcntl.h>   // _O_TEXT（_fdopen 的翻译模式）
#include <io.h>
#include <process.h>
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace go {
namespace {

class RandomEngine final : public ArenaEngine {
public:
    explicit RandomEngine(std::uint64_t seed) : rng_(seed) {}

    std::optional<go::Point> genmove(go::Game& game) override {
        // 已经对"当前行棋方"预告过的一步就下它：AI 视角显示的点 = 实战的点。
        if (planned_.has_value() && planned_for_ == game.to_move()) {
            const std::optional<go::Point> move = *planned_;
            planned_.reset();
            return move;
        }
        planned_.reset();
        return pick(game);
    }

    [[nodiscard]] std::optional<std::optional<go::Point>> preview_move(
        const go::Game& game) override {
        if (!planned_.has_value() || planned_for_ != game.to_move()) {
            planned_ = pick(game);
            planned_for_ = game.to_move();
        }
        return planned_;
    }

    void clear_board(int, float) override { planned_.reset(); }

    [[nodiscard]] std::string name() const override { return "random"; }

private:
    std::optional<go::Point> pick(const go::Game& game) {
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
        std::uniform_int_distribution<std::size_t> choice(0, stones.size() - 1);
        return stones[choice(rng_)];
    }

    std::mt19937_64 rng_;
    std::optional<std::optional<go::Point>> planned_;  // 已预告未下的一步
    go::Stone planned_for_ = go::Stone::black;
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

    [[nodiscard]] std::optional<std::optional<go::Point>> preview_move(
        const go::Game& game) override {
        if (!player_) {
            player_ = go::PolicyPlayer::load(path_, device_);
        }
        // argmax 与 genmove 同源：预告的点就是它将下的点。
        return std::optional<std::optional<go::Point>>{
            player_->select_move(game)};
    }

    void clear_board(int, float) override {}

    [[nodiscard]] std::optional<PolicyEvaluation> evaluate(const Game& game) override {
        if (!player_) {
            player_ = go::PolicyPlayer::load(path_, device_);
        }
        return player_->evaluate(game);
    }

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
        // GTP v2 规定小写 "pass"/"resign"，但实战引擎大小写不一（GnuGo 3.8
        // 的 genmove 返回 "PASS"）。resign 也按 pass 处理：竞技台照常数子
        // 收尾，不让一个词把整场对局打断。
        if (vertex.empty() || is_pass_like(vertex)) {
            return std::nullopt;
        }
        return parse_vertex(vertex, game.width());
    }

    static bool is_pass_like(const std::string& token) {
        std::string lower;
        lower.reserve(token.size());
        for (const char character : token) {
            lower.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(character))));
        }
        return lower == "pass" || lower == "resign";
    }

    [[nodiscard]] std::optional<std::optional<go::Point>> preview_move(
        const go::Game& game) override {
        const std::string color = game.to_move() == go::Stone::black ? "b" : "w";
        // reg_genmove：GTP 的"说出你要下哪，但别真下、别动你的盘面"。
        try {
            const std::string vertex = first_word(send("reg_genmove " + color));
            if (vertex.empty() || is_pass_like(vertex)) {
                return std::optional<std::optional<go::Point>>{
                    std::optional<go::Point>{}};
            }
            const go::Point point = parse_vertex(vertex, game.width());
            return std::optional<std::optional<go::Point>>{
                std::optional<go::Point>{point}};
        } catch (const std::exception&) {
            // 引擎不支持 reg_genmove：随机合法点当示意（合成分布）。
            const std::vector<std::optional<go::Point>> moves =
                game.legal_moves();
            std::vector<go::Point> stones;
            stones.reserve(moves.size());
            for (const auto& move : moves) {
                if (move.has_value()) {
                    stones.push_back(*move);
                }
            }
            if (stones.empty()) {
                return std::optional<std::optional<go::Point>>{
                    std::optional<go::Point>{}};
            }
            std::uniform_int_distribution<std::size_t> choice(
                0, stones.size() - 1);
            return std::optional<std::optional<go::Point>>{
                std::optional<go::Point>{stones[choice(rng_)]}};
        }
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
        // 父→子管道：写端 to_child_write 归父进程，读端 to_child_read 给子进程 stdin。
        // 子→父管道：写端 from_child_write 给子进程 stdout，读端 from_child_read 归父进程。
        // （两条管道的两端必须各自成对；接错会让 stdin/stdout 短路到同一根管道，
        //  父进程的读端立即 EOF、子进程又永远收不到 EOF 而泄漏。）
        HANDLE to_child_read = nullptr;
        HANDLE to_child_write = nullptr;
        if (!CreatePipe(&to_child_read, &to_child_write, &inherit, 0)) {
            throw std::runtime_error("cannot create gtp pipes");
        }
        HANDLE from_child_read = nullptr;
        HANDLE from_child_write = nullptr;
        if (!CreatePipe(&from_child_read, &from_child_write, &inherit, 0)) {
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
            // 与 Linux 分支同语义：5 秒内不退就强杀，绝不留孤儿（孤儿会继承
            // 宿主的 stderr 句柄，让上层读管道永远等不到 EOF）。
            if (WaitForSingleObject(process_, 5000) == WAIT_TIMEOUT) {
                TerminateProcess(process_, 1);
                WaitForSingleObject(process_, 5000);
            }
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
    std::mt19937_64 rng_{std::random_device{}()};
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
}  // namespace

std::optional<std::optional<Point>> ArenaEngine::preview_move(const Game&) {
    return std::nullopt;  // 默认无法预告
}

std::optional<PolicyEvaluation> ArenaEngine::evaluate(const Game& game) {
    // 默认评估 = 把"将下点"合成单点 100% 的分布（synthetic = true）：
    // 不支持评估的引擎（random / 外部 GTP）也让 GUI 的 AI 视角有内容可看，
    // 而不是报"不支持评估"。连预告都给不出时才返回 nullopt。
    const std::optional<std::optional<Point>> planned = preview_move(game);
    if (!planned.has_value()) {
        return std::nullopt;
    }
    PolicyEvaluation evaluation;
    evaluation.policy.assign(policy_class_count, 0.0F);
    std::size_t index = feature_plane_stride;  // 361 = pass
    if (planned->has_value()) {
        const Point point = planned->value();  // -> 已越过外层 optional
        index = static_cast<std::size_t>(point.y) * game.width() +
                static_cast<std::size_t>(point.x);
    }
    evaluation.policy[index] = 1.0F;
    evaluation.value = std::nullopt;
    evaluation.synthetic = true;
    return evaluation;
}

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


std::string engine_display_name(const std::string& spec) {
    if (spec == "random") {
        return "随机引擎";
    }
    if (spec.starts_with("model:")) {
        std::string rest = spec.substr(6);
        if (const std::size_t equals = rest.find('='); equals != std::string::npos) {
            rest.resize(equals);
        }
        const std::size_t slash = rest.find_last_of("/\\");
        std::string base =
            rest.substr(slash == std::string::npos ? 0 : slash + 1);
        for (const std::string_view suffix : {std::string_view{".nnmodel"},
                                              std::string_view{".nnckpt"}}) {
            if (base.ends_with(suffix)) {
                base.resize(base.size() - suffix.size());
                break;
            }
        }
        return base.empty() ? "模型" : base;
    }
    if (spec.starts_with("gtp:")) {
        const std::string_view command = std::string_view{spec}.substr(4);
        const std::size_t start = command.find_first_not_of(" \t");
        const std::size_t end = command.find_first_of(" \t", start);
        const std::string_view word =
            command.substr(start == std::string_view::npos ? 0 : start,
                           end == std::string_view::npos ? std::string_view::npos
                                                        : end - start);
        return word.empty() ? "GTP 引擎" : std::string{word};
    }
    return spec;
}

}  // namespace go
