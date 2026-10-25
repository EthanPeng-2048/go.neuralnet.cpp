// go_gtp_cli — GTP v2 适配器（M6）：让 Sabaki 等 GUI 可以直接与本项目的
// 策略模型对局。
//
// 实现范围（roadmap M6 DoD「Sabaki 亲手可下」）：
//   * 协议：GTP v2 行协议——`[id] command args`，响应 `= ... \n\n` /
//     `? ... \n\n`，`#` 开头为注释行；
//   * 命令：protocol_version / name / version / known_command /
//     list_commands / boardsize / clear_board / komi / play / genmove /
//     showboard / final_score / quit；
//   * 规则全部走 KataGo（go::Game），与 M1/M2/M6 特征同源；
//   * genmove：M3 v1 特征 → 策略模型 → 合法点 argmax（go::PolicyPlayer）；
//     `--visits N` 打开 M8 的 MCTS 搜索（N 次模拟）。
//
// 不实现：reg_genmove / ponder / time settings（time 命令只做应答不生效）。boardsize 支持 1–19，但**模型只在 19×19 上
// 可用**：其他尺寸的 genmove 返回错误并提示（特征契约限制，不是协议限制）。

#include <charconv>
#include <cctype>
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "go/game.hpp"
#include "go/mcts.hpp"
#include "go/policy.hpp"

namespace {

constexpr const char* kEngineName = "go.neuralnet.cpp";
constexpr const char* kEngineVersion = "0.2.0";

struct GtpOptions {
    std::string model;
    std::string device;  // 空 = CPU
    std::size_t visits = 0;  // >0 = M8 MCTS 搜索次数；0 = 纯先验 argmax
};

// GTP 坐标：字母 = 列（跳过 I），数字 = 行（自底向上，19 = 顶行）。
// go::Point 的 y=0 是顶行，因此 row = size - y。
std::string point_to_gtp(const go::Point& point, int size) {
    std::string result;
    char column = static_cast<char>('A' + point.x);
    if (column >= 'I') {
        column = static_cast<char>(column + 1);  // 跳过 I
    }
    result.push_back(column);
    result += std::to_string(size - point.y);
    return result;
}

std::optional<go::Point> gtp_to_point(std::string_view token, int size) {
    if (token == "pass" || token == "resign") {
        return std::nullopt;
    }
    if (token.size() < 2) {
        throw std::invalid_argument("invalid GTP vertex: " + std::string(token));
    }
    char column = static_cast<char>(std::toupper(static_cast<unsigned char>(token[0])));
    if (column == 'I') {
        throw std::invalid_argument("GTP columns skip the letter I");
    }
    if (column < 'A' || column > 'T') {
        throw std::invalid_argument("invalid GTP column: " + std::string(token));
    }
    int x = column - 'A';
    if (column > 'I') {
        --x;  // I 被跳过
    }
    int row = 0;
    const std::string_view digits = token.substr(1);
    const auto [end, error] = std::from_chars(
        digits.data(), digits.data() + digits.size(), row
    );
    if (error != std::errc{} || end != digits.data() + digits.size()) {
        throw std::invalid_argument("invalid GTP row: " + std::string(token));
    }
    if (x < 0 || x >= size || row < 1 || row > size) {
        throw std::invalid_argument("GTP vertex outside board: " + std::string(token));
    }
    return go::Point{x, size - row};
}

go::Stone parse_color(std::string_view token) {
    if (token == "b" || token == "B" || token == "black") {
        return go::Stone::black;
    }
    if (token == "w" || token == "W" || token == "white") {
        return go::Stone::white;
    }
    throw std::invalid_argument("color must be b/w or black/white");
}

std::vector<std::string> split_words(std::string_view line) {
    std::vector<std::string> words;
    std::size_t index = 0;
    while (index < line.size()) {
        while (index < line.size() &&
               std::isspace(static_cast<unsigned char>(line[index]))) {
            ++index;
        }
        const std::size_t start = index;
        while (index < line.size() &&
               !std::isspace(static_cast<unsigned char>(line[index]))) {
            ++index;
        }
        if (index > start) {
            words.emplace_back(line.substr(start, index - start));
        }
    }
    return words;
}

class Engine {
public:
    explicit Engine(GtpOptions options)
        : options_(std::move(options)),
          game_(19, 19, go::Stone::black, go::RuleSet::chinese(komi_)) {}

    // 处理一条命令，返回响应正文（不含 "= "/"? " 前缀）。
    // 返回 nullopt = quit。
    std::optional<std::pair<bool, std::string>> execute(
        const std::vector<std::string>& words
    ) {
        if (words.empty()) {
            return std::pair{true, std::string{}};
        }
        const std::string& command = words.front();
        try {
            if (command == "protocol_version") {
                return std::pair{true, std::string{"2"}};
            }
            if (command == "name") {
                return std::pair{true, std::string{kEngineName}};
            }
            if (command == "version") {
                return std::pair{true, std::string{kEngineVersion}};
            }
            if (command == "known_command") {
                return std::pair{true, std::string{
                    is_known(words.size() > 1 ? words[1] : "") ? "true" : "false"
                }};
            }
            if (command == "list_commands") {
                return std::pair{true, list_commands()};
            }
            if (command == "quit") {
                return std::nullopt;
            }
            if (command == "boardsize") {
                return boardsize(words);
            }
            if (command == "clear_board") {
                return clear_board();
            }
            if (command == "komi") {
                return komi(words);
            }
            if (command == "play") {
                return play(words);
            }
            if (command == "genmove") {
                return genmove(words);
            }
            if (command == "showboard") {
                return std::pair{true, game_.ascii()};
            }
            if (command == "final_score") {
                return final_score();
            }
            // time settings / time_left / final_status_list 等：应答成功但
            // 不生效（本引擎 M8 前无搜索、无时钟）。
            if (command == "time_settings" || command == "time_left") {
                return std::pair{true, std::string{}};
            }
            if (command == "reg_genmove") {
                // 回退到 genmove 的语义（本引擎无 ponder）。
                return genmove(words);
            }
            return std::pair{false, std::string{"unknown command: "} + command};
        } catch (const std::exception& error) {
            return std::pair{false, std::string{error.what()}};
        }
    }

private:
    static bool is_known(std::string_view command) {
        static const char* kCommands[] = {
            "protocol_version", "name", "version", "known_command",
            "list_commands", "quit", "boardsize", "clear_board", "komi",
            "play", "genmove", "showboard", "final_score", "time_settings",
            "time_left", "reg_genmove",
        };
        for (const char* candidate : kCommands) {
            if (command == candidate) {
                return true;
            }
        }
        return false;
    }

    static std::string list_commands() {
        std::string result =
            "protocol_version\nname\nversion\nknown_command\nlist_commands\n"
            "quit\nboardsize\nclear_board\nkomi\nplay\ngenmove\nshowboard\n"
            "final_score\ntime_settings\ntime_left\nreg_genmove";
        return result;
    }

    std::pair<bool, std::string> boardsize(const std::vector<std::string>& words) {
        if (words.size() != 2) {
            return {false, "usage: boardsize <size>"};
        }
        int size = 0;
        const auto [end, error] = std::from_chars(
            words[1].data(), words[1].data() + words[1].size(), size
        );
        if (error != std::errc{} || end != words[1].data() + words[1].size() ||
            size < 1 || size > 19) {
            return {false, "size must be an integer in [1,19]"};
        }
        size_ = size;
        game_ = go::Game(size_, size_, go::Stone::black, go::RuleSet::chinese(komi_));
        return {true, {}};
    }

    std::pair<bool, std::string> clear_board() {
        game_ = go::Game(size_, size_, go::Stone::black, go::RuleSet::chinese(komi_));
        return {true, {}};
    }

    std::pair<bool, std::string> komi(const std::vector<std::string>& words) {
        if (words.size() != 2) {
            return {false, "usage: komi <value>"};
        }
        std::size_t consumed = 0;
        const float value = std::stof(words[1], &consumed);
        if (consumed != words[1].size()) {
            return {false, "komi must be a number"};
        }
        komi_ = value;
        // GTP 惯例：komi 在空盘前设置；若已有落子，重建盘面会丢历史，
        // 因此拒绝带子改 komi。
        if (game_.move_count() > 0) {
            return {false, "cannot set komi after moves have been played"};
        }
        game_ = go::Game(size_, size_, go::Stone::black, go::RuleSet::chinese(komi_));
        return {true, {}};
    }

    std::pair<bool, std::string> play(const std::vector<std::string>& words) {
        if (words.size() != 3) {
            return {false, "usage: play <color> <vertex>"};
        }
        const go::Stone color = parse_color(words[1]);
        std::optional<go::Point> point;
        if (words[2] != "resign") {
            try {
                point = gtp_to_point(words[2], size_);
            } catch (const std::invalid_argument&) {
                return {false, "invalid vertex: " + words[2]};
            }
        }
        // GTP 的 play 是权威指令（GUI 可以回放/摆子），走 tolerant 路径，
        // 与 M2/M3 的 SGF 回放语义一致；pass（nullopt → PASS_LOC）同样走它。
        if (!game_.play_tolerant(color, point)) {
            return {false, "illegal move: " + words[2]};
        }
        return {true, {}};
    }

    std::pair<bool, std::string> genmove(const std::vector<std::string>& words) {
        if (words.size() != 2) {
            return {false, "usage: genmove <color>"};
        }
        const go::Stone color = parse_color(words[1]);
        if (game_.game_over()) {
            return {true, "pass"};
        }
        if (color != game_.to_move()) {
            const char turn = game_.to_move() == go::Stone::black ? 'b' : 'w';
            return {false, std::string("color to move is ") + turn};
        }
        if (size_ != 19) {
            return {false,
                    "this engine only plays 19x19 (M3 v1 feature contract); "
                    "boardsize is " + std::to_string(size_)};
        }
        if (!player_) {
            player_ = go::PolicyPlayer::load(options_.model, options_.device);
            if (options_.visits > 0) {
                evaluator_ = go::make_policy_evaluator(*player_);
                go::MctsConfig config;
                config.simulations = options_.visits;
                search_ = std::make_unique<go::MctsSearch>(*evaluator_, config);
            }
        }
        // --visits N：MCTS 选点；缺省仍是纯先验 argmax（M6 语义不变）。
        const std::optional<go::Point> move = search_
            ? search_->search(game_).best_move
            : player_->select_move(game_);
        if (!move.has_value()) {
            if (!game_.play(color, std::nullopt)) {
                return {false, "cannot pass"};
            }
            return {true, "pass"};
        }
        if (!game_.play(color, move)) {
            return {false, "engine selected an illegal move"};
        }
        return {true, point_to_gtp(*move, size_)};
    }

    std::pair<bool, std::string> final_score() {
        if (!game_.game_over()) {
            return {false, "game is not over"};
        }
        const go::Score score = game_.score();
        if (!score.winner.has_value()) {
            return {true, "0"};
        }
        const float margin = score.white_minus_black >= 0
            ? score.white_minus_black
            : -score.white_minus_black;
        const char letter = *score.winner == go::Stone::white ? 'W' : 'B';
        std::string result{letter, '+'};
        // 中国规则的分数可能带 .5；整数分不写小数点。
        if (margin == static_cast<float>(static_cast<int>(margin))) {
            result += std::to_string(static_cast<int>(margin));
        } else {
            result += std::to_string(margin);
        }
        return {true, result};
    }

    GtpOptions options_;
    int size_ = 19;
    float komi_ = 7.5F;
    go::Game game_;
    std::optional<go::PolicyPlayer> player_;
    std::unique_ptr<go::LeafEvaluator> evaluator_;
    std::unique_ptr<go::MctsSearch> search_;
};

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " --model FILE.nnmodel [--gpu[=DEVICE]] [--visits N]\n"
        << "       GTP v2 over stdin/stdout (Sabaki-compatible)\n"
        << "       --visits N: select moves with MCTS (N simulations);\n"
        << "       N=0 (default) selects raw policy argmax\n";
}

std::size_t parse_visits(std::string_view text) {
    std::size_t visits = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), visits);
    // 0 = 纯先验 argmax（与缺省一致）；只拒绝非数字。
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument("--visits must be a non-negative integer");
    }
    return visits;
}

GtpOptions parse_options(int argc, char** argv) {
    GtpOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--model") {
            if (++i >= argc) {
                throw std::invalid_argument("--model requires a path");
            }
            options.model = argv[i];
        } else if (arg.starts_with("--model=")) {
            options.model = std::string(arg.substr(8));
        } else if (arg == "--gpu") {
            options.device = "gpu";
        } else if (arg.starts_with("--gpu=")) {
            options.device = std::string(arg.substr(6));
        } else if (arg == "--cpu") {
            options.device.clear();
        } else if (arg == "--visits") {
            if (++i >= argc) {
                throw std::invalid_argument("--visits requires a number");
            }
            options.visits = parse_visits(argv[i]);
        } else if (arg.starts_with("--visits=")) {
            options.visits = parse_visits(std::string(arg.substr(9)));
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(arg));
        }
    }
    if (options.model.empty()) {
        throw std::invalid_argument("gtp requires --model FILE.nnmodel");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Engine engine(parse_options(argc, argv));
        std::string line;
        while (std::getline(std::cin, line)) {
            // 去掉行尾 \r（Windows GUI）。
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty() || line.front() == '#') {
                continue;
            }
            // `[id] command ...`：id 是可选的数字前缀。
            std::string_view body = line;
            std::optional<std::string> id;
            const std::size_t space = body.find_first_of(" \t");
            const std::string_view head = body.substr(0, space);
            bool numeric = !head.empty();
            for (const char c : head) {
                if (!std::isdigit(static_cast<unsigned char>(c))) {
                    numeric = false;
                    break;
                }
            }
            if (numeric) {
                id = std::string(head);
                body = space == std::string_view::npos ? std::string_view{} : body.substr(space);
            }
            const std::vector<std::string> words = split_words(body);
            const auto response = engine.execute(words);
            if (!response.has_value()) {
                break;  // quit
            }
            const std::string prefix = response->first ? "= " : "? ";
            std::cout << prefix;
            if (id.has_value()) {
                std::cout << *id << ' ';
            }
            std::cout << response->second << "\n\n";
            std::cout.flush();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
