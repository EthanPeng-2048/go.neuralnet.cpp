#include "go/game.hpp"
#include "go/features.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "external/nlohmann_json/json.hpp"

namespace {

struct ParsedMove {
    go::Stone color;
    std::optional<go::Point> point;
};

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

int parse_int(std::string_view text) {
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument("expected an integer: " + std::string(text));
    }
    return value;
}

float parse_float(std::string_view text) {
    std::size_t consumed = 0;
    const std::string copy(text);
    const float value = std::stof(copy, &consumed);
    if (consumed != copy.size()) {
        throw std::invalid_argument("expected a number: " + copy);
    }
    return value;
}

go::Stone parse_color(std::string_view text) {
    const std::string value = lower(std::string(text));
    if (value == "b" || value == "black") {
        return go::Stone::black;
    }
    if (value == "w" || value == "white") {
        return go::Stone::white;
    }
    throw std::invalid_argument("move color must be b/black or w/white");
}

ParsedMove parse_move(std::string_view text) {
    const std::size_t separator = text.find(':');
    if (separator == std::string_view::npos) {
        throw std::invalid_argument("move must use COLOR:X,Y or COLOR:pass");
    }
    ParsedMove move{parse_color(text.substr(0, separator)), std::nullopt};
    const std::string_view point = text.substr(separator + 1);
    if (lower(std::string(point)) == "pass") {
        return move;
    }
    const std::size_t comma = point.find(',');
    if (comma == std::string_view::npos) {
        throw std::invalid_argument("point must use zero-based X,Y");
    }
    move.point = go::Point{
        parse_int(point.substr(0, comma)),
        parse_int(point.substr(comma + 1)),
    };
    return move;
}

char stone_char(go::Stone stone) {
    switch (stone) {
    case go::Stone::black:
        return 'x';
    case go::Stone::white:
        return 'o';
    case go::Stone::empty:
        return '.';
    }
    return '?';
}

nlohmann::json stone_json(go::Stone stone) {
    switch (stone) {
    case go::Stone::black:
        return "b";
    case go::Stone::white:
        return "w";
    case go::Stone::empty:
        return nullptr;
    }
    return nullptr;
}

nlohmann::json game_json(const go::Game& game) {
    nlohmann::json board = nlohmann::json::array();
    for (int y = 0; y < game.height(); ++y) {
        std::string row;
        row.reserve(static_cast<std::size_t>(game.width()));
        for (int x = 0; x < game.width(); ++x) {
            row.push_back(stone_char(game.at({x, y})));
        }
        board.push_back(std::move(row));
    }

    const go::Score score = game.score();
    return nlohmann::json{
        {"width", game.width()},
        {"height", game.height()},
        {"moves", game.move_count()},
        {"to_move", stone_json(game.to_move())},
        {"game_over", game.game_over()},
        {"board", std::move(board)},
        {"score", {
            {"white_minus_black", score.white_minus_black},
            {"winner", score.winner ? stone_json(*score.winner) : nlohmann::json(nullptr)},
        }},
    };
}

go::RuleSet rule_set_from(std::string_view name, float komi) {
    const std::string normalized = lower(std::string(name));
    if (normalized == "chinese") {
        return go::RuleSet::chinese(komi);
    }
    if (normalized == "japanese") {
        return go::RuleSet::japanese(komi);
    }
    throw std::invalid_argument("--rules must be chinese or japanese");
}

go::Stone parse_stone_name(std::string_view text) {
    return parse_color(text);
}

std::string stone_name(go::Stone stone) {
    return stone == go::Stone::white ? "w" : "b";
}

void print_usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " replay [--size N] [--rules chinese|japanese] [--komi K] [--tolerant] "
           "COLOR:X,Y|COLOR:pass ...\n"
        << "       " << program << " trace --input JSON\n"
        << "       " << program << " trace-batch --input JSONL\n"
        << "       " << program << " features --input JSON\n";
}

int run_replay(int argc, char** argv) {
    int size = 19;
    go::RuleSet rules = go::RuleSet::chinese();
    bool tolerant = false;
    std::vector<ParsedMove> moves;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--size") {
            if (++i >= argc) {
                throw std::invalid_argument("--size requires a value");
            }
            size = parse_int(argv[i]);
        } else if (arg == "--rules") {
            if (++i >= argc) {
                throw std::invalid_argument("--rules requires a value");
            }
            rules = rule_set_from(argv[i], rules.komi);
        } else if (arg == "--komi") {
            if (++i >= argc) {
                throw std::invalid_argument("--komi requires a value");
            }
            rules.komi = parse_float(argv[i]);
        } else if (arg == "--tolerant") {
            tolerant = true;
        } else if (arg.starts_with("--")) {
            throw std::invalid_argument("unknown option: " + std::string(arg));
        } else {
            moves.push_back(parse_move(arg));
        }
    }

    go::Game game(size, size, go::Stone::black, rules);
    for (const ParsedMove& move : moves) {
        const bool accepted = tolerant
            ? game.play_tolerant(move.color, move.point)
            : game.play(move.color, move.point);
        if (!accepted) {
            throw std::invalid_argument(
                "move " + std::to_string(game.move_count() + 1) + " is not legal under the selected rules"
            );
        }
    }

    std::cout << game_json(game).dump() << '\n';
    return 0;
}

std::string read_text_input(const std::string& path) {
    if (path == "-") {
        return std::string{
            std::istreambuf_iterator<char>(std::cin),
            std::istreambuf_iterator<char>()
        };
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open trace input: " + path);
    }
    return std::string{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    };
}

std::string required_json_string(const nlohmann::json& object, const char* key) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_string()) {
        throw std::invalid_argument(std::string("missing string field: ") + key);
    }
    return it->get<std::string>();
}

int required_json_int(const nlohmann::json& object, const char* key) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_number_integer()) {
        throw std::invalid_argument(std::string("missing integer field: ") + key);
    }
    return it->get<int>();
}

nlohmann::json frame_json(const go::Game& game) {
    nlohmann::json rows = nlohmann::json::array();
    std::string board = game.ascii();
    std::size_t begin = 0;
    while (begin <= board.size()) {
        const std::size_t end = board.find('\n', begin);
        rows.push_back(board.substr(begin, end - begin));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }

    nlohmann::json ko = nullptr;
    if (const auto point = game.ko_point(); point.has_value()) {
        ko = nlohmann::json::array({point->x, point->y});
    }
    return nlohmann::json{
        {"board", std::move(rows)},
        {"to_move", stone_name(game.to_move())},
        {"move_number", game.move_count()},
        {"ko", std::move(ko)},
        {"black_stones_captured", game.captured_count(go::Stone::black)},
        {"white_stones_captured", game.captured_count(go::Stone::white)},
        {"game_over", game.game_over()},
    };
}

nlohmann::json trace_request(const nlohmann::json& request, bool tolerant) {
    const int size = required_json_int(request, "size");
    const std::string rules_name = required_json_string(request, "rules");
    const float komi = request.value("komi", 7.5F);
    const go::Stone to_move = parse_stone_name(required_json_string(request, "to_move"));

    const auto rows_it = request.find("rows");
    if (rows_it == request.end() || !rows_it->is_array() || rows_it->empty()) {
        throw std::invalid_argument("trace requires a non-empty rows array");
    }
    std::string rows;
    for (std::size_t i = 0; i < rows_it->size(); ++i) {
        if (!(*rows_it)[i].is_string()) {
            throw std::invalid_argument("trace rows must be strings");
        }
        if (i != 0) {
            rows.push_back('\n');
        }
        rows += (*rows_it)[i].get<std::string>();
    }

    go::Game game = go::Game::from_ascii(rows, to_move, rule_set_from(rules_name, komi));
    nlohmann::json frames = nlohmann::json::array();
    frames.push_back(frame_json(game));

    const auto moves_it = request.find("moves");
    if (moves_it != request.end()) {
        if (!moves_it->is_array()) {
            throw std::invalid_argument("trace moves must be an array");
        }
        for (const nlohmann::json& item : *moves_it) {
            const go::Stone color = parse_stone_name(required_json_string(item, "color"));
            std::optional<go::Point> point;
            const auto point_it = item.find("point");
            if (point_it != item.end() && !point_it->is_null()) {
                if (!point_it->is_array() || point_it->size() != 2) {
                    throw std::invalid_argument("move point must be [x,y] or null");
                }
                point = go::Point{
                    (*point_it)[0].get<int>(),
                    (*point_it)[1].get<int>(),
                };
            }
            const bool accepted = tolerant
                ? game.play_tolerant(color, point)
                : game.play(color, point);
            if (!accepted) {
                throw std::runtime_error(
                    "trace move " + std::to_string(game.move_count() + 1) + " was rejected"
                );
            }
            frames.push_back(frame_json(game));
        }
    }

    return nlohmann::json{
        {"size", size},
        {"rules", rules_name},
        {"komi", komi},
        {"frames", std::move(frames)},
    };
}

struct TraceOptions {
    std::string input_path;
    bool tolerant = true;
};

TraceOptions parse_trace_options(int argc, char** argv, std::string_view command) {
    TraceOptions options;
    bool has_input = false;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--input") {
            if (++i >= argc) {
                throw std::invalid_argument("--input requires a path or '-'");
            }
            options.input_path = argv[i];
            has_input = true;
        } else if (arg == "--strict") {
            options.tolerant = false;
        } else {
            throw std::invalid_argument(
                "unknown " + std::string(command) + " option: " + std::string(arg)
            );
        }
    }
    if (!has_input) {
        throw std::invalid_argument(std::string(command) + " requires --input");
    }
    return options;
}

int run_trace(int argc, char** argv) {
    const TraceOptions options = parse_trace_options(argc, argv, "trace");
    const nlohmann::json request = nlohmann::json::parse(
        read_text_input(options.input_path)
    );
    std::cout << trace_request(request, options.tolerant).dump() << '\n';
    return 0;
}

int run_trace_batch(int argc, char** argv) {
    const TraceOptions options = parse_trace_options(argc, argv, "trace-batch");
    std::ifstream file;
    std::istream* input = &std::cin;
    if (options.input_path != "-") {
        file.open(options.input_path);
        if (!file) {
            throw std::runtime_error("cannot open trace-batch input: " + options.input_path);
        }
        input = &file;
    }

    std::string line;
    std::size_t line_number = 0;
    while (std::getline(*input, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }
        try {
            std::cout << trace_request(
                nlohmann::json::parse(line),
                options.tolerant
            ).dump() << '\n';
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "trace-batch line " + std::to_string(line_number) + ": " + error.what()
            );
        }
    }
    if (input->bad()) {
        throw std::runtime_error("failed while reading trace-batch input");
    }
    return 0;
}

// M6 特征对拍：与 trace 相同的请求 JSON，但输出每一帧的 9×19×19 特征
// （小端 float32 的 hex），供 tests/test_m6_features.py 与 Python
// extract_features 逐位比对。hex 比 JSON 浮点更严格——不做十进制往返。
nlohmann::json features_request(const nlohmann::json& request) {
    const nlohmann::json traced = trace_request(request, true);
    const nlohmann::json& frames = traced.at("frames");

    // 重放同一局以获得逐帧的 go::Game（trace_request 不暴露 C++ 状态）。
    const std::string rules_name = required_json_string(request, "rules");
    const float komi = request.value("komi", 7.5F);
    const go::Stone to_move = parse_stone_name(required_json_string(request, "to_move"));

    std::string rows;
    const nlohmann::json& rows_it = request.at("rows");
    for (std::size_t i = 0; i < rows_it.size(); ++i) {
        if (i != 0) {
            rows.push_back('\n');
        }
        rows += rows_it[i].get<std::string>();
    }
    go::Game game = go::Game::from_ascii(rows, to_move, rule_set_from(rules_name, komi));

    std::vector<float> buffer(go::feature_element_count);
    nlohmann::json planes = nlohmann::json::array();
    auto emit = [&](const go::Game& current) {
        go::extract_features(current, buffer);
        static const char* digits = "0123456789abcdef";
        std::string hex;
        hex.reserve(buffer.size() * 2 * sizeof(float));
        for (float value : buffer) {
            const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
            for (unsigned shift = 0; shift < 32; shift += 8) {  // 小端字节序
                const std::uint32_t byte = (bits >> shift) & 0xFF;
                hex.push_back(digits[byte >> 4]);
                hex.push_back(digits[byte & 0xF]);
            }
        }
        planes.push_back(std::move(hex));
    };
    emit(game);

    const auto moves_it = request.find("moves");
    if (moves_it != request.end()) {
        for (const nlohmann::json& item : *moves_it) {
            const go::Stone color = parse_stone_name(required_json_string(item, "color"));
            std::optional<go::Point> point;
            const auto point_it = item.find("point");
            if (point_it != item.end() && !point_it->is_null()) {
                point = go::Point{
                    (*point_it)[0].get<int>(),
                    (*point_it)[1].get<int>(),
                };
            }
            if (!game.play_tolerant(color, point)) {
                throw std::runtime_error("features move was rejected");
            }
            emit(game);
        }
    }
    if (planes.size() != frames.size()) {
        throw std::runtime_error("features frame count mismatch");
    }
    return nlohmann::json{{"planes_hex_le_f32", std::move(planes)}};
}

int run_features(int argc, char** argv) {
    const TraceOptions options = parse_trace_options(argc, argv, "features");
    const nlohmann::json request = nlohmann::json::parse(
        read_text_input(options.input_path)
    );
    std::cout << features_request(request).dump() << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 2;
        }
        const std::string_view command = argv[1];
        if (command == "replay") {
            return run_replay(argc, argv);
        }
        if (command == "trace") {
            return run_trace(argc, argv);
        }
        if (command == "trace-batch") {
            return run_trace_batch(argc, argv);
        }
        if (command == "features") {
            return run_features(argc, argv);
        }
        print_usage(argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
