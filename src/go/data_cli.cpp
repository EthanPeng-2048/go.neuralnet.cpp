#include "go/nngames.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "external/nlohmann_json/json.hpp"
#include "neuralnet.cpp/dataset_format.hpp"

namespace {

using json = nlohmann::json;

struct PackOptions {
    std::string input;
    std::string output;
    go::DatasetSource source;
    std::uint64_t created_at = 0;
};

std::string required_string(const json& value, const char* key) {
    const auto it = value.find(key);
    if (it == value.end() || !it->is_string()) {
        throw std::runtime_error(std::string("missing string field: ") + key);
    }
    return it->get<std::string>();
}

std::string nn_required_string(const nn::KeyValueRecord& value, const char* key) {
    std::string result;
    if (!value.get(key, result)) {
        throw std::runtime_error(std::string("missing vocabulary string field: ") + key);
    }
    return result;
}

std::uint64_t nn_required_uint(const nn::KeyValueRecord& value, const char* key) {
    std::uint64_t result = 0;
    if (!value.get(key, result)) {
        throw std::runtime_error(std::string("missing vocabulary uint field: ") + key);
    }
    return result;
}

std::uint64_t uint_field(const json& value, const char* key, std::uint64_t fallback) {
    const auto it = value.find(key);
    if (it == value.end() || it->is_null()) {
        return fallback;
    }
    if (!it->is_number_unsigned() && !it->is_number_integer()) {
        throw std::runtime_error(std::string("field is not an integer: ") + key);
    }
    return it->get<std::uint64_t>();
}

std::int32_t int_field(const json& value, const char* key, std::int32_t fallback) {
    const auto it = value.find(key);
    if (it == value.end() || it->is_null()) {
        return fallback;
    }
    if (!it->is_number_integer()) {
        throw std::runtime_error(std::string("field is not an integer: ") + key);
    }
    return it->get<std::int32_t>();
}

std::string string_field(const json& value, const char* key, std::string fallback = {}) {
    const auto it = value.find(key);
    if (it == value.end() || it->is_null()) {
        return fallback;
    }
    if (!it->is_string()) {
        throw std::runtime_error(std::string("field is not a string: ") + key);
    }
    return it->get<std::string>();
}

std::vector<std::uint32_t> u32_array(const json& value, const char* key) {
    std::vector<std::uint32_t> result;
    const auto it = value.find(key);
    if (it == value.end() || it->is_null()) {
        return result;
    }
    if (!it->is_array()) {
        throw std::runtime_error(std::string("field is not an array: ") + key);
    }
    result.reserve(it->size());
    for (const json& item : *it) {
        if (!item.is_number_unsigned() && !item.is_number_integer()) {
            throw std::runtime_error(std::string("array item is not an integer: ") + key);
        }
        result.push_back(item.get<std::uint32_t>());
    }
    return result;
}

std::vector<std::uint64_t> u64_array(const json& value, const char* key) {
    std::vector<std::uint64_t> result;
    const auto it = value.find(key);
    if (it == value.end() || it->is_null()) {
        return result;
    }
    if (!it->is_array()) {
        throw std::runtime_error(std::string("field is not an array: ") + key);
    }
    result.reserve(it->size());
    for (const json& item : *it) {
        if (!item.is_number_unsigned() && !item.is_number_integer()) {
            throw std::runtime_error(std::string("array item is not an integer: ") + key);
        }
        result.push_back(item.get<std::uint64_t>());
    }
    return result;
}

go::GameRecord parse_record(const json& value) {
    go::GameRecord record;
    record.id = required_string(value, "id");
    record.source_path = required_string(value, "source_path");
    record.source_sha256 = required_string(value, "source_sha256");
    record.board_size = static_cast<int>(uint_field(value, "board_size", 19));
    record.komi_twice = int_field(value, "komi_twice", 11);
    record.handicap = static_cast<int>(uint_field(value, "handicap", 0));
    record.to_move = string_field(value, "to_move", "b") == "w"
        ? go::Stone::white
        : go::Stone::black;
    record.outcome = go::decode_outcome(string_field(value, "outcome", "unknown"));
    record.result = string_field(value, "result");
    record.rules = string_field(value, "rules", "japanese");
    record.black_player = string_field(value, "black_player");
    record.white_player = string_field(value, "white_player");
    record.event = string_field(value, "event");
    record.date = string_field(value, "date");
    record.setup_black = u32_array(value, "setup_black");
    record.setup_white = u32_array(value, "setup_white");
    record.setup_empty = u32_array(value, "setup_empty");
    record.moves = u64_array(value, "moves");
    return record;
}

json record_json(const go::GameRecord& record) {
    return json{
        {"id", record.id},
        {"source_path", record.source_path},
        {"source_sha256", record.source_sha256},
        {"board_size", record.board_size},
        {"komi_twice", record.komi_twice},
        {"handicap", record.handicap},
        {"to_move", record.to_move == go::Stone::white ? "w" : "b"},
        {"outcome", go::encode_outcome(record.outcome)},
        {"result", record.result},
        {"rules", record.rules},
        {"black_player", record.black_player},
        {"white_player", record.white_player},
        {"event", record.event},
        {"date", record.date},
        {"setup_black", record.setup_black},
        {"setup_white", record.setup_white},
        {"setup_empty", record.setup_empty},
        {"moves", record.moves},
    };
}

std::string option_value(int& index, int argc, char** argv) {
    if (++index >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index - 1]);
    }
    return argv[index];
}

PackOptions parse_pack_options(int argc, char** argv) {
    PackOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--input") {
            options.input = option_value(i, argc, argv);
        } else if (arg == "--output") {
            options.output = option_value(i, argc, argv);
        } else if (arg == "--source-name") {
            options.source.name = option_value(i, argc, argv);
        } else if (arg == "--source-url") {
            options.source.url = option_value(i, argc, argv);
        } else if (arg == "--archive-sha256") {
            options.source.archive_sha256 = option_value(i, argc, argv);
        } else if (arg == "--license") {
            options.source.license = option_value(i, argc, argv);
        } else if (arg == "--retrieved-at") {
            options.source.retrieved_at = std::stoull(option_value(i, argc, argv));
        } else if (arg == "--created-at") {
            options.created_at = std::stoull(option_value(i, argc, argv));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(arg));
        }
    }
    if (options.input.empty() || options.output.empty()) {
        throw std::invalid_argument("pack requires --input and --output");
    }
    if (options.source.name.empty()) {
        throw std::invalid_argument("pack requires --source-name");
    }
    if (options.created_at == 0) {
        options.created_at = static_cast<std::uint64_t>(
            std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())
        );
    }
    return options;
}

int pack(int argc, char** argv) {
    const PackOptions options = parse_pack_options(argc, argv);
    std::ifstream input(options.input);
    if (!input) {
        throw std::runtime_error("cannot open JSONL input: " + options.input);
    }

    go::GameDataset dataset;
    dataset.source = options.source;
    dataset.created_at = options.created_at;
    std::unordered_set<std::string> ids;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }
        try {
            go::GameRecord record = parse_record(json::parse(line));
            if (!ids.insert(record.id).second) {
                throw std::runtime_error("duplicate game id: " + record.id);
            }
            dataset.games.push_back(std::move(record));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                options.input + ":" + std::to_string(line_number) + ": " + error.what()
            );
        }
    }
    if (dataset.games.empty()) {
        throw std::runtime_error("refusing to write an empty .nngames file");
    }
    go::save_nngames(dataset, options.output);
    std::cout << json{
        {"output", options.output},
        {"games", dataset.games.size()},
    }.dump() << '\n';
    return 0;
}

std::string input_path(int argc, char** argv) {
    for (int i = 2; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--input") {
            return option_value(i, argc, argv);
        }
    }
    throw std::invalid_argument("command requires --input");
}

int stats(int argc, char** argv) {
    const std::string path = input_path(argc, argv);
    std::ifstream probe(path, std::ios::binary | std::ios::ate);
    if (!probe) {
        throw std::runtime_error("cannot open .nngames file: " + path);
    }
    const go::GameDataset dataset = go::load_nngames(path);
    std::size_t moves = 0;
    std::uint64_t positions = 0;
    for (const auto& game : dataset.games) {
        moves += game.moves.size();
        positions += game.moves.size() + 1;
    }
    std::cout << json{
        {"file", path},
        {"bytes", std::filesystem::file_size(path)},
        {"games", dataset.games.size()},
        {"moves", moves},
        {"positions", positions},
        {"source", {
            {"name", dataset.source.name},
            {"url", dataset.source.url},
            {"archive_sha256", dataset.source.archive_sha256},
            {"license", dataset.source.license},
        }},
    }.dump() << '\n';
    return 0;
}

int verify(int argc, char** argv) {
    std::string path;
    std::optional<std::size_t> sample;
    std::uint64_t seed = 0;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--input") {
            path = option_value(i, argc, argv);
        } else if (arg == "--sample") {
            sample = static_cast<std::size_t>(std::stoull(option_value(i, argc, argv)));
        } else if (arg == "--seed") {
            seed = std::stoull(option_value(i, argc, argv));
        } else {
            throw std::invalid_argument("unknown verify option: " + std::string(arg));
        }
    }
    if (path.empty()) {
        throw std::invalid_argument("verify requires --input");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open .nngames file: " + path);
    }
    const std::string bytes{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    };
    const go::GameDataset dataset = go::parse_nngames(bytes);
    const bool byte_identical = go::serialize_nngames(dataset) == bytes;
    const go::ReplaySummary replay = go::verify_replay(dataset, sample, seed);
    std::cout << json{
        {"games", dataset.games.size()},
        {"byte_identical", byte_identical},
        {"sampled_games", replay.sampled_games},
        {"replayed_moves", replay.replayed_moves},
        {"setup_stones", replay.setup_stones},
        {"captures", replay.captures},
        {"final_state_hash", replay.final_state_hash},
    }.dump() << '\n';
    if (!byte_identical) {
        throw std::runtime_error("nngames serialization is not byte-reversible");
    }
    return 0;
}

int export_game(int argc, char** argv) {
    std::string path;
    std::optional<std::size_t> index;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--input") {
            path = option_value(i, argc, argv);
        } else if (arg == "--index") {
            index = static_cast<std::size_t>(std::stoull(option_value(i, argc, argv)));
        } else {
            throw std::invalid_argument("unknown export option: " + std::string(arg));
        }
    }
    if (path.empty() || !index.has_value()) {
        throw std::invalid_argument("export requires --input and --index");
    }
    const go::GameDataset dataset = go::load_nngames(path);
    if (*index >= dataset.games.size()) {
        throw std::out_of_range("game index outside dataset");
    }
    std::cout << record_json(dataset.games[*index]).dump() << '\n';
    return 0;
}

std::string vocab_option_output(int argc, char** argv) {
    for (int i = 2; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--output") {
            return option_value(i, argc, argv);
        }
    }
    throw std::invalid_argument("command requires --output");
}

std::vector<std::string> go_vocab() {
    std::vector<std::string> vocab;
    vocab.reserve(367);
    for (int y = 0; y < 19; ++y) {
        for (int x = 0; x < 19; ++x) {
            std::ostringstream name;
            name << "P_" << std::setw(2) << std::setfill('0') << x
                 << '_' << std::setw(2) << y;
            vocab.push_back(name.str());
        }
    }
    vocab.push_back("<pass>");
    vocab.push_back("<bos>");
    vocab.push_back("<eos>");
    vocab.push_back("<decision>");
    vocab.push_back("<pad>");
    vocab.push_back("<unk>");
    return vocab;
}

int write_vocab(int argc, char** argv) {
    const std::string output = vocab_option_output(argc, argv);
    const std::vector<std::string> vocab = go_vocab();
    nn::KeyValueRecord record;
    record.set("type", "go");
    record.set("version", static_cast<std::uint64_t>(1));
    record.set("max_board_size", static_cast<std::uint64_t>(19));
    record.set("vocab", nn::encode_vocab_blob(vocab));
    record.set("merges", std::vector<std::uint64_t>{});
    record.set("markers", std::vector<std::uint64_t>{362, 363, 364, 361, 365, 366});
    const std::string bytes = nn::write_nnvocab(record);

    const std::string temporary = output + ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("cannot create vocabulary output: " + temporary);
        }
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!stream) {
            throw std::runtime_error("cannot write vocabulary output: " + temporary);
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, output, error);
    if (error) {
        std::filesystem::remove(output, error);
        error.clear();
        std::filesystem::rename(temporary, output, error);
    }
    if (error) {
        throw std::runtime_error("cannot replace vocabulary output: " + error.message());
    }
    std::cout << json{
        {"output", output},
        {"type", "go"},
        {"vocab_size", vocab.size()},
    }.dump() << '\n';
    return 0;
}

int inspect_vocab(int argc, char** argv) {
    const std::string path = input_path(argc, argv);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open vocabulary: " + path);
    }
    const std::string bytes{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    };
    auto parsed = nn::parse_nnvocab(bytes);
    if (!parsed) {
        throw std::runtime_error("cannot parse vocabulary: " + parsed.error().message);
    }
    const std::string type = nn_required_string(*parsed, "type");
    const std::string blob = nn_required_string(*parsed, "vocab");
    auto decoded_vocab = nn::decode_vocab_blob(blob);
    if (!decoded_vocab) {
        throw std::runtime_error("cannot decode vocabulary blob: " + decoded_vocab.error().message);
    }
    const std::vector<std::string>& vocab = *decoded_vocab;
    std::cout << json{
        {"file", path},
        {"type", type},
        {"version", nn_required_uint(*parsed, "version")},
        {"max_board_size", nn_required_uint(*parsed, "max_board_size")},
        {"vocab_size", vocab.size()},
    }.dump() << '\n';
    return 0;
}

void usage(const char* program) {
    std::cerr
        << "usage:\n"
        << "  " << program << " pack --input JSONL --output FILE --source-name NAME [...]\n"
        << "  " << program << " stats --input FILE\n"
        << "  " << program << " verify --input FILE [--sample N --seed S]\n"
        << "  " << program << " export --input FILE --index N\n"
        << "  " << program << " write-vocab --output FILE\n"
        << "  " << program << " inspect-vocab --input FILE\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage(argv[0]);
            return 2;
        }
        const std::string_view command = argv[1];
        if (command == "pack") {
            return pack(argc, argv);
        }
        if (command == "stats") {
            return stats(argc, argv);
        }
        if (command == "verify") {
            return verify(argc, argv);
        }
        if (command == "export") {
            return export_game(argc, argv);
        }
        if (command == "write-vocab") {
            return write_vocab(argc, argv);
        }
        if (command == "inspect-vocab") {
            return inspect_vocab(argc, argv);
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
