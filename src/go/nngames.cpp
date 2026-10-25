#include "go/nngames.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "neuralnet.cpp/model_keyvalue_record.hpp"

namespace go {
namespace {

constexpr std::uint32_t first_special_token = 361;
constexpr std::uint64_t move_token_count = 724;
constexpr int max_board_size = 19;

std::string game_key(std::size_t index) {
    std::ostringstream out;
    out << 'g' << std::setw(8) << std::setfill('0') << index;
    return out.str();
}

std::uint64_t required_uint(const nn::KeyValueRecord& record, const std::string& key) {
    std::uint64_t value = 0;
    if (!record.get(key, value)) {
        throw std::runtime_error("missing uint field: " + key);
    }
    return value;
}

std::string required_string(const nn::KeyValueRecord& record, const std::string& key) {
    std::string value;
    if (!record.get(key, value)) {
        throw std::runtime_error("missing string field: " + key);
    }
    return value;
}

std::vector<std::uint64_t> required_array(
    const nn::KeyValueRecord& record,
    const std::string& key
) {
    std::vector<std::uint64_t> value;
    if (!record.get(key, value)) {
        throw std::runtime_error("missing array field: " + key);
    }
    return value;
}

nn::KeyValueRecord required_record(
    const nn::KeyValueRecord& record,
    const std::string& key
) {
    nn::KeyValueRecord value;
    if (!record.get(key, value)) {
        throw std::runtime_error("missing record field: " + key);
    }
    return value;
}

std::vector<std::uint64_t> to_u64_array(const std::vector<std::uint32_t>& values) {
    return std::vector<std::uint64_t>(values.begin(), values.end());
}

std::vector<std::uint32_t> to_u32_array(
    const std::vector<std::uint64_t>& values,
    const std::string& field
) {
    std::vector<std::uint32_t> result;
    result.reserve(values.size());
    for (const std::uint64_t value : values) {
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("field exceeds uint32 range: " + field);
        }
        result.push_back(static_cast<std::uint32_t>(value));
    }
    return result;
}

nn::KeyValueRecord encode_game(const GameRecord& game) {
    nn::KeyValueRecord record;
    record.set("id", game.id);
    record.set("source_path", game.source_path);
    record.set("source_sha256", game.source_sha256);
    record.set("board_size", static_cast<std::uint64_t>(game.board_size));
    record.set("komi_twice", static_cast<std::uint64_t>(
        static_cast<std::int64_t>(game.komi_twice)
    ));
    record.set("handicap", static_cast<std::uint64_t>(game.handicap));
    record.set("to_move", static_cast<std::uint64_t>(
        game.to_move == Stone::white ? 2 : 1
    ));
    record.set("outcome", encode_outcome(game.outcome));
    record.set("result", game.result);
    record.set("rules", game.rules);
    record.set("black_player", game.black_player);
    record.set("white_player", game.white_player);
    record.set("event", game.event);
    record.set("date", game.date);
    record.set("setup_black", to_u64_array(game.setup_black));
    record.set("setup_white", to_u64_array(game.setup_white));
    record.set("setup_empty", to_u64_array(game.setup_empty));
    record.set("moves", game.moves);
    return record;
}

GameRecord decode_game(const nn::KeyValueRecord& record) {
    GameRecord game;
    game.id = required_string(record, "id");
    game.source_path = required_string(record, "source_path");
    game.source_sha256 = required_string(record, "source_sha256");
    game.board_size = static_cast<int>(required_uint(record, "board_size"));
    game.komi_twice = static_cast<std::int32_t>(
        static_cast<std::int64_t>(required_uint(record, "komi_twice"))
    );
    game.handicap = static_cast<int>(required_uint(record, "handicap"));
    game.to_move = required_uint(record, "to_move") == 2 ? Stone::white : Stone::black;
    game.outcome = decode_outcome(required_string(record, "outcome"));
    game.result = required_string(record, "result");
    game.rules = required_string(record, "rules");
    game.black_player = required_string(record, "black_player");
    game.white_player = required_string(record, "white_player");
    game.event = required_string(record, "event");
    game.date = required_string(record, "date");
    game.setup_black = to_u32_array(required_array(record, "setup_black"), "setup_black");
    game.setup_white = to_u32_array(required_array(record, "setup_white"), "setup_white");
    game.setup_empty = to_u32_array(required_array(record, "setup_empty"), "setup_empty");
    game.moves = required_array(record, "moves");
    if (game.board_size <= 0 || game.board_size > max_board_size) {
        throw std::runtime_error("invalid board size in game record: " + game.id);
    }
    return game;
}

nn::KeyValueRecord encode_source(const DatasetSource& source) {
    nn::KeyValueRecord record;
    record.set("name", source.name);
    record.set("url", source.url);
    record.set("archive_sha256", source.archive_sha256);
    record.set("license", source.license);
    record.set("retrieved_at", source.retrieved_at);
    return record;
}

DatasetSource decode_source(const nn::KeyValueRecord& record) {
    DatasetSource source;
    source.name = required_string(record, "name");
    source.url = required_string(record, "url");
    source.archive_sha256 = required_string(record, "archive_sha256");
    source.license = required_string(record, "license");
    source.retrieved_at = required_uint(record, "retrieved_at");
    return source;
}

RuleSet rules_for(const GameRecord& game) {
    const float komi = static_cast<float>(game.komi_twice) / 2.0F;
    if (game.rules == "chinese") {
        return RuleSet::chinese(komi);
    }
    if (game.rules == "japanese") {
        return RuleSet::japanese(komi);
    }
    // Historical records often omit RU. Use the broader territory ruleset for
    // replay while preserving the original "unknown" label in the record.
    return RuleSet::japanese(komi);
}

std::string setup_ascii(const GameRecord& game) {
    const std::size_t point_count =
        static_cast<std::size_t>(game.board_size) * game.board_size;
    std::vector<char> board(point_count, '.');
    auto place = [&](const std::vector<std::uint32_t>& points, char stone) {
        for (const std::uint32_t point : points) {
            const int x = static_cast<int>(point % 19);
            const int y = static_cast<int>(point / 19);
            if (x >= game.board_size || y >= game.board_size) {
                throw std::runtime_error("setup point outside board in game: " + game.id);
            }
            board[static_cast<std::size_t>(y * game.board_size + x)] = stone;
        }
    };
    place(game.setup_black, 'x');
    place(game.setup_white, 'o');
    place(game.setup_empty, '.');

    std::string result;
    result.reserve(point_count + game.board_size - 1);
    for (int y = 0; y < game.board_size; ++y) {
        result.append(
            board.begin() + static_cast<std::ptrdiff_t>(y * game.board_size),
            board.begin() + static_cast<std::ptrdiff_t>((y + 1) * game.board_size)
        );
        if (y + 1 < game.board_size) {
            result.push_back('\n');
        }
    }
    return result;
}

void hash_mix(std::uint64_t& hash, std::uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ULL;
}

}  // namespace

std::uint64_t encode_move(Stone color, std::optional<Point> point) {
    if (color == Stone::empty) {
        throw std::invalid_argument("move color cannot be empty");
    }
    const std::uint64_t point_token = point.has_value()
        ? static_cast<std::uint64_t>(point->y * 19 + point->x)
        : first_special_token;
    const std::uint64_t token = point_token * 2 + (color == Stone::white ? 1 : 0);
    if (token >= move_token_count) {
        throw std::invalid_argument("move token outside 19x19 vocabulary");
    }
    return token;
}

PlayedMove decode_move(std::uint64_t token) {
    if (token >= move_token_count) {
        throw std::invalid_argument("move token outside 19x19 vocabulary");
    }
    PlayedMove move;
    move.color = (token & 1U) ? Stone::white : Stone::black;
    const std::uint64_t point = token >> 1U;
    if (point == first_special_token) {
        move.point = std::nullopt;
    } else {
        move.point = Point{
            static_cast<int>(point % 19),
            static_cast<int>(point / 19),
        };
    }
    return move;
}

std::string encode_outcome(GameOutcome outcome) {
    switch (outcome) {
    case GameOutcome::unknown:
        return "unknown";
    case GameOutcome::black:
        return "black";
    case GameOutcome::white:
        return "white";
    case GameOutcome::draw:
        return "draw";
    }
    throw std::invalid_argument("invalid game outcome");
}

GameOutcome decode_outcome(std::string_view text) {
    if (text == "unknown") {
        return GameOutcome::unknown;
    }
    if (text == "black") {
        return GameOutcome::black;
    }
    if (text == "white") {
        return GameOutcome::white;
    }
    if (text == "draw") {
        return GameOutcome::draw;
    }
    throw std::runtime_error("invalid outcome: " + std::string(text));
}

std::string serialize_nngames(const GameDataset& dataset) {
    nn::KeyValueRecord games;
    for (std::size_t i = 0; i < dataset.games.size(); ++i) {
        games.set(game_key(i), encode_game(dataset.games[i]));
    }

    nn::KeyValueRecord root;
    root.set("format", "go.nngames");
    root.set("format_version", nngames_format_version);
    root.set("created_at", dataset.created_at);
    root.set("source", encode_source(dataset.source));
    root.set("game_count", static_cast<std::uint64_t>(dataset.games.size()));
    root.set("games", games);

    std::string result(nngames_magic);
    result += root.serialize_v2(nngames_magic.size());
    return result;
}

GameDataset parse_nngames(std::string_view bytes) {
    if (bytes.size() < nngames_magic.size() || bytes.substr(0, nngames_magic.size()) != nngames_magic) {
        throw std::runtime_error("not an NNGM v1 file");
    }
    auto parsed = nn::KeyValueRecord::parse_at(
        nn::kvrec_memory_source(bytes),
        nngames_magic.size()
    );
    if (!parsed) {
        throw std::runtime_error("cannot parse .nngames KVRecord: " + parsed.error().message);
    }
    const nn::KeyValueRecord root = std::move(*parsed);
    if (required_string(root, "format") != "go.nngames") {
        throw std::runtime_error("unexpected .nngames format field");
    }
    if (required_uint(root, "format_version") != nngames_format_version) {
        throw std::runtime_error("unsupported .nngames format version");
    }

    GameDataset dataset;
    dataset.created_at = required_uint(root, "created_at");
    dataset.source = decode_source(required_record(root, "source"));
    const std::size_t count = static_cast<std::size_t>(required_uint(root, "game_count"));
    const nn::KeyValueRecord games = required_record(root, "games");
    dataset.games.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        dataset.games.push_back(decode_game(required_record(games, game_key(i))));
    }
    return dataset;
}

GameDataset load_nngames(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open .nngames file: " + path);
    }
    std::string bytes{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    };
    return parse_nngames(bytes);
}

void save_nngames(const GameDataset& dataset, const std::string& path) {
    const std::string bytes = serialize_nngames(dataset);
    const std::string temporary = path + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot create .nngames file: " + temporary);
        }
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!output) {
            throw std::runtime_error("failed writing .nngames file: " + temporary);
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
    }
    if (error) {
        throw std::runtime_error("cannot replace .nngames file: " + error.message());
    }
}

ReplaySummary verify_replay(
    const GameDataset& dataset,
    std::optional<std::size_t> random_sample,
    std::uint64_t seed,
    const ReplayOptions& options
) {
    std::vector<std::size_t> indices(dataset.games.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    if (random_sample.has_value() && *random_sample < indices.size()) {
        std::mt19937_64 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(*random_sample);
    }
    const std::size_t count = indices.size();

    // 单局重放结果槽。每局棋相互独立（Game 的 Impl 归本局所有，
    // KataGo 的 makeBoardMoveTolerant 不碰共享可变状态），可按局并发
    // 填各自的槽；之后按重放序串行折叠，保证汇总值（尤其
    // final_state_hash 的逐字节混合序）与线程数、完成顺序逐位无关。
    struct Slot {
        std::size_t replayed_moves = 0;
        std::size_t setup_stones = 0;
        std::size_t captures = 0;
        std::string final_ascii;
        std::string failure;  // 空 = 该局重放成功
    };
    std::vector<Slot> slots(count);

    auto replay_one = [&dataset, &indices, &slots](std::size_t position) {
        const GameRecord& record = dataset.games[indices[position]];
        Slot& slot = slots[position];
        try {
            Game game = Game::from_ascii(
                setup_ascii(record),
                record.to_move,
                rules_for(record)
            );
            slot.setup_stones = record.setup_black.size()
                + record.setup_white.size() + record.setup_empty.size();

            for (const std::uint64_t token : record.moves) {
                const PlayedMove move = decode_move(token);
                if (!game.play_tolerant(move.color, move.point)) {
                    slot.failure =
                        "cannot replay move "
                        + std::to_string(game.move_count() + 1)
                        + " in game " + record.id;
                    return;
                }
                ++slot.replayed_moves;
            }

            const std::vector<PlayedMove> history = game.history();
            if (history.size() != record.moves.size()) {
                slot.failure = "history length mismatch in game: " + record.id;
                return;
            }
            for (std::size_t i = 0; i < history.size(); ++i) {
                if (encode_move(history[i].color, history[i].point)
                    != record.moves[i]) {
                    slot.failure =
                        "history value mismatch in game: " + record.id;
                    return;
                }
            }
            slot.captures = static_cast<std::size_t>(
                game.captured_count(Stone::black)
                + game.captured_count(Stone::white)
            );
            slot.final_ascii = game.ascii();
        } catch (const std::exception& error) {
            // 严格模式下这条消息会被原样抛出（类型统一为
            // runtime_error）；容错模式下作为该局的剔除原因。
            slot.failure = error.what();
        }
    };

    const std::size_t workers = std::min(options.threads, count);
    if (workers <= 1) {
        for (std::size_t position = 0; position < count; ++position) {
            replay_one(position);
        }
    } else {
        std::atomic<std::size_t> cursor{0};
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (std::size_t worker = 0; worker < workers; ++worker) {
            pool.emplace_back([&] {
                while (true) {
                    const std::size_t position = cursor.fetch_add(1);
                    if (position >= count) {
                        break;
                    }
                    replay_one(position);
                }
            });
        }
        for (std::thread& thread : pool) {
            thread.join();
        }
    }

    ReplaySummary summary;
    summary.sampled_games = count;
    for (std::size_t position = 0; position < count; ++position) {
        Slot& slot = slots[position];
        if (!slot.failure.empty()) {
            if (!options.skip_unreplayable) {
                // 严格模式：抛重放序中第一个失败 ⇒ 与串行逐位同语义。
                throw std::runtime_error(slot.failure);
            }
            summary.skipped.push_back(SkippedReplay{
                indices[position], dataset.games[indices[position]].id,
                slot.failure});
            continue;
        }
        summary.replayed_moves += slot.replayed_moves;
        summary.setup_stones += slot.setup_stones;
        summary.captures += slot.captures;
        for (const unsigned char c : slot.final_ascii) {
            hash_mix(summary.final_state_hash, c);
        }
    }
    return summary;
}

}  // namespace go
