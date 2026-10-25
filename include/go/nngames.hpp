#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "go/game.hpp"

namespace go {

inline constexpr std::string_view nngames_magic = "NNGM0001";
inline constexpr std::uint64_t nngames_format_version = 1;

enum class GameOutcome : std::uint8_t {
    unknown = 0,
    black = 1,
    white = 2,
    draw = 3,
};

struct DatasetSource {
    std::string name;
    std::string url;
    std::string archive_sha256;
    std::string license;
    std::uint64_t retrieved_at = 0;
};

struct GameRecord {
    std::string id;
    std::string source_path;
    std::string source_sha256;
    int board_size = 19;
    std::int32_t komi_twice = 11;
    int handicap = 0;
    Stone to_move = Stone::black;
    GameOutcome outcome = GameOutcome::unknown;
    std::string result;
    std::string rules = "japanese";
    std::string black_player;
    std::string white_player;
    std::string event;
    std::string date;
    std::vector<std::uint32_t> setup_black;
    std::vector<std::uint32_t> setup_white;
    std::vector<std::uint32_t> setup_empty;
    // Encoded as point * 2 + (white ? 1 : 0), with point 361 meaning pass.
    std::vector<std::uint64_t> moves;
};

struct GameDataset {
    DatasetSource source;
    std::uint64_t created_at = 0;
    std::vector<GameRecord> games;
};

// 重放校验的行为开关。默认值 = 原语义：单线程、首个失败即抛。
struct ReplayOptions {
    // 并发重放的线程数（1 = 串行）。每局棋相互独立，按局分配；
    // final_state_hash 仍按重放序串行折叠 ⇒ 与线程数逐位无关。
    std::size_t threads = 1;
    // true = 不可重放的棋局不抛错，记入 summary.skipped（供上层
    // 「坏局直接抛弃」策略剔除后重新打包）。
    bool skip_unreplayable = false;
};

struct SkippedReplay {
    std::size_t index = 0;  // dataset 下标
    std::string id;
    std::string reason;
};

struct ReplaySummary {
    std::size_t sampled_games = 0;
    std::size_t replayed_moves = 0;
    std::size_t setup_stones = 0;
    std::size_t captures = 0;
    std::uint64_t final_state_hash = 1469598103934665603ULL;
    // 仅 skip_unreplayable 模式下非空；严格模式下失败直接抛错。
    std::vector<SkippedReplay> skipped;
};

[[nodiscard]] std::uint64_t encode_move(Stone color, std::optional<Point> point);
[[nodiscard]] PlayedMove decode_move(std::uint64_t token);
[[nodiscard]] std::string encode_outcome(GameOutcome outcome);
[[nodiscard]] GameOutcome decode_outcome(std::string_view text);

[[nodiscard]] std::string serialize_nngames(const GameDataset& dataset);
[[nodiscard]] GameDataset parse_nngames(std::string_view bytes);
[[nodiscard]] GameDataset load_nngames(const std::string& path);
void save_nngames(const GameDataset& dataset, const std::string& path);

[[nodiscard]] ReplaySummary verify_replay(
    const GameDataset& dataset,
    std::optional<std::size_t> random_sample,
    std::uint64_t seed,
    const ReplayOptions& options = {}
);

}  // namespace go
