#include "go/features.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace go {
namespace {

constexpr float kEmpty = 0.0F;
constexpr float kOne = 1.0F;

}  // namespace

void extract_features(
    const Game& game,
    Stone view_color,
    std::span<float> output
) {
    if (output.size() != feature_element_count) {
        throw std::invalid_argument("feature output span has wrong size");
    }
    if (view_color == Stone::empty) {
        throw std::invalid_argument("feature view color must be black or white");
    }
    std::fill(output.begin(), output.end(), kEmpty);

    const int size = game.width();
    if (size != static_cast<int>(training_board_size) ||
        game.height() != static_cast<int>(training_board_size)) {
        throw std::invalid_argument("feature extraction requires a 19x19 board");
    }

    const Stone opponent =
        view_color == Stone::black ? Stone::white : Stone::black;

    // plane 0/1/7：行棋方 / 对方 / 空点
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const std::size_t cell =
                static_cast<std::size_t>(y) * size + static_cast<std::size_t>(x);
            const Stone stone = game.at(Point{x, y});
            if (stone == view_color) {
                output[0 * feature_plane_stride + cell] = kOne;
            } else if (stone == opponent) {
                output[1 * feature_plane_stride + cell] = kOne;
            } else {
                output[7 * feature_plane_stride + cell] = kOne;
            }
        }
    }

    // plane 2：白走填充
    if (view_color == Stone::white) {
        std::fill_n(
            output.begin() + 2 * feature_plane_stride,
            feature_plane_stride,
            kOne
        );
    }

    // plane 3：简单劫禁入点
    if (const auto ko = game.ko_point(); ko.has_value()) {
        output[3 * feature_plane_stride +
               static_cast<std::size_t>(ko->y) * size + ko->x] = kOne;
    }

    // plane 4..6：最近三手（chronological，plane 4 = 倒数第三手）。pass
    // 占用槽位但不置位——与 Python 的 `if point is None: continue` 一致。
    const std::vector<PlayedMove> moves = game.history();
    const std::size_t window = std::min<std::size_t>(3, moves.size());
    for (std::size_t offset = 0; offset < window; ++offset) {
        const PlayedMove& move = moves[moves.size() - window + offset];
        if (!move.point.has_value()) {
            continue;
        }
        output[(4 + offset) * feature_plane_stride +
               static_cast<std::size_t>(move.point->y) * size +
               move.point->x] = kOne;
    }

    // plane 8：边界距离（归一化与 Python 逐位一致）
    const float normalizer = std::max(1.0F, (size - 1) / 2.0F);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const float distance = static_cast<float>(std::min(
                std::min(x, y),
                std::min(size - 1 - x, size - 1 - y)
            ));
            output[8 * feature_plane_stride +
                   static_cast<std::size_t>(y) * size + x] = distance / normalizer;
        }
    }
}

void extract_features(const Game& game, std::span<float> output) {
    extract_features(game, game.to_move(), output);
}

}  // namespace go
