#pragma once

#include <cstddef>
#include <span>

#include "go/feature_cache.hpp"  // feature_plane_count / training_board_size
#include "go/game.hpp"

namespace go {

// M3 v1 特征平面的 C++ 提取器（M6）：GTP 适配器与 Arena 需要在对局中实时
// 把当前盘面喂给策略模型，不能经 Python/JSON 往返。本实现与训练管线的
// `python/go_pipeline/features.py::extract_features` 逐位一致（一致性由
// `go_rules_cli features` 与 tests/test_m6_features.py 对拍保证）。
//
// 平面顺序（9 平面，19×19）：
//   0 own_stones        行棋方棋子
//   1 opponent_stones   对方棋子
//   2 white_to_play     全盘填充（白走=1，黑走=0）
//   3 simple_ko_ban     简单劫禁入点
//   4..6 recent moves   最近三手（plane 4 = 倒数第三手，plane 6 = 最近一手；
//                       pass 占位但不置位，与 Python 一致）
//   7 empty             空点
//   8 border_distance   min(x, y, W-1-x, H-1-y) / ((size-1)/2)
//
// 布局与 `.ngf` 训练缓存一致：plane-major，平面内 row-major（y 行自顶向下，
// 索引 y*size + x），共 9*19*19 个 float。
inline constexpr std::size_t feature_plane_stride =
    static_cast<std::size_t>(training_board_size) * training_board_size;
inline constexpr std::size_t feature_element_count =
    static_cast<std::size_t>(feature_plane_count) * feature_plane_stride;

// 提取 `view_color` 视角下的 9 平面特征。`output` 必须恰好容纳
// feature_element_count 个 float。
void extract_features(
    const Game& game,
    Stone view_color,
    std::span<float> output
);

// 以当前行棋方为视角的便捷重载。
void extract_features(const Game& game, std::span<float> output);

}  // namespace go
