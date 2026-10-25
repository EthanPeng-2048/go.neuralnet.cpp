#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>

namespace go {

inline constexpr std::string_view feature_cache_magic = "NGFEAT01";
// M7 v2：加入每样本最近 K 手落子历史（u16 点位 id，meta 之后、planes 之前）。
// v1（无历史）保持可读：读取器按 version 分派 stride，v1 路径逐字节不变。
inline constexpr std::uint32_t feature_cache_version = 2;
inline constexpr std::uint32_t feature_cache_version_legacy = 1;
inline constexpr std::uint32_t feature_cache_header_size = 56;
inline constexpr std::uint32_t feature_cache_record_size_v1 = 13020;
inline constexpr std::uint32_t feature_plane_count = 9;
inline constexpr std::uint32_t training_board_size = 19;
inline constexpr std::uint32_t policy_class_count = 362;
// v2 头 reserved 起点（offset 32）写 u32 history_len + u32 history_vocab。
inline constexpr std::uint32_t feature_cache_default_history_len = 64;
inline constexpr std::uint32_t feature_cache_history_vocab = 367;

// v2 记录 stride = 24（meta）+ 2·K（历史 u16）+ 12996（planes）。
constexpr std::uint32_t feature_cache_record_size_v2(std::uint32_t history_len) {
    return 24 + 2 * history_len +
           feature_plane_count * training_board_size * training_board_size * 4;
}

struct FeatureSampleMeta {
    std::uint64_t sample_id = 0;
    std::uint32_t game_index = 0;
    std::uint32_t move_number = 0;
    std::uint16_t target = 0;
    std::int8_t value = 0;
    std::uint8_t flags = 0;
};

class FeatureCache {
public:
    static FeatureCache open(const std::filesystem::path& path);

    [[nodiscard]] std::size_t size() const noexcept { return sample_count_; }
    [[nodiscard]] std::size_t feature_count() const noexcept { return feature_count_; }
    [[nodiscard]] std::uint32_t board_size() const noexcept { return board_size_; }
    // v1 缓存为 0（无历史）；B 模型训练要求 > 0 并按模型 K 校验。
    [[nodiscard]] std::uint32_t history_len() const noexcept { return history_len_; }
    [[nodiscard]] std::uint32_t history_vocab() const noexcept {
        return history_vocab_;
    }
    [[nodiscard]] std::uint32_t format_version() const noexcept {
        return format_version_;
    }
    [[nodiscard]] std::uint32_t record_stride() const noexcept {
        return record_stride_;
    }
    FeatureSampleMeta meta(std::size_t index);
    void read_planes(std::size_t index, std::span<float> output);
    // `history` 可选：v2 缓存填 K 个点位 id（左 <pad> 补齐、右对齐，行 0 =
    // 最旧槽位）；v1 缓存收到非空 history 报错（禁止静默空历史）。
    void read_sample(
        std::size_t index,
        std::span<float> output,
        FeatureSampleMeta* metadata = nullptr,
        std::span<std::uint16_t> history = {}
    );
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    FeatureCache(
        std::filesystem::path path,
        std::ifstream stream,
        std::uint64_t data_offset,
        std::size_t feature_count,
        std::uint32_t board_size,
        std::size_t sample_count,
        std::uint32_t format_version,
        std::uint32_t history_len,
        std::uint32_t history_vocab,
        std::uint32_t record_stride
    );

    std::filesystem::path path_;
    std::ifstream stream_;
    std::uint64_t data_offset_ = 0;
    std::size_t feature_count_ = 0;
    std::uint32_t board_size_ = 0;
    std::size_t sample_count_ = 0;
    std::uint32_t format_version_ = feature_cache_version_legacy;
    std::uint32_t history_len_ = 0;
    std::uint32_t history_vocab_ = 0;
    std::uint32_t record_stride_ = feature_cache_record_size_v1;

    [[nodiscard]] FeatureSampleMeta read_header(std::size_t index);
};

}  // namespace go
