#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace go {

inline constexpr std::string_view feature_cache_magic = "NGFEAT01";
inline constexpr std::uint32_t feature_cache_version = 1;
inline constexpr std::uint32_t feature_cache_header_size = 56;
inline constexpr std::uint32_t feature_cache_record_size = 13020;
inline constexpr std::uint32_t feature_plane_count = 9;
inline constexpr std::uint32_t training_board_size = 19;
inline constexpr std::uint32_t policy_class_count = 362;

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

    [[nodiscard]] std::size_t size() const noexcept { return samples_.size(); }
    [[nodiscard]] std::size_t feature_count() const noexcept { return feature_count_; }
    [[nodiscard]] std::uint32_t board_size() const noexcept { return board_size_; }
    [[nodiscard]] const FeatureSampleMeta& meta(std::size_t index) const;
    void read_planes(std::size_t index, std::span<float> output);
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    FeatureCache(
        std::filesystem::path path,
        std::ifstream stream,
        std::uint64_t data_offset,
        std::size_t feature_count,
        std::uint32_t board_size,
        std::vector<FeatureSampleMeta> samples
    );

    std::filesystem::path path_;
    std::ifstream stream_;
    std::uint64_t data_offset_ = 0;
    std::size_t feature_count_ = 0;
    std::uint32_t board_size_ = 0;
    std::vector<FeatureSampleMeta> samples_;
};

}  // namespace go
