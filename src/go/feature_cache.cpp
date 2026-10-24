#include "go/feature_cache.hpp"

#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace go {
namespace {

#pragma pack(push, 1)
struct CacheHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint64_t record_count;
    std::uint32_t feature_count;
    std::uint32_t board_size;
    std::uint8_t reserved[24];
};

struct CacheRecordHeader {
    std::uint64_t sample_id;
    std::uint32_t game_index;
    std::uint32_t move_number;
    std::uint16_t target;
    std::int8_t value;
    std::uint8_t flags;
    std::uint8_t reserved[4];
};
#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<CacheHeader>);
static_assert(std::is_trivially_copyable_v<CacheRecordHeader>);
static_assert(sizeof(CacheHeader) == feature_cache_header_size);
static_assert(sizeof(CacheRecordHeader) == 24);
static_assert(sizeof(CacheRecordHeader) + feature_plane_count *
              training_board_size * training_board_size * sizeof(float) ==
              feature_cache_record_size);
static_assert(std::endian::native == std::endian::little,
              "NGFEAT01 v1 is little-endian");

std::string checked_path(const std::filesystem::path& path) {
    return path.string();
}

}  // namespace

FeatureCache::FeatureCache(
    std::filesystem::path path,
    std::ifstream stream,
    std::uint64_t data_offset,
    std::size_t feature_count,
    std::uint32_t board_size,
    std::size_t sample_count
)
    : path_(std::move(path)),
      stream_(std::move(stream)),
      data_offset_(data_offset),
      feature_count_(feature_count),
      board_size_(board_size),
      sample_count_(sample_count) {}

FeatureCache FeatureCache::open(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open feature cache: " + checked_path(path));
    }

    CacheHeader header{};
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (stream.gcount() != static_cast<std::streamsize>(sizeof(header))) {
        throw std::runtime_error("truncated feature cache header: " + checked_path(path));
    }
    if (std::memcmp(header.magic, feature_cache_magic.data(), sizeof(header.magic)) != 0) {
        throw std::runtime_error("invalid feature cache magic: " + checked_path(path));
    }
    if (header.version != feature_cache_version) {
        throw std::runtime_error(
            "unsupported feature cache version " + std::to_string(header.version)
        );
    }
    if (header.header_size != feature_cache_header_size) {
        throw std::runtime_error("unsupported feature cache header size");
    }
    if (header.feature_count != feature_plane_count *
            training_board_size * training_board_size) {
        throw std::runtime_error("unexpected feature count in cache");
    }
    if (header.board_size != training_board_size) {
        throw std::runtime_error("M4 feature cache must contain 19x19 samples");
    }
    if (header.record_count == 0 ||
        header.record_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("invalid feature cache record count");
    }
    const std::uint64_t expected_size =
        feature_cache_header_size +
        header.record_count * static_cast<std::uint64_t>(feature_cache_record_size);
    const std::uint64_t actual_size = std::filesystem::file_size(path);
    if (actual_size != expected_size) {
        throw std::runtime_error(
            "feature cache size mismatch: expected " + std::to_string(expected_size) +
            ", got " + std::to_string(actual_size)
        );
    }

    return FeatureCache(
        path,
        std::move(stream),
        feature_cache_header_size,
        header.feature_count,
        header.board_size,
        static_cast<std::size_t>(header.record_count)
    );
}

FeatureSampleMeta FeatureCache::read_header(std::size_t index) {
    if (index >= sample_count_) {
        throw std::out_of_range("feature cache sample index outside range");
    }
    const std::uint64_t offset =
        data_offset_ + static_cast<std::uint64_t>(index) * feature_cache_record_size;
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    CacheRecordHeader record{};
    stream_.read(reinterpret_cast<char*>(&record), sizeof(record));
    if (!stream_) {
        throw std::runtime_error("truncated feature cache record header");
    }
    if (record.target >= policy_class_count) {
        throw std::runtime_error("feature cache target outside policy classes");
    }
    if (record.value < -1 || record.value > 1) {
        throw std::runtime_error("feature cache value outside [-1,1]");
    }
    return {
        record.sample_id,
        record.game_index,
        record.move_number,
        record.target,
        record.value,
        record.flags,
    };
}

void FeatureCache::read_planes(std::size_t index, std::span<float> output) {
    read_sample(index, output);
}

void FeatureCache::read_sample(
    std::size_t index,
    std::span<float> output,
    FeatureSampleMeta* metadata
) {
    if (output.size() != feature_count_) {
        throw std::invalid_argument("feature cache output span has wrong size");
    }
    const FeatureSampleMeta value = read_header(index);
    if (metadata != nullptr) {
        *metadata = value;
    }
    const std::uint64_t offset =
        data_offset_ + static_cast<std::uint64_t>(index) * feature_cache_record_size +
        sizeof(CacheRecordHeader);
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!stream_) {
        throw std::runtime_error("cannot seek feature cache sample");
    }
    stream_.read(
        reinterpret_cast<char*>(output.data()),
        static_cast<std::streamsize>(output.size_bytes())
    );
    if (stream_.gcount() != static_cast<std::streamsize>(output.size_bytes())) {
        throw std::runtime_error("truncated feature cache sample");
    }
}

FeatureSampleMeta FeatureCache::meta(std::size_t index) {
    return read_header(index);
}

}  // namespace go
