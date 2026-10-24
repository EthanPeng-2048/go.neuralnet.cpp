#include "go/game.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "game/board.h"
#include "game/boardhistory.h"
#include "game/rules.h"

namespace go {
namespace {

void initialize_katago() {
    static std::once_flag once;
    std::call_once(once, [] {
        Board::initHash();
    });
}

Player to_katago_player(Stone stone) {
    switch (stone) {
    case Stone::black:
        return P_BLACK;
    case Stone::white:
        return P_WHITE;
    case Stone::empty:
        break;
    }
    throw std::invalid_argument("a move requires black or white");
}

Stone from_katago_color(Color color) {
    switch (color) {
    case C_BLACK:
        return Stone::black;
    case C_WHITE:
        return Stone::white;
    case C_EMPTY:
        return Stone::empty;
    default:
        throw std::invalid_argument("KataGo returned a wall color for a playable point");
    }
}

Loc to_katago_loc(const Board& board, std::optional<Point> point) {
    if (!point.has_value()) {
        return Board::PASS_LOC;
    }
    if (point->x < 0 || point->y < 0 || point->x >= board.x_size || point->y >= board.y_size) {
        return Board::NULL_LOC;
    }
    return Location::getLoc(point->x, point->y, board.x_size);
}

::Rules to_katago_rules(const RuleSet& source) {
    ::Rules result;
    switch (source.ko_rule) {
    case KoRule::simple:
        result.koRule = ::Rules::KO_SIMPLE;
        break;
    case KoRule::positional:
        result.koRule = ::Rules::KO_POSITIONAL;
        break;
    case KoRule::situational:
        result.koRule = ::Rules::KO_SITUATIONAL;
        break;
    case KoRule::spight:
        result.koRule = ::Rules::KO_SPIGHT;
        break;
    }
    switch (source.scoring) {
    case ScoringMethod::area:
        result.scoringRule = ::Rules::SCORING_AREA;
        break;
    case ScoringMethod::territory:
        result.scoringRule = ::Rules::SCORING_TERRITORY;
        break;
    }
    result.taxRule = ::Rules::TAX_NONE;
    result.multiStoneSuicideLegal = source.multi_stone_suicide_legal;
    result.friendlyPassOk = source.friendly_pass_ok;
    result.komi = source.komi;
    return result;
}

std::vector<std::string_view> split_rows(std::string_view text) {
    std::vector<std::string_view> rows;
    while (!text.empty()) {
        const std::size_t newline = text.find('\n');
        std::string_view row = text.substr(0, newline);
        if (!row.empty() && row.back() == '\r') {
            row.remove_suffix(1);
        }
        if (!row.empty()) {
            rows.push_back(row);
        }
        if (newline == std::string_view::npos) {
            break;
        }
        text.remove_prefix(newline + 1);
    }
    return rows;
}

}  // namespace

struct Game::Impl {
    Impl(Board board, Player player, const ::Rules& rules)
        : board(std::move(board)),
          history(this->board, player, rules, 0, BoardHistoryModes(false, false)) {}

    Board board;
    BoardHistory history;
};

RuleSet RuleSet::chinese(float komi) {
    return RuleSet{
        .ko_rule = KoRule::positional,
        .scoring = ScoringMethod::area,
        .multi_stone_suicide_legal = false,
        .friendly_pass_ok = true,
        .komi = komi,
    };
}

RuleSet RuleSet::japanese(float komi) {
    return RuleSet{
        .ko_rule = KoRule::situational,
        .scoring = ScoringMethod::territory,
        .multi_stone_suicide_legal = false,
        .friendly_pass_ok = false,
        .komi = komi,
    };
}

Game::Game(int width, int height, Stone to_move, RuleSet rules) {
    initialize_katago();
    if (width <= 0 || height <= 0 || width > Board::MAX_LEN || height > Board::MAX_LEN) {
        throw std::invalid_argument("invalid board size");
    }
    impl_ = std::make_unique<Impl>(
        Board(width, height),
        to_katago_player(to_move),
        to_katago_rules(rules)
    );
}

Game::Game(const Game& other)
    : impl_(other.impl_ ? std::make_unique<Impl>(*other.impl_) : nullptr) {}

Game::Game(Game&& other) noexcept = default;

Game& Game::operator=(const Game& other) {
    if (this != &other) {
        impl_ = other.impl_ ? std::make_unique<Impl>(*other.impl_) : nullptr;
    }
    return *this;
}

Game& Game::operator=(Game&& other) noexcept = default;

Game::~Game() = default;

Game Game::from_ascii(std::string_view rows, Stone to_move, RuleSet rules) {
    initialize_katago();
    const std::vector<std::string_view> lines = split_rows(rows);
    if (lines.empty()) {
        throw std::invalid_argument("position must contain at least one row");
    }

    const int width = static_cast<int>(lines.front().size());
    const int height = static_cast<int>(lines.size());
    if (width > Board::MAX_LEN || height > Board::MAX_LEN) {
        throw std::invalid_argument("position exceeds KataGo's compile-time board limit");
    }

    std::string normalized;
    normalized.reserve(static_cast<std::size_t>(width) * height + lines.size());
    for (std::size_t y = 0; y < lines.size(); ++y) {
        if (static_cast<int>(lines[y].size()) != width) {
            throw std::invalid_argument("all position rows must have the same width");
        }
        for (const char raw : lines[y]) {
            const char point = static_cast<char>(std::tolower(static_cast<unsigned char>(raw)));
            if (point == '.') {
                normalized.push_back('.');
            } else if (point == 'x') {
                normalized.push_back('x');
            } else if (point == 'o') {
                normalized.push_back('o');
            } else {
                throw std::invalid_argument("ASCII positions accept only '.', 'x', and 'o'");
            }
        }
        if (y + 1 < lines.size()) {
            normalized.push_back('\n');
        }
    }

    Game game;
    game.impl_ = std::make_unique<Impl>(
        Board::parseBoard(width, height, normalized),
        to_katago_player(to_move),
        to_katago_rules(rules)
    );
    return game;
}

int Game::width() const noexcept {
    return impl_->board.x_size;
}

int Game::height() const noexcept {
    return impl_->board.y_size;
}

Stone Game::at(Point point) const {
    return from_katago_color(impl_->board.colors[to_katago_loc(impl_->board, point)]);
}

Stone Game::to_move() const noexcept {
    return from_katago_color(static_cast<Color>(impl_->history.presumedNextMovePla));
}

std::size_t Game::move_count() const noexcept {
    return impl_->history.moveHistory.size();
}

bool Game::game_over() const noexcept {
    return impl_->history.isGameFinished;
}

std::optional<Point> Game::ko_point() const noexcept {
    if (impl_->board.ko_loc == Board::NULL_LOC) {
        return std::nullopt;
    }
    return Point{
        Location::getX(impl_->board.ko_loc, width()),
        Location::getY(impl_->board.ko_loc, width()),
    };
}

int Game::captured_count(Stone color) const noexcept {
    if (color == Stone::black) {
        return impl_->board.numBlackCaptures;
    }
    if (color == Stone::white) {
        return impl_->board.numWhiteCaptures;
    }
    return 0;
}

std::string Game::ascii() const {
    std::string result;
    result.reserve(static_cast<std::size_t>(width()) * height() + height() - 1);
    for (int y = 0; y < height(); ++y) {
        for (int x = 0; x < width(); ++x) {
            switch (at({x, y})) {
            case Stone::black:
                result.push_back('x');
                break;
            case Stone::white:
                result.push_back('o');
                break;
            case Stone::empty:
                result.push_back('.');
                break;
            }
        }
        if (y + 1 < height()) {
            result.push_back('\n');
        }
    }
    return result;
}

std::vector<PlayedMove> Game::history() const {
    std::vector<PlayedMove> result;
    result.reserve(impl_->history.moveHistory.size());
    for (const ::Move& move : impl_->history.moveHistory) {
        if (move.loc == Board::PASS_LOC) {
            result.push_back(PlayedMove{from_katago_color(move.pla), std::nullopt});
        } else {
            result.push_back(PlayedMove{
                from_katago_color(move.pla),
                Point{
                    Location::getX(move.loc, width()),
                    Location::getY(move.loc, width()),
                },
            });
        }
    }
    return result;
}

bool Game::is_legal(Stone color, std::optional<Point> point) const {
    if (game_over()) {
        return false;
    }
    const Loc loc = to_katago_loc(impl_->board, point);
    return loc != Board::NULL_LOC && impl_->history.isLegal(impl_->board, loc, to_katago_player(color));
}

bool Game::play(Stone color, std::optional<Point> point) {
    if (!is_legal(color, point)) {
        return false;
    }
    const Loc loc = to_katago_loc(impl_->board, point);
    impl_->history.makeBoardMoveAssumeLegal(impl_->board, loc, to_katago_player(color), nullptr);
    return true;
}

bool Game::play_tolerant(Stone color, std::optional<Point> point) {
    const Loc loc = to_katago_loc(impl_->board, point);
    if (loc == Board::NULL_LOC) {
        return false;
    }
    return impl_->history.makeBoardMoveTolerant(impl_->board, loc, to_katago_player(color));
}

std::vector<std::optional<Point>> Game::legal_moves() const {
    std::vector<std::optional<Point>> moves;
    moves.reserve(static_cast<std::size_t>(width()) * height() + 1);
    const Stone player = to_move();
    for (int y = 0; y < height(); ++y) {
        for (int x = 0; x < width(); ++x) {
            const Point point{x, y};
            if (is_legal(player, point)) {
                moves.emplace_back(point);
            }
        }
    }
    if (is_legal(player, std::nullopt)) {
        moves.emplace_back(std::nullopt);
    }
    return moves;
}

Score Game::score() const {
    Board scored_board = impl_->board;
    BoardHistory scored_history = impl_->history;
    scored_history.endAndScoreGameNow(scored_board);
    std::optional<Stone> winner;
    if (scored_history.winner == P_BLACK) {
        winner = Stone::black;
    } else if (scored_history.winner == P_WHITE) {
        winner = Stone::white;
    }
    return Score{scored_history.finalWhiteMinusBlackScore, winner};
}

std::vector<Stone> Game::area() const {
    Color native_area[Board::MAX_ARR_SIZE]{};
    impl_->history.getAreaNow(impl_->board, native_area);
    std::vector<Stone> result(static_cast<std::size_t>(width()) * height(), Stone::empty);
    for (int y = 0; y < height(); ++y) {
        for (int x = 0; x < width(); ++x) {
            const Loc loc = Location::getLoc(x, y, width());
            result[static_cast<std::size_t>(y * width() + x)] = from_katago_color(native_area[loc]);
        }
    }
    return result;
}

}  // namespace go
