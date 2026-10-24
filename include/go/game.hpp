#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace go {

enum class Stone : unsigned char {
    empty = 0,
    black = 1,
    white = 2,
};

struct Point {
    int x = 0;
    int y = 0;

    friend bool operator==(const Point&, const Point&) = default;
};

enum class KoRule {
    simple,
    positional,
    situational,
    spight,
};

enum class ScoringMethod {
    area,
    territory,
};

struct RuleSet {
    KoRule ko_rule = KoRule::positional;
    ScoringMethod scoring = ScoringMethod::area;
    bool multi_stone_suicide_legal = false;
    bool friendly_pass_ok = true;
    float komi = 7.5F;

    static RuleSet chinese(float komi = 7.5F);
    static RuleSet japanese(float komi = 6.5F);
};

struct Score {
    float white_minus_black = 0.0F;
    std::optional<Stone> winner;
};

struct PlayedMove {
    Stone color = Stone::empty;
    std::optional<Point> point;
};

// Thin lifecycle adapter over KataGo's Board and BoardHistory. All move
// legality, capture, ko/superko, pass, phase, and scoring decisions stay in
// KataGo; this class only owns a game and converts coordinates.
class Game {
public:
    explicit Game(
        int width = 19,
        int height = 19,
        Stone to_move = Stone::black,
        RuleSet rules = RuleSet::chinese()
    );
    Game(const Game& other);
    Game(Game&& other) noexcept;
    Game& operator=(const Game& other);
    Game& operator=(Game&& other) noexcept;
    ~Game();

    // Rows are top-to-bottom. Accepts '.', 'x'/'X', and 'o'/'O'.
    static Game from_ascii(
        std::string_view rows,
        Stone to_move,
        RuleSet rules
    );

    [[nodiscard]] int width() const noexcept;
    [[nodiscard]] int height() const noexcept;
    [[nodiscard]] Stone at(Point point) const;
    [[nodiscard]] Stone to_move() const noexcept;
    [[nodiscard]] std::size_t move_count() const noexcept;
    [[nodiscard]] bool game_over() const noexcept;
    [[nodiscard]] std::optional<Point> ko_point() const noexcept;
    [[nodiscard]] int captured_count(Stone color) const noexcept;
    [[nodiscard]] std::string ascii() const;
    [[nodiscard]] std::vector<PlayedMove> history() const;

    [[nodiscard]] bool is_legal(Stone color, std::optional<Point> point) const;

    // Strict play enforces turn order after the game has been scored.
    // Tolerant play follows KataGo's SGF-replay path and accepts an explicit
    // side regardless of turn order.
    bool play(Stone color, std::optional<Point> point);
    bool play_tolerant(Stone color, std::optional<Point> point);

    [[nodiscard]] std::vector<std::optional<Point>> legal_moves() const;
    [[nodiscard]] Score score() const;
    [[nodiscard]] std::vector<Stone> area() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace go
