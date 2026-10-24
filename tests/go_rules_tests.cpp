#include "go/game.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void play(go::Game& game, go::Stone color, std::optional<go::Point> point) {
    if (!game.play(color, point)) {
        throw std::runtime_error(
            "expected legal move " + std::to_string(game.move_count() + 1)
        );
    }
}

void test_capture() {
    go::Game game(5, 5);
    play(game, go::Stone::black, go::Point{1, 1});
    play(game, go::Stone::white, go::Point{2, 1});
    play(game, go::Stone::black, go::Point{1, 2});
    play(game, go::Stone::white, go::Point{0, 0});
    play(game, go::Stone::black, go::Point{3, 1});
    play(game, go::Stone::white, go::Point{4, 4});
    play(game, go::Stone::black, go::Point{2, 2});
    play(game, go::Stone::white, go::Point{0, 4});
    play(game, go::Stone::black, go::Point{2, 0});

    require(game.at({2, 1}) == go::Stone::empty, "captured white stone remained on the board");
}

void test_suicide() {
    const go::Game game = go::Game::from_ascii(
        ".....\n"
        ".xxx.\n"
        ".x.x.\n"
        ".xxx.\n"
        ".....",
        go::Stone::white,
        go::RuleSet::chinese()
    );
    require(!game.is_legal(go::Stone::white, go::Point{2, 2}), "suicide was accepted");
}

void test_ko_and_superko() {
    constexpr std::string_view ko_position =
        ".o.xxo\n"
        "oxxxo.\n"
        "o.x.oo\n"
        "xx.oo.\n"
        "oooo.o";

    {
        go::Game game = go::Game::from_ascii(
            ko_position,
            go::Stone::black,
            go::RuleSet::chinese(0.5F)
        );
        play(game, go::Stone::black, go::Point{5, 1});
        require(
            !game.is_legal(go::Stone::white, go::Point{5, 0}),
            "immediate ko recapture was accepted"
        );
    }

    {
        go::Game game = go::Game::from_ascii(
            ko_position,
            go::Stone::black,
            go::RuleSet::chinese(0.5F)
        );
        play(game, go::Stone::black, go::Point{5, 1});
        play(game, go::Stone::white, std::nullopt);
        play(game, go::Stone::black, go::Point{3, 2});
        play(game, go::Stone::white, go::Point{2, 0});
        play(game, go::Stone::black, go::Point{0, 0});
        play(game, go::Stone::white, go::Point{5, 0});
        play(game, go::Stone::black, std::nullopt);
        play(game, go::Stone::white, go::Point{1, 0});
        play(game, go::Stone::black, std::nullopt);
        play(game, go::Stone::white, go::Point{2, 0});
        require(
            !game.is_legal(go::Stone::black, go::Point{0, 0}),
            "positional-superko repeated position was accepted"
        );
    }
}

void test_scoring() {
    // Position and expected values are ported from KataGo v1.18.2's
    // tests/testrules.cpp ("Area/Territory scoring in the main phase").
    constexpr std::string_view scoring_position =
        "ox.ooo.\n"
        "oxxxxxx\n"
        "ooooooo\n"
        ".xoxx..\n"
        "ooox...\n"
        "x.oxxxx\n"
        ".xox...";

    {
        go::Game area = go::Game::from_ascii(
            scoring_position,
            go::Stone::black,
            go::RuleSet::chinese(0.5F)
        );
        require(std::abs(area.score().white_minus_black + 3.5F) < 0.001F, "area score mismatch");
        require(area.score().winner == go::Stone::black, "area winner mismatch");

        const std::vector<go::Stone> ownership = area.area();
        require(ownership.size() == 49, "area ownership size mismatch");
        require(ownership[0] == go::Stone::white, "occupied area ownership mismatch");

        play(area, go::Stone::black, go::Point{5, 3});
        require(std::abs(area.score().white_minus_black + 4.5F) < 0.001F, "updated area score mismatch");
    }

    {
        go::Game territory = go::Game::from_ascii(
            scoring_position,
            go::Stone::black,
            go::RuleSet::japanese(0.5F)
        );
        require(
            std::abs(territory.score().white_minus_black + 3.5F) < 0.001F,
            "territory score mismatch"
        );
        play(territory, go::Stone::black, go::Point{5, 3});
        require(
            std::abs(territory.score().white_minus_black + 3.5F) < 0.001F,
            "updated territory score mismatch"
        );
    }
}

void test_passes_end_game() {
    go::Game game(9, 9, go::Stone::black, go::RuleSet::chinese(7.5F));
    play(game, go::Stone::black, std::nullopt);
    require(!game.game_over(), "one pass ended an area-scoring game");
    play(game, go::Stone::white, std::nullopt);
    require(game.game_over(), "two passes did not end the game");
    require(!game.is_legal(go::Stone::black, std::nullopt), "move accepted after game over");
    const go::Score score = game.score();
    require(std::abs(score.white_minus_black - 7.5F) < 0.001F, "finished game score mismatch");
    require(score.winner == go::Stone::white, "finished game winner mismatch");
}

std::vector<std::optional<go::Point>> replay_moves() {
    // 200 distinct, well-spaced legal points. Coordinates are fixed so the
    // benchmark measures rule replay rather than random-number generation.
    std::vector<std::optional<go::Point>> moves;
    moves.reserve(200);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 19; ++x) {
            moves.emplace_back(go::Point{x, y});
        }
    }
    for (int x = 0; x < 10; ++x) {
        moves.emplace_back(go::Point{x, 10});
    }
    return moves;
}

void test_replay_and_latency() {
    const std::vector<std::optional<go::Point>> moves = replay_moves();
    std::vector<double> milliseconds;
    milliseconds.reserve(15);

    for (int sample = 0; sample < 15; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        go::Game game(19, 19);
        for (std::size_t i = 0; i < moves.size(); ++i) {
            const go::Stone color = i % 2 == 0 ? go::Stone::black : go::Stone::white;
            if (!game.play(color, moves[i])) {
                throw std::runtime_error("benchmark replay reached an unexpected illegal move");
            }
        }
        const auto stop = std::chrono::steady_clock::now();
        milliseconds.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count()
        );
        require(game.move_count() == moves.size(), "replay move count mismatch");
    }

    std::ranges::sort(milliseconds);
    const double median = milliseconds[milliseconds.size() / 2];
    std::cout << "M1 200-move replay median: " << median << " ms\n";
    if (std::getenv("GO_SKIP_PERF_ASSERT") == nullptr) {
        require(median < 5.0, "single-game replay exceeded the 5 ms budget");
    }
}

void benchmark_public_replay() {
    const auto start = std::chrono::steady_clock::now();
    go::Game game(19, 19);
    const std::vector<std::optional<go::Point>> moves = replay_moves();
    for (std::size_t i = 0; i < moves.size(); ++i) {
        const go::Stone color = i % 2 == 0 ? go::Stone::black : go::Stone::white;
        if (!game.play(color, moves[i])) {
            throw std::runtime_error("public replay benchmark move was illegal");
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    std::cout << "M1 replay_public_ms="
              << std::chrono::duration<double, std::milli>(stop - start).count()
              << " moves=" << moves.size() << '\n';
}

}  // namespace

int main() {
    try {
        test_capture();
        test_suicide();
        test_ko_and_superko();
        test_scoring();
        test_passes_end_game();
        test_replay_and_latency();
        benchmark_public_replay();
        std::cout << "M1 rule tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "M1 rule test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
