// go_gui_tests — GuiSession 状态机单测：对弈流程、悔棋、中国规则终局、
// 竞技台步进/换先/比分、SGF 与状态 JSON 契约。全部用 random 引擎跑，
// 不依赖模型文件；模型路径的 HTTP 端到端在 tests/test_gui_http.py。

#include "go/gui_session.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using json = nlohmann::json;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

go::gui::SessionConfig random_config() {
    go::gui::SessionConfig config;
    config.engine = "random";
    config.black_engine = "random";
    config.white_engine = "random";
    config.seed = 20261025;
    config.speed_ms = 50;
    return config;
}

void test_initial_state() {
    go::gui::GuiSession session(random_config());
    const json state = session.state();
    require(state["ok"].get<bool>(), "state 必须 ok");
    require(state["mode"] == "play", "默认进入对弈模式");
    require(state["size"] == 19, "棋盘必须是 19×19");
    require(state["komi"] == 7.5, "中国规则默认贴 7.5 目");
    require(state["to_move"] == "black", "黑先");
    require(state["move_number"] == 0, "开局 0 手");
    require(state["board"].get<std::string>().size() == 361, "board 是 361 格");
    require(state["clickable"].get<bool>(), "轮到人类时可点击");
    require(state["play"]["human_color"] == "black", "默认执黑");
    require(!state["play"]["engine_turn"].get<bool>(), "开局轮到人类");
}

void test_play_flow_and_undo() {
    go::gui::GuiSession session(random_config());
    json reply = session.human_move(3, 3);
    require(reply["ok"].get<bool>(), "人类落子应成功");
    require(reply["move_number"] == 1, "落子后 1 手");
    require(reply["board"].get<std::string>()[60] == 'B', "(3,3) 应是黑子");
    require(reply["last_move"]["x"] == 3 && reply["last_move"]["y"] == 3,
            "last_move 应指向 (3,3)");
    require(reply["play"]["engine_turn"].get<bool>(), "落子后轮到引擎");

    require(session.engine_move()["ok"].get<bool>(), "引擎应手应成功");
    const json after_engine = session.state();
    require(after_engine["move_number"] == 2, "引擎应手后 2 手");
    require(!after_engine["play"]["engine_turn"].get<bool>(), "回到人类回合");

    require(!session.human_move(3, 3)["ok"].get<bool>(), "不能下在已有子的点");
    require(!session.human_move(19, 0)["ok"].get<bool>(), "坐标越界应被拒");
    require(!session.engine_move()["ok"].get<bool>(), "人类回合引擎不能走");

    const json undone = session.undo();
    require(undone["ok"].get<bool>(), "悔棋应成功");
    require(undone["move_number"] == 0, "悔棋退回两手到开局");
    require(undone["to_move"] == "black", "悔棋后回到黑方");
}

void test_chinese_score_and_resign() {
    go::gui::GuiSession session(random_config());
    require(session.new_game("none", 7.5F)["ok"].get<bool>(), "自由摆棋开局");
    require(session.human_pass()["ok"].get<bool>(), "第一次停一手");
    const json after_one = session.state();
    require(!after_one["game_over"].get<bool>(), "一次停一手不终局");
    require(session.human_pass()["ok"].get<bool>(), "第二次停一手");
    const json final_state = session.state();
    require(final_state["game_over"].get<bool>(), "双方停一手应终局");
    require(final_state["result"]["reason"] == "two-passes", "终局原因 = 数子");
    require(final_state["result"]["winner"] == "white",
            "空棋盘 + 贴 7.5 目应白胜（中国规则数子）");
    require(final_state["result"]["margin"] == 7.5, "白胜 7.5 目");
    require(final_state["to_move"].is_null(), "终局后 to_move 为空");

    require(session.new_game("black", 7.5F)["ok"].get<bool>(), "新开对局");
    require(session.human_move(3, 3)["ok"].get<bool>(), "落子");
    const json resigned = session.resign();
    require(resigned["ok"].get<bool>(), "认输应成功");
    require(resigned["result"]["winner"] == "white", "认输后白胜");
    require(resigned["result"]["reason"] == "resign", "终局原因 = 认输");
}

void test_arena_flow() {
    go::gui::GuiSession session(random_config());
    require(session.set_mode("arena")["ok"].get<bool>(), "切到竞技台");
    json state = session.state();
    require(state["mode"] == "arena", "模式 = arena");
    require(state["arena"]["game_index"] == 1, "第 1 局（1 基）");
    require(state["arena"]["games_played"] == 0, "尚未完赛");
    require(!state["clickable"].get<bool>(), "竞技台不接受点按");

    require(!session.human_move(3, 3)["ok"].get<bool>(), "竞技台拒绝人类落子");

    require(session.arena_step()["ok"].get<bool>(), "单步应成功");
    state = session.state();
    require(state["move_number"] == 1, "单步走一手");
    require(!state["arena"]["running"].get<bool>(), "单步不启动自动对局");

    // 步进到终局（random vs random，防御上限 1444 手）。
    for (int step = 0; step < 2000; ++step) {
        if (session.state()["game_over"].get<bool>()) {
            break;
        }
        session.arena_step();
    }
    state = session.state();
    require(state["game_over"].get<bool>(), "步进应能下完一局");
    require(state["arena"]["games_played"] == 1, "完赛记 1 局");
    const std::string first_black = state["arena"]["black_name"].get<std::string>();
    const std::string first_white = state["arena"]["white_name"].get<std::string>();
    const std::string first_winner = state["result"]["winner"].get<std::string>();

    require(session.arena_next()["ok"].get<bool>(), "开下一局");
    state = session.state();
    require(state["arena"]["game_index"] == 2, "第 2 局");
    require(state["move_number"] == 0, "新局清盘");
    require(state["arena"]["black_name"] == first_white &&
                state["arena"]["white_name"] == first_black,
            "下一局黑白引擎换先");

    // 分数要跟着引擎走：赢的一方换到对面席位后，胜局也要挂到对面名下，
    // 否则记分板换边不换分，名字和胜局张冠李戴。
    if (first_winner == "black") {
        require(state["arena"]["wins"]["white"] == 1,
                "黑位引擎赢的 1 局应随换先记到白位名下");
        require(state["arena"]["wins"]["black"] == 0, "对面引擎没有胜局");
    } else {
        require(state["arena"]["wins"]["black"] == 1,
                "白位引擎赢的 1 局应随换先记到黑位名下");
        require(state["arena"]["wins"]["white"] == 0, "对面引擎没有胜局");
    }
    require(state["arena"]["games_played"] == 1, "换先不改变已赛场次");

    require(session.arena_reset()["ok"].get<bool>(), "重置比分");
    state = session.state();
    require(state["arena"]["games_played"] == 0, "重置后比分清零");
    require(state["arena"]["game_index"] == 1, "重置后回到第 1 局");
}

void test_sgf_export() {
    go::gui::GuiSession session(random_config());
    session.new_game("black", 7.5F);
    session.human_move(3, 3);
    session.engine_move();
    const std::string sgf = session.sgf();
    require(sgf.find("SZ[19]") != std::string::npos, "SGF 含棋盘尺寸");
    require(sgf.find("KM[7.5]") != std::string::npos, "SGF 含贴目");
    require(sgf.find("RU[Chinese]") != std::string::npos, "SGF 标注中国规则");
    require(sgf.find(";B[dd]") != std::string::npos, "SGF 含 (3,3) 的黑着");
    require(sgf.find(";W[") != std::string::npos, "SGF 含白着");
    require(sgf.find("PB[人类]") != std::string::npos, "SGF 黑方 = 人类");
}

void test_synthetic_preview_matches_play() {
    go::gui::GuiSession session(random_config());
    session.new_game("black", 7.5F);  // 人类执黑
    session.human_move(3, 3);         // 轮到引擎（白）

    // 不支持评估的引擎也要能给 AI 视角：单点 100% 的"将下点预告"。
    const json evaluation = session.evaluate();
    require(evaluation["ok"].get<bool>(), "random 引擎也能给出评估（合成预告）");
    require(evaluation["synthetic"].get<bool>(), "random 引擎是合成分布");
    require(evaluation["top"].size() == 1, "合成分布只有一个候选点");
    require(evaluation["top"][0]["p"] == 1.0, "候选点概率 100%");
    require(evaluation["value"].is_null(), "合成分布没有价值头");

    // 预告的点必须与引擎随后真下的点一致。
    const int preview_x = evaluation["top"][0].value("x", -1);
    const int preview_y = evaluation["top"][0].value("y", -1);
    const json played = session.engine_move();
    require(played["ok"].get<bool>(), "引擎应手应成功");
    if (preview_x >= 0) {
        require(played["last_move"]["x"] == preview_x &&
                    played["last_move"]["y"] == preview_y,
                "预告点应与实战着点一致");
    } else {
        require(played["history"].back().value("pass", false),
                "预告 pass 则实战也应 pass");
    }
}

}  // namespace

int main() {
    int failures = 0;
    const auto run = [&failures](const char* name, void (*test)()) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& exception) {
            std::cout << "FAIL " << name << ": " << exception.what() << '\n';
            ++failures;
        }
    };
    run("initial_state", test_initial_state);
    run("play_flow_and_undo", test_play_flow_and_undo);
    run("chinese_score_and_resign", test_chinese_score_and_resign);
    run("arena_flow", test_arena_flow);
    run("sgf_export", test_sgf_export);
    run("synthetic_preview_matches_play", test_synthetic_preview_matches_play);
    if (failures != 0) {
        std::cout << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "all gui session tests passed\n";
    return 0;
}
