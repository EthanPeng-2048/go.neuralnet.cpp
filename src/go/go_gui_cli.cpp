// go_gui_cli — 本地对弈室 GUI（web）：一个进程同时提供
//
//   * 对弈   —— 人类 vs 策略模型（或 random / 外部 GTP 引擎），点按落子、
//               悔棋、停一手、认输、SGF 导出；
//   * 竞技台 —— 引擎自动对局观战：后台线程按节奏落子、换先开新局、累计
//               比分、速度可调，AI 视角（候选着热力图）+ 形势判断（价值头）。
//
// 规则固定中国规则（go::RuleSet::chinese：位置超级劫 + 数子/面积计分 +
// 禁止自杀 + 贴 7.5 目），与挑战对手约定的口径一致；棋规全部走 KataGo
// （go::Game），模型选点走 go::PolicyPlayer，与 GTP/Arena 完全同源。
//
// 页面是单文件 HTML（gui/index.html），构建期嵌入二进制；服务默认只监听
// 127.0.0.1，浏览器打开打印的地址即可用。

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "external/nlohmann_json/json.hpp"

#include "go/gui_session.hpp"
#include "go/http_server.hpp"

#include "gui_page.hpp"

namespace {

using json = nlohmann::json;

struct Options {
    std::string host = "127.0.0.1";
    int port = 8765;
    std::string engine;         // play 模式对手；空 = 自动发现模型
    std::string black_engine;   // 竞技台第 1 局执黑；空 = 跟随 --engine
    std::string white_engine;   // 竞技台第 1 局执白；空 = random
    std::string model;          // --model PATH：所有槽位都用这个模型
    std::string device;         // GPU 设备名；空 = CPU
    float komi = 7.5F;          // 中国规则默认贴 7.5 目
    int speed_ms = 1200;
};

std::string with_device(const std::string& spec, const std::string& device) {
    if (device.empty() || !spec.starts_with("model:")) {
        return spec;
    }
    return spec + "=" + device;
}

// 没给 --engine 时在常见产物位置找模型；找不到就用 random（照样能下棋）。
std::string discover_model_spec() {
    constexpr std::string_view candidates[] = {
        "artifacts/m4/m4-c-validator.nnmodel",
        ".cache/m7-b-smoke.nnmodel",
        ".cache/m7-c-base.nnmodel",
    };
    for (const std::string_view candidate : candidates) {
        std::error_code error;
        if (std::filesystem::exists(candidate, error)) {
            return "model:" + std::string{candidate};
        }
    }
    return "random";
}

// PATH 里找 gnugo（Windows 同时找 gnugo.exe）。
bool find_gnugo() {
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr) {
        return false;
    }
    const std::string_view path{path_env};
#if defined(_WIN32)
    constexpr std::string_view separator = ";";
#else
    constexpr std::string_view separator = ":";
#endif
    constexpr std::string_view names[] = {"gnugo", "gnugo.exe"};
    std::size_t start = 0;
    while (start <= path.size()) {
        const std::size_t end = path.find(separator, start);
        const std::string_view directory = path.substr(
            start, end == std::string_view::npos ? end : end - start);
        for (const std::string_view name : names) {
            if (directory.empty()) {
                continue;
            }
            std::error_code error;
            if (std::filesystem::exists(
                    std::filesystem::path{directory} / std::string{name}, error)) {
                return true;
            }
        }
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return false;
}

// 默认对手引擎：装了 GnuGo 就用它（真实对手），否则空串（调用方再退回模型/random）。
std::string gnugo_spec() {
    return find_gnugo() ? std::string{"gtp:gnugo --mode gtp --quiet"} : std::string{};
}

float parse_float(std::string_view text) {
    std::size_t consumed = 0;
    const std::string copy{text};
    const float value = std::stof(copy, &consumed);
    if (consumed != copy.size()) {
        throw std::invalid_argument("expected a number: " + copy);
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument(
                    std::string("missing value for ") + argv[i - 1]);
            }
            return argv[i];
        };
        if (arg == "--host") {
            options.host = value();
        } else if (arg == "--port") {
            options.port = std::stoi(value());
        } else if (arg == "--engine") {
            options.engine = value();
        } else if (arg == "--black-engine") {
            options.black_engine = value();
        } else if (arg == "--white-engine") {
            options.white_engine = value();
        } else if (arg == "--model") {
            options.model = value();
        } else if (arg == "--device") {
            options.device = value();
        } else if (arg == "--komi") {
            options.komi = parse_float(value());
        } else if (arg == "--speed") {
            options.speed_ms = std::stoi(value());
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: go_gui_cli [--port 8765] [--host 127.0.0.1]\n"
                << "       [--model FILE] [--device NAME]\n"
                << "       [--engine ENGINE] [--black-engine ENGINE]"
                << " [--white-engine ENGINE]\n"
                << "       [--komi 7.5] [--speed 1200]\n"
                << "  ENGINE = random | model:FILE[=DEVICE] | gtp:COMMAND\n"
                << "  默认对手：本机装了 gnugo 就用 gtp:gnugo --mode gtp --quiet，\n"
                << "  否则用自动发现的模型（artifacts/m4、.cache 下的 .nnmodel），再退回 random\n"
                << "  竞技台默认：黑 = 项目模型，白 = 默认对手；--model 让所有槽位都用该模型\n"
                << "  规则固定中国规则（贴 7.5 目、数子计分）；--port 0 由系统分配端口\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(arg));
        }
    }
    return options;
}

go::gui::SessionConfig make_session_config(const Options& options) {
    go::gui::SessionConfig config;

    const std::string model_spec = with_device(
        options.model.empty() ? std::string{} : "model:" + options.model,
        options.device);
    const std::string discovered = with_device(discover_model_spec(), options.device);
    const std::string gnugo = gnugo_spec();

    // 默认对手（对弈对手 / 竞技台对手席）：GnuGo > 自动发现的模型 > random。
    // 装了 gnugo 就别拿随机引擎糊弄——GnuGo 是能真下棋的对手。
    std::string opponent =
        !gnugo.empty() ? gnugo : (discovered.empty() ? "random" : discovered);
    // 默认我方（竞技台）：项目模型 > 默认对手。
    std::string player = discovered.empty() ? opponent : discovered;
    if (!model_spec.empty()) {
        // --model：所有未显式指定的槽位都用这个模型。
        opponent = model_spec;
        player = model_spec;
    }

    config.engine = options.engine.empty() ? opponent : options.engine;
    config.black_engine =
        options.black_engine.empty() ? player : options.black_engine;
    config.white_engine =
        options.white_engine.empty() ? opponent : options.white_engine;
    config.komi = options.komi;
    config.speed_ms = options.speed_ms;
    return config;
}

go::gui::HttpReply json_reply(json body) {
    int status = 200;
    if (body.contains("ok") && !body["ok"].get<bool>()) {
        status = body.value("status", 400);
        body.erase("status");
    }
    return go::gui::HttpReply{
        status, "application/json; charset=utf-8", body.dump()};
}

json parse_body(const go::gui::HttpRequest& request) {
    if (request.body.empty()) {
        return json::object();
    }
    return json::parse(request.body);
}

go::gui::HttpReply route(go::gui::GuiSession& session,
                         const go::gui::HttpRequest& request) {
    const std::string& path = request.path;
    if (request.method == "GET") {
        if (path == "/" || path == "/index.html") {
            return go::gui::HttpReply{
                200, "text/html; charset=utf-8", go::gui::page::kGuiPage};
        }
        if (path == "/favicon.ico") {
            return go::gui::HttpReply{204, "text/plain", {}};
        }
        if (path == "/api/state") {
            return json_reply(session.state());
        }
        if (path == "/api/evaluate") {
            try {
                return json_reply(session.evaluate());
            } catch (const std::exception& exception) {
                return json_reply(json{{"ok", false},
                                       {"error", exception.what()},
                                       {"status", 409}});
            }
        }
        if (path == "/api/sgf") {
            return go::gui::HttpReply{
                200, "text/plain; charset=utf-8", session.sgf()};
        }
    } else if (request.method == "POST") {
        json body;
        try {
            body = parse_body(request);
        } catch (const std::exception& exception) {
            return json_reply(json{{"ok", false},
                                   {"error", std::string("JSON 解析失败：") +
                                                 exception.what()},
                                   {"status", 400}});
        }
        if (path == "/api/play") {
            return json_reply(session.human_move(
                body.value("x", -1), body.value("y", -1)));
        }
        if (path == "/api/pass") {
            return json_reply(session.human_pass());
        }
        if (path == "/api/engine-move") {
            return json_reply(session.engine_move());
        }
        if (path == "/api/undo") {
            return json_reply(session.undo());
        }
        if (path == "/api/resign") {
            return json_reply(session.resign());
        }
        if (path == "/api/new") {
            return json_reply(session.new_game(
                body.value("human_color", "black"),
                static_cast<float>(body.value("komi", 0.0))));
        }
        if (path == "/api/mode") {
            return json_reply(session.set_mode(body.value("mode", "play")));
        }
        if (path == "/api/config") {
            return json_reply(session.configure(
                body.value("speed_ms", 0), body.value("auto_next", true),
                static_cast<float>(body.value("komi", 0.0))));
        }
        if (path == "/api/arena/start") {
            return json_reply(session.arena_start());
        }
        if (path == "/api/arena/pause") {
            return json_reply(session.arena_pause());
        }
        if (path == "/api/arena/step") {
            return json_reply(session.arena_step());
        }
        if (path == "/api/arena/next") {
            return json_reply(session.arena_next());
        }
        if (path == "/api/arena/reset") {
            return json_reply(session.arena_reset());
        }
    }
    return go::gui::HttpReply{404, "text/plain", "not found\n"};
}

std::atomic<go::gui::HttpServer*> g_server{nullptr};

extern "C" void handle_signal(int) {
    if (go::gui::HttpServer* server = g_server.load()) {
        server->stop();  // 只置原子标志，信号安全
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        go::gui::GuiSession session(make_session_config(options));
        go::gui::HttpServer server(
            options.host, options.port,
            [&session](const go::gui::HttpRequest& request) {
                return route(session, request);
            });

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        std::cout << "go.gui listening on http://" << options.host << ":"
                  << server.port() << std::endl;
        server.run();
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "go_gui_cli: " << exception.what() << '\n';
        return 1;
    }
}
