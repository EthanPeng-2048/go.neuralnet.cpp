#pragma once

// ── 极简 HTTP/1.1 服务器（GUI 专用）──────────────────────────────────────
//
// 只做对弈室需要的最小集：GET / POST、每请求一连接（无 keep-alive）、
// Connection: close、固定 Content-Length。默认只监听 127.0.0.1——它不是
// 通用 Web 框架，也不打算对外网开放。业务逻辑全部在 GuiSession 里，
// 这一层只负责字节进出与请求/响应的编解码。

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace go::gui {

struct HttpRequest {
    std::string method;   // "GET" / "POST"
    std::string path;     // 不含查询串，如 "/api/state"
    std::string query;    // 查询串（不含 '?'）
    std::string body;
    std::map<std::string, std::string> headers;  // 键统一小写
};

struct HttpReply {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
};

class HttpServer {
public:
    using Handler = std::function<HttpReply(const HttpRequest&)>;

    // host 默认 127.0.0.1；port = 0 表示由系统分配（测试用）。
    HttpServer(std::string host, int port, Handler handler);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // 启动监听并阻塞处理连接，直到 stop()。返回实际监听端口。
    int run();
    // 关闭监听套接字（可从信号处理函数调用；内部只做 close + 原子置位）。
    void stop();

    [[nodiscard]] int port() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace go::gui
