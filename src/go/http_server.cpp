#include "go/http_server.hpp"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
using SocketHandle = SOCKET;
using SockLen = int;  // Winsock 的 getsockname 用 int*
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using SocketHandle = int;
using SockLen = socklen_t;
constexpr SocketHandle kInvalidSocket = -1;
#endif

namespace go::gui {
namespace {

constexpr std::size_t kMaxRequestBytes = 64 * 1024;

void close_socket(SocketHandle socket) {
#if defined(_WIN32)
    ::closesocket(socket);
#else
    ::close(socket);
#endif
}

std::string status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 409: return "Conflict";
        case 500: return "Internal Server Error";
        default: return "Status";
    }
}

std::string lowercase(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

std::string trim(const std::string& text) {
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

// 读到 "\r\n\r\n" 为止，再按 Content-Length 收满 body。超时/超限返回 false。
bool read_request(SocketHandle socket, HttpRequest& request) {
    std::string buffer;
    char chunk[4096];
    std::size_t header_end = std::string::npos;
    while ((header_end = buffer.find("\r\n\r\n")) == std::string::npos) {
        const int received = static_cast<int>(
            ::recv(socket, chunk, static_cast<int>(sizeof(chunk)), 0));
        if (received <= 0) {
            return false;
        }
        buffer.append(chunk, static_cast<std::size_t>(received));
        if (buffer.size() > kMaxRequestBytes) {
            return false;
        }
    }

    const std::size_t line_end = buffer.find("\r\n");
    const std::string request_line = buffer.substr(0, line_end);
    const std::size_t first_space = request_line.find(' ');
    const std::size_t second_space =
        first_space == std::string::npos ? std::string::npos
                                         : request_line.find(' ', first_space + 1);
    if (first_space == std::string::npos || second_space == std::string::npos) {
        return false;
    }
    request.method = request_line.substr(0, first_space);
    const std::string target =
        request_line.substr(first_space + 1, second_space - first_space - 1);
    if (const std::size_t query_start = target.find('?');
        query_start == std::string::npos) {
        request.path = target;
    } else {
        request.path = target.substr(0, query_start);
        request.query = target.substr(query_start + 1);
    }

    std::size_t position = line_end + 2;
    while (position < header_end) {
        const std::size_t end = buffer.find("\r\n", position);
        const std::string line = buffer.substr(position, end - position);
        position = end + 2;
        if (const std::size_t colon = line.find(':'); colon != std::string::npos) {
            request.headers[lowercase(trim(line.substr(0, colon)))] =
                trim(line.substr(colon + 1));
        }
    }

    std::size_t content_length = 0;
    if (const auto it = request.headers.find("content-length");
        it != request.headers.end()) {
        try {
            content_length = static_cast<std::size_t>(std::stoul(it->second));
        } catch (const std::exception&) {
            return false;
        }
        if (content_length > kMaxRequestBytes) {
            return false;
        }
    }

    const std::size_t body_start = header_end + 4;
    while (buffer.size() < body_start + content_length) {
        const int received = static_cast<int>(
            ::recv(socket, chunk, static_cast<int>(sizeof(chunk)), 0));
        if (received <= 0) {
            return false;
        }
        buffer.append(chunk, static_cast<std::size_t>(received));
        if (buffer.size() > kMaxRequestBytes) {
            return false;
        }
    }
    request.body = buffer.substr(body_start, content_length);
    return true;
}

bool send_all(SocketHandle socket, const std::string& data) {
    std::size_t sent = 0;
    while (sent < data.size()) {
        const int written = static_cast<int>(::send(
            socket, data.data() + sent, static_cast<int>(data.size() - sent), 0));
        if (written <= 0) {
            return false;
        }
        sent += static_cast<std::size_t>(written);
    }
    return true;
}

std::string render_reply(const HttpReply& reply) {
    std::ostringstream head;
    head << "HTTP/1.1 " << reply.status << " " << status_text(reply.status)
         << "\r\n"
         << "Content-Type: " << reply.content_type << "\r\n"
         << "Content-Length: " << reply.body.size() << "\r\n"
         << "Cache-Control: no-store\r\n"
         << "Connection: close\r\n\r\n";
    return head.str() + reply.body;
}

}  // namespace

struct HttpServer::Impl {
    std::string host;
    int port = 0;
    Handler handler;
    SocketHandle listen_socket = kInvalidSocket;
    std::atomic<bool> stopped{false};
#if defined(_WIN32)
    bool wsa_started = false;
#endif
};

HttpServer::HttpServer(std::string host, int port, Handler handler)
    : impl_(std::make_unique<Impl>()) {
    impl_->host = std::move(host);
    impl_->port = port;
    impl_->handler = std::move(handler);

#if defined(_WIN32)
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
        throw std::runtime_error("WSAStartup failed");
    }
    impl_->wsa_started = true;
#endif

    impl_->listen_socket = ::socket(AF_INET, SOCK_STREAM, 0);
    if (impl_->listen_socket == kInvalidSocket) {
        throw std::runtime_error("cannot create listening socket");
    }
    int reuse = 1;
    ::setsockopt(impl_->listen_socket, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&reuse), sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    if (::inet_pton(AF_INET, impl_->host.c_str(), &address.sin_addr) != 1) {
        throw std::runtime_error("invalid bind address: " + impl_->host);
    }
    if (::bind(impl_->listen_socket, reinterpret_cast<sockaddr*>(&address),
               sizeof(address)) != 0) {
        throw std::runtime_error("cannot bind " + impl_->host + ":" +
                                 std::to_string(port));
    }
    // port = 0 → 系统分配端口，回填给调用方（测试与 --port 0 都靠它）。
    sockaddr_in bound{};
    SockLen length = sizeof(bound);
    if (::getsockname(impl_->listen_socket, reinterpret_cast<sockaddr*>(&bound),
                      &length) == 0) {
        impl_->port = static_cast<int>(ntohs(bound.sin_port));
    }
    if (::listen(impl_->listen_socket, 16) != 0) {
        throw std::runtime_error("cannot listen on " + impl_->host);
    }
}

HttpServer::~HttpServer() {
    stop();
    if (impl_->listen_socket != kInvalidSocket) {
        close_socket(impl_->listen_socket);
        impl_->listen_socket = kInvalidSocket;
    }
#if defined(_WIN32)
    if (impl_->wsa_started) {
        WSACleanup();
    }
#endif
}

int HttpServer::port() const noexcept {
    return impl_->port;
}

void HttpServer::stop() {
    // 只做原子置位（信号处理函数里也安全）；accept 循环 200 ms 内自行退出。
    impl_->stopped.store(true);
}

int HttpServer::run() {
    while (!impl_->stopped.load()) {
        // select 带超时轮询：stop() 只需置位，不依赖 close 能打断 accept。
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(impl_->listen_socket, &read_set);
        timeval timeout{0, 200 * 1000};
#if defined(_WIN32)
        const int ready = ::select(0, &read_set, nullptr, nullptr, &timeout);
#else
        const int ready = ::select(impl_->listen_socket + 1, &read_set, nullptr,
                                   nullptr, &timeout);
#endif
        if (ready <= 0) {
            continue;  // 超时 / 信号中断：回到循环检查停止标志
        }
        SocketHandle client = ::accept(impl_->listen_socket, nullptr, nullptr);
        if (client == kInvalidSocket) {
            continue;
        }
        // 每连接一个工作线程；连接短命（Connection: close），线程即走即销。
        std::thread([this, client] {
            HttpRequest request;
            HttpReply reply;
            try {
                if (!read_request(client, request)) {
                    reply = HttpReply{400, "text/plain", "bad request\n"};
                } else {
                    reply = impl_->handler(request);
                }
            } catch (const std::exception& exception) {
                reply = HttpReply{
                    500, "text/plain", std::string("internal error: ") +
                                           exception.what() + "\n"};
            }
            send_all(client, render_reply(reply));
            close_socket(client);
        }).detach();
    }
    return impl_->port;
}

}  // namespace go::gui
