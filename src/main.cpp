#include <neuralnet.cpp/nn.hpp>

#include <cstdio>
#include <string_view>

int main(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::string_view(argv[i]) == "--version")
        {
            std::puts("go.neuralnet.cpp 0.1.0 (M0, neuralnet.cpp v1.12.0)");
            return 0;
        }
    }

    auto engine = nn::CpuEngine::create();
    if (!engine)
    {
        std::fprintf(stderr, "go.neuralnet.cpp M0: CPU engine creation failed\n");
        return 1;
    }

    std::puts("go.neuralnet.cpp M0 smoke: neuralnet::nn linked");
    return 0;
}
