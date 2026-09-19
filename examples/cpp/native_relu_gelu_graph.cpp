#include "../../native/aexrt.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

static float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

int main() {
    if (!aexrt_d3d12_probe()) {
        std::cerr << "no D3D12 adapter available\n";
        return 1;
    }

    aexrt::Device device(0);
    std::vector<float> x = {-4.0f, 2.0f, -1.0f, 8.0f, 0.0f, -0.25f};

    auto graph = aexrt::compile_relu_gelu_graph(device, static_cast<uint64_t>(x.size()));
    auto y = aexrt::run(device, graph, x);

    float max_diff = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        float expected = gelu(std::max(x[i], 0.0f));
        max_diff = std::max(max_diff, std::fabs(y[i] - expected));
    }

    std::cout << "AEXRT C++ Relu->Gelu graph max diff: " << max_diff
              << " dispatches: " << graph.dispatch_count()
              << " buffers: " << graph.buffer_count() << "\n";
    return max_diff < 1e-5f ? 0 : 2;
}
