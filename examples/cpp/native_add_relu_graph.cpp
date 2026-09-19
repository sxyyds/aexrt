#include "../../native/aexrt.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    if (!aexrt_d3d12_probe()) {
        std::cerr << "no D3D12 adapter available\n";
        return 1;
    }

    aexrt::Device device(0);
    std::vector<float> a = {-4.0f, 2.0f, -1.0f, 8.0f, 0.0f, -0.25f};
    std::vector<float> b = {1.0f, -3.0f, 5.0f, -4.0f, 0.5f, 2.0f};

    auto graph = aexrt::compile_add_relu_graph(device, static_cast<uint64_t>(a.size()));
    auto y = aexrt::run(device, graph, a, b);

    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float expected = std::max(a[i] + b[i], 0.0f);
        max_diff = std::max(max_diff, std::fabs(y[i] - expected));
    }

    std::cout << "AEXRT C++ Add->Relu graph max diff: " << max_diff
              << " dispatches: " << graph.dispatch_count()
              << " buffers: " << graph.buffer_count() << "\n";
    return max_diff == 0.0f ? 0 : 2;
}
