#include "../../native/aexrt.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

int main() {
    if (!aexrt_d3d12_probe()) {
        std::cerr << "no D3D12 adapter available\n";
        return 1;
    }

    aexrt::Device device(0);
    std::vector<float> x = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
    std::vector<float> gate = {0.5f, -1.5f, 2.5f, -3.5f, 4.5f, -5.5f};
    std::vector<float> residual = {1.0f, -0.5f, 0.25f, -0.75f, 0.125f, 2.0f};

    AexrtNodeDesc nodes[] = {
        {AEXRT_OP_ADD_FLOAT32, 0, 2, 3},
        {AEXRT_OP_SIGMOID_FLOAT32, 1, 0, 4},
        {AEXRT_OP_MUL_FLOAT32, 3, 4, 5},
        {AEXRT_OP_TANH_FLOAT32, 5, 0, 6},
    };
    AexrtGraphDesc desc{};
    desc.element_count = static_cast<uint64_t>(x.size());
    desc.input_count = 3;
    desc.node_count = static_cast<uint32_t>(sizeof(nodes) / sizeof(nodes[0]));
    desc.nodes = nodes;
    desc.output_value = 6;

    aexrt::Graph graph(aexrt_compile_graph(device.get(), &desc));
    auto y = aexrt::run(device, graph, std::vector<std::vector<float>>{x, gate, residual});

    float max_diff = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        float expected = std::tanh((x[i] + residual[i]) * sigmoid(gate[i]));
        max_diff = std::max(max_diff, std::fabs(y[i] - expected));
    }

    std::cout << "AEXRT C++ 3-input fused graph max diff: " << max_diff
              << " dispatches: " << graph.dispatch_count()
              << " buffers: " << graph.buffer_count() << "\n";
    return max_diff < 2e-5f && graph.dispatch_count() == 1 && graph.buffer_count() == 4 ? 0 : 2;
}
