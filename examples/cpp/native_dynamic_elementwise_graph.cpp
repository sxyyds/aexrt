#include "../../native/aexrt.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

static float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

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

    AexrtNodeDesc nodes[] = {
        {AEXRT_OP_ADD_FLOAT32, 0, 1, 2},
        {AEXRT_OP_SUB_FLOAT32, 2, 1, 3},
        {AEXRT_OP_DIV_FLOAT32, 3, 1, 4},
        {AEXRT_OP_TANH_FLOAT32, 4, 0, 5},
        {AEXRT_OP_SIGMOID_FLOAT32, 1, 0, 6},
        {AEXRT_OP_MUL_FLOAT32, 5, 6, 7},
        {AEXRT_OP_GELU_FLOAT32, 7, 0, 8},
    };
    AexrtGraphDesc desc{};
    desc.element_count = static_cast<uint64_t>(x.size());
    desc.input_count = 2;
    desc.node_count = static_cast<uint32_t>(sizeof(nodes) / sizeof(nodes[0]));
    desc.nodes = nodes;
    desc.output_value = 8;

    aexrt::Graph graph(aexrt_compile_graph(device.get(), &desc));
    auto y = aexrt::run(device, graph, x, gate);

    float max_diff = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        float residual = x[i] + gate[i];
        float centered = residual - gate[i];
        float scaled = centered / gate[i];
        float candidate = std::tanh(scaled);
        float weight = sigmoid(gate[i]);
        float expected = gelu(candidate * weight);
        max_diff = std::max(max_diff, std::fabs(y[i] - expected));
    }

    std::cout << "AEXRT C++ dynamic elementwise DAG max diff: " << max_diff
              << " dispatches: " << graph.dispatch_count()
              << " buffers: " << graph.buffer_count() << "\n";
    return max_diff < 2e-5f && graph.dispatch_count() == 1 && graph.buffer_count() == 3 ? 0 : 2;
}
