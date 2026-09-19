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

    std::vector<float> x = {-2.0f, -0.5f, 0.0f, 3.0f, 9.0f};
    auto input = aexrt::upload_float32(device, x);
    auto output = aexrt::allocate_float32_uav(device, static_cast<uint64_t>(x.size()));

    auto dispatch = aexrt::prepare_relu_float32(device, input, output, static_cast<uint64_t>(x.size()));
    aexrt::execute(device, dispatch);
    auto y = aexrt::download_float32(device, output, static_cast<uint64_t>(x.size()));

    float max_diff = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        float expected = std::max(x[i], 0.0f);
        max_diff = std::max(max_diff, std::fabs(y[i] - expected));
    }

    std::cout << "AEXRT pure C++ native ReLU max diff: " << max_diff << "\n";
    return max_diff == 0.0f ? 0 : 2;
}
