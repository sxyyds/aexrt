#include "../../native/aexrt.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

int main() {
    if (!aexrt_d3d12_probe()) {
        std::cerr << "no D3D12 adapter available\n";
        return 1;
    }

    constexpr uint32_t n = 1;
    constexpr uint32_t c = 2;
    constexpr uint32_t h = 4;
    constexpr uint32_t w = 4;
    constexpr uint32_t oc = 3;
    constexpr uint32_t kh = 3;
    constexpr uint32_t kw = 3;
    constexpr uint32_t oh = 4;
    constexpr uint32_t ow = 4;

    std::vector<float> x(n * c * h * w);
    std::vector<float> weight(oc * c * kh * kw);
    std::vector<float> bias(oc);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (static_cast<int>(i % 11) - 5) * 0.125f;
    for (size_t i = 0; i < weight.size(); ++i) weight[i] = (static_cast<int>(i % 7) - 3) * 0.05f;
    for (size_t i = 0; i < bias.size(); ++i) bias[i] = (static_cast<int>(i) - 1) * 0.1f;

    std::vector<float> expected_linear(n * oc * oh * ow);
    std::vector<float> expected(n * oc * oh * ow);
    for (uint32_t co = 0; co < oc; ++co) {
        for (uint32_t y = 0; y < oh; ++y) {
            for (uint32_t xw = 0; xw < ow; ++xw) {
                float acc = bias[co];
                for (uint32_t ci = 0; ci < c; ++ci) {
                    for (uint32_t ky = 0; ky < kh; ++ky) {
                        int iy = static_cast<int>(y + ky) - 1;
                        if (iy < 0 || iy >= static_cast<int>(h)) continue;
                        for (uint32_t kx = 0; kx < kw; ++kx) {
                            int ix = static_cast<int>(xw + kx) - 1;
                            if (ix < 0 || ix >= static_cast<int>(w)) continue;
                            size_t xi = ((ci * h + static_cast<uint32_t>(iy)) * w + static_cast<uint32_t>(ix));
                            size_t wi = (((co * c + ci) * kh + ky) * kw + kx);
                            acc += x[xi] * weight[wi];
                        }
                    }
                }
                expected_linear[(co * oh + y) * ow + xw] = acc;
                expected[(co * oh + y) * ow + xw] = silu(acc);
            }
        }
    }

    aexrt::Device device(0);
    auto xb = aexrt::upload_float32(device, x);
    auto wb = aexrt::upload_float32(device, weight);
    auto bb = aexrt::upload_float32(device, bias);
    auto linear_yb = aexrt::allocate_float32_uav(device, expected_linear.size());
    auto yb = aexrt::allocate_float32_uav(device, expected.size());

    AexrtConv2DDesc desc{};
    desc.batch = n;
    desc.in_channels = c;
    desc.in_h = h;
    desc.in_w = w;
    desc.out_channels = oc;
    desc.out_h = oh;
    desc.out_w = ow;
    desc.kernel_h = kh;
    desc.kernel_w = kw;
    desc.stride_h = 1;
    desc.stride_w = 1;
    desc.pad_top = 1;
    desc.pad_left = 1;
    desc.dilation_h = 1;
    desc.dilation_w = 1;
    desc.groups = 1;

    aexrt::conv2d_float32(device, xb, wb, bb, linear_yb, desc);
    auto linear_y = aexrt::download_float32(device, linear_yb, expected_linear.size());
    float linear_max_diff = 0.0f;
    for (size_t i = 0; i < linear_y.size(); ++i) {
        linear_max_diff = std::max(linear_max_diff, std::fabs(linear_y[i] - expected_linear[i]));
    }
    std::cout << "AEXRT C++ Conv2D max diff: " << linear_max_diff << "\n";
    if (linear_max_diff >= 2e-5f) {
        return 3;
    }

    auto dispatch = aexrt::prepare_conv2d_silu_upload_float32(device, wb, bb, yb, desc);
    aexrt::execute_upload(device, dispatch, x);
    auto y = aexrt::download_float32(device, yb, expected.size());

    float max_diff = 0.0f;
    for (size_t i = 0; i < y.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(y[i] - expected[i]));
    }
    std::cout << "AEXRT C++ Conv2D+SiLU max diff: " << max_diff << "\n";
    if (max_diff >= 2e-5f) {
        return 2;
    }

    constexpr uint32_t concat_h = 2;
    constexpr uint32_t concat_w = 3;
    constexpr uint32_t c0 = 2;
    constexpr uint32_t c1 = 1;
    constexpr uint32_t concat_out_channels = 4;
    std::vector<float> concat_input((c0 + c1) * concat_h * concat_w);
    std::vector<float> concat_weight(concat_out_channels * (c0 + c1));
    std::vector<float> concat_bias(concat_out_channels);
    for (size_t i = 0; i < concat_input.size(); ++i) concat_input[i] = (static_cast<int>(i % 9) - 4) * 0.1f;
    for (size_t i = 0; i < concat_weight.size(); ++i) concat_weight[i] = (static_cast<int>(i % 5) - 2) * 0.07f;
    for (size_t i = 0; i < concat_bias.size(); ++i) concat_bias[i] = (static_cast<int>(i) - 1) * 0.03f;
    std::vector<float> concat_expected(concat_out_channels * concat_h * concat_w);
    for (uint32_t co = 0; co < concat_out_channels; ++co) {
        for (uint32_t yy = 0; yy < concat_h; ++yy) {
            for (uint32_t xx = 0; xx < concat_w; ++xx) {
                float acc = concat_bias[co];
                uint32_t spatial = yy * concat_w + xx;
                for (uint32_t ci = 0; ci < c0 + c1; ++ci) {
                    acc += concat_input[ci * concat_h * concat_w + spatial] * concat_weight[co * (c0 + c1) + ci];
                }
                concat_expected[(co * concat_h + yy) * concat_w + xx] = silu(acc);
            }
        }
    }
    auto concat_parent = aexrt::upload_float32(device, concat_input);
    auto concat_left = aexrt::create_float32_view(device, concat_parent, 0, c0 * concat_h * concat_w);
    auto concat_right = aexrt::create_float32_view(device, concat_parent, c0 * concat_h * concat_w, c1 * concat_h * concat_w);
    auto concat_wb = aexrt::upload_float32(device, concat_weight);
    auto concat_bb = aexrt::upload_float32(device, concat_bias);
    auto concat_yb = aexrt::allocate_float32_uav(device, concat_expected.size());
    std::vector<AexrtBuffer*> concat_inputs = {concat_left.get(), concat_right.get()};
    std::vector<uint32_t> concat_channels = {c0, c1};
    aexrt::concat_conv1x1_float32(
        device,
        concat_inputs,
        concat_channels,
        concat_wb,
        concat_bb,
        concat_yb,
        1,
        concat_h,
        concat_w,
        concat_out_channels,
        true);
    auto concat_y = aexrt::download_float32(device, concat_yb, concat_expected.size());
    float concat_max_diff = 0.0f;
    for (size_t i = 0; i < concat_y.size(); ++i) {
        concat_max_diff = std::max(concat_max_diff, std::fabs(concat_y[i] - concat_expected[i]));
    }
    std::cout << "AEXRT C++ View+Concat+1x1Conv+SiLU max diff: " << concat_max_diff << "\n";
    return concat_max_diff < 2e-5f ? 0 : 4;
}
