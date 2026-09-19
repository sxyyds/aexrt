#include "../../native/aexrt.hpp"

#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static bool ends_with_ci(const std::string& text, const std::string& suffix) {
    if (suffix.size() > text.size()) return false;
    for (size_t i = 0; i < suffix.size(); ++i) {
        const unsigned char a = static_cast<unsigned char>(text[text.size() - suffix.size() + i]);
        const unsigned char b = static_cast<unsigned char>(suffix[i]);
        if (std::tolower(a) != std::tolower(b)) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (!aexrt_d3d12_probe()) {
        std::cerr << "no D3D12 adapter available\n";
        return 1;
    }

    aexrt::Device device(0);
    const AexrtD3D12Capabilities capabilities = device.capabilities();
    std::cout << "AEXRT D3D12 capabilities: shader_model=0x"
              << std::hex << capabilities.highest_shader_model << std::dec
              << " native_fp16=" << capabilities.native_fp16_supported
              << " wave_mma_tier=" << capabilities.wave_mma_tier
              << " dxc=" << capabilities.dxc_available << "\n";
    std::string package_path = "examples\\cs2V8_320.aexrt";
    uint32_t benchmark_runs = 0;
    bool profile = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--runs") {
            if (i + 1 >= argc) {
                std::cerr << "missing value for --runs\n";
                return 1;
            }
            benchmark_runs = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--profile") {
            profile = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "native_yolo_package.exe [model.onnx|model.aexrt] [--runs N] [--profile]\n";
            return 0;
        } else {
            package_path = arg;
        }
    }
    if (profile) {
        _putenv_s("AEXRT_NATIVE_D3D12_PROFILE_REPLAY", "1");
    }
    const bool use_onnx = ends_with_ci(package_path, ".onnx");
    aexrt::YoloModel model;
    try {
        model = use_onnx
            ? aexrt::compile_yolo_from_onnx(device, package_path)
            : aexrt::load_engine(device, package_path);
    } catch (const std::exception& error) {
        std::cerr << "failed to load AEXRT model '" << package_path << "': " << error.what() << "\n";
        return 2;
    }

    std::cout << "AEXRT C++ YOLO " << (use_onnx ? "ONNX" : "package")
              << " loaded: mode=" << static_cast<uint32_t>(model.package_mode())
              << " anchors=" << model.anchor_count()
              << " classes=" << model.class_count()
              << " input_elements=" << model.input_element_count()
              << " graph_nodes=" << model.graph_node_count()
              << " values=" << model.graph_value_count()
              << " constants=" << model.constant_count()
              << " prepared_commands=" << model.prepared_command_count()
              << " supported_commands=" << model.supported_prepared_command_count()
              << " unsupported_commands=" << model.unsupported_prepared_command_count()
              << " executable=" << (model.executable() ? 1 : 0) << "\n";

    if (!model.executable()) {
        return 0;
    }

    if (model.package_mode() == AEXRT_YOLO_PACKAGE_NATIVE_D3D12_GRAPH) {
        std::vector<float> input(model.input_element_count(), 0.0f);
        auto detections = aexrt::run_yolo(device, model, input, 16);
        std::cout << "AEXRT C++ TileFlow: conv1x1=" << model.tileflow_conv1x1_count()
                  << " spatial3x3=" << model.tileflow_3x3_spatial_count()
                  << " pack4_3x3=" << model.tileflow_3x3_pack4_count()
                  << " pack8_3x3=" << model.tileflow_3x3_pack8_count()
                  << " implicit_gemm_3x3=" << model.tileflow_3x3_implicit_gemm_count()
                  << " shape40=" << model.tileflow_3x3_implicit_gemm_40x40_count()
                  << " shape20=" << model.tileflow_3x3_implicit_gemm_20x20_count()
                  << " shape10=" << model.tileflow_3x3_implicit_gemm_10x10_count()
                  << " exact40_64=" << model.tileflow_3x3_exact_40x40_64x64_count()
                  << " exact20_64=" << model.tileflow_3x3_exact_20x20_64x64_count()
                  << " exact10_128=" << model.tileflow_3x3_exact_10x10_128x128_count()
                  << " exact20_128x64=" << model.tileflow_3x3_exact_20x20_128x64_count()
                  << " exact10_256x64=" << model.tileflow_3x3_exact_10x10_256x64_count()
                  << " native_fp16=" << model.tileflow_native_fp16_count()
                  << " skipped=" << model.prepared_skipped_command_count()
                  << " late_concat_conv1x1=" << model.late_concat_conv1x1_fusion_count()
                  << " concat_residual_cv2=" << model.concat_residual_conv1x1_fusion_count()
                  << " paired_conv3x3=" << model.paired_conv3x3_fusion_count()
                  << " c2f_bottleneck=" << model.c2f_bottleneck_superblock_count()
                  << " c2f_tail=" << model.c2f_tail_residual_fusion_count()
                  << " head_fusion=" << (model.head_fusion_enabled() ? 1 : 0)
                  << "\n";
        if (benchmark_runs != 0) {
            double total_ms = 0.0;
            double total_memcpy_ms = 0.0;
            double total_submit_ms = 0.0;
            double total_fence_ms = 0.0;
            double total_readback_ms = 0.0;
            uint32_t timed_runs = 0;
            for (uint32_t i = 0; i < benchmark_runs; ++i) {
                const auto t0 = std::chrono::high_resolution_clock::now();
                detections = aexrt::run_yolo(device, model, input, 16);
                const auto t1 = std::chrono::high_resolution_clock::now();
                total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                const AexrtYoloRunTiming timing = model.last_run_timing();
                if (timing.valid) {
                    total_memcpy_ms += timing.memcpy_ms;
                    total_submit_ms += timing.submit_ms;
                    total_fence_ms += timing.fence_ms;
                    total_readback_ms += timing.readback_ms;
                    ++timed_runs;
                }
            }
            const double avg_ms = total_ms / benchmark_runs;
            std::cout << "AEXRT C++ native graph avg_infer=" << std::fixed << std::setprecision(3)
                      << avg_ms << " ms fps=" << (avg_ms > 0.0 ? 1000.0 / avg_ms : 0.0)
                      << " runs=" << benchmark_runs << "\n";
            if (timed_runs != 0) {
                const double avg_memcpy_ms = total_memcpy_ms / timed_runs;
                const double avg_submit_ms = total_submit_ms / timed_runs;
                const double avg_fence_ms = total_fence_ms / timed_runs;
                const double avg_readback_ms = total_readback_ms / timed_runs;
                const double accounted_ms = avg_memcpy_ms + avg_submit_ms + avg_fence_ms + avg_readback_ms;
                std::cout << "AEXRT C++ native graph avg_breakdown: memcpy=" << std::setprecision(4)
                          << avg_memcpy_ms << " ms submit=" << avg_submit_ms
                          << " ms fence=" << avg_fence_ms
                          << " ms readback=" << avg_readback_ms
                          << " ms accounted=" << accounted_ms
                          << " ms unaccounted=" << (avg_ms - accounted_ms)
                          << " ms runs=" << timed_runs << "\n";
            }
        }
        if (profile) {
            const uint32_t event_count = model.profile_event_count();
            double total_gpu_ms = 0.0;
            for (uint32_t i = 0; i < event_count; ++i) {
                total_gpu_ms += model.profile_event_ms(i);
            }
            std::cout << "AEXRT C++ GPU dispatch profile: events=" << event_count
                      << " summed_gpu_ms=" << std::fixed << std::setprecision(4) << total_gpu_ms << "\n";
            for (uint32_t i = 0; i < event_count; ++i) {
                std::cout << "  [" << std::setw(3) << i << "] "
                          << std::fixed << std::setprecision(4) << model.profile_event_ms(i)
                          << " ms  " << model.profile_event_label(i) << "\n";
            }
        }
        std::cout << "AEXRT C++ native graph detections: " << detections.size() << "\n";
        return 0;
    }

    if (model.package_mode() != AEXRT_YOLO_PACKAGE_OUTPUT0_POSTPROCESS) {
        return 0;
    }

    std::vector<float> output0(model.input_element_count(), 0.0f);
    const uint32_t anchors = model.anchor_count();
    if (model.class_count() < 2 || anchors < 5) {
        std::cout << "AEXRT C++ YOLO package loaded: anchors=" << anchors
                  << " classes=" << model.class_count()
                  << " input_elements=" << model.input_element_count() << "\n";
        return 0;
    }
    auto set_box = [&](uint32_t a, float cx, float cy, float w, float h) {
        output0[0 * anchors + a] = cx;
        output0[1 * anchors + a] = cy;
        output0[2 * anchors + a] = w;
        output0[3 * anchors + a] = h;
    };
    auto set_score = [&](uint32_t a, uint32_t cls, float score) {
        output0[(4 + cls) * anchors + a] = score;
    };

    set_box(0, 10.0f, 10.0f, 10.0f, 10.0f);
    set_box(1, 11.0f, 10.0f, 10.0f, 10.0f);
    set_box(2, 50.0f, 50.0f, 8.0f, 8.0f);
    set_box(3, 80.0f, 80.0f, 8.0f, 8.0f);
    set_box(4, 70.0f, 70.0f, 8.0f, 8.0f);
    set_score(0, 0, 0.9f);
    set_score(1, 0, 0.8f);
    set_score(2, 1, 0.7f);
    set_score(3, 0, 0.2f);
    set_score(4, 0, 0.4f);

    auto detections = aexrt::run_yolo(device, model, output0, 4);
    std::cout << "AEXRT C++ YOLO package detections: " << detections.size() << "\n";
    for (const auto& det : detections) {
        std::cout << "  cls=" << static_cast<int>(det.class_id)
                  << " score=" << det.score
                  << " xyxy=[" << det.x1 << ", " << det.y1 << ", " << det.x2 << ", " << det.y2 << "]\n";
    }

    if (detections.size() != 3) {
        return 2;
    }
    if (static_cast<int>(detections[0].class_id) != 0 || std::fabs(detections[0].score - 0.9f) > 1e-6f) {
        return 3;
    }
    if (static_cast<int>(detections[1].class_id) != 1 || std::fabs(detections[1].score - 0.7f) > 1e-6f) {
        return 4;
    }
    if (static_cast<int>(detections[2].class_id) != 0 || std::fabs(detections[2].score - 0.4f) > 1e-6f) {
        return 5;
    }
    return 0;
}
