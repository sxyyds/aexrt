// dml_compare: AEXRT native D3D12 engine vs ONNX Runtime DirectML EP 基准对比。
// 用途（P0 可比性）：同机、同模型、同输入，测 e2e 墙钟时间 + AEXRT GPU 时间分解。
// 所有产物（报告）写入 build\benchmarks，不写 C 盘。
//
// 用法:
//   dml_compare.exe --list-adapters
//   dml_compare.exe [--runs N] [--warmup M] [--models a,b,c] [--skip-dml] [--skip-aexrt]
//                   [--adapter N | --warp] [--dml-device N] [--out report.md]

#include "../../native/aexrt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h>

#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"

namespace {

/* 用 Win32 进程环境块，确保 native DLL 里 GetEnvironmentVariableA 一定可见。 */
void set_env(const char* name, const char* value) {
    ::SetEnvironmentVariableA(name, value);
}

void clear_env(const char* name) {
    ::SetEnvironmentVariableA(name, nullptr);
}

struct ModelEntry {
    const char* name;
    const char* engine_path;
    const char* onnx_path;
};

const ModelEntry kDefaultModels[] = {
    {"cs2V8_320", "examples\\cs2V8_320.aexrt", "models\\cs2V8_320.onnx"},
    {"apex10w_yolov5_320", "examples\\apex10w_yolov5_320.aexrt", "models\\apex10w_yolov5_320.onnx"},
    {"DYv11s_414_320", "examples\\DYv11s_414_320.aexrt", "models\\DYv11s_414_320.onnx"},
};

struct Stats {
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

Stats compute_stats(std::vector<double> samples) {
    Stats stats;
    if (samples.empty()) {
        return stats;
    }
    std::sort(samples.begin(), samples.end());
    const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    stats.mean_ms = sum / static_cast<double>(samples.size());
    stats.min_ms = samples.front();
    stats.max_ms = samples.back();
    const size_t mid = samples.size() / 2;
    stats.median_ms = (samples.size() % 2 != 0)
        ? samples[mid]
        : (samples[mid - 1] + samples[mid]) * 0.5;
    return stats;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// 确定性输入：简单 LCG，两个后端使用完全相同的字节。
void fill_input(std::vector<float>& input, uint32_t seed) {
    uint32_t state = seed != 0 ? seed : 1u;
    for (auto& value : input) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<float>(static_cast<int32_t>(state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
    }
}

std::string wide_to_utf8(const wchar_t* text) {
    if (!text) return {};
    const int size = ::WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) return {};
    std::string out(static_cast<size_t>(size - 1), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), size, nullptr, nullptr);
    return out;
}

std::wstring utf8_to_wide(const std::string& text) {
    if (text.empty()) return {};
    const int size = ::MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (size <= 1) return {};
    std::wstring out(static_cast<size_t>(size - 1), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, out.data(), size);
    return out;
}

std::string vendor_name(uint32_t vendor_id) {
    switch (vendor_id) {
        case 0x10DE: return "NVIDIA";
        case 0x1002: return "AMD";
        case 0x8086: return "Intel";
        case 0x1414: return "Microsoft";
        default: return "unknown";
    }
}

struct AdapterRow {
    uint32_t raw_index;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t is_software;
    uint32_t dedicated_video_memory_mb;
    std::string description;
    std::string caps;
};

// 逐个适配器创建设备并查询能力，生成 vendor matrix。
std::vector<AdapterRow> probe_adapters() {
    std::vector<AdapterRow> rows;
    AexrtAdapterInfo infos[16]{};
    const uint32_t count = aexrt_d3d12_enumerate_adapters(infos, 16);
    for (uint32_t i = 0; i < count && i < 16; ++i) {
        AdapterRow row;
        row.raw_index = infos[i].raw_index;
        row.vendor_id = infos[i].vendor_id;
        row.device_id = infos[i].device_id;
        row.is_software = infos[i].is_software;
        row.dedicated_video_memory_mb = infos[i].dedicated_video_memory_mb;
        row.description = wide_to_utf8(infos[i].description);
        char env_value[16]{};
        std::snprintf(env_value, sizeof(env_value), "%u", infos[i].raw_index);
        set_env("AEXRT_NATIVE_D3D12_ADAPTER", env_value);
        try {
            aexrt::Device device(0);
            const AexrtD3D12Capabilities caps = device.capabilities();
            std::ostringstream stream;
            stream << "SM=0x" << std::hex << caps.highest_shader_model << std::dec
                   << " fp16=" << caps.native_fp16_supported
                   << " wave_mma=" << caps.wave_mma_tier
                   << " dxc=" << caps.dxc_available;
            row.caps = stream.str();
        } catch (const std::exception&) {
            row.caps = "device creation failed";
        }
        rows.push_back(std::move(row));
    }
    clear_env("AEXRT_NATIVE_D3D12_ADAPTER");
    return rows;
}

struct AexrtResult {
    bool ok = false;
    std::string error;
    Stats e2e;
    Stats gpu;
    Stats memcpy_ms;
    Stats submit_ms;
    Stats fence_ms;
    Stats readback_ms;
    uint64_t input_elements = 0;
    uint32_t profile_events = 0;
    uint32_t detection_count = 0;
    std::vector<std::pair<std::string, double>> top_events;
};

AexrtResult run_aexrt(const ModelEntry& entry, uint32_t adapter_index, uint32_t warmup, uint32_t runs,
                      uint32_t event_top_n, uint32_t pipeline_frames) {
    AexrtResult result;
    try {
        aexrt::Device device(adapter_index);
        aexrt::YoloModel model = aexrt::load_engine(device, entry.engine_path);
        if (!model.executable()) {
            result.error = "engine not executable on this device";
            return result;
        }
        result.input_elements = model.input_element_count();
        std::vector<float> input(static_cast<size_t>(result.input_elements));
        fill_input(input, 0xC2A11u);
        std::vector<double> e2e_samples;
        std::vector<double> gpu_samples;
        std::vector<double> memcpy_samples;
        std::vector<double> submit_samples;
        std::vector<double> fence_samples;
        std::vector<double> readback_samples;
        e2e_samples.reserve(runs);
        gpu_samples.reserve(runs);

        auto detections = aexrt::run_yolo(device, model, input, 64);
        result.detection_count = static_cast<uint32_t>(detections.size());
        for (uint32_t i = 0; i < warmup; ++i) {
            detections = aexrt::run_yolo(device, model, input, 64);
        }
        for (uint32_t i = 0; i < runs; ++i) {
            const double t0 = now_ms();
            detections = aexrt::run_yolo(device, model, input, 64);
            const double t1 = now_ms();
            e2e_samples.push_back(t1 - t0);
            const AexrtYoloRunTiming timing = model.last_run_timing();
            if (timing.valid != 0) {
                memcpy_samples.push_back(timing.memcpy_ms);
                submit_samples.push_back(timing.submit_ms);
                fence_samples.push_back(timing.fence_ms);
                readback_samples.push_back(timing.readback_ms);
            }
            double gpu_total = 0.0;
            const uint32_t events = model.profile_event_count();
            for (uint32_t e = 0; e < events; ++e) {
                gpu_total += model.profile_event_ms(e);
            }
            gpu_samples.push_back(gpu_total);
        }
        /* N-buffer 模式下 run 返回的是上一帧结果：排空取最后一帧，
         * 并以排空结果作为 detection_count（与同步模式同源）。 */
        if (pipeline_frames >= 2u) {
            std::vector<AexrtYoloDetection> flushed;
            if (model.flush_pipeline(flushed)) {
                result.detection_count = static_cast<uint32_t>(flushed.size());
            }
        }
        result.profile_events = model.profile_event_count();
        if (event_top_n > 0) {
            std::vector<std::pair<std::string, double>> events;
            events.reserve(result.profile_events);
            for (uint32_t e = 0; e < result.profile_events; ++e) {
                events.emplace_back(model.profile_event_label(e) != nullptr
                                        ? std::string(model.profile_event_label(e))
                                        : std::string("event") + std::to_string(e),
                                    model.profile_event_ms(e));
            }
            std::sort(events.begin(), events.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
            result.top_events.assign(events.begin(),
                                     events.begin() + static_cast<long>(std::min<size_t>(events.size(), event_top_n)));
        }
        result.e2e = compute_stats(std::move(e2e_samples));
        result.gpu = compute_stats(std::move(gpu_samples));
        result.memcpy_ms = compute_stats(std::move(memcpy_samples));
        result.submit_ms = compute_stats(std::move(submit_samples));
        result.fence_ms = compute_stats(std::move(fence_samples));
        result.readback_ms = compute_stats(std::move(readback_samples));
        result.ok = true;
    } catch (const std::exception& error) {
        result.error = error.what();
    }
    return result;
}

struct DmlResult {
    bool ok = false;
    std::string error;
    Stats e2e;
    uint64_t input_elements = 0;
    uint64_t output_elements = 0;
    double output_checksum = 0.0;
};

DmlResult run_dml(const ModelEntry& entry, uint32_t dml_device, uint32_t warmup, uint32_t runs) {
    DmlResult result;
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dml_compare");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(options, dml_device));

        const std::wstring wide_path = utf8_to_wide(entry.onnx_path);
        Ort::Session session(env, wide_path.c_str(), options);

        Ort::AllocatorWithDefaultOptions allocator;
        const size_t input_count = session.GetInputCount();
        if (input_count == 0) {
            result.error = "model has no inputs";
            return result;
        }
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto input_type = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        if (input_type.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            result.error = "model input is not float32";
            return result;
        }
        std::vector<int64_t> shape = input_type.GetShape();
        for (auto& dim : shape) {
            if (dim <= 0) dim = 1;  // 动态维度按 1 处理
        }
        uint64_t elements = 1;
        for (const auto dim : shape) {
            elements *= static_cast<uint64_t>(dim);
        }
        result.input_elements = elements;

        std::vector<float> input(static_cast<size_t>(elements));
        fill_input(input, 0xC2A11u);
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto tensor = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), shape.data(), shape.size());
        const char* input_names[] = {input_name.get()};
        const size_t output_count = session.GetOutputCount();
        std::vector<Ort::AllocatedStringPtr> output_name_holders;
        std::vector<const char*> output_names;
        output_name_holders.reserve(output_count);
        output_names.reserve(output_count);
        for (size_t i = 0; i < output_count; ++i) {
            output_name_holders.push_back(session.GetOutputNameAllocated(i, allocator));
            output_names.push_back(output_name_holders.back().get());
        }

        auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1,
                                   output_names.data(), output_names.size());
        for (uint32_t i = 0; i < warmup; ++i) {
            outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1,
                                  output_names.data(), output_names.size());
        }

        std::vector<double> samples;
        samples.reserve(runs);
        for (uint32_t i = 0; i < runs; ++i) {
            const double t0 = now_ms();
            outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1,
                                  output_names.data(), output_names.size());
            const double t1 = now_ms();
            samples.push_back(t1 - t0);
        }
        if (!outputs.empty()) {
            const float* data = outputs.front().GetTensorData<float>();
            const auto out_info = outputs.front().GetTensorTypeAndShapeInfo();
            result.output_elements = out_info.GetElementCount();
            double checksum = 0.0;
            const uint64_t limit = std::min<uint64_t>(result.output_elements, 65536);
            for (uint64_t i = 0; i < limit; ++i) {
                checksum += static_cast<double>(data[i]);
            }
            result.output_checksum = checksum;
        }
        result.e2e = compute_stats(std::move(samples));
        result.ok = true;
    } catch (const Ort::Exception& error) {
        result.error = error.what();
    } catch (const std::exception& error) {
        result.error = error.what();
    }
    return result;
}

std::string fmt(double value, const char* suffix = " ms") {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << value << suffix;
    return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
    uint32_t runs = 100;
    uint32_t warmup = 20;
    uint32_t adapter_index = 0;
    uint32_t dml_device = 0;
    uint32_t event_top_n = 15;
    uint32_t pipeline_frames = 1;
    bool list_adapters = false;
    bool skip_aexrt = false;
    bool skip_dml = false;
    bool use_warp = false;
    std::string out_path = "build\\benchmarks\\dml_compare_report.md";
    std::vector<std::string> requested_models;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--runs") {
            runs = static_cast<uint32_t>(std::stoul(next_value("--runs")));
        } else if (arg == "--warmup") {
            warmup = static_cast<uint32_t>(std::stoul(next_value("--warmup")));
        } else if (arg == "--adapter") {
            adapter_index = static_cast<uint32_t>(std::stoul(next_value("--adapter")));
        } else if (arg == "--dml-device") {
            dml_device = static_cast<uint32_t>(std::stoul(next_value("--dml-device")));
        } else if (arg == "--events") {
            event_top_n = static_cast<uint32_t>(std::stoul(next_value("--events")));
        } else if (arg == "--pipeline") {
            pipeline_frames = static_cast<uint32_t>(std::stoul(next_value("--pipeline")));
            if (pipeline_frames < 1u || pipeline_frames > 4u) {
                std::cerr << "--pipeline must be within [1, 4]\n";
                return 1;
            }
        } else if (arg == "--models") {
            std::stringstream stream(next_value("--models"));
            std::string token;
            while (std::getline(stream, token, ',')) {
                if (!token.empty()) requested_models.push_back(token);
            }
        } else if (arg == "--out") {
            out_path = next_value("--out");
        } else if (arg == "--list-adapters") {
            list_adapters = true;
        } else if (arg == "--skip-aexrt") {
            skip_aexrt = true;
        } else if (arg == "--skip-dml") {
            skip_dml = true;
        } else if (arg == "--warp") {
            use_warp = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "dml_compare.exe [--runs N] [--warmup M] [--models a,b,c]\n"
                << "               [--adapter N | --warp] [--dml-device N] [--out file.md]\n"
                << "               [--pipeline 1..4] [--skip-aexrt] [--skip-dml] [--list-adapters]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return 1;
        }
    }

    if (use_warp) {
        set_env("AEXRT_NATIVE_D3D12_WARP", "1");
    }
    // 打开整模型 GPU 时间戳 profile（时间分解用）。
    set_env("AEXRT_NATIVE_D3D12_PROFILE_REPLAY", "1");
    // N-buffer 流水线帧数（默认 1 = 同步语义；>=2 时 run 返回上一帧结果，
    // 循环结束后 flush_pipeline 排空并回收最后一帧）。
    {
        char env_value[16]{};
        std::snprintf(env_value, sizeof(env_value), "%u", pipeline_frames);
        set_env("AEXRT_NATIVE_D3D12_PIPELINE_FRAMES", env_value);
    }

    if (list_adapters) {
        const std::vector<AdapterRow> rows = probe_adapters();
        std::cout << "raw  vendor    dev_id  sw  vram_mb  caps                          description\n";
        std::cout << "---- --------  ------  --  -------  ----------------------------  ----------------\n";
        for (const auto& row : rows) {
            std::cout << std::setw(4) << row.raw_index << "  " << std::setw(8) << std::left
                      << vendor_name(row.vendor_id) << std::right << "  " << std::setw(6) << std::hex
                      << row.device_id << std::dec << "  " << std::setw(2) << row.is_software << "  "
                      << std::setw(7) << row.dedicated_video_memory_mb << "  " << std::setw(28) << std::left
                      << row.caps << std::right << "  " << row.description << "\n";
        }
        std::cout << "\nselect with: AEXRT_NATIVE_D3D12_ADAPTER=<raw> or --adapter <hardware index> or --warp\n";
        return 0;
    }

    std::vector<ModelEntry> models;
    for (const auto& entry : kDefaultModels) {
        if (requested_models.empty() ||
            std::find(requested_models.begin(), requested_models.end(), entry.name) != requested_models.end()) {
            models.push_back(entry);
        }
    }
    if (models.empty()) {
        std::cerr << "no matching models\n";
        return 1;
    }

    std::vector<AdapterRow> adapter_rows = probe_adapters();

    std::ostringstream report;
    report << "# AEXRT vs ONNX Runtime DirectML comparison\n\n";
    report << "- adapters:\n";
    for (const auto& row : adapter_rows) {
        report << "    - raw " << row.raw_index << ": " << vendor_name(row.vendor_id) << " "
               << row.description << (row.is_software != 0 ? " (software)" : "") << " [" << row.caps
               << "]\n";
    }
    report << "- onnxruntime: " << OrtGetApiBase()->GetVersionString() << " (DirectML EP, device "
           << dml_device << ")\n";
    report << "- directml runtime: Windows inbox DirectML.dll\n";
    report << "- aexrt device: " << (use_warp ? "WARP software" : "hardware adapter " + std::to_string(adapter_index))
           << "\n";
    report << "- warmup / runs: " << warmup << " / " << runs << "\n";
    report << "- aexrt pipeline frames: " << pipeline_frames
           << (pipeline_frames >= 2u ? " (N-buffer, run returns previous frame; flush at end)" : " (synchronous)")
           << "\n";
    report << "- fairness: same deterministic input bytes; both paths include host->device upload; "
              "single thread; first-load compile excluded via warmup\n\n";

    for (const auto& entry : models) {
        std::cout << "=== " << entry.name << " ===\n";
        report << "## " << entry.name << "\n\n";
        AexrtResult aexrt;
        DmlResult dml;
        if (!skip_aexrt) {
            std::cout << "running AEXRT engine (" << warmup << " warmup + " << runs
                      << " runs, pipeline=" << pipeline_frames << ")...\n";
            aexrt = run_aexrt(entry, adapter_index, warmup, runs, event_top_n, pipeline_frames);
            if (aexrt.ok) {
                std::cout << "  AEXRT e2e median " << fmt(aexrt.e2e.median_ms)
                          << " (mean " << fmt(aexrt.e2e.mean_ms) << ", min " << fmt(aexrt.e2e.min_ms)
                          << "), gpu " << fmt(aexrt.gpu.median_ms) << ", events "
                          << aexrt.profile_events << ", dets " << aexrt.detection_count << "\n";
            } else {
                std::cout << "  AEXRT FAILED: " << aexrt.error << "\n";
            }
        }
        if (!skip_dml) {
            std::cout << "running DirectML EP (" << warmup << " warmup + " << runs << " runs)...\n";
            dml = run_dml(entry, dml_device, warmup, runs);
            if (dml.ok) {
                std::cout << "  DML   e2e median " << fmt(dml.e2e.median_ms)
                          << " (mean " << fmt(dml.e2e.mean_ms) << ", min " << fmt(dml.e2e.min_ms)
                          << "), out_elems " << dml.output_elements << "\n";
            } else {
                std::cout << "  DML   FAILED: " << dml.error << "\n";
            }
        }

        report << "| backend | e2e median ms | e2e mean ms | e2e min ms | fps (median) | note |\n";
        report << "|---|---|---|---|---|---|\n";
        if (aexrt.ok) {
            report << "| AEXRT native D3D12 | " << fmt(aexrt.e2e.median_ms, "") << " | "
                   << fmt(aexrt.e2e.mean_ms, "") << " | " << fmt(aexrt.e2e.min_ms, "") << " | "
                   << std::fixed << std::setprecision(1) << 1000.0 / aexrt.e2e.median_ms << " | gpu "
                   << fmt(aexrt.gpu.median_ms, "") << " ms, " << aexrt.profile_events
                   << " profile events, dets " << aexrt.detection_count << " |\n";
        } else if (!skip_aexrt) {
            report << "| AEXRT native D3D12 | failed | | | | " << aexrt.error << " |\n";
        }
        if (dml.ok) {
            report << "| ORT DirectML EP | " << fmt(dml.e2e.median_ms, "") << " | "
                   << fmt(dml.e2e.mean_ms, "") << " | " << fmt(dml.e2e.min_ms, "") << " | "
                   << std::fixed << std::setprecision(1) << 1000.0 / dml.e2e.median_ms << " | out "
                   << dml.output_elements << " elems |\n";
        } else if (!skip_dml) {
            report << "| ORT DirectML EP | failed | | | | " << dml.error << " |\n";
        }
        if (aexrt.ok && dml.ok) {
            const double speedup = dml.e2e.median_ms / aexrt.e2e.median_ms;
            report << "\nspeedup (DML median / AEXRT median): " << std::fixed << std::setprecision(2)
                   << speedup << "x\n";
        }
        if (aexrt.ok) {
            report << "\nAEXRT timing breakdown (median ms): memcpy "
                   << fmt(aexrt.memcpy_ms.median_ms, "") << ", submit " << fmt(aexrt.submit_ms.median_ms, "")
                   << ", fence " << fmt(aexrt.fence_ms.median_ms, "") << ", readback "
                   << fmt(aexrt.readback_ms.median_ms, "") << ", gpu "
                   << fmt(aexrt.gpu.median_ms, "") << "\n";
            if (!aexrt.top_events.empty()) {
                report << "\ntop GPU events (last run):\n\n```\n";
                for (const auto& [label, ms] : aexrt.top_events) {
                    report << "  " << std::fixed << std::setprecision(3) << std::setw(8) << ms << " ms  "
                           << label << "\n";
                }
                report << "```\n";
            }
        }
        report << "\n";
    }

    std::cout << "\n" << report.str();

    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(out_path).parent_path(), ec);
    std::ofstream file(out_path, std::ios::binary);
    if (!file) {
        std::cerr << "failed to open report file: " << out_path << "\n";
        return 1;
    }
    file << report.str();
    std::cout << "saved: " << out_path << "\n";
    return 0;
}
