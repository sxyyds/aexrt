// combo_repro: 重现 conv_autotune 组合应用失败的定位工具。
//
// 逐前缀应用 override 集合（{}，{w0}，{w0,w1}，...），每次 warmup+run，
// 定位首个失败前缀；失败时打印 last_error 与设备移除原因。
//
// 用法: combo_repro.exe model.aexrt map.json [--adapter N]
// map.json: {"cmd_index": alg, ...}（conv_autotune --emit-map 产物）

#include "../../native/aexrt.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h>

namespace {

void set_env(const char* name, const char* value) { ::SetEnvironmentVariableA(name, value); }

std::string read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

/* 极简 JSON 对 {"k": v, ...} 的解析（无嵌套，键值均为整数）。 */
std::map<uint32_t, uint32_t> parse_map(const std::string& text) {
    std::map<uint32_t, uint32_t> out;
    size_t pos = 0;
    while (true) {
        pos = text.find('"', pos);
        if (pos == std::string::npos) break;
        const size_t key_end = text.find('"', pos + 1);
        if (key_end == std::string::npos) break;
        const std::string key = text.substr(pos + 1, key_end - pos - 1);
        pos = text.find(':', key_end);
        if (pos == std::string::npos) break;
        size_t value_end = text.find_first_of(",}", pos);
        if (value_end == std::string::npos) value_end = text.size();
        const std::string value = text.substr(pos + 1, value_end - pos - 1);
        try {
            out[static_cast<uint32_t>(std::stoul(key))] = static_cast<uint32_t>(std::stoul(value));
        } catch (...) {
        }
        pos = value_end;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: combo_repro.exe model.aexrt map.json [--adapter N] [--restore-first]\n";
        return 1;
    }
    uint32_t adapter = 0;
    bool restore_first = false;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--adapter" && i + 1 < argc) adapter = static_cast<uint32_t>(std::stoul(argv[++i]));
        if (arg == "--restore-first") restore_first = true;
    }
    set_env("AEXRT_NATIVE_D3D12_PROFILE_REPLAY", "1");
    set_env("AEXRT_NATIVE_D3D12_PIPELINE_FRAMES", "1");

    const std::map<uint32_t, uint32_t> overrides = parse_map(read_file(argv[2]));
    if (overrides.empty()) {
        std::cerr << "empty map\n";
        return 1;
    }
    std::cout << "map: " << overrides.size() << " overrides\n";

    try {
        aexrt::Device device(adapter);
        aexrt::YoloModel model = aexrt::load_engine(device, argv[1]);
        if (!model.executable()) {
            std::cerr << "engine not executable\n";
            return 2;
        }
        const uint64_t elements = model.input_element_count();
        std::vector<float> input(static_cast<size_t>(elements));
        uint32_t state = 0xC2A11u;
        for (auto& v : input) {
            state = state * 1664525u + 1013904223u;
            v = static_cast<float>(static_cast<int32_t>(state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
        std::vector<AexrtYoloDetection> dets;
        auto run_ok = [&](const char* tag) -> bool {
            dets.clear();
            dets.resize(64);
            uint32_t n = 0;
            const int ok = aexrt_yolo_run(device.get(), model.get(), input.data(), input.size(), dets.data(), 64, &n);
            if (!ok) {
                const char* reason = aexrt_yolo_get_last_error(model.get());
                std::cout << "  [" << tag << "] RUN FAILED: "
                          << (reason && reason[0] ? reason : "no reason") << "\n";
                return false;
            }
            dets.resize(n);
            return true;
        };

        /* 全基线 sanity */
        if (!run_ok("baseline")) return 3;
        std::cout << "baseline ok, dets " << dets.size() << "\n";

        /* 逐前缀应用 */
        uint32_t applied = 0;
        for (const auto& [cmd_index, alg] : overrides) {
            if (!aexrt_yolo_override_conv_algorithm(model.get(), cmd_index, alg)) {
                std::cout << "override cmd#" << cmd_index << " alg=" << alg << " rejected\n";
                continue;
            }
            ++applied;
            std::cout << "prefix " << applied << ": cmd#" << cmd_index << " -> alg " << alg << " ... ";
            for (int i = 0; i < 2; ++i) {
                if (!run_ok("combo")) {
                    std::cout << "FAILED at prefix " << applied << " (cmd#" << cmd_index
                              << " alg=" << alg << ")\n";
                    return 4;
                }
            }
            std::cout << "ok (dets " << dets.size() << ")\n";
        }
        std::cout << "all prefixes ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "exception: " << error.what() << "\n";
        return 5;
    }
}
