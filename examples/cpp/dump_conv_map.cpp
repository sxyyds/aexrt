#include "../../native/aexrt.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: dump_conv_map.exe model.aexrt [out.json]\n";
        return 1;
    }
    try {
        aexrt::Device device(0);
        auto model = aexrt::load_engine(device, argv[1]);
        const uint32_t count = aexrt_yolo_conv_plan_count(model.get());
        std::map<uint32_t, uint32_t> out;
        uint32_t fused = 0;
        for (uint32_t i = 0; i < count; ++i) {
            AexrtConvPlanInfo info{};
            info.struct_size = sizeof(info);
            if (!aexrt_yolo_conv_plan_info(model.get(), i, &info)) continue;
            if (info.in_fusion_group) {
                ++fused;
                continue;
            }
            out[info.command_index] = info.planned_algorithm;
        }
        std::ostream* os = &std::cout;
        std::ofstream file;
        if (argc > 2) {
            file.open(argv[2], std::ios::binary);
            if (!file) {
                std::cerr << "cannot write " << argv[2] << "\n";
                return 2;
            }
            os = &file;
        }
        *os << "{\n";
        size_t n = 0;
        for (const auto& item : out) {
            *os << "    \"" << item.first << "\": " << item.second
                << (++n < out.size() ? "," : "") << "\n";
        }
        *os << "}\n";
        std::cerr << "commands=" << count << " map=" << out.size()
                  << " fused_skipped=" << fused << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}
