// conv_autotune: per-device 卷积算法自动调优工具（P1.3）。
//
// 流程：加载 .aexrt 引擎 -> 枚举 CONV/CONV_SILU 命令 -> 唯一形状去重 ->
// 逐候选 override + 整图实测（含数值门禁）-> 应用全部最优 -> 复测 ->
// 输出报告（build\benchmarks）与 tuned 头文件（集成到用户程序）。
//
// 用法:
//   conv_autotune.exe examples\DYv11s_414_320.aexrt [--adapter N | --warp]
//                      [--runs 10] [--out report.md] [--emit-header tuned.h]

#include "../../native/aexrt.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <windows.h>

namespace {

void set_env(const char* name, const char* value) { ::SetEnvironmentVariableA(name, value); }
void clear_env(const char* name) { ::SetEnvironmentVariableA(name, nullptr); }

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct ShapeKey {
    uint32_t silu;
    uint32_t in_c;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t out_c;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t k;
    uint32_t s;
    uint32_t groups;
    uint32_t dilation;
    /* groups/dilation 必须进 key：同形状的普通卷积与分组卷积共享 key 时，
     * 普通（groups=1）代表测得的 winograd/特化算法会被错误套用到分组卷积上，
     * 按 groups=1 的权重布局寻址 → GPU 越界（实测设备移除）。 */
    bool operator<(const ShapeKey& o) const {
        return std::tie(silu, in_c, in_h, in_w, out_c, out_h, out_w, k, s, groups, dilation) <
               std::tie(o.silu, o.in_c, o.in_h, o.in_w, o.out_c, o.out_h, o.out_w, o.k, o.s,
                        o.groups, o.dilation);
    }
    std::string str() const {
        std::ostringstream stream;
        stream << (silu ? "silu " : "lin  ") << in_c << "x" << in_h << "x" << in_w << " -> "
               << out_c << "x" << out_h << "x" << out_w << " k" << k << "s" << s
               << " g" << groups << " d" << dilation;
        return stream.str();
    }
};

struct PlanEntry {
    AexrtConvPlanInfo info{};
    ShapeKey key{};
};

struct CandidateResult {
    uint32_t algorithm = 0;
    bool runnable = false;
    bool gate_passed = false;
    double median_ms = -1.0;
};

double median_of(std::vector<double> samples) {
    if (samples.empty()) return -1.0;
    std::sort(samples.begin(), samples.end());
    const size_t mid = samples.size() / 2;
    return samples.size() % 2 != 0 ? samples[mid] : (samples[mid - 1] + samples[mid]) * 0.5;
}

/* 从 GPU 时间戳 profile 里取指定命令的事件耗时（信噪比远高于整图墙钟）。
 * 防御：只取首个命中事件；超过整图量级的值视为异常丢弃。 */
double profile_event_ms_for_cmd(aexrt::YoloModel& model, uint32_t command_index, double e2e_ms) {
    const uint32_t count = model.profile_event_count();
    if (count == 0) return -1.0;
    std::ostringstream prefix;
    prefix << "cmd#" << command_index << " ";
    const std::string prefix_str = prefix.str();
    for (uint32_t i = 0; i < count; ++i) {
        const char* label = model.profile_event_label(i);
        if (!label || !label[0]) continue;
        if (std::strncmp(label, prefix_str.c_str(), prefix_str.size()) == 0) {
            const double ms = model.profile_event_ms(i);
            if (e2e_ms > 0 && ms > e2e_ms * 0.5) return -1.0;  /* 异常大的事件，不可信 */
            return ms;
        }
    }
    return -1.0;
}

/* 数值门禁：检测框数量与数值一致（score 差 > 1e-3 或框差 > 0.05px 视为不通过）。 */
bool detections_match(
    const std::vector<AexrtYoloDetection>& a,
    const std::vector<AexrtYoloDetection>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i].score - b[i].score) > 1e-3f) return false;
        if (std::abs(a[i].x1 - b[i].x1) > 0.05f || std::abs(a[i].y1 - b[i].y1) > 0.05f ||
            std::abs(a[i].x2 - b[i].x2) > 0.05f || std::abs(a[i].y2 - b[i].y2) > 0.05f) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    /* 崩溃诊断：无缓冲 stdout，保证崩溃前的进度可见。 */
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string engine_path;
    uint32_t adapter_index = 0;
    uint32_t runs = 10;
    uint32_t warmup = 2;
    bool use_warp = false;
    std::string out_path = "build\\benchmarks\\conv_autotune_report.md";
    std::string header_path;
    std::string map_path;
    uint32_t shape_limit = 0;  /* 0 = 不限制；调试用：只处理前 K 个唯一形状 */
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--adapter") {
            adapter_index = static_cast<uint32_t>(std::stoul(next_value("--adapter")));
        } else if (arg == "--runs") {
            runs = static_cast<uint32_t>(std::stoul(next_value("--runs")));
        } else if (arg == "--warmup") {
            warmup = static_cast<uint32_t>(std::stoul(next_value("--warmup")));
        } else if (arg == "--out") {
            out_path = next_value("--out");
        } else if (arg == "--emit-header") {
            header_path = next_value("--emit-header");
        } else if (arg == "--emit-map") {
            map_path = next_value("--emit-map");
        } else if (arg == "--limit") {
            shape_limit = static_cast<uint32_t>(std::stoul(next_value("--limit")));
        } else if (arg == "--warp") {
            use_warp = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "conv_autotune.exe model.aexrt [--adapter N | --warp] [--runs N]\n"
                         "                       [--out report.md] [--emit-header tuned.h]\n";
            return 0;
        } else {
            engine_path = arg;
        }
    }
    if (engine_path.empty()) {
        std::cerr << "engine path required\n";
        return 1;
    }
    if (use_warp) set_env("AEXRT_NATIVE_D3D12_WARP", "1");
    /* 打开 GPU 时间戳 profile：候选耗时按事件级测量，避免整图墙钟噪声。 */
    set_env("AEXRT_NATIVE_D3D12_PROFILE_REPLAY", "1");
    /* 调优需要逐帧确定性（数值门禁按帧比对检测）：强制同步执行路径。 */
    set_env("AEXRT_NATIVE_D3D12_PIPELINE_FRAMES", "1");

    try {
        aexrt::Device device(adapter_index);
        aexrt::YoloModel model = aexrt::load_engine(device, engine_path);
        if (!model.executable()) {
            std::cerr << "engine not executable\n";
            return 2;
        }
        std::cout << "engine loaded\n";

        const uint32_t plan_count = aexrt_yolo_conv_plan_count(model.get());
        std::cout << "conv commands: " << plan_count << "\n";
        if (plan_count == 0) {
            std::cerr << "no tunable conv commands\n";
            return 0;
        }

        std::vector<PlanEntry> plans;
        plans.reserve(plan_count);
        std::map<ShapeKey, size_t> key_to_index;
        for (uint32_t i = 0; i < plan_count; ++i) {
            PlanEntry entry;
            entry.info.struct_size = sizeof(AexrtConvPlanInfo);
            if (!aexrt_yolo_conv_plan_info(model.get(), i, &entry.info)) {
                std::cerr << "conv plan info failed at " << i << "\n";
                return 2;
            }
            const AexrtConv2DDesc& d = entry.info.desc;
            entry.key = ShapeKey{entry.info.kind_is_silu, d.in_channels, d.in_h, d.in_w,
                                 d.out_channels, d.out_h, d.out_w, d.kernel_h, d.stride_h,
                                 d.groups, d.dilation_h};
            if (key_to_index.find(entry.key) == key_to_index.end()) {
                key_to_index[entry.key] = plans.size();
                plans.push_back(entry);
            }
        }
        std::cout << "unique shapes: " << plans.size() << "\n";
        if (shape_limit > 0 && plans.size() > shape_limit) {
            plans.resize(shape_limit);
            std::cout << "limited to first " << shape_limit << " shapes\n";
        }

        /* 确定性输入 */
        const uint64_t elements = model.input_element_count();
        std::vector<float> input(static_cast<size_t>(elements));
        uint32_t state = 0xC2A11u;
        for (auto& v : input) {
            state = state * 1664525u + 1013904223u;
            v = static_cast<float>(static_cast<int32_t>(state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
        bool run_failed = false;
        auto run_once = [&](std::vector<AexrtYoloDetection>& dets) -> double {
            dets.resize(64);
            uint32_t n = 0;
            const double t0 = now_ms();
            const int ok = aexrt_yolo_run(device.get(), model.get(), input.data(),
                                          static_cast<uint64_t>(input.size()), dets.data(), 64, &n);
            const double t1 = now_ms();
            run_failed = (ok == 0);
            if (run_failed) {
                dets.clear();
                return -1.0;
            }
            dets.resize(n);
            return t1 - t0;
        };

        std::vector<AexrtYoloDetection> baseline_dets;
        for (uint32_t i = 0; i < warmup; ++i) {
            run_once(baseline_dets);
        }
        if (run_failed) {
            std::cerr << "baseline run failed\n";
            return 2;
        }
        std::vector<double> samples;
        for (uint32_t i = 0; i < runs + 4; ++i) {
            const double t = run_once(baseline_dets);
            if (t >= 0) samples.push_back(t);
        }
        const double baseline_ms = median_of(samples);
        std::cout << "baseline e2e median " << std::fixed << std::setprecision(3) << baseline_ms
                  << " ms, dets " << baseline_dets.size() << "\n";

        struct TuneRow {
            ShapeKey key;
            uint32_t current_algorithm;
            uint32_t best_algorithm;
            double best_ms;
            double current_ms_estimate;
            std::vector<CandidateResult> results;
        };
        std::vector<TuneRow> rows;

        for (auto& plan : plans) {
            TuneRow row;
            row.key = plan.key;
            row.current_algorithm = plan.info.current_algorithm;
            row.best_algorithm = plan.info.current_algorithm;
            row.best_ms = -1.0;
            if (plan.info.in_fusion_group != 0u) {
                /* 融合组内命令：算法由融合执行决定，单独 override 会破坏物理
                 * 执行（甚至崩溃），直接保留当前算法。 */
                rows.push_back(std::move(row));
                std::cout << "  " << plan.key.str() << ": [in fusion group] keep alg="
                          << row.current_algorithm << "\n";
                continue;
            }
            /* 当前算法的事件级耗时（GPU 时间戳 profile，信噪比远高于整图墙钟） */
            {
                std::vector<AexrtYoloDetection> dets;
                std::vector<double> ev;
                for (uint32_t i = 0; i < warmup + runs; ++i) {
                    const double t = run_once(dets);
                    if (t < 0 || run_failed) continue;
                    const double e = profile_event_ms_for_cmd(model, plan.info.command_index, t);
                    if (e >= 0 && i >= warmup) ev.push_back(e);
                }
                row.current_ms_estimate = median_of(ev);
            }

            for (uint32_t c = 0; c < plan.info.candidate_count; ++c) {
                const uint32_t candidate = plan.info.candidates[c];
                CandidateResult res;
                res.algorithm = candidate;
                std::cout << "    cand alg=" << candidate << "\n";
                if (!aexrt_yolo_override_conv_algorithm(model.get(), plan.info.command_index, candidate)) {
                    row.results.push_back(res);
                    continue;
                }
                std::vector<AexrtYoloDetection> dets;
                const double gate_ms = run_once(dets);
                res.runnable = !run_failed;
                if (!res.runnable || gate_ms < 0 || !detections_match(dets, baseline_dets)) {
                    /* 执行失败（fail-closed）或数值门禁失败：恢复并跳过。
                     * 恢复目标必须是固化的 planned_algorithm（见基线快照注释）。 */
                    aexrt_yolo_override_conv_algorithm(model.get(), plan.info.command_index, plan.info.planned_algorithm);
                    for (uint32_t i = 0; i < warmup; ++i) run_once(dets);
                    row.results.push_back(res);
                    continue;
                }
                res.gate_passed = true;
                std::vector<double> ev;
                for (uint32_t i = 0; i < warmup + runs; ++i) {
                    const double t = run_once(dets);
                    if (t < 0 || run_failed) continue;
                    const double e = profile_event_ms_for_cmd(model, plan.info.command_index, t);
                    if (e >= 0 && i >= warmup) ev.push_back(e);
                }
                res.median_ms = median_of(ev);
                aexrt_yolo_override_conv_algorithm(model.get(), plan.info.command_index, plan.info.planned_algorithm);
                for (uint32_t i = 0; i < warmup; ++i) run_once(dets);
                row.results.push_back(res);
            }
            /* 显著度决策：候选事件耗时改善超过 5% 才切换，避免噪声赢家 */
            for (const auto& res : row.results) {
                if (!res.gate_passed || res.median_ms < 0) continue;
                if (row.best_ms < 0 || res.median_ms < row.best_ms) {
                    row.best_ms = res.median_ms;
                    row.best_algorithm = res.algorithm;
                }
            }
            if (row.best_ms < 0 || row.current_ms_estimate < 0 ||
                row.best_ms > row.current_ms_estimate * 0.95) {
                row.best_algorithm = row.current_algorithm;
                row.best_ms = row.current_ms_estimate;
            }
            rows.push_back(std::move(row));
            std::cout << "  " << row.key.str() << ": current alg=" << row.current_algorithm
                      << " best alg=" << row.best_algorithm
                      << " (" << std::fixed << std::setprecision(4) << row.current_ms_estimate
                      << " -> " << row.best_ms << " ms gpu)\n";
        }

        /* 先快照胜出列表（应用 override 后 current_algorithm 会随之变化）。
         * all_winners 供 map 输出（渐进回退会掏空 winners，但编译期安全
         * 过滤使完整胜者列表可安全固化）。 */
        std::vector<std::pair<uint32_t, uint32_t>> winners;
        std::vector<std::pair<uint32_t, uint32_t>> all_winners;
        for (uint32_t i = 0; i < plan_count; ++i) {
            AexrtConvPlanInfo info{};
            info.struct_size = sizeof(AexrtConvPlanInfo);
            if (!aexrt_yolo_conv_plan_info(model.get(), i, &info)) continue;
            const AexrtConv2DDesc& d = info.desc;
            ShapeKey key{info.kind_is_silu, d.in_channels, d.in_h, d.in_w,
                         d.out_channels, d.out_h, d.out_w, d.kernel_h, d.stride_h,
                         d.groups, d.dilation_h};
            for (const auto& row : rows) {
                if (!(row.key < key) && !(key < row.key)) {
                    if (row.best_algorithm != info.current_algorithm) {
                        winners.emplace_back(info.command_index, row.best_algorithm);
                        all_winners.emplace_back(info.command_index, row.best_algorithm);
                    }
                    break;
                }
            }
        }

        /* 基线快照：所有可调命令的固化算法（渐进回退时恢复用）。
         * 必须用 planned_algorithm（引擎固化的 planned_kernel）——
         * current_algorithm 是含 fail-closed 回退的生效值，用它还原会把
         * 回退值写进 planned_kernel/物理计划，偏离固化的 barrier 计划
         * （实测 combo 阶段 barrier 重放失败且失败状态无法自愈）。
         * 同时记录融合组内命令（apply 阶段绝不 override）。 */
        std::map<uint32_t, uint32_t> original_algorithms;
        std::set<uint32_t> grouped_commands;
        for (uint32_t i = 0; i < plan_count; ++i) {
            AexrtConvPlanInfo info{};
            info.struct_size = sizeof(AexrtConvPlanInfo);
            if (!aexrt_yolo_conv_plan_info(model.get(), i, &info)) continue;
            if (info.in_fusion_group != 0u) {
                grouped_commands.insert(info.command_index);
                continue;
            }
            original_algorithms[info.command_index] = info.planned_algorithm;
        }

        /* 应用全部胜出 override；组合复测失败时渐进回退（逐个摘除再测），
         * 兜住「单候选通过但组合与 barrier 计划失配」的引擎级交互。 */
        uint32_t applied = 0;
        auto apply_all = [&]() {
            applied = 0;
            /* 先全部恢复基线，再应用幸存 winners（override 状态持久，须显式还原）。
             * 融合组内命令的 kernel 由融合组决定，绝不单独 override。 */
            for (const auto& [cmd_index, alg] : original_algorithms) {
                aexrt_yolo_override_conv_algorithm(model.get(), cmd_index, alg);
            }
            for (const auto& winner : winners) {
                if (grouped_commands.count(winner.first) != 0) continue;
                if (aexrt_yolo_override_conv_algorithm(model.get(), winner.first, winner.second)) {
                    std::cout << "  apply cmd#" << winner.first << " -> alg " << winner.second << "\n";
                    ++applied;
                }
            }
        };
        auto combo_runs = [&]() {
            std::vector<AexrtYoloDetection> dets;
            for (uint32_t i = 0; i < warmup; ++i) {
                run_once(dets);
                if (run_failed) return false;
            }
            for (uint32_t i = 0; i < 4; ++i) {
                run_once(dets);
                if (run_failed) return false;
            }
            return true;
        };
        apply_all();
        std::cout << "overrides applied: " << applied << "\n";
        /* 渐进回退：组合跑不通就逐个摘除胜出 override 重试，直到可跑或清空。 */
        if (applied > 0 && !combo_runs()) {
            std::cerr << "combo failed, falling back to incremental rejection\n";
            while (!winners.empty()) {
                const char* reason = aexrt_yolo_get_last_error(model.get());
                std::cerr << "  rejecting winner cmd#" << winners.front().first << " alg="
                          << winners.front().second << " (" << (reason && reason[0] ? reason : "?") << ")\n";
                winners.erase(winners.begin());
                apply_all();
                if (combo_runs()) break;
            }
        }

        /* tuned 复测 + 终门禁 */
        std::vector<AexrtYoloDetection> tuned_dets;
        for (uint32_t i = 0; i < warmup; ++i) run_once(tuned_dets);
        std::vector<double> ts;
        for (uint32_t i = 0; i < runs + 4; ++i) {
            const double t = run_once(tuned_dets);
            if (t >= 0 && !run_failed) ts.push_back(t);
            if (run_failed) {
                const char* reason = aexrt_yolo_get_last_error(model.get());
                std::cerr << "tuned run failed: " << (reason && reason[0] ? reason : "?") << "\n";
                break;
            }
        }
        const double tuned_ms = median_of(ts);
        const bool final_gate = detections_match(tuned_dets, baseline_dets) && tuned_ms > 0;
        std::cout << "tuned e2e median " << std::fixed << std::setprecision(3) << tuned_ms
                  << " ms (baseline " << baseline_ms << "), final gate "
                  << (final_gate ? "PASS" : "FAIL") << "\n";

        /* 报告 */
        std::ostringstream report;
        report << "# conv_autotune report\n\n- engine: " << engine_path << "\n";
        report << "- device: " << (use_warp ? "WARP" : "adapter " + std::to_string(adapter_index))
               << "\n- unique shapes: " << plans.size() << ", overrides applied: " << applied << "\n";
        report << "- baseline e2e median: " << baseline_ms << " ms; tuned: " << tuned_ms
               << " ms; speedup " << std::setprecision(3) << baseline_ms / tuned_ms << "x\n";
        report << "- final numeric gate: " << (final_gate ? "PASS" : "FAIL") << "\n\n";
        report << "| shape | current alg | best alg | current ms | best ms | blocked candidates |\n";
        report << "|---|---|---|---|---|---|\n";
        for (const auto& row : rows) {
            std::ostringstream blocked;
            for (const auto& res : row.results) {
                if (!res.gate_passed && res.algorithm != row.current_algorithm) {
                    blocked << res.algorithm << " ";
                }
            }
            report << "| " << row.key.str() << " | " << row.current_algorithm << " | "
                   << row.best_algorithm << " | " << std::fixed << std::setprecision(3)
                   << row.current_ms_estimate << " | " << row.best_ms << " | " << blocked.str()
                   << " |\n";
        }
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(out_path).parent_path(), ec);
        std::ofstream file(out_path, std::ios::binary);
        file << report.str();
        std::cout << "saved: " << out_path << "\n";

        /* JSON map：供 aexrtc build --algo-map 固化进引擎。
         * map 直接输出单候选胜者（各自已过数值门禁）——组合级兼容由编译期
         * 安全过滤兜底（in-fusion-group / producer-of-group / winograd-target
         * 跳过）：运行时组合失败常源于 stale head-fusion 记录（override 不
         * 重算 head fusion），而编译期从 override 后的 plan 全链重算，无此问题
         * （实测 cmd#55 运行时组合必败、编译期固化安全）。 */
        if (!map_path.empty()) {
            std::ofstream m(map_path, std::ios::binary);
            m << "{\n";
            for (size_t i = 0; i < all_winners.size(); ++i) {
                m << "    \"" << all_winners[i].first << "\": " << all_winners[i].second
                  << (i + 1 < all_winners.size() ? "," : "") << "\n";
            }
            m << "}\n";
            std::cout << "saved: " << map_path << " (" << all_winners.size() << " overrides)\n";
        }
        if (!header_path.empty()) {
            std::ofstream h(header_path, std::ios::binary);
            h << "// 由 conv_autotune 生成：per-device 卷积算法覆盖表。\n";
            h << "// 用法：aexrt_yolo_load_engine 成功后，对每条记录调用\n";
            h << "// aexrt_yolo_override_conv_algorithm(model, entry.command_index, entry.algorithm)。\n";
            h << "typedef struct AexrtTunedConvEntry {\n    unsigned command_index;\n    unsigned algorithm;\n} AexrtTunedConvEntry;\n\n";
            h << "static const AexrtTunedConvEntry kTunedConvPlan[] = {\n";
            for (const auto& winner : winners) {
                h << "    {" << winner.first << ", " << winner.second << "},\n";
            }
            h << "};\nstatic const unsigned kTunedConvPlanCount = sizeof(kTunedConvPlan) / sizeof(kTunedConvPlan[0]);\n";
            std::cout << "saved: " << header_path << "\n";
        }
        clear_env("AEXRT_NATIVE_D3D12_WARP");
        return final_gate ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "conv_autotune failed: " << error.what();
        const char* reason = aexrt_yolo_get_last_error(nullptr);
        if (reason && reason[0]) {
            std::cerr << " | runtime: " << reason;
        }
        std::cerr << "\n";
        clear_env("AEXRT_NATIVE_D3D12_WARP");
        return 2;
    }
    return 0;
}
