# AEXRT Kernel Tile Spec（PC / 移动端共用）

本文件定义 PC D3D12（HLSL）与移动 Vulkan（GLSL）共用的**自研 tile / 量化 /
调优契约**。执行路径不链接 cuDNN、oneDNN、DirectML 或任何第三方推理库；
DirectML/ORT 仅作为外部基准测试对象。

## 1. 计算 tile

| 维度 | 约定 |
|---|---|
| 输出 tile | position × OC，常见 32×32；1x1 短空间用 32×32 / 4-OC 线程 |
| 线程映射 | `pos_l = tid % TS`，`oc_group = tid / TS`；warp 内 pos 连续，权重广播 |
| K 分块 | KB = 16 uint（64 个 int8 通道）或 32 uint（1x1） |
| LDS A | 输入 tile + halo，按 `[row][KB]`；**行跨度必须 +1 padding**，否则同列 32 路 bank conflict |
| LDS B | 权重 tile `[32 oc][9 taps × KB]`，行跨度 `9*KB + 1` |
| 权重布局 | `[oc][tap][icg]`，uint 打包 4 个 int8；1x1 为 `[oc][icg]` |
| 累加 | int32 `dot4add_i8packed`；每 16 通道（4 icg）转一次 float 并乘 group scale |
| 输出 | NCHW；1 线程写 4/8/16 个 OC，按 `out_plane` 跨步 |

### 为什么是 +1 padding

`groupshared uint tile[32][32]` 的行跨度 32 word，`(row*32 + k) % 32 = k`，
所有行同列命中同一 bank，A 读取是 32 路冲突，B 是 8 路冲突。改为 33 word 后
`(row*33 + k) % 32 = (row + k) % 32`，同一 warp 内行不同即 bank 不同。

### OC blocking 的取舍

- 1×1：4 OC/线程在 10×10/20×20 因 warp 占用翻倍收益明显；40×40 收益依模型而定。
- concat_conv1x1：16 OC/线程把激活重复读取降 4×，但线程数降 4×；RTX 5060 实测中性，
  保留为 opt-in（`AEXRT_NATIVE_D3D12_ENABLE_CONCAT_CONV1X1_VEC4X4`）。
- 3×3：stride-1 B-tile 进 LDS 中性偏正（同时修复缺失的 halo barrier）；
  stride-2 因 28.4 KB LDS 压 occupancy，回退 global 权重流。

## 2. 量化契约（W8A8）

- **producer-side 预量化**：上一层 conv 的 epilogue 直接按下一层的
  activation scale 量化输出，避免单独的 quantize pass。
- activation scale：per-tensor 动态 amax / 127（带安全系数）；每 16 通道
  一个 group scale（pg16）。
- weight：per-OC 对称 int8 + per-OC scale。
- 检测头（box/cls 最后一层）保持 fp16/fp32，不参与 int8。
- 数值门禁：每层模拟误差 + 真实图像端到端检测数与分数差；失败的命令回退 fp16。

## 3. 逐层 autotune

- 统一 schema：`(command_index, shape_key, candidate_algos, measured_ms,
  numeric_gate, chosen_algo)`。
- PC：`conv_autotune --emit-map` → JSON；只固化 `in_fusion_group==0` 且
  通过数值门的命令。
- 移动：`--emit-policy FILE` / `--policy FILE`（与 PC algo map 同一思路）。
  离线一次录 6 个候选变体，轮转交错测整图时间；逐层 timestamp 只在相对整图
  冠军有 >25% 优势时才生效，最终用整图延迟自校验，回退到可信的整图冠军。
  归因时必须用 `VkEvent`（set+wait）建立执行依赖：`vkCmdPipelineBarrier`
  只保证内存可见性，相邻 dispatch 仍可重叠，时间戳会错配。
- 不在 fusion 组内单独 override：组内 kernel 由融合组拥有，否则物理计划失配。

## 4. 执行策略按设备选择（同一契约，不同实现）

| 设备 | W8A8 实测 | 采用 |
|---|---|---|
| RTX 5060（Vulkan/dp4a） | 1.18 ms vs fp16 2.11 ms（1.79×） | W8A8 |
| Mali-G925（SDOT） | 8.45 ms vs fp16 5.0–5.5 ms | fp16（AXM3 仍可加载） |

结论：量化**格式与导出器**共用，**执行策略**按设备实测选择。移动端
`OpSDot(PackedVectorFormat4x8Bit)` 的每指令吞吐低于 fp16 点积，W8A8 只在
权重带宽占主导的 stem 上打平；因此 Mali 的策略表全选 fp16，PC 选 int8。
这一策略由 `--emit-policy` 的整图测量自动得出，不是硬编码。

## 5. PC ↔ 移动映射

| 概念 | PC (HLSL/D3D12) | 移动 (GLSL/Vulkan) |
|---|---|---|
| 工作组 | `[numthreads(128/256)]` | `local_size_x = 64/128` |
| 共享内存 | `groupshared` | `shared` |
| int8 dot | `dot4add_i8packed` (SM6.4) | `shaderIntegerDotProduct` / `dotPacked4x8EXT` |
| 权重打包 | uint = 4×int8 | uint = 4×int8 |
| 调度单位 | command list / PSO | command buffer / pipeline |
| 调优 | per-shape algo map | per-layer variant table |

移动端 Mali 的 subgroup=16、32KB shared、fp16 2× 吞吐；PC 的 warp=32、
32KB static groupshared。tile 尺寸按 `warp/subgroup、shared 上限、OC 余数`
生成变体，而不是复制同一份常量。

## 6. 当前已验证结论

- LDS +1 padding：1×1 稳定收益；3×3 stride-1 中性偏正。
- 4-OC 1×1 短空间变体：10×10/20×20 收益，40×40 依模型。
- stride-2 3×3 B-staging：占用受限，回退。
- concat 16-OC blocking：RTX 5060 中性，opt-in。
- W8A8：桌面 1.79×；Mali SDOT 1.4–1.6× 慢于 fp16 点积，移动端不采用。
- 移动端空间复用（2×2×8、1×2×8）：Mali 均负结果；累加器/占用率是瓶颈。
- **PC 剖面（cs2V8，2026-11-18）**：int8 3×3 32 条 0.53 ms（~4 TMAC/s），
  int8 1×1 17 条 0.24 ms（短空间层 1.6–2.1 TMAC/s），stride2 6 条 0.13 ms。
  1×1 的效率由权重/位置流量决定，不是 tile 尺寸：concat_conv1x1 的
  tileflow / vec4 / vec4x4 / direct_pack4 四个变体在 DY 上实测相差 <3%
  （0.079–0.085 ms/条），换 concat 形态无收益。
- **位置配对 int8 1×1（实验，已回退）**：64 pos × 32 oc tile、2 pos × 8 oc/
  线程，LDS 读数/点积从 1.125 降到 0.625，但 cs2V8/DY/apex 实测 0–1%，
  且只有 2/4 层满足 `oc%32==0 && spatial>512`。当前 1×1 的瓶颈不在 LDS
  复用率，短空间层已用 4-OC 变体跑到 1.6–2.1 TMAC/s。代码未保留。
- **DY plan 可复现了**：用 shipped 引擎 dump 的 algo map + 反融合配方，
  从 ONNX 重建得到 **2.99–3.00 ms**（shipped 3.01 ms），arena 大小逐字节一致：

  ```
  AEXRTC_UNFUSE_C2F_TAIL=1 AEXRTC_UNFUSE_SMALL_CONCAT=1 \
  AEXRTC_UNFUSE_PAIRED_CONV=1 AEXRTC_NO_MEASURED_FP16=1 \
  aexrtc build models/DYv11s_414_320.onnx -o dy.aexrt \
      --algo-map build/tmp/dy_plan_map.json
  ```

  同一配方不带 map 是 5.28 ms（1.76× 慢）：DY 的性能基本由每命令 plan 决定，
  自动 planner 在该模型上明显劣于录制 plan。cs2V8/apex 的 shipped 引擎
  （127/115 命令）与当前导出器的命令数（124/117）不一致，暂不能逐字节复现。
- **concat_conv1x1 融合的方向性结论**：在未调优的 fresh plan 上，保留融合
  5.16 ms vs 全反融合 5.31 ms（融合约 +3%），所以它不是“负优化”；但 shipped
  DY plan 本身不走该融合。先把融合 plan 用 `conv_autotune --emit-map` 调优
  （5.27 → 3.52 ms）再固化，仍比 shipped 的 2.99 ms 慢 17% —— 融合把该节点
  从 int8 LTS 降级成 fp32（`StructuredBuffer<float4>` 权重），**丢掉了 int8**。
  结论：保持反融合；要再赢只能做真正的 int8 concat superblock。
- **stride-2 3×3 4-oc 变体（lts4-s2）**：256 线程、8×4 pos × 32 oc、每线程
  4 oc（`AEXRT_NATIVE_D3D12_ENABLE_3X3S2_LTS4=1`）。stride-2 层位置数只有
  stride-1 的 1/4，8-oc 版本只暴露 8–13 warps/SM；4-oc 把驻留 warp 翻倍。
  DY 实测 3.017/3.015 → 3.009/2.976 ms（≈0.8%，噪声内），cs2V8 中性；
  检测输出与 8-oc 版本**逐位一致**（19 个稠密检测逐字段差 0）。因收益在
  噪声内，保持 env-gated，不改默认。
- **PC 内核数值验证工具**：`native_yolo_package --dump-dets FILE --max-dets N`
  导出检测为二进制（count + 6×f32/条）。配合 `aexrtc build --conf 1e-6`
  可把随机输入变成 19 条稠密检测，作为换内核时的逐位/逐字段对比信号
  （之前 0 detections 使任何内核都“通过”）。
- **矩阵单元（2026-11-19）**：Mali-G925 暴露 VK_KHR_cooperative_matrix
  （fp16 16×32×32、int8 4×16×16），Slang 生成的模块可编译；但该驱动在
  ≥~320 个 workgroup 时出现固定 ~10 ms 退化（多实现复现），真实 conv 的
  网格规模不可用 → 移动端矩阵单元重写搁置。PC D3D12 无对应接口（WaveMMA
  不可用），host Vulkan 仅作参考。
- **apex neck lattice**：`AEXRT_NATIVE_D3D12_DISABLE_NECK_LATTICE` A/B 显示
  融合是 ~1–2% 的小赢（2.03/2.04 → 2.05/2.075 ms），不是负优化；apex 的
  剩余差距分布在 1×1 类与单条 lattice 内核，无单点可回收。
- **stride-2 半 KB B-tile（2026-11-19）**：8-icg K 块 + 权重 LDS 暂存
  （A 4.9KB + B 9.3KB）。首版 shader 在 HLSL 里用了 groupshared 指针，
  DXC 直接拒绝（HLSL 不支持指针）；改为直接二维索引后编译通过，数值与
  基线**逐位一致**。K-split 关闭时 ltsb 2.98 ms vs 旧 lts 3.21 ms（stride2
  类快 ~7%）；但默认的 K-split 路径本身就是 2.94–3.00 ms，两者打平；
  cs2V8 上 ltsb 甚至略慢（1.24 vs 1.22 ms）。结论：**默认保持 K-split，
  ltsb 仅 opt-in**，不再作为默认候选。
- **1×1 K-split 扩展（2026-11-19，已回退）**：ks partial kernel 本就有
  `KS_TAPS=1` 分支，但把 `use_ks` 扩到 family 1 后数值错误（稠密检测
  19→4），combine/索引对 1×1 布局不成立；已回退，不再作为 1×1 的候选。
- **head decode/NMS 不是 PC 的目标**：DY 的 post 三个 kernel 合计
  ~0.005 ms（decode 0.0023 / nms_mark 0.0014 / topk 0.0011），优化空间可忽略。
- **K-split 因子可调**：`AEXRT_NATIVE_D3D12_INT8_KSPLITS=1..8`（默认 4）。
  DY 上 4 最优（2 慢 1.5–2.5%，8 慢 ~7%），apex 中性；默认不变。
  非交错 A/B 会被 DVFS 误导，必须交替测。
