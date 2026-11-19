# AEXRT-M · Vulkan 移动推理引擎

天玑 9500 / Mali-G925 上的自研 Vulkan compute 推理路径。与桌面 AEXRT（D3D12）
共享同一套思路：自有二进制引擎、静态执行计划、换线友好的内核与实测自动调优。

宿主验证：RTX 5060 Laptop，本路径 fp16 为 ~1.6 ms GPU。
真机（天玑 9500 / Mali-G925-Immortalis MC12）：cs2V8_320 **GPU 5.12 ms 中位 /
4.78 ms 最佳**（`--autotune` 选中 auto 混合变体；autotune 内 min 4.50 ms）。
数值门禁在主机 Vulkan 上通过（6 个 head 张量 vs ONNX Runtime，NRMSE ≤ 1e-3）。

## 1. 架构

```
ONNX ──make_subsets.py──> AXM1 flat ──yolo_flat──> Vulkan compute ──> dets.bin
                                │
                     header + per-value scale table
                     layer table + fp16 weight blob
```

### 1.1 AXM1 平面模型

- 116 字节头：层数/值数/输入输出/类别数/检测头 3 级 box·cls vid/stride/DFL 权重。
- 每值一个 f32 量化 scale（W8A8 预留；fp16 路径忽略）。
- 层记录：conv 26 B，其余 22 B；权重按 `[oc][kh][kw][ic]` 打包为 fp16，
  每行 4 元素对齐，支持 `f16vec4` 直读。
- 导出器做**连接图级**融合，而不是节点顺序相邻匹配：
  - `Conv -> Sigmoid -> Mul` 折成 `act=1`（SiLU），`Conv -> Sigmoid` 折成 `act=2`；
  - 共享 Sigmoid 输出时退化为独立 unary silu 层，不误删共享节点；
  - Split 展开成零拷贝 slice 视图；检测头 Concat/Reshape 在 DCE 中删除。
- 当前 cs2V8_320：63 conv（57 个带 SiLU）、13 concat、3 maxpool、2 resize、
  16 slice，共 103 层；权重 6.03 MB。

### 1.2 位置主序 + 通道偏移放置

所有激活按 `value[pos * ctot + ch_off + c]` 访问：

- concat 的生产者**直接写进 concat 缓冲区**（放置阶段求 `(root, channel offset)`）；
- slice/split 是纯视图，不占缓冲、不拷贝；
- 卷积/元素操作的输入输出都带 `ch_off`、`ctot`，因此同一段内存可被多个值
  以不同通道窗口共享。

这消除了移动端最贵的 concat 拷贝，也让后续融合（残差、跨层复用）有统一寻址。

### 1.3 静态命令图

固定 shape 下，整个图只录制一次：

- 所有 descriptor set 在录制期分配并常驻；
- 每帧只有一次 `vkQueueSubmit` + fence；
- 依赖用 per-value 读写 epoch 跟踪，只在真依赖处插 `vkCmdPipelineBarrier`，
  且 barrier 放在**消费者之前**（旧实现放在生产者之后是错的，会导致同
  command buffer 内的 RAW 竞争，这是本轮修复的关键正确性缺陷）；
- 帧首/帧尾各一个 GPU timestamp，benchmark 报告 `gpu-ts` 与 wall 两种口径。

### 1.4 加载时自动调优

`--autotune` 会为 base / fp16-dot / 2×2 quartet 三种卷积变体各录一条 command
buffer，**轮转交错**跑 1 轮预热 + 9 轮计时，每轮轮换起始变体后取各变体最小值（消除 Mali DVFS/热漂移），选最快的一条执行。选择结果体现在
启动日志的 `autotune: using variant N`。这让同一份二进制在 Mali（偏好 fp16 打包
运算）和独显（偏好高线程数 fp32）上都能落到合适的核。

### 1.5 GPU 检测头 + 有界后处理

- decode 内核在 GPU 上完成 sigmoid、DFL softmax/期望、box 解码和原子压缩；
- 候选写入 device-local 缓冲，帧尾 `vkCmdCopyBuffer` 到 host-cached staging，
  避免 CPU 直接读 VRAM（独显上约 0.5 µs/word，实测可把 7.6 ms 读回降到 0.2 ms）；
- CPU 端：确定性 score/类别/坐标排序 → 每类 top-128 → 贪心 NMS；
  top-K 丢弃的框分数低于同类所有保留框，只可能被抑制，不会改变保留结果；
- 输出检测按分数降序写入 `[count][x1,y1,x2,y2,score,cls] * n`。

## 2. 内核

| 文件 | 说明 |
|---|---|
| `kernels/conv_i8.comp` | 1 位置 × 8 OC；fp16 存储/权重，fp32 累加；带标量兜底路径（stem C=3、非 4 对齐放置） |
| `kernels/conv_f16.comp` | 同上映射；每 16 通道一组用 fp16 dot + fp16 部分和，再累加 fp32；大空间默认 |
| `kernels/conv_f16_oc4.comp` | 1 位置 × 4 OC；10×10 / 20×20 空间提高工作组数，auto 默认 |
| `kernels/conv_f16_oc1.comp` | 1 位置 × 1 OC；10×10 实验变体，实测更慢，仅 `--conv-variant f16oc1` |
| `kernels/conv_stem.comp` | C=3 RGB stem：每 tap 补零到 4 通道 + f16vec4 权重 + fp16 dot4；真机 0.31 vs 0.91 ms |
| `kernels/conv_quad.comp` | 2×2 输出块 × 8 OC，权重复用 4×；Mali/独显均变慢，仅实验 |
| `kernels/elementwise.comp` | scalar：maxpool / nearest resize / add / silu / sigmoid；vector：每线程 4 通道（`--vector-elem`） |
| `kernels/decode.comp` | YOLOv8 DFL 头解码；候选数硬上限，绝不越界写 |
| `kernels/copy.comp` | Phase-1 遗留，当前未使用 |

## 3. 构建

```bash
# SPIR-V + C 头（自动找 NDK shader-tools 的 glslc）
py mobile/tools/build.py shaders

# Android aarch64（NDK r27/r28）
py mobile/tools/build.py android --api 35

# Windows 主机验证（MSVC；自动从系统 vulkan-1.dll 生成 import lib）
py mobile/tools/build.py host

# 导出平面模型（默认 cut 在检测头 concat 之前，自动 DCE）
py mobile/tools/make_subsets.py
```

产物：

- `build/mobile/yolo_flat`（Android ELF）
- `build/mobile/yolo_flat_host.exe`（Windows 验证）
- `build/tmp/cs2v8_full.bin`（AXM1）
- `build/mobile/vk/include`（从 NDK 复制的最小 Vulkan 头，供 MSVC）

## 4. 数值验证（主机）

一键门禁（shaders/model/host 已存在时）：

```bash
py mobile/tools/run_tests.py          # 或 --build 先重编
```

分步命令：

```bash
# 1) 生成 fp32 NCHW 输入
py -3 -c "import numpy as np; np.random.default_rng(1234).random((1,3,320,320),dtype=np.float32).tofile('build/tmp/input.f32')"

# 2) 跑主机 Vulkan 并导出三级 box/cls 头张量
./build/mobile/yolo_flat_host.exe build/tmp/cs2v8_full.bin build/tmp/input.f32 build/tmp/dets.bin 1 \
    --conv-variant base --dump-heads build/tmp/heads_final

# 3) 与 ONNX Runtime CPU 对比（位置主序，NRMSE/cosine 门禁）
py -3 mobile/tools/verify_host.py --heads build/tmp/heads_final

# 4) 解码逐框对比（全候选，排序后比较）
./build/mobile/yolo_flat_host.exe build/tmp/cs2v8_full.bin build/tmp/input.f32 build/tmp/dets_nms.bin 1 \
    --conf 0.00001 --dump-cands build/tmp/cands_nms.bin
py -3 mobile/tools/verify_decode.py --oracle build/tmp/heads_final --cands build/tmp/cands_nms.bin

# 5) C 排序+NMS 与 NumPy 参考逐项一致
py -3 mobile/tools/verify_nms.py --cands build/tmp/cands_nms.bin --dets build/tmp/dets_nms.bin
```

实测结果（RTX 5060，随机输入）：

- 6 个检测头张量：NRMSE 2.4e-4 ~ 9.6e-4，cosine ≥ 0.999998，PASS；
- 2100 个候选：box 最大误差 8e-5，类别完全一致，PASS；
- NMS：全部保留框逐项最大误差 0，PASS（保留数随 variant 略有差异，属正常）。

## 5. 性能

```bash
# 50 次计时；输出 wall 与 GPU timestamp 两种口径
./build/mobile/yolo_flat_host.exe build/tmp/cs2v8_full.bin build/tmp/input.f32 build/tmp/dets.bin 50 --conv-variant base
# 加载时在设备上实测 base/f16/oc4/auto/quad 并交错选择
./build/mobile/yolo_flat_host.exe ... 10 --autotune
# 每层 GPU 时间（top 18）
./build/mobile/yolo_flat_host.exe ... 3 --conv-variant base --profile
```

真机（Mali-G925，cs2V8_320，1.01 GMAC，交错 autotune + 10 次基准）：

| 变体 | autotune min | 备注 |
|---|---|---|
| auto（stem + oc4/f16 逐层混合） | **4.50 ms** | 选中 |
| f16（全 8 OC） | 4.80 ms | |
| f16oc4（全 4 OC） | 5.40 ms | 小空间受益、大空间受损 |
| base（fp32 累加） | 9.40 ms | |
| 2×2 quartet | 15.93 ms | |

同次运行最终 benchmark：**gpu-ts median 5.12 ms / best 4.78 ms**；wall median 6.46 ms。
Mali DVFS 下整机数值波动可达 10–20%，以同一进程内的交错 autotune 或隔离子模型为准
（stem 单独模型：0.31 ms vs 0.91 ms）。

安静机器上的 RTX 5060 结果（cs2V8_320，1.01 GMAC）：

| 变体 | gpu-ts median | gpu-ts best |
|---|---|---|
| base（fp16 存储 + fp32 累加） | ~1.62 ms | **~1.57 ms** |
| f16-dot | ~1.77 ms | ~1.71 ms |
| 2×2 quartet | ~5.5 ms | — |

桌面 AEXRT D3D12（W8A8 + Winograd + 深度融合）同卡 1.08 ms。移动路径与它的
差距主要在 int8 与更激进的层融合，而不是调度框架。

> 主机数值会受桌面 compositor/其他 GPU 进程影响；`gpu-ts best` 或锁定时钟后的
> `median` 才可用于跨变体比较。Windows 下 wall 时间波动可达数倍。

## 6. 设备实测（已执行）

设备：OnePlus PLC110 / mt6991（天玑 9500），Android 16，
`Mali-G925-Immortalis MC12`，Vulkan 1.3.278，timestampPeriod 76.923 ns。

```bash
py mobile/tools/deploy.py --serial <id> --iters 30
adb shell "cd /data/local/tmp && ./aexrt-m cs2v8_full.bin input.f32 dets.bin 10 --autotune"
```

本轮结果：

- 交错 autotune：auto 4.495 / f16 4.800 / f16oc4 5.399 / base 9.404 / quad 15.928 ms；
- 最终 auto：gpu-ts median 5.12 ms，best 4.78 ms；
- 隔离 stem 子模型：0.31 ms（专用 kernel）vs 0.91 ms（标量 fallback）。

注意：

- `--profile` 的逐层 timestamp 只覆盖 dispatch 执行区间，不含 dispatch 间隙
  （Mali 上约 9 µs/次，77 次合计 ~0.7 ms），适合相对热点排序，不是整帧延迟；
- 整机 wall 时间受 DVFS/温度影响很大，跨变体比较必须用同一进程内的交错测量；
- 退出时 `FORTIFY: pthread_mutex_lock called on a destroyed mutex` 来自 Mali 驱动
  teardown（进程不销毁 Vulkan 对象直接退出），不影响结果。

## 7. W8A8（AXM3）与逐层策略

### 7.1 结果（同一套导出器 / 校准 / 格式）

| 设备 | fp16 auto | W8A8 int8 | 结论 |
|---|---|---|---|
| RTX 5060（Vulkan，同一 harness） | 2.11 ms | **1.18 ms（1.79×）** | 桌面走 W8A8，dp4a/SDOT 有效 |
| Mali-G925 真机 | 5.0–5.5 ms | 8.45 ms | **SDOT 反而慢**，移动端走 fp16 |

逐层（`--profile`，同进程）：3×3 64→64 40×40：int8 0.314 ms vs fp16 0.218 ms；
3×3 256→64 10×10：0.310 vs 0.195 ms；stem 3→16 320×320：0.210 vs 0.238 ms
（int8 略快）。Mali 上 `OpSDot(PackedVectorFormat4x8Bit)` 的每指令吞吐低于
fp16 点积，W8A8 只在权重/激活带宽占主导的 stem 上打平。

数值门禁（`py mobile/tools/run_tests.py --int8`）：

- 6 个 head 对 ORT：cos ≥ 0.999，box NRMSE 2–4%，cls NRMSE 1–2%；
- GPU decode 与自身 head 的 NumPy oracle：box 最大差 6e-5；
- CPU NMS 与参考实现：0 差异。

量化契约：`calibrate.py` 采样各 value 的 amax → `make_subsets.py --int8` 生成
AXM3；per-OC 权重 int8 + fp32 scale，per-value 激活 scale = `amax/127*1.15`；
slice/concat/add/maxpool/resize 组内强制同一 scale（否则零拷贝视图会按错误
scale 反量化）。导出的第一个 conv 之前，输入 scale 由校准集的输入 amax 给出。

### 7.2 逐层策略（离线选型，运行时零 autotune）

PC 侧用 `conv_autotune --emit-map` + `aexrtc build --algo-map` 固化每层算法；
移动端现在用同一套思路：

```
py mobile/tools/deploy.py --serial <id> --emit-policy build/tmp/policy_mali.txt
py mobile/tools/deploy.py --serial <id> --policy build/tmp/policy_mali.txt
```

- `--emit-policy`：录 6 个候选（auto/f16/f16oc4/base/quad/pair），轮转交错
  测量整图时间，并采集逐层 timestamp；逐层选择仅在相对整图冠军有 >25% 优势时
  生效；
- 归因测量用 `VkEvent` 建立执行依赖（内存 barrier 不阻止相邻 dispatch 重叠，
  直接读 timestamp 会把同形状层测成 0.0005/0.08 ms 两种结果）；
- 最后用整图时间自校验：逐层混合若比整图冠军慢 >2%，自动改写成整图冠军的
  全量 map，保证落盘策略可信。

Mali 实测：整图冠军为 auto，落盘为 63 层全 auto；启动时间 **970 ms vs
1680 ms**（省 0.7 s），稳态 gpu-ts 与 autotune 进程一致（5.9–6.0 vs 5.2–6.0 ms，
差异在 DVFS 噪声内）。

### 7.3 N 帧流水线（吞吐模式）

`--pipeline N`（1–4）为每一帧准备独立的 value arena、候选/staging 缓冲和 fence，
同一张图录 N 份 command buffer；权重只读共享。提交时轮转 set，第 i 帧在 GPU 上
执行时第 i+1 帧已经在队列里，fence 尾巴不再落在关键路径上。

**Mali-G925 实测**（policy 固定为 63 层 auto，40–60 帧）：

| 模式 | 每帧时间 | 吞吐 |
|---|---|---|
| `--pipeline 1`（串行） | wall 7.37–8.28 ms（best 5.61） | ~121–136 fps |
| `--pipeline 2` | **5.21–5.31 ms** | **188–192 fps** |
| `--pipeline 3` | 5.20–5.35 ms | 187–192 fps |
| `--pipeline 4` | 5.24 ms | 191 fps |

- N=2 即到平台：再多的 set 不再提升（队列深度 2 足够掩盖 fence 尾巴），只增加内存；
- 相对串行中位数提升约 **1.4–1.6×**，且每帧时间比串行的 best 还低；
- 串行的 gpu-ts median 5.72–6.46 ms，说明 1.5–2 ms/帧 的提交/栅栏尾巴被完全掩盖；
- 正确性：`--conf 0.00001` 下 pipeline 1/2/3 的 detections **逐字节一致**
  （364 candidates / 92 kept），host 侧同样一致并已加入 `run_tests.py` 门禁。

host 上收益为 0/负（fp16 1.87 → 1.94–2.16，int8 1.24 → 1.42）：独显 GPU 已饱和，
N 份工作集翻倍抵消了栅栏收益。所以该模式按设备选择，默认关闭。

复现：

```
py mobile/tools/build.py android
py mobile/tools/deploy.py --serial <id> --iters 60 \
    --policy build/tmp/policy_mali.txt --pipeline 2
```

### 7.4 流式输入（`--stream`）

`--stream --frames FILE`（FILE = 多帧 fp32 NCHW 拼接）把流水线变成真实流式路径：
每帧只在对应 set 的 fence 完成后，把新帧写进该 set 的 **cached staging**，图的
开头用一次 GPU `vkCmdCopyBuffer` 送进 device-local fp16 输入缓冲。

- 为什么不直接写 mapped 输入缓冲：Mali 上直接写 device 内存约 0.4 ms/帧，
  独显经 PCIe >1 ms/帧；cached staging + GPU copy 把这段降到 ~0.1 ms。
- 转换：aarch64 用 `vcvt_f16_f32`（每指令 4 像素）做 NCHW→position-major fp16。

| 平台 | 静态 pipeline 2 | `--stream`（含转换） | 差值 |
|---|---|---|---|
| Mali-G925 | 5.37 ms/帧（186 fps） | **5.49 ms/帧（182 fps）** | +0.12 ms（2%） |
| RTX 5060 | 1.91 ms/帧 | 2.03 ms/帧 | +0.12 ms |

正确性：流式运行最后一帧的 detections 与“单独跑该帧”**逐字节一致**
（Mali 402 candidates / 89 kept；host 407 / 94）；`AEXRT_STREAM_DUMP=prefix`
会把逐帧 NMS 结果写成 `prefix_<f>.bin`，第 0 帧同样与单独跑该帧逐字节一致
（设备 2116 字节）。两条都已加入 `run_tests.py` 门禁（fp16 与 W8A8 都过）。

**逐帧后处理**：每提交第 f 帧后，host 立即等待第 f-1 帧的 fence 并做 NMS，
CPU 后处理与第 f 帧的 GPU 执行重叠：

| Mali，30 帧 | 每帧时间 |
|---|---|
| 默认置信度（无候选） | 5.65–5.69 ms |
| `--conf 0.00001`（每帧 2100 候选，NMS ~2.3 ms） | 5.49–5.73 ms |

即最坏情况下 2.3 ms/帧 的 CPU NMS 被完全隐藏，吞吐不变。剩余缺口只是把
逐帧 detections 通过库 API 交给调用方——当前 runtime 仍是基准/验证形态。

### 7.5 矩阵单元（VK_KHR_cooperative_matrix）：工具链已解，但 Mali 驱动有规模悬崖

`--caps` 探针在 Mali-G925 上确认 **VK_KHR_cooperative_matrix** 可用：

| 配置 | Mali-G925 | RTX 5060 (Vulkan) |
|---|---|---|
| fp16 A/B → fp32 acc | 16×32×32、4×8×8 | 16×16×16、16×8×16、16×8×8 |
| fp32 A/B → fp32 acc | 16×16×16、4×4×4 | 无 |
| int8 A/B → int32 acc | 4×16×16 | 16×16×32（含混类型） |

**工具链**：NDK 的 glslc/spirv-as 不支持 coopmat；改用 PyPI `slangpy` wheel
内置的 Slang 编译器（`mobile/tools/slang_glsl_spv.py` 通过其 C API 生成
SPIR-V）。手写模块在 Mali 上始终被拒（`VK_ERROR_INITIALIZATION_FAILED`，
排除项见 git 历史），但 **Slang 生成的模块 Mali 可以编译并运行** —— 说明是
编码细节，不是驱动缺失。

**当前驱动指纹**（`--caps` 已打印 driverVersion）：Mali-G925，Vulkan
api 1.3.278 / **driver 49.1.0**（r49p1），系统 OTA
`PLC110_11.C.30_1300_202604101246`（Android 16，2026-04-10 构建）。
GPU 驱动随 OxygenOS vendor 镜像发布，只能通过系统 OTA 更新；更新后用
一条命令复测：`./aexrt-m --bench-coopmat live.spv 8192 16 32 32 2`。

**Mali 实测结论（关键）**：coopmat dispatch 在 **≤ ~256 个 workgroup** 时
正常（但被 launch 开销主导，测不出有意义的 TFLOPS）；**≥ ~320 个 workgroup
后出现固定 ~10 ms 的退化**，且末尾 timestamp 不再写入。该悬崖在 1/4/32
MMA 链、静态/动态循环、有无逐迭代 store、16/32 线程工作组下均复现，
且 512/1024 组会出现 5 µs 的“成功”读数（末尾 timestamp 丢失），更像驱动
在 coopmat dispatch 上出错而非单纯变慢；同一规模下普通 dispatch 正常
（0.037 ms）。结论：**该驱动上的 coopmat 无法用于真实 conv**（需要数百到
数千 workgroup），内核重写计划搁置；等驱动更新后复测。

host 侧（NVIDIA Vulkan）coopmat 可正常规模化，仅作参考。

`--bench-coopmat SPV GROUPS M N K ESZ` 已内置（含 wall-clock 回退：驱动丢
末尾 timestamp 时改用提交墙钟）。Slang 模块用
`py -3 mobile/tools/slang_glsl_spv.py <slang-compiler.dll> <shader.slang> <out.spv> slang` 生成。

## 8. 库 API（in-process）

`yolo_flat.c` 用 `-DAEXRT_M_NO_MAIN` 编译即为可嵌入库，接口在
`mobile/runtime/aexrt_m.h`：

```c
AexrtMSession *s = aexrt_m_open("model.bin", "policy.txt", 2, 0.25f);
float frame[3 * 320 * 320];       /* fp32 NCHW */
AexrtMDet dets[64];
int n = aexrt_m_run(s, frame, dets, 64);
aexrt_m_close(s);
```

- 单进程单会话：Vulkan 设备、pipeline、录制好的命令图常驻（与 app engine 的
  用法一致）；`aexrt_m_run` 同步：写 cached staging → 提交 → 等待 → per-class NMS。
- `frames_in_flight > 1` 分配独立 value arena（异步模式的预留；当前 run 仍逐帧
  等待后才返回 detections）。
- `conf_threshold <= 0` 使用内置 0.25；`policy_path = NULL` 使用逐层 auto 规则。
- **Android 集成**（`.so` 构建、dlopen 冒烟测试、JNI wrapper、调用契约、性能参考）：
  见 [`docs/MOBILE_INTEGRATION.md`](../docs/MOBILE_INTEGRATION.md)。
- 构建：`py mobile/tools/build.py api-host` → `build/mobile/example_api.exe`；
  `py mobile/tools/build.py api-android` → `build/mobile/example_api_android`
  （aarch64，可直接 adb push 运行）。
- 验证：
  - host：`run_tests.py` 新增 API 门禁——同帧、同 conf 下 API 输出与 CLI
    **逐字节一致**；2 帧轮转（93/91 dets，最后一帧与 CLI 相同）；W8A8
    模型（92 dets）同样逐字节一致；
  - **Mali 真机**：`example_api_android` 在设备上跑同一帧、conf=1e-5，
    88 detections，与 CLI 输出 **逐字节一致**（2116 字节）。
- setup 日志由 `qprintf` 静音（`g_quiet`），库调用不会污染调用方 stdout。
- 延迟：真机同步 API 与 CLI 单帧同量级（热态 9–13 ms，含输入转换、提交、
  等待、NMS）；CLI 的 `--pipeline 2` 用多帧在飞达到 5.4–5.8 ms/帧 吞吐。
  `aexrt_m_open` 会先跑一次零帧预热（shader cache/DVFS），避免首帧 ~2×。
  异步 submit/poll API 尚未暴露，`frames_in_flight` 已预留独立 arena。

## 9. 已验证的负结果

| 尝试 | 结果 |
|---|---|
| 2×2 × 8 OC quartet | 独显慢 3~6×（线程数降 4×，每层 dispatch 太小，填充不足）；保留为待测变体 |
| fp16 dot（16 深度） | 独显慢 ~8%；**Mali 真机 auto 4.50 ms vs base 9.40 ms（2.1×）** |
| 1-OC/线程（10×10） | 真机更慢（权重/输入指令比恶化）；仅保留 `--conv-variant f16oc1` |
| 全模型 4-OC | 小空间受益、大空间受损；逐层 auto 混合优于任何纯变体 |
| Winograd F(2×2,3×3)（Mali 真机） | type-10 单层 0.75–2.86 ms，整模型 29–31 ms，直接 f16 为 ~5.6 ms（慢 5–6×）。2×2 tile 使线程数降 4×，10×10 仅 6–7 个工作组；transform/M 累加为 fp32 标量/vec2。默认关闭，`--winograd` 仅实验 |
| 残差 Add 折进卷积 epilogue | 少 6 次 dispatch，但 97 层模型比 103 层慢 ~6%；已回退，保留结论 |
| 4 通道向量化 elementwise | 独显持平/略慢（线程数降 4×）；`--vector-elem` 可选 |
| CPU 直接读 VRAM 候选缓冲 | 2100 候选读回 7.6 ms；改 staging 后 0.2 ms |
| W8A8 int8（Mali-G925 真机） | SDOT 吞吐低于 fp16 点积：整模型 8.45 ms vs fp16 5.0–5.5 ms；3×3 层慢 1.4–1.6×。保留 AXM3 契约与内核（PC 有效），移动端策略选 fp16 |
| 1×2 × 8 OC pair（Mali 真机） | 权重复用 2×，但累加器翻倍、线程数减半：6.99 ms vs auto 5.04 ms；寄存器/占用率仍是瓶颈 |

## 10. 下一步

1. **W8A8 已完成，移动端不采用**：桌面 W8A8 1.79× 已实测；Mali 的 SDOT 慢于
   fp16，除非改用 fp16 计算 + int8 权重（W8A16，本机权重带宽并非瓶颈，预期无效）。
2. **逐层 autotune 已完成持久化**：`--emit-policy/--policy` 已上线；后续若出现
   新内核变体，直接加入候选列表即可。
3. **Winograd**：F(2×2) 已在 Mali 实测为负结果（见上）。若继续，必须换数据流：
   tile 内共享 input transform、fp16 向量累加/更宽 OC 复用，或改 F(4×4)；
   不能直接沿用当前每线程独立 transform 的实现。
4. **N 帧流水线**：双 command buffer/fence，把 CPU NMS 与下一帧 GPU 重叠；
   当前是单帧串行，吞吐未优化。
5. **GPU NMS / top-k**：候选多时把 CPU O(n²) 彻底移出关键路径。
6. **持久化内核/多图层批处理**：把 10×10 头部的多次小 dispatch 合并，降低
   移动端 dispatch/barrier 固定开销。
