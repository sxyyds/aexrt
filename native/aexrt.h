#pragma once

#include <stdint.h>

#ifdef _WIN32
#ifdef AEXRT_NATIVE_EXPORTS
#define AEXRT_API __declspec(dllexport)
#else
#define AEXRT_API __declspec(dllimport)
#endif
#else
#define AEXRT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AexrtDevice AexrtDevice;
typedef struct AexrtBuffer AexrtBuffer;
typedef struct AexrtPreparedDispatch AexrtPreparedDispatch;
typedef struct AexrtCompiledGraph AexrtCompiledGraph;
typedef struct AexrtYoloModel AexrtYoloModel;

typedef struct AexrtYoloDetection {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float class_id;
} AexrtYoloDetection;

typedef struct AexrtD3D12Capabilities {
    uint32_t struct_size;
    uint32_t highest_shader_model;
    uint32_t native_fp16_supported;
    uint32_t wave_mma_tier;
    uint32_t dxc_available;
    uint32_t reserved[3];
} AexrtD3D12Capabilities;

typedef struct AexrtYoloRunTiming {
    uint32_t struct_size;
    uint32_t valid;
    double memcpy_ms;   /* CPU input copy into the persistently mapped upload buffer. */
    double submit_ms;   /* ID3D12CommandQueue::ExecuteCommandLists. */
    double fence_ms;    /* Queue Signal plus the CPU completion wait. */
    double readback_ms; /* Profile and detection result map/copy. */
    uint64_t reserved[4];
} AexrtYoloRunTiming;

typedef enum AexrtYoloPackageMode {
    AEXRT_YOLO_PACKAGE_UNKNOWN = 0,
    AEXRT_YOLO_PACKAGE_OUTPUT0_POSTPROCESS = 1,
    AEXRT_YOLO_PACKAGE_NATIVE_D3D12_GRAPH = 2
} AexrtYoloPackageMode;

typedef enum AexrtYoloOutputLayout {
    AEXRT_YOLO_OUTPUT_LAYOUT_UNKNOWN = 0,
    AEXRT_YOLO_OUTPUT_LAYOUT_CHANNELS_FIRST = 1,
    AEXRT_YOLO_OUTPUT_LAYOUT_CHANNELS_LAST = 2
} AexrtYoloOutputLayout;

typedef enum AexrtOpKind {
    AEXRT_OP_RELU_FLOAT32 = 1,
    AEXRT_OP_GELU_FLOAT32 = 2,
    AEXRT_OP_ADD_FLOAT32 = 3,
    AEXRT_OP_SUB_FLOAT32 = 4,
    AEXRT_OP_MUL_FLOAT32 = 5,
    AEXRT_OP_DIV_FLOAT32 = 6,
    AEXRT_OP_SIGMOID_FLOAT32 = 7,
    AEXRT_OP_TANH_FLOAT32 = 8
} AexrtOpKind;

typedef struct AexrtNodeDesc {
    AexrtOpKind op;
    uint32_t input0;
    uint32_t input1;
    uint32_t output;
} AexrtNodeDesc;

typedef struct AexrtScalarConstantDesc {
    uint32_t value;
    float scalar;
} AexrtScalarConstantDesc;

typedef struct AexrtGraphDesc {
    uint64_t element_count;
    uint32_t input_count;
    uint32_t node_count;
    const AexrtNodeDesc* nodes;
    uint32_t output_value;
    uint32_t constant_count;
    const AexrtScalarConstantDesc* constants;
} AexrtGraphDesc;

typedef struct AexrtConv2DDesc {
    uint32_t batch;
    uint32_t in_channels;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t out_channels;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_top;
    uint32_t pad_left;
    uint32_t dilation_h;
    uint32_t dilation_w;
    uint32_t groups;
} AexrtConv2DDesc;

/* DXGI 适配器描述，用于 vendor matrix 报告与设备选择。 */
typedef struct AexrtAdapterInfo {
    uint32_t struct_size;
    uint32_t raw_index;               /* EnumAdapters1 原始序号（软件适配器计入）。 */
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t is_software;
    uint32_t dedicated_video_memory_mb;
    wchar_t description[128];
} AexrtAdapterInfo;

AEXRT_API int aexrt_d3d12_probe(void);
AEXRT_API AexrtDevice* aexrt_d3d12_create_device(uint32_t adapter_index);
/*
 * 设备适配器选择支持的环境变量：
 *   AEXRT_NATIVE_D3D12_WARP=1       使用 D3D12 WARP 软件设备。
 *   AEXRT_NATIVE_D3D12_ADAPTER=<n>  使用原始 DXGI 序号（软件适配器计入）。
 * 未设置时按纯硬件序号（高性能优先）选择 adapter_index。
 */
AEXRT_API uint32_t aexrt_d3d12_enumerate_adapters(AexrtAdapterInfo* out_infos, uint32_t max_infos);
AEXRT_API int aexrt_d3d12_get_capabilities(
    AexrtDevice* device,
    AexrtD3D12Capabilities* capabilities);
AEXRT_API void aexrt_d3d12_destroy_device(AexrtDevice* device);

AEXRT_API AexrtBuffer* aexrt_d3d12_upload_float32(
    AexrtDevice* device,
    const float* data,
    uint64_t element_count);

AEXRT_API AexrtBuffer* aexrt_d3d12_allocate_float32_uav(
    AexrtDevice* device,
    uint64_t element_count);

AEXRT_API AexrtBuffer* aexrt_d3d12_create_float32_view(
    AexrtDevice* device,
    AexrtBuffer* buffer,
    uint64_t element_offset,
    uint64_t element_count);

AEXRT_API void aexrt_d3d12_destroy_buffer(AexrtBuffer* buffer);

AEXRT_API int aexrt_d3d12_relu_float32(
    AexrtDevice* device,
    AexrtBuffer* input,
    AexrtBuffer* output,
    uint64_t element_count);

AEXRT_API int aexrt_d3d12_conv2d_silu_float32(
    AexrtDevice* device,
    AexrtBuffer* input,
    AexrtBuffer* weight,
    AexrtBuffer* bias,
    AexrtBuffer* output,
    const AexrtConv2DDesc* desc);

AEXRT_API int aexrt_d3d12_conv2d_float32(
    AexrtDevice* device,
    AexrtBuffer* input,
    AexrtBuffer* weight,
    AexrtBuffer* bias,
    AexrtBuffer* output,
    const AexrtConv2DDesc* desc);

AEXRT_API int aexrt_d3d12_concat_conv1x1_float32(
    AexrtDevice* device,
    AexrtBuffer* const* inputs,
    const uint32_t* input_channels,
    uint32_t input_count,
    AexrtBuffer* weight,
    AexrtBuffer* bias,
    AexrtBuffer* output,
    uint32_t batch,
    uint32_t height,
    uint32_t width,
    uint32_t out_channels,
    uint32_t activation);

AEXRT_API AexrtPreparedDispatch* aexrt_d3d12_prepare_relu_float32(
    AexrtDevice* device,
    AexrtBuffer* input,
    AexrtBuffer* output,
    uint64_t element_count);

AEXRT_API AexrtPreparedDispatch* aexrt_d3d12_prepare_conv2d_silu_float32(
    AexrtDevice* device,
    AexrtBuffer* input,
    AexrtBuffer* weight,
    AexrtBuffer* bias,
    AexrtBuffer* output,
    const AexrtConv2DDesc* desc);

AEXRT_API AexrtPreparedDispatch* aexrt_d3d12_prepare_conv2d_silu_upload_float32(
    AexrtDevice* device,
    AexrtBuffer* weight,
    AexrtBuffer* bias,
    AexrtBuffer* output,
    const AexrtConv2DDesc* desc,
    uint32_t ring_size);

AEXRT_API int aexrt_d3d12_execute_prepared(
    AexrtDevice* device,
    AexrtPreparedDispatch* dispatch);

AEXRT_API int aexrt_d3d12_execute_prepared_upload_float32(
    AexrtDevice* device,
    AexrtPreparedDispatch* dispatch,
    const float* input,
    uint64_t element_count);

AEXRT_API void aexrt_d3d12_destroy_prepared(AexrtPreparedDispatch* dispatch);

AEXRT_API int aexrt_d3d12_download_float32(
    AexrtDevice* device,
    AexrtBuffer* buffer,
    float* out,
    uint64_t element_count);

AEXRT_API AexrtCompiledGraph* aexrt_compile_graph(
    AexrtDevice* device,
    const AexrtGraphDesc* desc);

AEXRT_API AexrtCompiledGraph* aexrt_load_graph_json(
    AexrtDevice* device,
    const char* path);

AEXRT_API uint32_t aexrt_graph_dispatch_count(
    AexrtCompiledGraph* graph);

AEXRT_API uint32_t aexrt_graph_buffer_count(
    AexrtCompiledGraph* graph);

AEXRT_API int aexrt_run(
    AexrtDevice* device,
    AexrtCompiledGraph* graph,
    const float* input,
    float* output);

AEXRT_API int aexrt_run2(
    AexrtDevice* device,
    AexrtCompiledGraph* graph,
    const float* input0,
    const float* input1,
    float* output);

AEXRT_API int aexrt_run_n(
    AexrtDevice* device,
    AexrtCompiledGraph* graph,
    const float* const* inputs,
    uint32_t input_count,
    float* output);

AEXRT_API void aexrt_destroy_graph(AexrtCompiledGraph* graph);

AEXRT_API AexrtYoloModel* aexrt_yolo_compile_from_package(
    AexrtDevice* device,
    const char* package_path);

AEXRT_API AexrtYoloModel* aexrt_yolo_load_engine(
    AexrtDevice* device,
    const char* engine_path);

AEXRT_API AexrtYoloModel* aexrt_yolo_compile_from_onnx(
    AexrtDevice* device,
    const char* onnx_path);

AEXRT_API uint64_t aexrt_yolo_input_element_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_class_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_anchor_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_channel_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_output_layout(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_has_objectness(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_package_mode(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_graph_node_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_graph_value_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_constant_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_prepared_command_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_supported_prepared_command_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_unsupported_prepared_command_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_prepared_skipped_command_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_late_concat_conv1x1_fusion_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_concat_residual_conv1x1_fusion_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_paired_conv3x3_fusion_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_c2f_bottleneck_superblock_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_c2f_tail_residual_fusion_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_head_fusion_enabled(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_head_final_conv_fusion_enabled(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_conv1x1_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_3x3_spatial_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_3x3_pack4_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_3x3_pack8_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_3x3_implicit_gemm_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_3x3_implicit_gemm_40x40_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_3x3_implicit_gemm_20x20_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_3x3_implicit_gemm_10x10_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_3x3_exact_40x40_64x64_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_3x3_exact_20x20_64x64_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_3x3_exact_10x10_128x128_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_3x3_exact_20x20_128x64_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_3x3_exact_10x10_256x64_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_tileflow_native_fp16_count(
    AexrtYoloModel* model);

AEXRT_API int aexrt_yolo_is_executable(
    AexrtYoloModel* model);

AEXRT_API int aexrt_yolo_run(
    AexrtDevice* device,
    AexrtYoloModel* model,
    const float* input,
    uint64_t input_element_count,
    AexrtYoloDetection* detections,
    uint32_t max_detections,
    uint32_t* out_detection_count);

/* 调试接口：读回最近一次 run 后的原始输出值（引擎数值验证用）。 */
AEXRT_API int aexrt_yolo_debug_output(AexrtYoloModel* model, float* out, uint64_t elements);
AEXRT_API int aexrt_yolo_debug_value(AexrtYoloModel* model, uint32_t value_id, float* out, uint64_t elements);
AEXRT_API int aexrt_yolo_debug_page(AexrtYoloModel* model, uint32_t page_id, float* out, uint64_t elements);

/* 上一次 aexrt_yolo_run / 物理计划校验失败的原因（fail-closed 诊断出口）。 */
AEXRT_API const char* aexrt_yolo_get_last_error(AexrtYoloModel* model);

AEXRT_API int aexrt_yolo_get_last_run_timing(
    AexrtYoloModel* model,
    AexrtYoloRunTiming* timing);

/*
 * N-buffer 流水线排空（AEXRT_NATIVE_D3D12_PIPELINE_FRAMES >= 2 时有意义）：
 * 等待最后一个 in-flight 帧完成并回收其检测结果。同步模式（frames=1）
 * 下无 in-flight 帧，返回 0 个检测。
 */
AEXRT_API int aexrt_yolo_flush_pipeline(
    AexrtYoloModel* model,
    AexrtYoloDetection* detections,
    uint32_t max_detections,
    uint32_t* out_detection_count);

AEXRT_API uint32_t aexrt_yolo_profile_event_count(
    AexrtYoloModel* model);

AEXRT_API const char* aexrt_yolo_profile_event_label(
    AexrtYoloModel* model,
    uint32_t index);

AEXRT_API double aexrt_yolo_profile_event_ms(
    AexrtYoloModel* model,
    uint32_t index);

/* autotune：卷积命令枚举与算法覆盖（P1.3，per-device kernel 选择）。 */
typedef struct AexrtConvPlanInfo {
    uint32_t struct_size;
    uint32_t command_index;        /* replay 命令序号（用于 override） */
    uint32_t kind_is_silu;
    AexrtConv2DDesc desc;          /* conv 形状 */
    uint32_t current_algorithm;    /* 当前生效算法 ID（含 fail-closed 回退，非固化值） */
    uint32_t candidate_count;
    uint32_t candidates[32];       /* 形状/设备可行的候选算法 ID */
    uint32_t in_fusion_group;      /* 1 = 命令位于物理融合组内，override 会破坏融合执行，调用方必须跳过 */
    uint32_t planned_algorithm;    /* 引擎固化的 planned_kernel（还原基线必须用它，用 current_algorithm 还原会把回退值写进计划） */
} AexrtConvPlanInfo;

AEXRT_API uint32_t aexrt_yolo_conv_plan_count(AexrtYoloModel* model);
AEXRT_API int aexrt_yolo_conv_plan_info(
    AexrtYoloModel* model,
    uint32_t index,
    AexrtConvPlanInfo* out_info);
/*
 * 覆盖一条 CONV/CONV_SILU 命令的算法并失效 prepared replay（下次 run 重录）。
 * 候选跨权重布局时可能产生数值差异，调用方必须做数值门禁；
 * 物理计划校验失败时运行时 fail-closed，调用方应跳过该候选。
 */
AEXRT_API int aexrt_yolo_override_conv_algorithm(
    AexrtYoloModel* model,
    uint32_t command_index,
    uint32_t algorithm_id);

/* Zero-copy input: returns the persistent CPU-mapped upload pointer for the
 * model input (fp32, row-major CHW, input_element_count elements). Writing
 * the frame directly here skips the aexrt_yolo_run input memcpy. */
AEXRT_API void* aexrt_yolo_input_ptr(
    AexrtYoloModel* model,
    uint64_t* out_elements);

AEXRT_API uint64_t aexrt_yolo_pipeline_cache_size(
    AexrtYoloModel* model);

AEXRT_API int aexrt_yolo_export_pipeline_cache(
    AexrtYoloModel* model,
    void* output,
    uint64_t output_nbytes);

AEXRT_API uint32_t aexrt_yolo_dxil_cache_hit_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_shader_cache_hit_count(
    AexrtYoloModel* model);

AEXRT_API uint32_t aexrt_yolo_pso_cache_hit_count(
    AexrtYoloModel* model);

AEXRT_API void aexrt_yolo_destroy(AexrtYoloModel* model);

#ifdef __cplusplus
}
#endif
