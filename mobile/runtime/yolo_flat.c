/* AEXRT-M mobile runtime: AXM1 flat model -> Vulkan compute -> detections.
 *
 * Host-side architecture note: the command buffer and every descriptor set
 * are built ONCE for a fixed shape (see record_graph). Per-frame work is a
 * single queue submit plus a fence wait, so the benchmark measures GPU time,
 * not command recording. Value placement is position-major and channel
 * offset based, which lets concat producers write straight into the concat
 * buffer and slices stay zero-copy views.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <vulkan/vulkan.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include "aexrt_m.h"

#include "../kernels/conv_i8_spv.h"
#include "../kernels/conv_quad_spv.h"
#include "../kernels/conv_f16_spv.h"
#include "../kernels/conv_f16_oc4_spv.h"
#include "../kernels/conv_stem_spv.h"
#include "../kernels/conv_wino_spv.h"
#include "../kernels/conv_wino2_spv.h"
#include "../kernels/conv_i8q_spv.h"
#include "../kernels/conv_stem_i8q_spv.h"
#include "../kernels/elementwise_i8_spv.h"
#include "../kernels/decode_i8_spv.h"
#include "../kernels/conv_f16_oc1_spv.h"
#include "../kernels/conv_pair_spv.h"
#include "../kernels/elementwise_spv.h"
#include "../kernels/decode_spv.h"

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
static double now_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}
#endif

#define CHK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    printf("FAIL %s -> %d (line %d)\n", #x, (int)r_, __LINE__); return 1; } } while (0)

/* ---- IEEE half <-> float (host portability; Android clang has _Float16,
 *      MSVC does not, and bit casts are safer than type punning) ---- */
static void ensure_dir(const char *path) {
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0777);
#endif
}

static uint16_t f2h(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t e = (int32_t)((u >> 23) & 0xffu) - 127;
    uint32_t m = u & 0x7fffffu;
    if (e == 128) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0u));
    int32_t h = e + 15;
    if (h >= 31) return (uint16_t)(sign | 0x7c00u);
    if (h <= 0) {
        if (h < -10) return (uint16_t)sign;
        m |= 0x800000u;
        uint32_t sh = (uint32_t)(14 - h);
        uint32_t r = m >> sh;
        uint32_t rem = m & ((1u << sh) - 1u);
        uint32_t half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1u))) r++;
        return (uint16_t)(sign | r);
    }
    uint32_t r = sign | ((uint32_t)h << 10) | (m >> 13);
    uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) r++;
    return (uint16_t)r;
}
static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, u;
    if (e == 0) {
        if (m == 0) u = sign;
        else {
            e = 127 - 15 + 1;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            u = sign | (e << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (e == 31) u = sign | 0x7f800000u | (m << 13);
    else u = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static int g_quiet;
static int g_api_mode;

static void qprintf(const char *fmt, ...) {
    if (g_quiet) return;
    va_list ap; va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

/* fp32 NCHW -> the runtime's fp16 position-major input layout [pos*3 + c].
 * aarch64: vcvt_f16_f32 converts four pixels per instruction; the scalar f2h
 * loop cost ~0.5 ms/frame at 320x320, which showed up directly in streaming. */
static void load_frame_fp16(const float *src, uint16_t *dst, uint32_t h, uint32_t w) {
    uint32_t sp = h * w;
#if defined(__aarch64__)
    uint32_t p = 0;
    for (; p + 3u < sp; p += 4u) {
        float16x4_t hr = vcvt_f16_f32(vld1q_f32(src + p));
        float16x4_t hg = vcvt_f16_f32(vld1q_f32(src + sp + p));
        float16x4_t hb = vcvt_f16_f32(vld1q_f32(src + 2u * sp + p));
        uint16_t r[4], g[4], b[4];
        vst1_u16(r, vreinterpret_u16_f16(hr));
        vst1_u16(g, vreinterpret_u16_f16(hg));
        vst1_u16(b, vreinterpret_u16_f16(hb));
        for (uint32_t k = 0; k < 4u; ++k) {
            dst[(p + k) * 3u + 0u] = r[k];
            dst[(p + k) * 3u + 1u] = g[k];
            dst[(p + k) * 3u + 2u] = b[k];
        }
    }
    for (; p < sp; ++p) {
        dst[p * 3u + 0u] = f2h(src[p]);
        dst[p * 3u + 1u] = f2h(src[sp + p]);
        dst[p * 3u + 2u] = f2h(src[2u * sp + p]);
    }
#else
    for (uint32_t p = 0; p < sp; ++p)
        for (uint32_t c = 0; c < 3; ++c)
            dst[p * 3 + c] = f2h(src[c * sp + p]);
#endif
}

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; VkDeviceSize sz; } Buf;

static VkInstance inst;
static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkCommandBuffer cmd;
static VkCommandBuffer cmds[6];
static VkFence fence;
#define MAXSETS 4
static VkFence pipe_fences[MAXSETS];
static VkDescriptorPool dpool;
static VkPipeline pipes[16];
static VkPipelineLayout layouts[16];
static VkDescriptorSetLayout setl[16];
static uint32_t qfam;
static VkQueryPool qpool, gpuqp;
static VkEvent seq_ev;      /* policy attribution: true execution serialization */
static int profiling;
static int force_bar;   /* emit-policy: isolate each dispatch for attribution */
static int have_quad = 1;
static int have_dot;
static int have_coopmat;
static int have_vmm, have_ssc;
static int force_adapter = -1;
static int submit_and_wait(VkCommandBuffer cb);
static int g_vec_elem = 0;
static uint32_t prof_n;
static uint32_t *prof_label;
static double ts_period_ns;
/* Per-layer execution policy: an offline measured variant map, the mobile
 * counterpart of the PC engine's algo map. -1 means "use the global rule". */
static int *layer_var;
static int have_policy;

static int mkbuf_ex(VkDeviceSize sz, VkBufferUsageFlags usage, Buf *b, int host_read) {
    if (sz == 0) sz = 16;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = sz; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHK(vkCreateBuffer(dev, &bi, 0, &b->buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, b->buf, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
    int ty = -1;
    /* GPU-only buffers: device-local mapped memory (unified on mobile, VRAM
     * on a discrete GPU). CPU readback buffers: HOST_CACHED system memory —
     * reading VRAM from the CPU is an uncached PCIe access (~0.5 us/word). */
    const VkMemoryPropertyFlags want_read[2] = {
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT };
    const VkMemoryPropertyFlags want_gpu[2] = {
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT };
    const VkMemoryPropertyFlags *want = host_read ? want_read : want_gpu;
    for (int w = 0; w < 2 && ty < 0; ++w)
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
            if (!(req.memoryTypeBits & (1u << i)) || (f & want[w]) != want[w]) continue;
            if (host_read && w == 0 && (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
            ty = (int)i; break;
        }
    if (ty < 0) { printf("no host-visible memory type for %llu bytes\n",
                         (unsigned long long)sz); return 1; }
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size; ai.memoryTypeIndex = (uint32_t)ty;
    CHK(vkAllocateMemory(dev, &ai, 0, &b->mem));
    CHK(vkBindBufferMemory(dev, b->buf, b->mem, 0));
    CHK(vkMapMemory(dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map));
    b->sz = req.size;
    memset(b->map, 0, (size_t)req.size);
    return 0;
}
static int mkbuf(VkDeviceSize sz, VkBufferUsageFlags usage, Buf *b) {
    return mkbuf_ex(sz, usage, b, 0);
}

static int mkpipe(int idx, const uint32_t *spv, size_t sz, int nb, int pc_sz) {
    VkDescriptorSetLayoutBinding lb[8];
    memset(lb, 0, sizeof(lb));
    for (int i = 0; i < nb; ++i) {
        lb[i].binding = (uint32_t)i;
        lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lb[i].descriptorCount = 1;
        lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl.bindingCount = (uint32_t)nb; dsl.pBindings = lb;
    CHK(vkCreateDescriptorSetLayout(dev, &dsl, 0, &setl[idx]));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pc_sz};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1; pl.pSetLayouts = &setl[idx];
    pl.pushConstantRangeCount = pc_sz > 0 ? 1 : 0;
    pl.pPushConstantRanges = pc_sz > 0 ? &pcr : NULL;
    CHK(vkCreatePipelineLayout(dev, &pl, 0, &layouts[idx]));
    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = sz; sm.pCode = spv;
    VkShaderModule mod;
    CHK(vkCreateShaderModule(dev, &sm, 0, &mod));
    VkComputePipelineCreateInfo cp = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = mod; cp.stage.pName = "main";
    cp.layout = layouts[idx];
    CHK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cp, 0, &pipes[idx]));
    vkDestroyShaderModule(dev, mod, 0);
    return 0;
}

static int dispatch(int pi, Buf **bufs, int nb, const void *push, int pc_sz,
                    uint32_t wx, int bar, uint32_t label) {
    for (int q = 0; q < nb; ++q)
        if (!bufs[q]->buf) { printf("NULL buf binding %d pipe %d\n", q, pi); return 1; }
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &setl[pi];
    VkDescriptorSet set;
    CHK(vkAllocateDescriptorSets(dev, &ai, &set));
    VkWriteDescriptorSet wr[8];
    VkDescriptorBufferInfo bi[8];
    memset(wr, 0, sizeof(wr));
    memset(bi, 0, sizeof(bi));
    for (int i = 0; i < nb; ++i) {
        bi[i].buffer = bufs[i]->buf;
        bi[i].offset = 0;
        bi[i].range = VK_WHOLE_SIZE;
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = set; wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &bi[i];
    }
    /* The dependency barrier must sit BEFORE the consuming dispatch: a
     * producer recorded earlier in the same command buffer is not ordered
     * against this dispatch otherwise. */
    if (bar) {
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, NULL, 0, NULL);
        /* A memory barrier does not stop the next dispatch from overlapping
         * this one; per-layer attribution needs an execution dependency. */
        if (force_bar && seq_ev) {
            vkCmdSetEvent(cmd, seq_ev, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            vkCmdWaitEvents(cmd, 1, &seq_ev,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, NULL, 0, NULL, 0, NULL);
        }
    }
    int prof = profiling && label != 0xFFFFFFFFu;
    if (prof)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            qpool, prof_n * 2u);
    vkUpdateDescriptorSets(dev, (uint32_t)nb, wr, 0, NULL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[pi]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layouts[pi],
                            0, 1, &set, 0, NULL);
    if (pc_sz > 0)
        vkCmdPushConstants(cmd, layouts[pi], VK_SHADER_STAGE_COMPUTE_BIT,
                           0, (uint32_t)pc_sz, push);
    vkCmdDispatch(cmd, wx, 1, 1);
    if (prof) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            qpool, prof_n * 2u + 1u);
        prof_label[prof_n++] = label;
    }
    return 0;
}

/* per-value read/write epochs: skip the barrier when this dispatch cannot
 * touch anything written since the last barrier (sibling concat producers,
 * split fan-out, independent head branches). */
static uint32_t *wr_ep, *rd_ep, seq_no;
static int haz(uint32_t in0, uint32_t in1, uint32_t out, uint32_t t_bar) {
    if (force_bar) return 1;
    int h = (wr_ep[in0] > t_bar) || (wr_ep[out] > t_bar) || (rd_ep[out] > t_bar);
    if (in1 < 0xFFFFFFFFu) h = h || (wr_ep[in1] > t_bar);
    return h;
}
static void mark(uint32_t in0, uint32_t in1, uint32_t out) {
    wr_ep[out] = ++seq_no;
    rd_ep[in0] = ++seq_no;
    if (in1 < 0xFFFFFFFFu) rd_ep[in1] = ++seq_no;
}

typedef struct { uint32_t type, act, stride, pad, oc, ic, kh, kw, in0, out, extra, w_off, b_off, s_off; } Layer;

/* ---- session state -------------------------------------------------------
 * Shared by the CLI and the library API (aexrt_m_open/run/close). Single
 * instance: the runtime owns one Vulkan device and one recorded graph. The
 * CLI fills these from argv; the API fills them from aexrt_m_open(). */
static const char *model_path, *input_path, *out_path;
static const char *dump_dir, *dump_cands, *policy_path, *emit_policy_path, *frames_path;
static uint32_t bench = 1;
static int conv_variant, nsets = 1, stream_mode, want_profile, autotune;
static float conf_thr = 0.25f, iou_thr = 0.45f;
static long msz;
static uint8_t *md;
static int is_i8;
static uint32_t nlayers, nvals, in_vid, out_vid, lt_sz, nc, nlevels, in_h, in_w;
static uint32_t box_vid[3], cls_vid[3];
static float *vscale;
static uint8_t *vdtype;
static uint32_t lt_off, wt_off;
static float *input_frame;
static Layer *layers;
static uint32_t *vsize, *vh, *vw, *vc;
static uint32_t *pb, *po;
static Buf *vb_sets, *vb;
static float *frames;
static uint32_t nframes, frame_elems;
static Buf wbuf;
static Buf candb_s[MAXSETS], cntb_s[MAXSETS], cands_s[MAXSETS], cnts_s[MAXSETS];
static Buf cntb, cands, cnts;
static Buf in_stage[MAXSETS];
static int g_session_ready;
static uint32_t g_api_next_set, g_api_submitted;

typedef struct { double ms; uint32_t label; } ProfRow;
static int prof_cmp(const void *pa, const void *pb) {
    const ProfRow *a = pa, *b = pb;
    return a->ms < b->ms ? 1 : a->ms > b->ms ? -1 : 0;
}

static void print_profile(const Layer *layers, const uint32_t *vh,
                          const uint32_t *vw, const uint32_t *vc) {
    if (!profiling || prof_n == 0) return;
    uint64_t *ts = malloc((size_t)prof_n * 2 * 8);
    VkResult r = vkGetQueryPoolResults(dev, qpool, 0, prof_n * 2,
                                       (size_t)prof_n * 2 * 8, ts, 8,
                                       VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) { printf("profile readback failed %d\n", (int)r); free(ts); return; }
    ProfRow *rows = malloc((size_t)prof_n * sizeof(ProfRow));
    double total = 0.0;
    for (uint32_t q = 0; q < prof_n; ++q) {
        uint64_t a = ts[q * 2], b = ts[q * 2 + 1];
        rows[q].ms = (double)(b - a) * ts_period_ns / 1e6;
        rows[q].label = prof_label[q];
        total += rows[q].ms;
    }
    qsort(rows, prof_n, sizeof(ProfRow), prof_cmp);
    printf("profile: %u dispatches, %.2f ms accounted\n", prof_n, total);
    for (uint32_t q = 0; q < prof_n && q < 18; ++q) {
        uint32_t lab = rows[q].label;
        if (lab & 0x80000000u) {
            printf("  %6.3f ms  %5.1f%%  decode L%u\n", rows[q].ms,
                   100.0 * rows[q].ms / (total > 0 ? total : 1), lab & 0x7fffffffu);
            continue;
        }
        const Layer *L = &layers[lab];
        uint32_t ih = vh[L->in0], iw = vw[L->in0];
        uint32_t oh = vh[L->out], ow = vw[L->out];
        double mac = (double)L->oc * L->ic * L->kh * L->kw * oh * ow;
        printf("  %6.3f ms  %5.1f%%  L%-3u t%u %3u->%-4u k%ux%u s%u %ux%u->%ux%u  %.1f MMAC\n",
               rows[q].ms, 100.0 * rows[q].ms / (total > 0 ? total : 1),
               lab, L->type, L->ic, L->oc, L->kh, L->kw, L->stride,
               iw, ih, ow, oh, mac / 1e6);
    }
    free(rows);
    free(ts);
}

/* Capability dump: the kernel roadmap depends on which instruction families
 * the device actually exposes. VK_KHR_cooperative_matrix is the only path to
 * matrix units beyond plain dot4 (WaveMMA is D3D12-only). */
static void print_caps(void) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pdev, NULL, &n, NULL);
    VkExtensionProperties *exts = malloc((size_t)n * sizeof(*exts));
    if (exts) vkEnumerateDeviceExtensionProperties(pdev, NULL, &n, exts);
    int has_coop = 0, has_dot = 0, has_subgroup = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const char *e = exts[i].extensionName;
        if (!strcmp(e, "VK_KHR_cooperative_matrix")) has_coop = 1;
        if (!strcmp(e, "VK_KHR_shader_integer_dot_product")) has_dot = 1;
        if (!strcmp(e, "VK_EXT_subgroup_size_control")) has_subgroup = 1;
        printf("ext: %s\n", e);
    }
    free(exts);
    printf("extensions: cooperative_matrix=%d integer_dot_product=%d subgroup_size_control=%d\n",
           has_coop, has_dot, has_subgroup);
    if (!has_coop) return;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &cmf;
    vkGetPhysicalDeviceFeatures2(pdev, &f2);
    printf("cooperativeMatrix=%d robustBufferAccess=%d\n",
           cmf.cooperativeMatrix, cmf.cooperativeMatrixRobustBufferAccess);
    {
        VkPhysicalDeviceVulkan12Features c12 = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceSubgroupSizeControlFeatures cssc = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        VkPhysicalDeviceFeatures2 cf2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        c12.pNext = &cssc; cf2.pNext = &c12;
        vkGetPhysicalDeviceFeatures2(pdev, &cf2);
        printf("vulkanMemoryModel=%d subgroupSizeControl=%d computeFullSubgroups=%d\n",
               c12.vulkanMemoryModel, cssc.subgroupSizeControl, cssc.computeFullSubgroups);
    }
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR get_props =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!get_props) { printf("coopmat property query unavailable\n"); return; }
    uint32_t pc = 0;
    VkResult r = get_props(pdev, &pc, NULL);
    printf("coopmat properties: count=%u result=%d\n", pc, (int)r);
    if (r != VK_SUCCESS || pc == 0) return;
    if (pc > 64) pc = 64;
    VkCooperativeMatrixPropertiesKHR *props = calloc(pc, sizeof(*props));
    for (uint32_t i = 0; i < pc; ++i)
        props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    if (get_props(pdev, &pc, props) == VK_SUCCESS) {
        for (uint32_t i = 0; i < pc; ++i)
            printf("  coopmat M=%u N=%u K=%u A=%d B=%d C=%d R=%d sat=%u scope=%d\n",
                   props[i].MSize, props[i].NSize, props[i].KSize, (int)props[i].AType,
                   (int)props[i].BType, (int)props[i].CType, (int)props[i].ResultType,
                   props[i].saturatingAccumulation, (int)props[i].scope);
    }
    free(props);
}

/* Cooperative-matrix microbenchmark: one workgroup runs `mul` chained MMAs
 * per subgroup. Reports matrix-unit TFLOP/s for the module passed in (the
 * NDK toolchain cannot compile GL_KHR_cooperative_matrix, so the SPIR-V is
 * generated by mobile/tools/mk_coopmat_spv.py). */
static int bench_coopmat(const char *path, uint32_t groups, uint32_t m,
                         uint32_t n, uint32_t k, uint32_t esz, uint32_t mul) {
    if (!have_coopmat) { printf("VK_KHR_cooperative_matrix not available\n"); return 1; }
    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 20 || sz % 4) { printf("bad spv size %ld\n", sz); return 1; }
    uint32_t *code = malloc((size_t)sz);
    if (!code || fread(code, 1, (size_t)sz, f) != (size_t)sz) { printf("short spv read\n"); return 1; }
    fclose(f);
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = (size_t)sz; smci.pCode = code;
    VkShaderModule sm;
    if (vkCreateShaderModule(dev, &smci, NULL, &sm) != VK_SUCCESS) {
        printf("shader module rejected\n"); return 1;
    }
    VkDescriptorSetLayoutBinding bind[3];
    memset(bind, 0, sizeof(bind));
    for (int i = 0; i < 3; ++i) {
        bind[i].binding = (uint32_t)i;
        bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bind[i].descriptorCount = 1;
        bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = 3; dlci.pBindings = bind;
    VkDescriptorSetLayout dsl;
    if (vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl) != VK_SUCCESS) return 1;
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout pl;
    if (vkCreatePipelineLayout(dev, &plci, NULL, &pl) != VK_SUCCESS) return 1;
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm; cpci.stage.pName = "main";
    if (getenv("AEXRT_COOP_PIPE_FLAGS"))
        cpci.flags = VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT |
                     VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
    cpci.layout = pl;
    VkPipeline pipe;
    VkResult pr = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe);
    if (pr != VK_SUCCESS) { printf("pipeline creation failed %d\n", (int)pr); return 1; }

    Buf ba, bb, bc;
    memset(&ba, 0, sizeof(ba)); memset(&bb, 0, sizeof(bb)); memset(&bc, 0, sizeof(bc));
    if (mkbuf((VkDeviceSize)m * k * esz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ba)) return 1;
    if (mkbuf((VkDeviceSize)k * n * esz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bb)) return 1;
    if (mkbuf((VkDeviceSize)m * n * esz + 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bc)) return 1;
    for (uint32_t i = 0; i < m * k; ++i) {
        if (esz == 2) ((uint16_t *)ba.map)[i] = 0x3c00;   /* 1.0h */
        else ((float *)ba.map)[i] = 1.0f;
    }
    for (uint32_t i = 0; i < k * n; ++i) {
        if (esz == 2) ((uint16_t *)bb.map)[i] = 0x3800;   /* 0.5h */
        else ((float *)bb.map)[i] = 0.5f;
    }
    memset(bc.map, 0, (size_t)m * n * esz);

    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &dsl;
    VkDescriptorSet set;
    if (vkAllocateDescriptorSets(dev, &ai, &set) != VK_SUCCESS) return 1;
    VkDescriptorBufferInfo bi[3] = {{ba.buf, 0, VK_WHOLE_SIZE}, {bb.buf, 0, VK_WHOLE_SIZE},
                                    {bc.buf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[3];
    memset(wr, 0, sizeof(wr));
    for (int i = 0; i < 3; ++i) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = set; wr[i].dstBinding = (uint32_t)i; wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(dev, 3, wr, 0, NULL);

    VkPhysicalDeviceSubgroupProperties sp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &sp;
    vkGetPhysicalDeviceProperties2(pdev, &p2);
    const uint32_t sub_per_wg = 32u / (sp.subgroupSize ? sp.subgroupSize : 32u);
    double flops = 2.0 * (double)m * n * k * (double)mul * (double)sub_per_wg * (double)groups;

    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        VkCommandBufferBeginInfo beg = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHK(vkBeginCommandBuffer(cmd, &beg));
        vkCmdResetQueryPool(cmd, gpuqp, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, gpuqp, 0);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
        vkCmdDispatch(cmd, groups, 1, 1);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, gpuqp, 1);
        CHK(vkEndCommandBuffer(cmd));
        double w0 = now_ms();
        if (submit_and_wait(cmd)) return 1;
        double wall = now_ms() - w0;
        uint64_t ts[2] = {0, 0};
        vkGetQueryPoolResults(dev, gpuqp, 0, 2, sizeof(ts), ts, 8,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        double ms = (double)(ts[1] - ts[0]) * ts_period_ns / 1e6;
        /* Above a workgroup-count threshold this driver stops writing the
         * trailing timestamp; fall back to the submit wall time. */
        if (ts[1] <= ts[0] || ms <= 0) ms = wall;
        if (ms > 0 && ms < best) best = ms;
    }
    printf("coopmat: %s m=%u n=%u k=%u esz=%u mul=%u groups=%u sub/wg=%u\n",
           path, m, n, k, esz, mul, groups, sub_per_wg);
    printf("coopmat: %.3f ms  %.2f TFLOP/s  (%.2f TMAC/s)\n",
           best, flops / (best * 1e-3) / 1e12, flops / 2.0 / (best * 1e-3) / 1e12);
    return 0;
}

static int gpu_init(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "aexrt-m";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    CHK(vkCreateInstance(&ici, NULL, &inst));

    uint32_t n = 0;
    CHK(vkEnumeratePhysicalDevices(inst, &n, NULL));
    if (n == 0) { qprintf("no Vulkan physical device\n"); return 1; }
    VkPhysicalDevice *pds = malloc(n * sizeof(*pds));
    CHK(vkEnumeratePhysicalDevices(inst, &n, pds));
    int best = -1, best_score = -1;
    for (uint32_t i = 0; i < n; ++i) {
        if (force_adapter >= 0 && (uint32_t)force_adapter != i) continue;
        VkPhysicalDeviceVulkan12Features f12 = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDevice16BitStorageFeatures f16s = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
        VkPhysicalDeviceShaderIntegerDotProductFeatures dot = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
        VkPhysicalDeviceSubgroupSizeControlFeatures ssc = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        dot.pNext = &ssc;
        f16s.pNext = &dot;
        f12.pNext = &f16s;
        VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &f12;
        vkGetPhysicalDeviceFeatures2(pds[i], &f2);
        if (!f12.shaderFloat16 || !f16s.storageBuffer16BitAccess) continue;
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pds[i], &p);
        if (getenv("AEXRT_GPU_DEBUG"))
            qprintf("[gpu] idx %u: %s type %d\n", i, p.deviceName, (int)p.deviceType);
        if (p.apiVersion < VK_API_VERSION_1_2) continue;
        int score = (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 2 :
                    (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 1 : 0;
        if (score > best_score) {
            best_score = score; best = (int)i;
            have_dot = dot.shaderIntegerDotProduct ? 1 : 0;
            have_vmm = f12.vulkanMemoryModel ? 1 : 0;
            have_ssc = (ssc.subgroupSizeControl && ssc.computeFullSubgroups) ? 1 : 0;
        }
    }
    if (best < 0) {
        qprintf("no Vulkan device with shaderFloat16 + storageBuffer16BitAccess\n");
        free(pds);
        return 1;
    }
    pdev = pds[best];
    free(pds);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pdev, &props);
    qprintf("GPU: %s (api %u.%u.%u, driver %u.%u.%u)\n", props.deviceName,
           VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion),
           VK_VERSION_PATCH(props.apiVersion),
           VK_VERSION_MAJOR(props.driverVersion), VK_VERSION_MINOR(props.driverVersion),
           VK_VERSION_PATCH(props.driverVersion));
    ts_period_ns = (double)props.limits.timestampPeriod;
    qprintf("timestampPeriod %.3f ns\n", ts_period_ns);
    if (props.limits.maxPushConstantsSize < 128) {
        qprintf("maxPushConstantsSize %u < 128\n", props.limits.maxPushConstantsSize);
        return 1;
    }

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qn, NULL);
    VkQueueFamilyProperties *qf = malloc(qn * sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qn, qf);
    int found = -1;
    for (uint32_t i = 0; i < qn; ++i)
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].queueCount > 0) {
            found = (int)i; break;
        }
    free(qf);
    if (found < 0) { qprintf("no compute queue family\n"); return 1; }
    qfam = (uint32_t)found;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features f12 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.shaderFloat16 = VK_TRUE;
    VkPhysicalDevice16BitStorageFeatures f16s = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    f16s.storageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
    dot.shaderIntegerDotProduct = have_dot ? VK_TRUE : VK_FALSE;
    f16s.pNext = &dot;
    f12.pNext = &f16s;
    /* cooperative matrix is optional: enable the extension and chain its
     * feature struct only when the device advertises it */
    {
        uint32_t en = 0;
        vkEnumerateDeviceExtensionProperties(pdev, NULL, &en, NULL);
        VkExtensionProperties *ep = en ? malloc((size_t)en * sizeof(*ep)) : NULL;
        if (ep) vkEnumerateDeviceExtensionProperties(pdev, NULL, &en, ep);
        for (uint32_t i = 0; i < en; ++i)
            if (!strcmp(ep[i].extensionName, "VK_KHR_cooperative_matrix")) have_coopmat = 1;
        free(ep);
    }
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    cmf.cooperativeMatrix = have_coopmat ? VK_TRUE : VK_FALSE;
    if (have_coopmat) dot.pNext = &cmf;
    /* Mali's coopmat path requires the Vulkan memory model and full-subgroup
     * control (Dawn's documented requirements for the same feature). */
    f12.vulkanMemoryModel = have_vmm ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceSubgroupSizeControlFeatures ssc_enable = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    ssc_enable.subgroupSizeControl = have_ssc ? VK_TRUE : VK_FALSE;
    ssc_enable.computeFullSubgroups = have_ssc ? VK_TRUE : VK_FALSE;
    {
        void *tail = have_coopmat ? (void *)&cmf : (void *)&dot;
        if (have_coopmat) cmf.pNext = &ssc_enable; else dot.pNext = &ssc_enable;
        (void)tail;
    }
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f12;
    const char *dev_ext[1];
    if (have_coopmat) {
        dev_ext[0] = "VK_KHR_cooperative_matrix";
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = dev_ext;
    }
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    CHK(vkCreateDevice(pdev, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = qfam;
    CHK(vkCreateCommandPool(dev, &pci, NULL, &pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 6;
    CHK(vkAllocateCommandBuffers(dev, &cai, cmds));
    cmd = cmds[0];
    VkFenceCreateInfo ff = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHK(vkCreateFence(dev, &ff, NULL, &fence));
    VkEventCreateInfo evi = {VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
    CHK(vkCreateEvent(dev, &evi, NULL, &seq_ev));
    for (int i = 0; i < MAXSETS; ++i) {
        VkFenceCreateInfo pfi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        CHK(vkCreateFence(dev, &pfi, NULL, &pipe_fences[i]));
    }
    VkDescriptorPoolSize dpsz = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 2048; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dpsz;
    CHK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
    VkQueryPoolCreateInfo gq = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    gq.queryType = VK_QUERY_TYPE_TIMESTAMP;
    gq.queryCount = 2;
    CHK(vkCreateQueryPool(dev, &gq, NULL, &gpuqp));
    return 0;
}

static int submit_and_wait(VkCommandBuffer cb) {
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkResetFences(dev, 1, &fence);
    CHK(vkQueueSubmit(queue, 1, &si, fence));
    CHK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    return 0;
}

/* Submit without waiting; `reuse` means this fence may still be pending from
 * the same buffer set's previous frame and must be reaped first. */
static int submit_async(VkCommandBuffer cb, VkFence f, int reuse) {
    if (reuse) CHK(vkWaitForFences(dev, 1, &f, VK_TRUE, UINT64_MAX));
    CHK(vkResetFences(dev, 1, &f));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    CHK(vkQueueSubmit(queue, 1, &si, f));
    return 0;
}

/* Median GPU-timestamp time of a recorded command buffer; used by the
 * load-time autotuner to compare kernel variants on the actual device. */
static int dcmp(const void *pa, const void *pb);
static Buf *g_cntb;
static void usage(const char *argv0) {
    printf("usage: %s <model.bin> <input.f32> <detections.bin> [bench] [options]\n"
           "  bench              number of timed iterations (default 1)\n"
           "  --dump-heads DIR   write per-level box/cls tensors as fp32\n"
           "  --dump-cands FILE  write the raw GPU candidate list\n"
           "  --conv-variant V   auto | base | f16 | f16oc4 | f16oc1 | quad | pair (default auto)\n"
           "  --policy FILE      apply a per-layer variant map (offline autotune)\n"
           "  --emit-policy FILE measure every variant per layer and write the map\n"
           "  --pipeline N       N frames in flight (1..4; throughput mode)\n"
           "  --stream           per-frame input + N frames in flight (needs >=2)\n"
           "  --frames FILE      concatenated fp32 NCHW frames for --stream\n"
           "  --caps             print device capability/extensions and exit\n"
           "  --adapter N        force physical device index (with --caps/bench)\n"
           "  --bench-coopmat SPV GROUPS M N K ESZ   matrix-unit microbenchmark\n"
           "  --conf P           confidence threshold 0..1 (default 0.25)\n"
           "  --iou P            NMS IoU threshold (default 0.45)\n", argv0);
}

typedef struct { float x1, y1, x2, y2, score; int cls; } Det;

static int dcmp(const void *pa, const void *pb) {
    double a = *(const double *)pa, b = *(const double *)pb;
    return a < b ? -1 : a > b ? 1 : 0;
}

static int det_cmp(const void *pa, const void *pb) {
    const Det *a = pa, *b = pb;
    if (a->score != b->score) return a->score > b->score ? -1 : 1;
    /* deterministic tie-break: same-score boxes must keep identical order
     * across C qsort and the reference NMS implementation */
    if (a->cls != b->cls) return a->cls < b->cls ? -1 : 1;
    if (a->x1 != b->x1) return a->x1 < b->x1 ? -1 : 1;
    if (a->y1 != b->y1) return a->y1 < b->y1 ? -1 : 1;
    if (a->x2 != b->x2) return a->x2 < b->x2 ? -1 : 1;
    if (a->y2 != b->y2) return a->y2 < b->y2 ? -1 : 1;
    return 0;
}

/* Greedy per-class NMS over a GPU candidate list. `out` receives the kept
 * detections in score order (the same order the file dump uses). Returns the
 * kept count. Bounded top-K per class keeps the O(n^2) pass cheap without
 * changing any kept detection above the cutoff. */
static uint32_t nms_candidates(const float *cand, uint32_t ndet, uint32_t ncls,
                               float iou_thr, Det *out, uint32_t max_out) {
    if (ndet > 4096u) ndet = 4096u;
    Det *dets = malloc(sizeof(Det) * (ndet ? ndet : 1));
    if (!dets) return 0;
    for (uint32_t i = 0; i < ndet; ++i) {
        dets[i].x1 = cand[i * 6 + 0]; dets[i].y1 = cand[i * 6 + 1];
        dets[i].x2 = cand[i * 6 + 2]; dets[i].y2 = cand[i * 6 + 3];
        dets[i].score = cand[i * 6 + 4];
        dets[i].cls = (int)(cand[i * 6 + 5] + 0.5f);
    }
    qsort(dets, ndet, sizeof(Det), det_cmp);
    {
        const uint32_t topk = 128;
        uint32_t n2 = 0;
        uint32_t *ccls = calloc(ncls ? ncls : 1, 4);
        Det *sel = malloc(sizeof(Det) * (ndet ? ndet : 1));
        for (uint32_t i = 0; i < ndet; ++i) {
            int c = dets[i].cls;
            if (c < 0 || c >= (int)ncls) continue;
            if (ccls[c]++ >= topk) continue;
            sel[n2++] = dets[i];
        }
        free(dets);
        dets = sel;
        ndet = n2;
        free(ccls);
    }
    unsigned char *dead = calloc(ndet ? ndet : 1, 1);
    uint32_t nkeep = 0;
    for (uint32_t i = 0; i < ndet; ++i) {
        if (dead[i]) continue;
        if (nkeep < max_out) out[nkeep] = dets[i];
        ++nkeep;
        for (uint32_t j = i + 1; j < ndet; ++j) {
            if (dead[j] || dets[j].cls != dets[i].cls) continue;
            float ix1 = dets[i].x1 > dets[j].x1 ? dets[i].x1 : dets[j].x1;
            float iy1 = dets[i].y1 > dets[j].y1 ? dets[i].y1 : dets[j].y1;
            float ix2 = dets[i].x2 < dets[j].x2 ? dets[i].x2 : dets[j].x2;
            float iy2 = dets[i].y2 < dets[j].y2 ? dets[i].y2 : dets[j].y2;
            float iw = ix2 - ix1, ih = iy2 - iy1;
            if (iw <= 0.f || ih <= 0.f) continue;
            float inter = iw * ih;
            float uni = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1)
                      + (dets[j].x2 - dets[j].x1) * (dets[j].y2 - dets[j].y1) - inter;
            if (uni > 0.f && inter / uni > iou_thr) dead[j] = 1;
        }
    }
    free(dead);
    free(dets);
    return nkeep < max_out ? nkeep : max_out;
}

/* Model load, GPU bring-up, buffer placement and graph recording. Fills
 * the session globals; called by both main() and aexrt_m_open(). */
static int session_setup(void) {
    /* ---- model ---- */
    FILE *f = fopen(model_path, "rb");
    if (!f) { qprintf("cannot open model: %s\n", model_path); return 1; }
    fseek(f, 0, SEEK_END);
    msz = ftell(f);
    fseek(f, 0, SEEK_SET);
    md = malloc((size_t)msz);
    if (!md) { qprintf("malloc %ld failed\n", msz); return 1; }
    if (fread(md, 1, (size_t)msz, f) != (size_t)msz) { qprintf("short model read\n"); return 1; }
    fclose(f);
    if (msz < 116 || (memcmp(md, "AXM1", 4) && memcmp(md, "AXM2", 4) &&
                      memcmp(md, "AXM3", 4))) { qprintf("bad magic\n"); return 1; }
    is_i8 = memcmp(md, "AXM3", 4) == 0;
    memcpy(&nlayers, md + 4, 4);
    memcpy(&nvals, md + 8, 4);
    memcpy(&in_vid, md + 12, 2);
    memcpy(&out_vid, md + 14, 2);
    memcpy(&nc, md + 16, 2);
    memcpy(&nlevels, md + 18, 2);
    memcpy(&lt_sz, md + 20, 4);
    memcpy(&in_h, md + 24, 2);
    memcpy(&in_w, md + 26, 2);
    if (!in_h) { in_h = in_w = 320; }        /* older export: single size field */
    if (!in_w) in_w = in_h;
    memset(box_vid, 0, sizeof(box_vid));
    memset(cls_vid, 0, sizeof(cls_vid));
    if (nlevels > 3) nlevels = 3;
    for (uint32_t L = 0; L < nlevels; ++L) {
        memcpy(&box_vid[L], md + 28 + L * 8, 2);
        memcpy(&cls_vid[L], md + 28 + L * 8 + 2, 2);
    }
    vscale = calloc(nvals ? nvals : 1, 4);
    vdtype = calloc(nvals ? nvals : 1, 1);
    if (116 + nvals * 4 > (uint32_t)msz) { qprintf("bad scale table\n"); return 1; }
    memcpy(vscale, md + 116, (size_t)nvals * 4);
    lt_off = 116 + nvals * 4;
    if (is_i8) {
        uint32_t dt_off = 116 + nvals * 4;
        if (dt_off + nvals > (uint32_t)msz) { qprintf("bad dtype table\n"); return 1; }
        memcpy(vdtype, md + dt_off, nvals);
        lt_off = dt_off + ((nvals + 3) & ~3u);
    }
    wt_off = lt_off + lt_sz + 4;
    if (wt_off > (uint32_t)msz) { qprintf("bad model table\n"); return 1; }
    qprintf("model: %ld bytes, layers=%u vals=%u nc=%u levels=%u input=%ux%u\n",
           msz, nlayers, nvals, nc, nlevels, in_w, in_h); fflush(stdout);

    /* ---- per-layer policy (offline autotune map) ---- */
    layer_var = malloc((size_t)(nlayers ? nlayers : 1) * sizeof(int));
    for (uint32_t i = 0; i < nlayers; ++i) layer_var[i] = -1;
    if (policy_path) {
        FILE *pf = fopen(policy_path, "r");
        if (!pf) { qprintf("cannot open policy %s\n", policy_path); return 1; }
        char pline[256];
        int applied = 0, bad_sig = 0;
        while (fgets(pline, sizeof(pline), pf)) {
            if (pline[0] == '#') {
                unsigned pn_l = 0, pn_v = 0, pw = 0, ph = 0;
                if (sscanf(pline, "# aexrt-m policy v1 nlayers=%u nvals=%u input=%ux%u",
                           &pn_l, &pn_v, &pw, &ph) == 4 &&
                    (pn_l != nlayers || pn_v != nvals || pw != in_w || ph != in_h)) {
                    qprintf("policy/model mismatch (policy %u/%u %ux%u, model %u/%u %ux%u)\n",
                           pn_l, pn_v, pw, ph, nlayers, nvals, in_w, in_h);
                    bad_sig = 1;
                }
                continue;
            }
            unsigned idx = 0; int var = 0;
            if (sscanf(pline, "%u %d", &idx, &var) == 2 && idx < nlayers &&
                var >= 0 && var <= 6) {
                layer_var[idx] = var;
                applied++;
            }
        }
        fclose(pf);
        if (bad_sig) return 1;
        have_policy = 1;
        autotune = 0;   /* a persisted map replaces the startup sweep */
        qprintf("policy: %d layer overrides from %s\n", applied, policy_path);
        fflush(stdout);
    }
    if (stream_mode && nsets < 2) nsets = 2;
    if (nsets > 1) {
        if (emit_policy_path) {
            qprintf("--pipeline and --emit-policy are exclusive\n");
            return 1;
        }
        autotune = 0;   /* one fixed plan per set; throughput mode */
        qprintf("%s: %d frame sets\n", stream_mode ? "stream" : "pipeline", nsets);
        fflush(stdout);
    }

    /* ---- input (fp32 NCHW); the library API feeds frames per call ---- */
    if (input_path) {
    FILE *fi = fopen(input_path, "rb");
    if (!fi) { qprintf("cannot open input: %s\n", input_path); return 1; }
    fseek(fi, 0, SEEK_END);
    long isz = ftell(fi);
    fseek(fi, 0, SEEK_SET);
    if (isz != (long)(in_h * in_w * 3 * 4)) {
        qprintf("input size %ld != %ux%ux3x4\n", isz, in_w, in_h); return 1;
    }
    input_frame = malloc((size_t)isz);
    if (!input_frame || fread(input_frame, 1, (size_t)isz, fi) != (size_t)isz) {
        qprintf("short input read\n"); return 1;
    }
    fclose(fi);
    qprintf("input: %ld bytes\n", isz); fflush(stdout);
    }

    /* ---- GPU ---- */
    if (gpu_init()) return 1;
    qprintf("Vulkan ok\n"); fflush(stdout);
    if (is_i8 && !have_dot) { qprintf("model needs shaderIntegerDotProduct\n"); return 1; }
    if (mkpipe(0, conv_i8_spv, sizeof(conv_i8_spv), 7, 80)) return 1;
    if (mkpipe(1, elementwise_spv, sizeof(elementwise_spv), 6, 84)) return 1;
    if (mkpipe(3, decode_spv, sizeof(decode_spv), 4, 44)) return 1;
    if (mkpipe(4, conv_quad_spv, sizeof(conv_quad_spv), 7, 80)) {
        qprintf("quad pipeline unavailable; base/f16 remain\n");
        have_quad = 0;
    }
    if (mkpipe(5, conv_f16_spv, sizeof(conv_f16_spv), 7, 80)) return 1;
    if (mkpipe(6, conv_f16_oc4_spv, sizeof(conv_f16_oc4_spv), 7, 80)) return 1;
    if (mkpipe(7, conv_stem_spv, sizeof(conv_stem_spv), 7, 80)) return 1;
    if (mkpipe(8, conv_f16_oc1_spv, sizeof(conv_f16_oc1_spv), 7, 80)) return 1;
    if (mkpipe(9, conv_wino_spv, sizeof(conv_wino_spv), 7, 80)) return 1;
    if (mkpipe(10, conv_wino2_spv, sizeof(conv_wino2_spv), 7, 80)) return 1;
    if (mkpipe(15, conv_pair_spv, sizeof(conv_pair_spv), 7, 80)) return 1;
    if (have_dot) {
        if (mkpipe(11, conv_i8q_spv, sizeof(conv_i8q_spv), 8, 96)) return 1;
        if (mkpipe(12, conv_stem_i8q_spv, sizeof(conv_stem_i8q_spv), 8, 96)) return 1;
        if (mkpipe(13, elementwise_i8_spv, sizeof(elementwise_i8_spv), 3, 84)) return 1;
        if (mkpipe(14, decode_i8_spv, sizeof(decode_i8_spv), 4, 52)) return 1;
    }
    qprintf("pipelines ok\n"); fflush(stdout);

    /* ---- layer table + per-value shape ---- */
    layers = calloc(nlayers ? nlayers : 1, sizeof(Layer));
    vsize = calloc(nvals ? nvals : 1, 4);
    vh = calloc(nvals ? nvals : 1, 4);
    vw = calloc(nvals ? nvals : 1, 4);
    vc = calloc(nvals ? nvals : 1, 4);
    uint32_t *concat_n = calloc(nlayers ? nlayers : 1, 4);
    uint32_t **concat_ins = calloc(nlayers ? nlayers : 1, sizeof(uint32_t *));
    uint32_t **concat_ch = calloc(nlayers ? nlayers : 1, sizeof(uint32_t *));
    if (!layers || !vsize || !vh || !vw || !vc) { qprintf("oom\n"); return 1; }

    in_vid &= 0xffff; out_vid &= 0xffff;
    vsize[in_vid] = in_h * in_w * 3;
    vh[in_vid] = in_h; vw[in_vid] = in_w; vc[in_vid] = 3;

    uint32_t off = lt_off;
    for (uint32_t i = 0; i < nlayers; ++i) {
        if (off + 2 > (uint32_t)msz) { qprintf("layer table overflow\n"); return 1; }
        uint8_t *p = md + off;
        Layer *L = &layers[i];
        uint32_t t = p[0];
        L->type = t;
        if (getenv("AEXRT_TRACE")) qprintf("L%u t%u off%u\n", i, t, off);
        if (t == 1 || t == 9 || t == 10 || t == 11) {
            if (off + 26 > (uint32_t)msz) { qprintf("conv record overflow\n"); return 1; }
            L->act = p[1]; L->stride = p[2]; L->pad = p[3];
            memcpy(&L->oc, p + 4, 2); memcpy(&L->ic, p + 6, 2);
            L->kh = p[8]; L->kw = p[9];
            memcpy(&L->in0, p + 10, 2); memcpy(&L->out, p + 12, 2);
            memcpy(&L->w_off, p + 14, 4); memcpy(&L->b_off, p + 18, 4);
            memcpy(&L->s_off, p + 22, 4);
            off += 26;
        } else if (t == 2) {
            uint8_t n = p[1];
            concat_n[i] = n;
            concat_ins[i] = calloc(n, 4);   /* memcpy fills only 2 bytes */
            concat_ch[i] = calloc(n, 4);
            uint32_t total = 0;
            for (uint8_t j = 0; j < n; ++j) {
                memcpy(&concat_ins[i][j], p + 4 + j * 4, 2);
                memcpy(&concat_ch[i][j], p + 4 + j * 4 + 2, 2);
                uint32_t v = concat_ins[i][j];
                if (v >= nvals) {
                    qprintf("concat layer %u off %u vid %u out of range (n=%u)\n",
                           i, off, v, (unsigned)n);
                    return 1;
                }
                total += vsize[v];
            }
            memcpy(&L->out, p + 4 + n * 4, 2);
            if (L->out >= nvals) { qprintf("concat out out of range\n"); return 1; }
            vsize[L->out] = total;
            uint32_t v0 = concat_ins[i][0];
            vh[L->out] = vh[v0]; vw[L->out] = vw[v0];
            uint32_t ctot = 0;
            for (uint8_t j = 0; j < n; ++j) {
                uint32_t v = concat_ins[i][j];
                if (vh[v] != vh[L->out] || vw[v] != vw[L->out]) {
                    qprintf("concat layer %u: v%u is %ux%u, first is %ux%u\n",
                           i, v, vh[v], vw[v], vh[L->out], vw[L->out]);
                    return 1;
                }
                ctot += concat_ch[i][j];
            }
            vc[L->out] = ctot;
            off += 4 + n * 4 + 4;
            continue;
        } else {
            if (off + 22 > (uint32_t)msz) { qprintf("layer record overflow\n"); return 1; }
            memcpy(&L->oc, p + 4, 2); memcpy(&L->ic, p + 6, 2);
            L->kh = p[8]; L->kw = p[9];
            memcpy(&L->in0, p + 10, 2); memcpy(&L->out, p + 12, 2);
            memcpy(&L->extra, p + 14, 2);
            L->stride = p[2]; L->pad = p[3]; L->act = p[1];
            off += 22;
        }
        uint32_t vi = L->in0, vo = L->out;
        if (vi >= nvals || vo >= nvals) { qprintf("layer vid out of range\n"); return 1; }
        uint32_t ih = vh[vi], iw = vw[vi];
        if (t == 1 || t == 10 || t == 11) {  /* conv / winograd / int8 */
            if (L->ic != vc[vi]) {
                qprintf("conv ic mismatch layer %u: %u vs value %u\n", i, L->ic, vc[vi]);
                return 1;
            }
            uint32_t sh = L->stride ? L->stride : 1;
            uint32_t oh = (ih + 2 * L->pad - L->kh) / sh + 1;
            uint32_t ow = (iw + 2 * L->pad - L->kw) / sh + 1;
            vh[vo] = oh; vw[vo] = ow; vc[vo] = L->oc;
            vsize[vo] = L->oc * oh * ow;
        } else if (t == 7) {                 /* slice = zero-copy view */
            if (L->ic == 0) { qprintf("slice ic=0\n"); return 1; }
            vh[vo] = ih; vw[vo] = iw; vc[vo] = L->oc;
            vsize[vo] = L->oc * ih * iw;
        } else if (t == 3) {                 /* maxpool with explicit window */
            uint32_t sh = L->stride ? L->stride : 1;
            uint32_t ph = L->pad;
            uint32_t oh = (ih + 2 * ph - L->kh) / sh + 1;
            uint32_t ow = (iw + 2 * ph - L->kw) / sh + 1;
            vh[vo] = oh; vw[vo] = ow; vc[vo] = L->oc;
            vsize[vo] = L->oc * oh * ow;
        } else if (t == 4) {                 /* nearest integer upsample */
            uint32_t sc = L->stride ? L->stride : 2;
            vh[vo] = ih * sc; vw[vo] = iw * sc; vc[vo] = L->oc;
            vsize[vo] = L->oc * vh[vo] * vw[vo];
        } else if (t == 5 || t == 6 || t == 8) {
            if (L->oc != vc[vi]) {
                qprintf("unary channel mismatch layer %u: %u vs %u\n", i, L->oc, vc[vi]);
                return 1;
            }
            vh[vo] = ih; vw[vo] = iw; vc[vo] = L->oc;
            vsize[vo] = L->oc * ih * iw;
        } else {
            qprintf("unknown layer type %u\n", t); return 1;
        }
    }

    /* ---- placement: (root buffer, channel offset) ---- */
    uint32_t *rtf = malloc(nvals * 4);
    uint32_t *rto = calloc(nvals, 4);
    pb = malloc(nvals * 4);
    po = calloc(nvals, 4);
    uint32_t *plb = malloc(nvals * 4);
    int32_t *plo = malloc(nvals * 4);
    for (uint32_t v = 0; v < nvals; ++v) { rtf[v] = v; plb[v] = 0xFFFFFFFFu; plo[v] = -1; }
    for (uint32_t i = 0; i < nlayers; ++i)
        if (layers[i].type == 7) {
            rtf[layers[i].out] = rtf[layers[i].in0];
            rto[layers[i].out] = rto[layers[i].in0] + layers[i].extra;
        }
    int conflicts = 0;
    for (uint32_t i = 0; i < nlayers; ++i) {
        Layer *L = &layers[i];
        if (L->type != 2) continue;
        uint32_t dst = 0;
        for (uint32_t j = 0; j < concat_n[i]; ++j) {
            uint32_t v = concat_ins[i][j];
            if (v >= nvals || vsize[v] == 0) continue;
            uint32_t r = rtf[v];
            int32_t want = (int32_t)dst - (int32_t)rto[v];
            if (plb[r] == 0xFFFFFFFFu) { plb[r] = L->out; plo[r] = want; }
            else if (plb[r] != L->out || plo[r] != want) {
                qprintf("place conflict v%u root %u (dst %u off %d vs %u/%d)\n",
                       v, r, dst, want, plb[r], plo[r]);
                conflicts++;
            }
            dst += concat_ch[i][j];
        }
    }
    for (uint32_t v = 0; v < nvals; ++v) {
        uint32_t r = rtf[v];
        if (plb[r] != 0xFFFFFFFFu) {
            pb[v] = plb[r];
            po[v] = (uint32_t)((int32_t)plo[r] + (int32_t)rto[v]);
        } else { pb[v] = r; po[v] = rto[v]; }
    }
    if (conflicts) qprintf("WARNING: %d placement conflicts (results may be wrong)\n", conflicts);

    uint32_t effective_out = out_vid;
    if (out_vid >= nvals || vsize[out_vid] == 0) {
        for (int j = (int)nlayers - 1; j >= 0; --j)
            if (layers[j].out < nvals && vsize[layers[j].out] > 0) {
                effective_out = layers[j].out; break;
            }
        if (out_vid < nvals)
            qprintf("out has no size, fallback v%u (%u elems)\n",
                   effective_out, vsize[effective_out]);
    }
    out_vid = effective_out;

    /* ---- buffers: one allocation per root value, replicated per frame set ----
     * Pipeline mode keeps N independent value arenas so frame i+1 can run on
     * the GPU while frame i is still being read back by the host. Weights are
     * read-only and stay shared. */
    vb_sets = calloc((size_t)nsets * nvals, sizeof(Buf));
    if (!vb_sets) { qprintf("oom\n"); return 1; }
    for (int s = 0; s < nsets; ++s) {
        Buf *vs = &vb_sets[(size_t)s * nvals];
        for (uint32_t i = 0; i < nvals; ++i) {
            if (vsize[i] == 0 || pb[i] != i) continue;
            VkDeviceSize bytes = (VkDeviceSize)vsize[i] * (vdtype[i] ? 1u : 2u);
            if (mkbuf(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vs[i])) {
                qprintf("buf alloc fail set %d v[%u] (%u elems)\n", s, i, vsize[i]);
                return 1;
            }
        }
    }
    vb = vb_sets;   /* set 0: the dump / NMS paths read this one */
    qprintf("buffers ok (%u values x %d set%s)\n", nvals, nsets,
           nsets > 1 ? "s" : "");
    fflush(stdout);

    if (is_i8 && vdtype[in_vid] != 0) { qprintf("int8 model input must be fp16\n"); return 1; }
    /* ---- input: fp32 NCHW on disk -> fp16 position-major (every set) ---- */
    if (input_frame) {
        for (int s = 0; s < nsets; ++s) {
            uint16_t *dst = (uint16_t *)vb_sets[(size_t)s * nvals + in_vid].map;
            load_frame_fp16(input_frame, dst, in_h, in_w);
        }
    }
    /* ---- optional multi-frame source for --stream ---- */
    frames = NULL;
    nframes = 0;
    frame_elems = in_h * in_w * 3u;
    if (stream_mode && frames_path) {
        FILE *ff = fopen(frames_path, "rb");
        if (!ff) { qprintf("cannot open frames: %s\n", frames_path); return 1; }
        fseek(ff, 0, SEEK_END);
        long fsz = ftell(ff);
        fseek(ff, 0, SEEK_SET);
        if (fsz <= 0 || (unsigned long)fsz % ((unsigned long)frame_elems * 4u) != 0) {
            qprintf("frames size %ld is not a multiple of one frame (%u floats)\n",
                   fsz, frame_elems);
            return 1;
        }
        nframes = (uint32_t)((unsigned long)fsz / ((unsigned long)frame_elems * 4u));
        frames = malloc((size_t)fsz);
        if (!frames || fread(frames, 1, (size_t)fsz, ff) != (size_t)fsz) {
            qprintf("short frames read\n"); return 1;
        }
        fclose(ff);
        qprintf("frames: %u x %u floats from %s\n", nframes, frame_elems, frames_path);
        if (nframes < bench) {
            qprintf("frames file has %u frames, need %u\n", nframes, bench);
            return 1;
        }
    }

    /* ---- one weight arena (interleaved weights+biases in the blob) ---- */
    uint64_t blob_end = 0;
    for (uint32_t i = 0; i < nlayers; ++i) {
        if (layers[i].type != 1 && layers[i].type != 10 && layers[i].type != 11) continue;
        uint32_t Cpad = layers[i].ic < 4 ? 4u : layers[i].ic;
        uint32_t R = layers[i].type == 10 ? 16u * layers[i].ic
                                         : Cpad * layers[i].kh * layers[i].kw;
        uint32_t Rp = (R + 3) & ~3u;
        uint64_t w_end = (uint64_t)layers[i].w_off +
                         (uint64_t)layers[i].oc * Rp * (layers[i].type == 11 ? 1u : 2u);
        uint64_t b_end = (uint64_t)layers[i].b_off * 4 + (uint64_t)layers[i].oc * 4;
        uint64_t s_end = (uint64_t)layers[i].s_off * 4 + (uint64_t)layers[i].oc * 4;
        if (w_end > blob_end) blob_end = w_end;
        if (b_end > blob_end) blob_end = b_end;
        if (s_end > blob_end) blob_end = s_end;
    }
    memset(&wbuf, 0, sizeof(wbuf));
    if (blob_end > 0) {
        if ((uint64_t)wt_off + blob_end > (uint64_t)msz) {
            qprintf("weight blob out of range\n"); return 1;
        }
        if (mkbuf((VkDeviceSize)(blob_end + 4096 + 64), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &wbuf))
            return 1;
        memcpy(wbuf.map, md + wt_off, (size_t)blob_end);
        qprintf("weight arena: %llu bytes\n", (unsigned long long)blob_end);
    }

    /* ---- decode buffers: one candidate set per frame set ---- */
    memset(candb_s, 0, sizeof(candb_s));
    memset(cntb_s, 0, sizeof(cntb_s));
    memset(cands_s, 0, sizeof(cands_s));
    memset(cnts_s, 0, sizeof(cnts_s));
    if (nlevels) {
        /* GPU writes candidates in device-local memory; one transfer copies
         * them to host-cached staging, avoiding slow CPU reads of VRAM. */
        for (int s = 0; s < nsets; ++s) {
            if (mkbuf(4096u * 6u * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &candb_s[s])) return 1;
            if (mkbuf(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &cntb_s[s])) return 1;
            if (mkbuf_ex(4096u * 6u * 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &cands_s[s], 1)) return 1;
            if (mkbuf_ex(16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &cnts_s[s], 1)) return 1;
        }
    }
    /* set-0 aliases used by the host decode / NMS tail */
    cntb = cntb_s[0];
    cands = cands_s[0]; cnts = cnts_s[0];

    /* Streaming input staging: host writes land in cached memory, the graph
     * starts with a GPU copy into the device-local fp16 input buffer. Writing
     * the frame straight into the mapped device buffer costs ~0.4 ms/frame on
     * Mali (uncached) and much more on a discrete GPU over PCIe. */
    memset(in_stage, 0, sizeof(in_stage));
    if (stream_mode || g_api_mode) {
        for (int s = 0; s < nsets; ++s) {
            VkDeviceSize ib = (VkDeviceSize)vsize[in_vid] * 2u;
            if (mkbuf_ex(ib, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &in_stage[s], 1)) return 1;
        }
    }

    /* ---- record the whole graph once ---- */
    if (emit_policy_path && ts_period_ns > 0.0) want_profile = 1;
    profiling = want_profile && ts_period_ns > 0.0;
    if (profiling) {
        VkQueryPoolCreateInfo qci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = 2u * (nlayers + 8u);
        CHK(vkCreateQueryPool(dev, &qci, NULL, &qpool));
        prof_label = malloc((size_t)(nlayers + 8) * 4);
        prof_n = 0;
    }
    wr_ep = calloc(nvals, 4);
    rd_ep = calloc(nvals, 4);
    const int emit = emit_policy_path != NULL;
    if (emit && is_i8) { qprintf("--emit-policy needs an fp16 model\n"); return 1; }
    int vars[7];
    int nvar = 0;
    if (emit) {
        /* every candidate variant, one command buffer each; the recorder
         * keeps per-dispatch timestamps so the map can be built per layer */
        vars[nvar++] = 0;
        vars[nvar++] = 3;
        vars[nvar++] = 4;
        vars[nvar++] = 1;
        if (have_quad) vars[nvar++] = 2;
        vars[nvar++] = 6;
    } else if (autotune && !profiling && !is_i8) {
        vars[nvar++] = 0;                  /* auto: per-layer f16/oc4 mix */
        vars[nvar++] = 3;                  /* fp16 dot, 8 OC */
        vars[nvar++] = 4;                  /* fp16 dot, 4 OC (short spatial) */
        vars[nvar++] = 1;                  /* fp32 accumulate */
        if (have_quad) vars[nvar++] = 2;   /* 2x2 quartet */
        vars[nvar++] = 6;                  /* 1x2 pair, weight reuse */
    } else {
        vars[nvar++] = conv_variant;
    }
    /* Emit mode adds one more pass after the measurement passes: the same
     * graph re-recorded with the per-layer map applied. Pipeline mode instead
     * records the same graph once per frame set. */
    const int npass = nsets > 1 ? nsets : (nvar + (emit ? 1 : 0));
    double *bestms = emit ? malloc((size_t)nlayers * 7 * sizeof(double)) : NULL;
    if (bestms)
        for (size_t k = 0; k < (size_t)nlayers * 7; ++k) bestms[k] = 1e30;
    uint32_t prof_n_all[7] = {0, 0, 0, 0, 0, 0, 0};
    uint32_t *prof_label_all[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    double gvar_ms[7] = {1e30, 1e30, 1e30, 1e30, 1e30, 1e30, 1e30};
    double var_ms[6] = {0, 0, 0, 0, 0, 0};
    const int pipe_mode = nsets > 1;
    const int base_variant = conv_variant;
    for (int v = 0; v < npass; ++v) {
    const int policy_pass = emit && v == nvar;
    const int set = pipe_mode ? v : 0;
    Buf *cbv = &candb_s[set], *kbv = &cntb_s[set];
    Buf *csv = &cands_s[set], *ksv = &cnts_s[set];
    /* attribution passes run serialized so a timestamp is one layer's cost */
    force_bar = (emit && !policy_pass) ? 1 : 0;
    if (policy_pass) {
        /* build the per-layer map from the measured passes */
        FILE *pf = fopen(emit_policy_path, "w");
        if (!pf) { qprintf("cannot write policy %s\n", emit_policy_path); return 1; }
        fprintf(pf, "# aexrt-m policy v1 nlayers=%u nvals=%u input=%ux%u\n",
                nlayers, nvals, in_w, in_h);
        int counts[7] = {0, 0, 0, 0, 0, 0, 0};
        double pred = 0.0;
        int gbest = 0;
        for (int k = 1; k < 7; ++k)
            if (gvar_ms[k] < gvar_ms[gbest]) gbest = k;
        qprintf("emit: whole-graph winner %d (%.3f ms)\n", gbest, gvar_ms[gbest]);
        for (uint32_t i = 0; i < nlayers; ++i) {
            Layer *L = &layers[i];
            if (L->type != 1 && L->type != 9 && L->type != 10 && L->type != 11) continue;
            /* Default to the trusted whole-graph winner.  Per-dispatch
             * timestamps sit near the timer resolution on fast layers and on
             * Mali they exclude inter-dispatch gaps, so a per-layer override
             * needs a large, unambiguous margin to be believed. */
            double gv = bestms[(size_t)i * 7 + gbest];
            int bv = gbest; double bms = gv;
            for (int k = 0; k < 7; ++k) {
                double ms = bestms[(size_t)i * 7 + k];
                if (ms < bms * 0.75 && gv > 0.005) { bms = ms; bv = k; }
            }
            layer_var[i] = bv;
            counts[bv]++;
            if (bms < 1e29) pred += bms;
        }
        if (getenv("AEXRT_POLICY_TRACE")) {
            qprintf("policy trace: layer  v0 v1 v2 v3 v4 v5 v6 (ms)\n");
            for (uint32_t i = 0; i < nlayers; ++i) {
                Layer *L = &layers[i];
                if (L->type != 1) continue;
                qprintf("L%-3u %u->%u %ux%u:", i, L->ic, L->oc, vw[L->out], vh[L->out]);
                for (int k = 0; k < 7; ++k) {
                    double m = bestms[(size_t)i * 7 + k];
                    if (m < 1e29) qprintf(" %.4f", m); else qprintf(" -");
                }
                qprintf("\n");
            }
            fflush(stdout);
        }
        for (uint32_t i = 0; i < nlayers; ++i) {
            Layer *L = &layers[i];
            if (L->type != 1 && L->type != 9 && L->type != 10 && L->type != 11) continue;
            double ms = bestms[(size_t)i * 7 + layer_var[i]];
            fprintf(pf, "%u %d  # t%u %u->%u k%ux%u s%u %ux%u ch%.4f ms%s\n",
                    i, layer_var[i], L->type, L->ic, L->oc, L->kh, L->kw, L->stride,
                    vw[L->out], vh[L->out],
                    ms < 1e29 ? ms : 0.0, ms < 1e29 ? "" : " (unmeasured)");
        }
        fclose(pf);
        have_policy = 1;
        qprintf("policy: %s (predicted %.3f ms; 0:%d 1:%d 2:%d 3:%d 4:%d 5:%d 6:%d)\n",
               emit_policy_path, pred, counts[0], counts[1], counts[2],
               counts[3], counts[4], counts[5], counts[6]);
        /* per-variant accounted totals: they should track each variant's
         * measured whole-graph time if attribution is coherent */
        for (int k = 0; k < 7; ++k) {
            double s = 0.0; int n = 0;
            for (uint32_t i = 0; i < nlayers; ++i) {
                Layer *L = &layers[i];
                if (L->type != 1 && L->type != 9 && L->type != 10 && L->type != 11) continue;
                if (bestms[(size_t)i * 7 + k] < 1e29) { s += bestms[(size_t)i * 7 + k]; n++; }
            }
            if (n) qprintf("  variant %d: %d layers, accounted %.3f ms\n", k, n, s);
        }
        fflush(stdout);
        vkResetCommandBuffer(cmds[0], 0);
    }
    conv_variant = pipe_mode ? base_variant : (policy_pass ? 0 : vars[v]);
    cmd = cmds[pipe_mode ? set : (policy_pass ? 0 : v)];
    if (pipe_mode) vb = &vb_sets[(size_t)set * nvals];
    g_cntb = nlevels ? kbv : NULL;
    seq_no = 0;
    memset(wr_ep, 0, (size_t)nvals * 4);
    memset(rd_ep, 0, (size_t)nvals * 4);
    prof_n = 0;
    uint32_t t_bar = 0;
    uint32_t quad_layers = 0, base_layers = 0, f16_layers = 0, oc4_layers = 0, oc1_layers = 0, stem_layers = 0, wino_layers = 0, pair_layers = 0;
    double t_rec0 = now_ms();
    VkCommandBufferBeginInfo beg = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHK(vkBeginCommandBuffer(cmd, &beg));
    vkCmdResetQueryPool(cmd, gpuqp, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, gpuqp, 0);
    if (profiling)
        vkCmdResetQueryPool(cmd, qpool, 0, 2u * (nlayers + 8u));
    if (in_stage[set].buf) {
        /* staging (host-writable, cached) -> device-local fp16 input */
        VkMemoryBarrier mh = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
            VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mh, 0, NULL, 0, NULL);
        VkBufferCopy ibc = {0, 0, (VkDeviceSize)vsize[in_vid] * 2u};
        vkCmdCopyBuffer(cmd, in_stage[set].buf, vb[in_vid].buf, 1, &ibc);
        VkMemoryBarrier mc2 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mc2, 0, NULL, 0, NULL);
    }
    /* host writes (input + weights) happen-before the first dispatch */
    {
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
            VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, NULL, 0, NULL);
    }
    const int rec_trace = getenv("AEXRT_REC_TRACE") != NULL;
    for (uint32_t i = 0; i < nlayers; ++i) {
        Layer *L = &layers[i];
        if (rec_trace) { qprintf("rec v%d L%u t%u\n", conv_variant, i, L->type); fflush(stdout); }
        if (L->out >= nvals || vsize[L->out] == 0) continue;
        if (L->in0 >= nvals || vsize[L->in0] == 0) continue;
        uint32_t in_b = pb[L->in0], out_b = pb[L->out];
        if (in_b >= nvals || out_b >= nvals) continue;
        uint32_t ih = vh[L->in0], iw = vw[L->in0];
        uint32_t oh = vh[L->out], ow = vw[L->out];
        if (L->type == 1 || L->type == 10 || L->type == 11) {
            uint32_t push[20] = {1, L->ic, ih, iw, L->oc, oh, ow,
                                 L->kh, L->kw, L->stride, L->stride, L->pad, L->pad,
                                 L->act, L->w_off / 8u, L->b_off, po[L->in0], po[L->out],
                                 vc[out_b], vc[in_b]};
            Buf *bufs[7] = {&vb[in_b], &wbuf, &wbuf, &vb[out_b], &vb[in_b], &wbuf, &vb[out_b]};
            int eligible = (L->ic % 4 == 0) && (vc[in_b] % 4 == 0) &&
                           (po[L->in0] % 4 == 0) && (vc[out_b] % 4 == 0) &&
                           (po[L->out] % 4 == 0) && (L->ic >= 8);
            int variant = (conv_variant == 1) ? 0 :
                          (conv_variant == 2) ? ((eligible && have_quad) ? 2 : 0) :
                          (conv_variant == 3) ? (eligible ? 3 : 0) :
                          (conv_variant == 4) ? (eligible ? 4 : 0) :
                          (conv_variant == 5) ? (eligible ? 5 : 0) :
                          (conv_variant == 6) ? (eligible ? 6 : 0) :
                          (eligible ? (oh * ow <= 400 ? 4 : 3) : 0); /* auto */
            if (layer_var && layer_var[i] > 0) {
                /* 0 in the map means "keep the auto rule", not the base kernel */
                variant = layer_var[i];
                if (!eligible || (variant == 2 && !have_quad)) variant = 0;
            }
            int hb = haz(in_b, 0xFFFFFFFFu, out_b, t_bar);
            if (hb) t_bar = seq_no;   /* barrier covers everything up to here */
            uint32_t total = ((L->oc + 7) / 8) * oh * ow;
            if (L->type == 11) {
                int stem = (L->ic < 4);
                if (!stem && (L->ic % 4 != 0 || vc[in_b] % 4 != 0 ||
                              po[L->in0] % 4 != 0 || vc[out_b] % 4 != 0 ||
                              po[L->out] % 4 != 0)) {
                    qprintf("int8 placement not 4-aligned layer %u\n", i);
                    return 1;
                }
                uint32_t p8[24] = {1, L->ic, ih, iw, L->oc, oh, ow, L->kh, L->kw,
                                   L->stride, L->stride, L->pad, L->pad, L->act,
                                   L->w_off / 4u, L->b_off, po[L->in0], po[L->out],
                                   vc[out_b], vc[in_b], fbits(vscale[L->in0]),
                                   fbits(vscale[L->out]), 1u, L->s_off};
                Buf *bufs8[8] = {&vb[in_b], &wbuf, &wbuf, &vb[out_b], &vb[in_b],
                                 &wbuf, &vb[out_b], &wbuf};
                uint32_t total8 = ((L->oc + 7) / 8) * oh * ow;
                if (dispatch(stem ? 12 : 11, bufs8, 8, p8, sizeof(p8),
                             (total8 + 127) / 128, hb, i)) return 1;
                mark(in_b, 0xFFFFFFFFu, out_b);
                continue;
            }
            if (L->type == 10) {
                uint32_t tiles = ((oh + 1) / 2) * ((ow + 1) / 2);
                if (getenv("AEXRT_WINO2")) {
                    uint32_t totalw2 = ((L->oc + 1) / 2) * tiles;
                    if (dispatch(10, bufs, 7, push, sizeof(push), (totalw2 + 127) / 128, hb, i)) return 1;
                } else {
                    uint32_t totalw = ((L->oc + 3) / 4) * tiles;
                    if (dispatch(9, bufs, 7, push, sizeof(push), (totalw + 127) / 128, hb, i)) return 1;
                }
                wino_layers++;
                mark(in_b, 0xFFFFFFFFu, out_b);
                continue;
            }
            if (L->ic == 3 && !getenv("AEXRT_NO_STEM")) {
                /* 3-channel RGB stem: dedicated f16vec4 weight path. */
                if (dispatch(7, bufs, 7, push, sizeof(push), (total + 127) / 128, hb, i)) return 1;
                stem_layers++;
                mark(in_b, 0xFFFFFFFFu, out_b);
                continue;
            }
            if (variant == 2) {
                uint32_t tiles = ((oh + 1) / 2) * ((ow + 1) / 2);
                total = ((L->oc + 7) / 8) * tiles;
                if (dispatch(4, bufs, 7, push, sizeof(push), (total + 63) / 64, hb, i)) return 1;
                quad_layers++;
            } else if (variant == 5) {
                uint32_t total1 = L->oc * oh * ow;
                if (dispatch(8, bufs, 7, push, sizeof(push), (total1 + 127) / 128, hb, i)) return 1;
                oc1_layers++;
            } else if (variant == 6) {
                uint32_t totalp = ((L->oc + 7) / 8) * oh * ((ow + 1) / 2);
                if (dispatch(15, bufs, 7, push, sizeof(push), (totalp + 127) / 128, hb, i)) return 1;
                pair_layers++;
            } else if (variant == 4) {
                uint32_t total4 = ((L->oc + 3) / 4) * oh * ow;
                if (dispatch(6, bufs, 7, push, sizeof(push), (total4 + 127) / 128, hb, i)) return 1;
                oc4_layers++;
            } else if (variant == 3) {
                if (dispatch(5, bufs, 7, push, sizeof(push), (total + 127) / 128, hb, i)) return 1;
                f16_layers++;
            } else {
                if (dispatch(0, bufs, 7, push, sizeof(push), (total + 127) / 128, hb, i)) return 1;
                base_layers++;
            }
            mark(in_b, 0xFFFFFFFFu, out_b);
        } else if (L->type == 3) {
            uint32_t sh = L->stride ? L->stride : 1;
            uint32_t C = L->oc;
            int vec = g_vec_elem && (C % 4 == 0) && (vc[in_b] % 4 == 0) && (po[L->in0] % 4 == 0) &&
                      (vc[out_b] % 4 == 0) && (po[L->out] % 4 == 0);
            uint32_t elems = vec ? (C / 4) * oh * ow : C * oh * ow;
            uint32_t push[21] = {vec ? 6u : 1u, elems, ih, iw, oh, ow, C, C,
                                 po[L->in0], 0, po[L->out], vc[in_b], 0, vc[out_b],
                                 L->kh, L->kw, sh, sh, L->pad, L->pad, 0};
            Buf *bufs[6] = {&vb[in_b], &vb[in_b], &vb[out_b],
                            &vb[in_b], &vb[in_b], &vb[out_b]};
            int hb = haz(in_b, 0xFFFFFFFFu, out_b, t_bar);
            if (hb) t_bar = seq_no;   /* barrier covers everything up to here */
            if (is_i8) {
                if ((C & 3u) || (vc[in_b] & 3u) || (po[L->in0] & 3u) ||
                    (vc[out_b] & 3u) || (po[L->out] & 3u)) {
                    qprintf("int8 maxpool misaligned c=%u\n", C);
                    return 1;
                }
                uint32_t elems8 = (C / 4) * oh * ow;
                uint32_t p8[21] = {1, elems8, ih, iw, oh, ow, C, C,
                                   po[L->in0], 0, po[L->out], vc[in_b], 0, vc[out_b],
                                   L->kh, L->kw, sh, sh, L->pad, L->pad, 0};
                Buf *b8[3] = {&vb[in_b], &vb[in_b], &vb[out_b]};
                if (dispatch(13, b8, 3, p8, sizeof(p8), (elems8 + 127) / 128, hb, i)) return 1;
                mark(in_b, 0xFFFFFFFFu, out_b);
                continue;
            }
            if (dispatch(1, bufs, vec ? 6 : 3, push, sizeof(push), (elems + 127) / 128, hb, i)) return 1;
            mark(in_b, 0xFFFFFFFFu, out_b);
        } else if (L->type == 4) {
            uint32_t sc = L->stride ? L->stride : 2;
            uint32_t C = L->oc;
            int vec = g_vec_elem && (C % 4 == 0) && (vc[in_b] % 4 == 0) && (po[L->in0] % 4 == 0) &&
                      (vc[out_b] % 4 == 0) && (po[L->out] % 4 == 0);
            uint32_t elems = vec ? (C / 4) * oh * ow : C * oh * ow;
            uint32_t push[21] = {vec ? 7u : 2u, elems, ih, iw, oh, ow, C, C,
                                 po[L->in0], 0, po[L->out], vc[in_b], 0, vc[out_b],
                                 0, 0, 0, 0, 0, 0, sc};
            Buf *bufs[6] = {&vb[in_b], &vb[in_b], &vb[out_b],
                            &vb[in_b], &vb[in_b], &vb[out_b]};
            int hb = haz(in_b, 0xFFFFFFFFu, out_b, t_bar);
            if (hb) t_bar = seq_no;   /* barrier covers everything up to here */
            if (is_i8) {
                if ((C & 3u) || (vc[in_b] & 3u) || (po[L->in0] & 3u) ||
                    (vc[out_b] & 3u) || (po[L->out] & 3u)) {
                    qprintf("int8 resize misaligned c=%u\n", C);
                    return 1;
                }
                uint32_t elems8 = (C / 4) * oh * ow;
                uint32_t p8[21] = {2, elems8, ih, iw, oh, ow, C, C,
                                   po[L->in0], 0, po[L->out], vc[in_b], 0, vc[out_b],
                                   0, 0, 0, 0, 0, 0, sc};
                Buf *b8[3] = {&vb[in_b], &vb[in_b], &vb[out_b]};
                if (dispatch(13, b8, 3, p8, sizeof(p8), (elems8 + 127) / 128, hb, i)) return 1;
                mark(in_b, 0xFFFFFFFFu, out_b);
                continue;
            }
            if (dispatch(1, bufs, vec ? 6 : 3, push, sizeof(push), (elems + 127) / 128, hb, i)) return 1;
            mark(in_b, 0xFFFFFFFFu, out_b);
        } else if (L->type == 5) {
            uint32_t in1_b = pb[L->extra];
            uint32_t C = L->oc;
            int vec = g_vec_elem && (C % 4 == 0) && (vc[in_b] % 4 == 0) && (po[L->in0] % 4 == 0) &&
                      (vc[in1_b] % 4 == 0) && (po[L->extra] % 4 == 0) &&
                      (vc[out_b] % 4 == 0) && (po[L->out] % 4 == 0);
            uint32_t elems = vec ? (C / 4) * oh * ow : C * oh * ow;
            uint32_t push[21] = {vec ? 8u : 3u, elems, ih, iw, oh, ow, C, C,
                                 po[L->in0], po[L->extra], po[L->out],
                                 vc[in_b], vc[in1_b], vc[out_b], 0, 0, 0, 0, 0, 0, 0};
            Buf *bufs[6] = {&vb[in_b], &vb[in1_b], &vb[out_b],
                            &vb[in_b], &vb[in1_b], &vb[out_b]};
            int hb = haz(in_b, in1_b, out_b, t_bar);
            if (hb) t_bar = seq_no;   /* barrier covers everything up to here */
            if (is_i8) {
                if ((C & 3u) || (vc[in_b] & 3u) || (po[L->in0] & 3u) ||
                    (vc[in1_b] & 3u) || (po[L->extra] & 3u) ||
                    (vc[out_b] & 3u) || (po[L->out] & 3u)) {
                    qprintf("int8 add misaligned c=%u\n", C);
                    return 1;
                }
                uint32_t elems8 = (C / 4) * oh * ow;
                uint32_t p8[21] = {3, elems8, ih, iw, oh, ow, C, C,
                                   po[L->in0], po[L->extra], po[L->out],
                                   vc[in_b], vc[in1_b], vc[out_b], 0, 0, 0, 0, 0, 0, 0};
                Buf *b8[3] = {&vb[in_b], &vb[in1_b], &vb[out_b]};
                if (dispatch(13, b8, 3, p8, sizeof(p8), (elems8 + 127) / 128, hb, i)) return 1;
                mark(in_b, in1_b, out_b);
                continue;
            }
            if (dispatch(1, bufs, vec ? 6 : 3, push, sizeof(push), (elems + 127) / 128, hb, i)) return 1;
            mark(in_b, in1_b, out_b);
        } else if (L->type == 6 || L->type == 8) {
            if (is_i8) { qprintf("int8 unary activation not implemented\n"); return 1; }
            uint32_t C = L->oc;
            int vec = g_vec_elem && (C % 4 == 0) && (vc[in_b] % 4 == 0) && (po[L->in0] % 4 == 0) &&
                      (vc[out_b] % 4 == 0) && (po[L->out] % 4 == 0);
            uint32_t elems = vec ? (C / 4) * oh * ow : C * oh * ow;
            uint32_t unary = (L->type == 6 ? 4u : 5u) + (vec ? 5u : 0u);
            uint32_t push[21] = {unary, elems, ih, iw, oh, ow, C, C,
                                 po[L->in0], 0, po[L->out],
                                 vc[in_b], 0, vc[out_b], 0, 0, 0, 0, 0, 0, 0};
            Buf *bufs[6] = {&vb[in_b], &vb[in_b], &vb[out_b],
                            &vb[in_b], &vb[in_b], &vb[out_b]};
            int hb = haz(in_b, 0xFFFFFFFFu, out_b, t_bar);
            if (hb) t_bar = seq_no;   /* barrier covers everything up to here */
            if (dispatch(1, bufs, vec ? 6 : 3, push, sizeof(push), (elems + 127) / 128, hb, i)) return 1;
            mark(in_b, 0xFFFFFFFFu, out_b);
        }
    }
    /* ---- GPU decode fused into the same command buffer ---- */
    if (nlevels) {
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, NULL, 0, NULL);
        for (uint32_t L = 0; L < nlevels; ++L) {
            if (box_vid[L] >= nvals || cls_vid[L] >= nvals) continue;
            uint32_t b = box_vid[L], c = cls_vid[L];
            if (vsize[b] == 0 || vsize[c] == 0) continue;
            uint32_t a = vh[b] * vw[b];
            Buf *bufs[4] = {&vb[pb[b]], &vb[pb[c]], cbv, kbv};
            if (is_i8) {
                /* box/cls are int8-typed values; their scales are per-value. */
                uint32_t p8[13] = {a, vw[b], vh[b], in_w, nc,
                                   (uint32_t)(conf_thr * 1000.0f + 0.5f),
                                   po[b], po[c], vc[pb[b]], vc[pb[c]], 4096,
                                   fbits(vscale[b]), fbits(vscale[c])};
                if (dispatch(14, bufs, 4, p8, sizeof(p8), (a + 127) / 128, 1, 0x80000000u | L)) return 1;
                continue;
            }
            uint32_t push[11] = {a, vw[b], vh[b], in_w, nc,
                                 (uint32_t)(conf_thr * 1000.0f + 0.5f),
                                 po[b], po[c], vc[pb[b]], vc[pb[c]], 4096};
            if (dispatch(3, bufs, 4, push, sizeof(push), (a + 127) / 128, 1, 0x80000000u | L)) return 1;
        }
        VkMemoryBarrier tb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &tb, 0, NULL, 0, NULL);
        VkBufferCopy bc = {0, 0, 4096u * 6u * 4u};
        vkCmdCopyBuffer(cmd, cbv->buf, csv->buf, 1, &bc);
        VkBufferCopy bc2 = {0, 0, 16};
        vkCmdCopyBuffer(cmd, kbv->buf, ksv->buf, 1, &bc2);
        VkMemoryBarrier hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
    }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, gpuqp, 1);
    CHK(vkEndCommandBuffer(cmd));
    double t_rec1 = now_ms();
    qprintf("variant %d: recorded in %.2f ms (conv: f16 %u, oc4 %u, oc1 %u, stem %u, wino %u, quad %u, pair %u, base %u)\n",
           conv_variant, t_rec1 - t_rec0, f16_layers, oc4_layers, oc1_layers, stem_layers, wino_layers, quad_layers, pair_layers, base_layers);
    if (emit && !policy_pass) {
        prof_n_all[v] = prof_n;
        prof_label_all[v] = malloc((size_t)prof_n * 4);
        if (prof_n) memcpy(prof_label_all[v], prof_label, (size_t)prof_n * 4);
        /* Rotate over every variant recorded so far and take the per-layer
         * min: sequential measurement lets DVFS drift decide the winner. */
        for (int rep = 0; rep < 3; ++rep) {
            for (int k = 0; k <= v; ++k) {
                int idx = (rep + k) % (v + 1);
                if (submit_and_wait(cmds[idx])) return 1;
                uint64_t gts[2] = {0, 0};
                vkGetQueryPoolResults(dev, gpuqp, 0, 2, sizeof(gts), gts, 8,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                double gm = (double)(gts[1] - gts[0]) * ts_period_ns / 1e6;
                if (gm > 0 && gm < gvar_ms[vars[idx]]) gvar_ms[vars[idx]] = gm;
                uint64_t *ts = malloc((size_t)prof_n_all[idx] * 2 * 8);
                if (!ts) return 1;
                if (vkGetQueryPoolResults(dev, qpool, 0, prof_n_all[idx] * 2,
                                          (size_t)prof_n_all[idx] * 2 * 8, ts, 8,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
                    for (uint32_t q = 0; q < prof_n_all[idx]; ++q) {
                        uint32_t lab = prof_label_all[idx][q];
                        if ((lab & 0x80000000u) || lab >= nlayers) continue;
                        double ms = (double)(ts[q * 2 + 1] - ts[q * 2]) * ts_period_ns / 1e6;
                        double *slot = &bestms[(size_t)lab * 7 + vars[idx]];
                        if (ms < *slot) *slot = ms;
                    }
                }
                free(ts);
            }
        }
    }
    }
    if (emit) {
        if (submit_and_wait(cmd)) return 1;
        double ebest = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            if (submit_and_wait(cmd)) return 1;
            uint64_t ts[2] = {0, 0};
            vkGetQueryPoolResults(dev, gpuqp, 0, 2, sizeof(ts), ts, 8,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            double ms = (double)(ts[1] - ts[0]) * ts_period_ns / 1e6;
            if (ms < ebest) ebest = ms;
        }
        qprintf("policy pass: gpu %.3f ms\n", ebest);
        int gbest = 0;
        for (int k = 1; k < 7; ++k)
            if (gvar_ms[k] < gvar_ms[gbest]) gbest = k;
        if (gvar_ms[gbest] < 1e29)
            qprintf("emit: whole-graph best %d = %.3f ms\n", gbest, gvar_ms[gbest]);
        if (gvar_ms[gbest] < 1e29 && ebest > gvar_ms[gbest] * 1.02) {
            /* The per-layer mix lost to a single global variant: ship the
             * trusted map instead of a map that only looks good on paper. */
            FILE *pf = fopen(emit_policy_path, "w");
            if (!pf) { qprintf("cannot rewrite policy %s\n", emit_policy_path); return 1; }
            fprintf(pf, "# aexrt-m policy v1 nlayers=%u nvals=%u input=%ux%u\n",
                    nlayers, nvals, in_w, in_h);
            for (uint32_t i = 0; i < nlayers; ++i) {
                Layer *L = &layers[i];
                if (L->type != 1 && L->type != 9 && L->type != 10 && L->type != 11) continue;
                layer_var[i] = gbest;
                fprintf(pf, "%u %d  # t%u %u->%u k%ux%u s%u %ux%u\n",
                        i, gbest, L->type, L->ic, L->oc, L->kh, L->kw, L->stride,
                        vw[L->out], vh[L->out]);
            }
            fclose(pf);
            qprintf("policy: per-layer mix regressed (%.3f vs %.3f), rewrote all-%d\n",
                   ebest, gvar_ms[gbest], gbest);
        }
        fflush(stdout);
        return 0;
    }
    if (nvar > 1) {
        double bm[6];
        for (int v = 0; v < nvar; ++v) {
            bm[v] = 1e30;
            if (g_cntb) memset(g_cntb->map, 0, 4);
            if (submit_and_wait(cmds[v])) return 1;   /* warm up each variant */
        }
        /* Round-robin the timed submits and rotate the starting variant every
         * round: sequential A/B on Mali is skewed by DVFS/thermal drift. */
        for (int round = 0; round < 9; ++round) {
            for (int k = 0; k < nvar; ++k) {
                int v = (round + k) % nvar;
                if (g_cntb) memset(g_cntb->map, 0, 4);
                if (submit_and_wait(cmds[v])) return 1;
                uint64_t ts[2] = {0, 0};
                vkGetQueryPoolResults(dev, gpuqp, 0, 2, sizeof(ts), ts, 8,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                double ms = (double)(ts[1] - ts[0]) * ts_period_ns / 1e6;
                if (ms < bm[v]) bm[v] = ms;
            }
        }
        int bestv = 0;
        for (int v = 0; v < nvar; ++v) {
            var_ms[v] = bm[v];
            qprintf("autotune: variant %d -> %.3f ms\n", vars[v], var_ms[v]);
            if (var_ms[v] < var_ms[bestv]) bestv = v;
        }
        cmd = cmds[bestv];
        conv_variant = vars[bestv];
        qprintf("autotune: using variant %d\n", conv_variant);
    } else {
        cmd = cmds[0];
    }

    return 0;
}

/* ---- in-process library API (aexrt_m.h) ---------------------------------- */
struct AexrtMSession { int token; };
static struct AexrtMSession g_api_token;
static int g_session_ever;
static Det g_api_dets[4096];

AexrtMSession *aexrt_m_open(const char *model, const char *policy, int frames,
                            float conf) {
    if (g_session_ever || !model) return NULL;
    g_quiet = 1;
    g_api_mode = 1;
    model_path = model;
    input_path = NULL; out_path = NULL;
    policy_path = policy;
    emit_policy_path = NULL; frames_path = NULL;
    nsets = frames > 1 ? (frames > MAXSETS ? MAXSETS : frames) : 1;
    stream_mode = 0; want_profile = 0; autotune = 0; conv_variant = 0;
    bench = 1;
    if (conf > 0.0f) conf_thr = conf;
    if (session_setup()) { g_quiet = 0; return NULL; }
    g_session_ready = 1;
    g_session_ever = 1;
    g_api_next_set = 0; g_api_submitted = 0;
    g_quiet = 0;
    /* One dummy inference warms the shader caches and DVFS, so the caller's
     * first real frame sees steady-state latency instead of ~2x. */
    {
        float *zero = calloc((size_t)3u * in_h * in_w, sizeof(float));
        if (zero) {
            (void)aexrt_m_run((AexrtMSession *)&g_api_token, zero, NULL, 0);
            free(zero);
            g_api_next_set = 0; g_api_submitted = 0;
        }
    }
    return (AexrtMSession *)&g_api_token;
}

int aexrt_m_input_size(AexrtMSession *sess) {
    if (!sess || !g_session_ready) return 0;
    return (int)(3u * in_h * in_w);
}

void aexrt_m_shape(AexrtMSession *sess, int *classes, int *width, int *height) {
    if (!sess || !g_session_ready) return;
    if (classes) *classes = (int)nc;
    if (width) *width = (int)in_w;
    if (height) *height = (int)in_h;
}

int aexrt_m_run(AexrtMSession *sess, const float *frame, AexrtMDet *dets, int max_dets) {
    if (!sess || !g_session_ready || !frame) return -1;
    int set = (int)(g_api_next_set % (uint32_t)nsets);
    if (g_api_submitted >= (uint32_t)nsets &&
        vkWaitForFences(dev, 1, &pipe_fences[set], VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return -1;
    uint16_t *dst = (uint16_t *)in_stage[set].map;
    load_frame_fp16(frame, dst, in_h, in_w);
    if (nlevels) memset(cntb_s[set].map, 0, 4);
    if (submit_async(cmds[set], pipe_fences[set], 0)) return -1;
    if (vkWaitForFences(dev, 1, &pipe_fences[set], VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return -1;
    ++g_api_submitted;
    ++g_api_next_set;
    uint32_t raw = *(volatile uint32_t *)cnts_s[set].map;
    if (!dets || max_dets <= 0) return 0;
    uint32_t cap = max_dets > 4096 ? 4096u : (uint32_t)max_dets;
    uint32_t k = nms_candidates((const float *)cands_s[set].map, raw, nc, iou_thr,
                                g_api_dets, cap);
    for (uint32_t i = 0; i < k; ++i) {
        dets[i].x1 = g_api_dets[i].x1; dets[i].y1 = g_api_dets[i].y1;
        dets[i].x2 = g_api_dets[i].x2; dets[i].y2 = g_api_dets[i].y2;
        dets[i].score = g_api_dets[i].score; dets[i].cls = g_api_dets[i].cls;
    }
    return (int)k;
}

void aexrt_m_close(AexrtMSession *sess) {
    if (!sess) return;
    /* One session per process: the Vulkan device, pipelines and recorded
     * command buffers live for the process lifetime (as in an app engine). */
    g_session_ready = 0;
}

#ifndef AEXRT_M_NO_MAIN
int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--adapter") && i + 1 < argc) {
            force_adapter = atoi(argv[i + 1]);
            continue;
        }
        if (!strcmp(argv[i], "--caps")) {
            if (gpu_init()) return 1;
            print_caps();
            return 0;
        }
        if (!strcmp(argv[i], "--bench-coopmat") && i + 6 < argc) {
            const char *spv = argv[i + 1];
            uint32_t groups = (uint32_t)strtoul(argv[i + 2], NULL, 10);
            uint32_t m = (uint32_t)strtoul(argv[i + 3], NULL, 10);
            uint32_t n = (uint32_t)strtoul(argv[i + 4], NULL, 10);
            uint32_t k = (uint32_t)strtoul(argv[i + 5], NULL, 10);
            uint32_t esz = (uint32_t)strtoul(argv[i + 6], NULL, 10);
            if (gpu_init()) return 1;
            return bench_coopmat(spv, groups, m, n, k, esz, 32);
        }
    }
    if (argc < 4) { usage(argv[0]); return 1; }
    model_path = argv[1]; input_path = argv[2]; out_path = argv[3];
    for (int i = 4; i < argc; ++i) {
        if (!strcmp(argv[i], "--dump-heads") && i + 1 < argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--dump-cands") && i + 1 < argc) dump_cands = argv[++i];
        else if (!strcmp(argv[i], "--conf") && i + 1 < argc) conf_thr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--iou") && i + 1 < argc) iou_thr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--profile")) want_profile = 1;
        else if (!strcmp(argv[i], "--policy") && i + 1 < argc) policy_path = argv[++i];
        else if (!strcmp(argv[i], "--emit-policy") && i + 1 < argc) emit_policy_path = argv[++i];
        else if (!strcmp(argv[i], "--pipeline") && i + 1 < argc) {
            nsets = atoi(argv[++i]);
            if (nsets < 1) nsets = 1;
            if (nsets > MAXSETS) nsets = MAXSETS;
        }
        else if (!strcmp(argv[i], "--stream")) stream_mode = 1;
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames_path = argv[++i];
        else if (!strcmp(argv[i], "--autotune")) autotune = 1;
        else if (!strcmp(argv[i], "--vector-elem")) g_vec_elem = 1;
        else if (!strcmp(argv[i], "--conv-variant") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "base")) conv_variant = 1;
            else if (!strcmp(v, "quad")) conv_variant = 2;
            else if (!strcmp(v, "f16")) conv_variant = 3;
            else if (!strcmp(v, "f16oc4")) conv_variant = 4;
            else if (!strcmp(v, "f16oc1")) conv_variant = 5;
            else if (!strcmp(v, "pair")) conv_variant = 6;
            else if (!strcmp(v, "auto")) conv_variant = 0;
            else { usage(argv[0]); return 1; }
        }
        else if (!strncmp(argv[i], "--", 2)) { usage(argv[0]); return 1; }
        else bench = (uint32_t)atoi(argv[i]);
    }
    if (bench < 1) bench = 1;
    printf("AEXRT-M starting...\n"); fflush(stdout);

    if (session_setup()) return 1;
    if (emit_policy_path) return 0;
    /* ---- benchmark ---- */
    double gpu_ms = 0.0;
    if (stream_mode) {
        /* True streaming: the host writes a new frame into the free set's
         * input buffer only after that set's previous frame retired, so the
         * conversion cost is on the critical path exactly as in a real app.
         * Frames come from --frames (concatenated fp32 NCHW) or repeat the
         * single loaded input. The decode/NMS tail below runs on the last
         * frame's candidate staging. */
        uint32_t nf = bench;
        Det *frame_dets = malloc(sizeof(Det) * 4096);
        uint32_t *kept = calloc(nf ? nf : 1, 4);
        double t0 = now_ms();
        for (uint32_t f = 0; f < nf; ++f) {
            int s = (int)(f % (uint32_t)nsets);
            if (f >= (uint32_t)nsets)
                CHK(vkWaitForFences(dev, 1, &pipe_fences[s], VK_TRUE, UINT64_MAX));
            const float *src = frames ? frames + (size_t)f * frame_elems : input_frame;
            uint16_t *dst = (uint16_t *)in_stage[s].map;
            load_frame_fp16(src, dst, in_h, in_w);
            if (nlevels) memset(cntb_s[s].map, 0, 4);
            if (submit_async(cmds[s], pipe_fences[s], 0)) return 1;
            /* Post-process the previous frame while this one runs: its set is
             * not needed again for another nsets-1 frames. */
            if (f >= 1u) {
                int p = (int)((f - 1u) % (uint32_t)nsets);
                CHK(vkWaitForFences(dev, 1, &pipe_fences[p], VK_TRUE, UINT64_MAX));
                uint32_t n = *(volatile uint32_t *)cnts_s[p].map;
                kept[f - 1u] = nms_candidates((const float *)cands_s[p].map, n, nc,
                                              iou_thr, frame_dets, 4096);
                const char *sdp = getenv("AEXRT_STREAM_DUMP");
                if (sdp) {
                    char sp[1024];
                    snprintf(sp, sizeof(sp), "%s_%u.bin", sdp, f - 1u);
                    FILE *fd = fopen(sp, "wb");
                    if (fd) {
                        fwrite(&kept[f - 1u], 4, 1, fd);
                        for (uint32_t k = 0; k < kept[f - 1u]; ++k) {
                            float r[6] = {frame_dets[k].x1, frame_dets[k].y1,
                                          frame_dets[k].x2, frame_dets[k].y2,
                                          frame_dets[k].score, (float)frame_dets[k].cls};
                            fwrite(r, 4, 6, fd);
                        }
                        fclose(fd);
                    }
                }
            }
        }
        for (int s = 0; s < nsets; ++s)
            CHK(vkWaitForFences(dev, 1, &pipe_fences[s], VK_TRUE, UINT64_MAX));
        double t1 = now_ms();
        uint32_t last = (nf - 1u) % (uint32_t)nsets;
        cands = cands_s[last]; cnts = cnts_s[last];
        uint32_t kmin = ~0u, kmax = 0; double ksum = 0.0;
        for (uint32_t f = 0; f + 1u < nf; ++f) {
            if (kept[f] < kmin) kmin = kept[f];
            if (kept[f] > kmax) kmax = kept[f];
            ksum += kept[f];
        }
        printf("stream: %d sets, %u frames, %.2f ms/frame incl. input convert + NMS "
               "(%.1f fps)\n", nsets, nf, (t1 - t0) / (double)nf,
               1000.0 * (double)nf / (t1 - t0));
        if (nf > 1u)
            printf("stream: per-frame NMS kept %u..%u (avg %.1f) over %u frames\n",
                   kmin, kmax, ksum / (double)(nf - 1u), nf - 1u);
        free(frame_dets);
        free(kept);
        gpu_ms = (t1 - t0) / (double)nf;
    } else if (nsets > 1) {
        /* N independent buffer sets alternate on one queue: frame i+1 starts
         * while frame i is still executing, so the fence tail of each frame is
         * hidden. Wall time / frames is the throughput metric; per-frame GPU
         * timestamps overlap and are not meaningful here. */
        uint32_t frames = bench;
        for (int s = 0; s < nsets; ++s) {
            if (nlevels) memset(cntb_s[s].map, 0, 4);
            if (submit_async(cmds[s], pipe_fences[s], 0)) return 1;
        }
        for (int s = 0; s < nsets; ++s)
            CHK(vkWaitForFences(dev, 1, &pipe_fences[s], VK_TRUE, UINT64_MAX));
        double t0 = now_ms();
        for (uint32_t f = 0; f < frames; ++f) {
            int s = (int)(f % (uint32_t)nsets);
            if (f >= (uint32_t)nsets)
                CHK(vkWaitForFences(dev, 1, &pipe_fences[s], VK_TRUE, UINT64_MAX));
            if (nlevels) memset(cntb_s[s].map, 0, 4);
            if (submit_async(cmds[s], pipe_fences[s], 0)) return 1;
        }
        for (int s = 0; s < nsets; ++s)
            CHK(vkWaitForFences(dev, 1, &pipe_fences[s], VK_TRUE, UINT64_MAX));
        double t1 = now_ms();
        double per = (t1 - t0) / (double)frames;
        printf("pipeline: %d sets, %.2f ms/frame, %.1f fps (%u frames, wall)\n",
               nsets, per, 1000.0 / per, frames);
        gpu_ms = per;
    } else {
        if (nlevels) memset(cntb.map, 0, 4);
        /* DVFS ramp: a single warmup leaves the timed median clocked low when
         * no autotune sweep preceded it (cold policy runs looked 20% slower). */
        for (int w = 0; w < 10; ++w)
            if (submit_and_wait(cmd)) return 1;
        double best = 1e30, sum = 0.0;
        double *times = malloc((size_t)bench * sizeof(double));
        double *gts = malloc((size_t)bench * sizeof(double));
        for (uint32_t it = 0; it < bench; ++it) {
            if (nlevels) memset(cntb.map, 0, 4);
            double a = now_ms();
            if (submit_and_wait(cmd)) return 1;
            double b = now_ms();
            double ms = b - a;
            times[it] = ms;
            sum += ms;
            if (ms < best) best = ms;
            uint64_t ts[2] = {0, 0};
            if (ts_period_ns > 0.0)
                vkGetQueryPoolResults(dev, gpuqp, 0, 2, sizeof(ts), ts, 8,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            gts[it] = ts_period_ns > 0.0 ? (double)(ts[1] - ts[0]) * ts_period_ns / 1e6 : ms;
        }
        qsort(times, bench, sizeof(double), dcmp);
        qsort(gts, bench, sizeof(double), dcmp);
        double med = times[bench / 2];
        double gmed = gts[bench / 2];
        double gbest = gts[0];
        printf("GPU done: wall median %.2f / best %.2f ms | gpu-ts median %.2f / best %.2f / mean %.2f ms (%u iters)\n",
               med, best, gmed, gbest, sum / bench, bench);
        gpu_ms = gmed;
        free(times);
        free(gts);
    }
    if (profiling) print_profile(layers, vh, vw, vc);

    /* ---- optional per-value dump for layer-by-layer validation ---- */
    {
        const char *dv = getenv("AEXRT_DUMP_VALUE");
        if (dv && dump_dir) {
            uint32_t v = (uint32_t)atoi(dv);
            if (v < nvals && vsize[v]) {
                uint32_t b = pb[v];
                char path[1024];
                snprintf(path, sizeof(path), "%s/v%u.f32", dump_dir, v);
                FILE *fo = fopen(path, "wb");
                if (fo) {
                    int8_t *q = (int8_t *)vb[b].map;
                    uint16_t *h = (uint16_t *)vb[b].map;
                    uint32_t stride = vc[b];
                    uint32_t sp = vh[v] * vw[v], chn = vc[v];
                    float s = vscale[v];
                    for (uint32_t pos = 0; pos < sp; ++pos)
                        for (uint32_t c = 0; c < chn; ++c) {
                            uint32_t k = pos * stride + po[v] + c;
                            float x = vdtype[v] ? (float)q[k] * s : h2f(h[k]);
                            fwrite(&x, 4, 1, fo);
                        }
                    fclose(fo);
                    printf("dump value v%u (%u elems, %s)\n", v, sp * chn,
                           vdtype[v] ? "int8" : "fp16");
                }
            }
        }
    }

    /* ---- optional head dumps for numeric validation ---- */
    if (dump_dir) {
        char path[1024];
        ensure_dir(dump_dir);
        for (uint32_t L = 0; L < nlevels; ++L) {
            uint32_t b = box_vid[L], c = cls_vid[L];
            if (b >= nvals || c >= nvals || vsize[b] == 0 || vsize[c] == 0) continue;
            uint32_t ab = vh[b] * vw[b], ac = vh[c] * vw[c];
            uint32_t cb = vc[pb[b]], cc = vc[pb[c]];
            uint16_t *bb = (uint16_t *)vb[pb[b]].map + po[b];
            uint16_t *bc = (uint16_t *)vb[pb[c]].map + po[c];
            int8_t *bq = (int8_t *)vb[pb[b]].map + po[b];
            int8_t *cq = (int8_t *)vb[pb[c]].map + po[c];
            float sb = vscale[b], sc = vscale[c];
            snprintf(path, sizeof(path), "%s/L%u_box.f32", dump_dir, L);
            FILE *fo = fopen(path, "wb");
            for (uint32_t p = 0; p < ab; ++p)
                for (uint32_t k = 0; k < 64; ++k) {
                    float v = is_i8 ? (float)bq[p * cb + k] * sb : h2f(bb[p * cb + k]);
                    fwrite(&v, 4, 1, fo);
                }
            fclose(fo);
            snprintf(path, sizeof(path), "%s/L%u_cls.f32", dump_dir, L);
            fo = fopen(path, "wb");
            for (uint32_t p = 0; p < ac; ++p)
                for (uint32_t k = 0; k < nc; ++k) {
                    float v = is_i8 ? (float)cq[p * cc + k] * sc : h2f(bc[p * cc + k]);
                    fwrite(&v, 4, 1, fo);
                }
            fclose(fo);
            printf("dump L%u: anchors %u gw %u ctot %u/%u\n", L, ab, vw[b], cb, cc);
        }
    }

    if (!nlevels) {
        /* debug path: dump the whole placement buffer as fp32 */
        uint32_t ob2 = pb[out_vid];
        uint32_t elems = vsize[out_vid];
        uint16_t *out = (uint16_t *)vb[ob2].map + po[out_vid];
        int8_t *out8 = (int8_t *)vb[ob2].map + po[out_vid];
        float so = vscale[out_vid];
        FILE *fo = fopen(out_path, "wb");
        if (!fo) { printf("cannot write %s\n", out_path); return 1; }
        for (uint32_t i = 0; i < elems; ++i) {
            float v = is_i8 ? (float)out8[i] * so : h2f(out[i]);
            fwrite(&v, 4, 1, fo);
        }
        fclose(fo);
        float mx = -1e30f, mn = 1e30f;
        for (uint32_t i = 0; i < elems; ++i) {
            float v = is_i8 ? (float)out8[i] * so : h2f(out[i]);
            if (v > mx) mx = v;
            if (v < mn) mn = v;
        }
        printf("output: v%u elems %u in buf v%u off %u\n",
               out_vid, vsize[out_vid], ob2, po[out_vid]);
        printf("range: [%.4f, %.4f]\n", mn, mx);
        printf("AEXRT-M mobile: PASS\n");
        return 0;
    }

    /* ---- CPU sort + per-class greedy NMS ---- */
    double t_cpu0 = now_ms();
    uint32_t ndet = *(volatile uint32_t *)cnts.map;
    if (ndet > 4096u) ndet = 4096u;
    float *cand = (float *)cands.map;
    if (dump_cands) {
        FILE *fc = fopen(dump_cands, "wb");
        if (fc) {
            fwrite(&ndet, 4, 1, fc);
            fwrite(cand, 4, (size_t)ndet * 6, fc);
            fclose(fc);
            printf("dumped %u raw candidates to %s\n", ndet, dump_cands);
        }
    }
    Det *dets = malloc(sizeof(Det) * (ndet ? ndet : 1));
    double t_sort0 = now_ms();
    uint32_t nkeep = nms_candidates(cand, ndet, nc, iou_thr, dets, ndet ? ndet : 1);
    double t_cpu1 = now_ms();
    if (profiling)
        printf("nms: %.2f ms (%u candidates)\n", t_cpu1 - t_sort0, ndet);

    FILE *fo = fopen(out_path, "wb");
    if (!fo) { printf("cannot write %s\n", out_path); return 1; }
    uint32_t rec = nkeep;
    fwrite(&rec, 4, 1, fo);
    for (uint32_t i = 0; i < nkeep; ++i) {
        float r[6] = {dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2,
                      dets[i].score, (float)dets[i].cls};
        fwrite(r, 4, 6, fo);
    }
    fclose(fo);
    printf("decode: %u candidates, NMS kept %u in %.2f ms\n", ndet, nkeep, t_cpu1 - t_cpu0);
    printf("latency: gpu %.2f ms + decode/nms %.2f ms\n", gpu_ms, t_cpu1 - t_cpu0);
    for (uint32_t i = 0; i < nkeep && i < 10; ++i) {
        printf("det: cls %d conf %.3f box [%.1f, %.1f, %.1f, %.1f]\n",
               dets[i].cls, dets[i].score, dets[i].x1, dets[i].y1,
               dets[i].x2, dets[i].y2);
    }
    printf("AEXRT-M mobile: PASS\n");
    return 0;
}
#endif /* AEXRT_M_NO_MAIN */
