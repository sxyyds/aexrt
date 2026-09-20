/* AEXRT-M Phase-1: Vulkan HAL skeleton end-to-end test (arm64-v8a).
 *
 * Validates the full HAL pipeline the inference engine will use:
 *   1. instance + Vulkan 1.3 logical device (shaderInt8 + dot enabled)
 *   2. unified-memory buffers (HOST_VISIBLE|HOST_COHERENT|DEVICE_LOCAL)
 *   3. compute pipeline from embedded SPIR-V (descriptor set, pipeline)
 *   4. record -> submit -> fence wait -> readback
 *   5. numerical check of the quantized pg16 GEMM vs CPU reference
 * Production engine code (native/aexrt_vulkan_runtime.cpp) grows from this. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>

#include "quant_test_spv.h"

#define CHK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    printf("FAIL %s -> %d (line %d)\n", #x, (int)r_, __LINE__); return 1; } } while (0)

#define K 256
#define GROUPS (K / 16)
#define OC 8

static VkInstance inst;
static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkCommandBuffer cmd;

typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    void* map;
    VkDeviceSize size;
} MapBuf;

static int mk_buf(VkDeviceSize sz, VkBufferUsageFlags usage, MapBuf* out) {
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = sz;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHK(vkCreateBuffer(dev, &bi, NULL, &out->buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, out->buf, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
    int type = -1;
    for (uint32_t pass = 0; pass < 2 && type < 0; ++pass) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if (!(req.memoryTypeBits & (1u << i))) continue;
            VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
            if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
                (pass == 0 ? (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) : 1)) {
                type = (int)i;
                break;
            }
        }
    }
    if (type < 0) { printf("FAIL: no host-visible coherent memory\n"); return 1; }
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = (uint32_t)type;
    CHK(vkAllocateMemory(dev, &ai, NULL, &out->mem));
    CHK(vkBindBufferMemory(dev, out->buf, out->mem, 0));
    CHK(vkMapMemory(dev, out->mem, 0, sz, 0, &out->map));
    out->size = sz;
    return 0;
}

int main(void) {
    /* 1. instance + device with int8 + dot features enabled */
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "aexrt-m-hal";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ci.pApplicationInfo = &app;
    CHK(vkCreateInstance(&ci, NULL, &inst));
    uint32_t n = 1;
    CHK(vkEnumeratePhysicalDevices(inst, &n, &pdev));
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan13Features f13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.shaderIntegerDotProduct = VK_TRUE;
    f13.maintenance4 = VK_TRUE;
    VkPhysicalDeviceShaderFloat16Int8Features ffi = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES };
    ffi.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    ffi.shaderInt8 = VK_TRUE;
    ffi.shaderFloat16 = VK_TRUE;
    f13.pNext = &ffi;
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pNext = &f13;
    CHK(vkCreateDevice(pdev, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, 0, 0, &queue);
    VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = 0;
    CHK(vkCreateCommandPool(dev, &pci, NULL, &pool));
    VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    CHK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    printf("HAL: device + queue + pool ok\n");

    /* 2. mapped unified buffers */
    MapBuf A, W, AS, WS, OUT;
    if (mk_buf(K, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &A)) return 1;
    if (mk_buf((VkDeviceSize)K * OC, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &W)) return 1;
    if (mk_buf(GROUPS * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &AS)) return 1;
    if (mk_buf(OC * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &WS)) return 1;
    if (mk_buf(64 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &OUT)) return 1;
    printf("HAL: 5 mapped buffers ok (type: unified device-local where available)\n");

    /* 3. descriptor set + compute pipeline */
    VkDescriptorSetLayoutBinding lb[5];
    for (int i = 0; i < 5; ++i) {
        lb[i].binding = (uint32_t)i;
        lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lb[i].descriptorCount = 1;
        lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dsl.bindingCount = 5;
    dsl.pBindings = lb;
    VkDescriptorSetLayout set_layout;
    CHK(vkCreateDescriptorSetLayout(dev, &dsl, NULL, &set_layout));
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set_layout;
    VkPipelineLayout layout;
    CHK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = sizeof(quant_test_spv);
    smci.pCode = (const uint32_t*)quant_test_spv;
    VkShaderModule module;
    CHK(vkCreateShaderModule(dev, &smci, NULL, &module));
    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    VkPipeline pipe;
    CHK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    printf("HAL: SPIR-V pipeline created (int8 arithmetic shader)\n");

    VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    VkDescriptorPool dpool;
    CHK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &set_layout;
    VkDescriptorSet set;
    CHK(vkAllocateDescriptorSets(dev, &dsai, &set));
    VkWriteDescriptorSet wr[5];
    VkDescriptorBufferInfo bi[5];
    VkBuffer bufs[5] = { A.buf, W.buf, AS.buf, WS.buf, OUT.buf };
    for (int i = 0; i < 5; ++i) {
        bi[i].buffer = bufs[i];
        bi[i].offset = 0;
        bi[i].range = VK_WHOLE_SIZE;
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = set;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(dev, 5, wr, 0, NULL);

    /* 4. host data + reference (deterministic) */
    signed char* a = (signed char*)A.map;
    signed char* w = (signed char*)W.map;
    float* as = (float*)AS.map;
    float* ws = (float*)WS.map;
    unsigned st = 12345u;
    for (int i = 0; i < K; ++i) { st = st * 1664525u + 1013904223u; a[i] = (signed char)((int)((st >> 20) % 200) - 100); }
    for (int i = 0; i < K * OC; ++i) { st = st * 1664525u + 1013904223u; w[i] = (signed char)((int)((st >> 20) % 60) - 30); }
    for (int g = 0; g < GROUPS; ++g) as[g] = 0.01f + 0.005f * (float)g;      /* amax/127 proxies */
    for (int o = 0; o < OC; ++o) ws[o] = 0.02f + 0.001f * (float)o;
    memset(OUT.map, 0, OUT.size);

    /* CPU reference mirroring the pg16 shader math */
    float ref[64];
    for (int t = 0; t < 64; ++t) {
        uint oc = (t / 16u) % OC;   /* match shader's (tid>>3)+oc*4 pattern loosely: compute per-out slot */
        (void)oc;
        ref[t] = 0.0f;  /* actual check below uses first 32 outputs written by groups 0..7 */
    }
    float refv[64];
    for (uint o = 0; o < OC; ++o) {
        for (uint s = 0; s < 4u; ++s) {
            uint tid_slot = (s & 7u) * 128u + (s >> 3u) + o * 4u;  /* shader's output index for tid s */
            float acc = 0.0f;
            for (uint g = 0; g < GROUPS; ++g) {
                int gsum = 0;
                for (uint e = 0; e < 16u; ++e) gsum += (int)a[g * 16 + e] * (int)w[o * K + g * 16 + e];
                acc += (float)gsum * as[g];
            }
            refv[s] = acc * ws[o];
            (void)tid_slot;
        }
        /* outputs for oc=o live at [o*4 .. o*4+3] */
    }

    /* 5. record + submit + fence */
    VkCommandBufferBeginInfo beg = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    CHK(vkBeginCommandBuffer(cmd, &beg));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    vkCmdDispatch(cmd, OC, 1, 1);
    CHK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    CHK(vkCreateFence(dev, &fci, NULL, &fence));
    CHK(vkQueueSubmit(queue, 1, &si, fence));
    CHK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    printf("HAL: dispatch + fence ok\n");

    /* 6. verify against CPU reference (out[oc*8 + lane]) */
    float* o = (float*)OUT.map;
    int checked = 0, bad = 0;
    for (uint oc = 0; oc < OC; ++oc) {
        float acc = 0.0f;
        for (uint g = 0; g < GROUPS; ++g) {
            int gsum = 0;
            for (uint e = 0; e < 16u; ++e) gsum += (int)a[g * 16 + e] * (int)w[oc * K + g * 16 + e];
            acc += (float)gsum * as[g];
        }
        float expect = acc * ws[oc];
        for (uint lane = 0; lane < 8u; ++lane) {
            uint idx = oc * 8u + lane;
            if (fabsf(o[idx] - expect) > 1e-3f * (1.0f + fabsf(expect))) { ++bad; if (bad < 4) printf("  mismatch idx=%u gpu=%.5f cpu=%.5f", idx, o[idx], expect); }

            ++checked;
        }
    }
    printf("verify: %d outputs checked, %d mismatches\n", checked, bad);
    printf(bad == 0 ? "AEXRT-M Phase-1 HAL: PASS (pg16 quantized GEMM on Mali)\n"
                    : "AEXRT-M Phase-1 HAL: FAIL\n");
    return bad == 0 ? 0 : 1;
}
