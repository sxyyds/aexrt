/* AEXRT-M mobile YOLO runtime: reads flat model, executes on Vulkan. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include "../kernels/conv_spv.h"
#include "../kernels/elementwise_spv.h"
#include "../kernels/copy_spv.h"

#define CHK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    printf("FAIL %s -> %d (line %d)\n", #x, (int)r_, __LINE__); return 1; } } while (0)

typedef struct { VkBuffer buf; VkDeviceMemory mem; void* map; VkDeviceSize sz; } Buf;
static VkInstance inst; static VkPhysicalDevice pdev; static VkDevice dev;
static VkQueue queue; static VkCommandPool pool; static VkCommandBuffer cmd;
static VkFence fence; static VkDescriptorPool dpool;
static VkPipeline pipes[3]; static VkPipelineLayout layouts[3]; static VkDescriptorSetLayout setl[3];

static int mkbuf(VkDeviceSize sz, VkBufferUsageFlags u, Buf* b) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = sz; bi.usage = u; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHK(vkCreateBuffer(dev, &bi, 0, &b->buf));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, b->buf, &req);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
    int ty = -1;
    for (uint32_t p2 = 0; p2 < 2 && ty < 0; ++p2)
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((req.memoryTypeBits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                && (p2==0 ? (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) : 1)) { ty=(int)i; break; }
    if (ty<0) return 1;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size; ai.memoryTypeIndex = (uint32_t)ty;
    CHK(vkAllocateMemory(dev, &ai, 0, &b->mem));
    CHK(vkBindBufferMemory(dev, b->buf, b->mem, 0));
    CHK(vkMapMemory(dev, b->mem, 0, sz, 0, &b->map));
    b->sz = sz; return 0;
}
static int mkpipe(int idx, const uint32_t* spv, size_t sz, int nb, int pc_sz) {
    VkDescriptorSetLayoutBinding lb[8];
    for (int i = 0; i < nb; ++i) { lb[i].binding=(uint32_t)i; lb[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; lb[i].descriptorCount=1; lb[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dsl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl.bindingCount=(uint32_t)nb; dsl.pBindings=lb;
    CHK(vkCreateDescriptorSetLayout(dev, &dsl, 0, &setl[idx]));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pc_sz};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount=1; pl.pSetLayouts=&setl[idx];
    pl.pushConstantRangeCount = pc_sz>0?1:0; pl.pPushConstantRanges = pc_sz>0?&pcr:0;
    CHK(vkCreatePipelineLayout(dev, &pl, 0, &layouts[idx]));
    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = sz; sm.pCode = spv;
    VkShaderModule mod; CHK(vkCreateShaderModule(dev, &sm, 0, &mod));
    VkComputePipelineCreateInfo cp = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cp.stage.module = mod; cp.stage.pName = "main";
    cp.layout = layouts[idx];
    CHK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cp, 0, &pipes[idx]));
    vkDestroyShaderModule(dev, mod, 0); return 0;
}
static int dispatch(int pi, Buf** bufs, int nb, void* push, int pc_sz, uint32_t wx) {
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool=dpool; ai.descriptorSetCount=1; ai.pSetLayouts=&setl[pi];
    VkDescriptorSet set; CHK(vkAllocateDescriptorSets(dev, &ai, &set));
    VkWriteDescriptorSet wr[8]; VkDescriptorBufferInfo bi[8];
    for (int i=0;i<nb;++i) { bi[i].buffer=bufs[i]->buf; bi[i].offset=0; bi[i].range=VK_WHOLE_SIZE;
        wr[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet=set; wr[i].dstBinding=(uint32_t)i;
        wr[i].descriptorCount=1; wr[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo=&bi[i]; }
    vkUpdateDescriptorSets(dev, (uint32_t)nb, wr, 0, 0);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[pi]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layouts[pi], 0, 1, &set, 0, 0);
    if (pc_sz>0) vkCmdPushConstants(cmd, layouts[pi], VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pc_sz, push);
    vkCmdDispatch(cmd, wx, 1, 1); return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s model input output\n", argv[0]); return 1; }
    printf("AEXRT-M starting...\n"); fflush(stdout);

    FILE* f = fopen(argv[1], "rb");
    if (!f) { printf("cannot open model: %s\n", argv[1]); return 1; }
    fseek(f,0,SEEK_END); long msz = ftell(f); fseek(f,0,SEEK_SET);
    uint8_t* md = malloc(msz);
    if (!md) { printf("malloc %ld failed\n", msz); return 1; }
    fread(md,1,msz,f); fclose(f);
    printf("model: %ld bytes loaded\n", msz); fflush(stdout);

    if (memcmp(md, "AXM1", 4)) { printf("bad magic\n"); return 1; }
    uint32_t nlayers, nvals, in_vid, out_vid, lt_sz, wt_off;
    memcpy(&nlayers, md+4, 4); memcpy(&nvals, md+8, 4);
    memcpy(&in_vid, md+12, 2); memcpy(&out_vid, md+14, 2);
    memcpy(&lt_sz, md+16, 4);
    memcpy(&wt_off, md+20+lt_sz, 4);
    printf("layers=%u vals=%u in=%u out=%u\n", nlayers, nvals, in_vid, out_vid);
    fflush(stdout);

    FILE* fi = fopen(argv[2], "rb");
    if (!fi) { printf("cannot open input\n"); return 1; }
    fseek(fi,0,SEEK_END); long isz = ftell(fi); fseek(fi,0,SEEK_SET);
    float* inp = malloc(isz);
    fread(inp,1,isz,fi); fclose(fi);
    printf("input: %ld bytes\n", isz); fflush(stdout);

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "aexrt-m"; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    CHK(vkCreateInstance(&ici, 0, &inst));
    uint32_t pn=1; CHK(vkEnumeratePhysicalDevices(inst, &pn, &pdev));
    float prio=1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex=0; qci.queueCount=1; qci.pQueuePriorities=&prio;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    CHK(vkCreateDevice(pdev, &dci, 0, &dev));
    vkGetDeviceQueue(dev,0,0,&queue);
    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pci.queueFamilyIndex = 0;
    CHK(vkCreateCommandPool(dev,&pci,0,&pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool=pool; cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount=1;
    CHK(vkAllocateCommandBuffers(dev,&cai,&cmd));
    VkFenceCreateInfo ff = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHK(vkCreateFence(dev,&ff,0,&fence));
    VkDescriptorPoolSize dpsz = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets=1024; dpci.poolSizeCount=1; dpci.pPoolSizes=&dpsz;
    CHK(vkCreateDescriptorPool(dev,&dpci,0,&dpool));
    printf("Vulkan ok\n"); fflush(stdout);

    if (mkpipe(0, conv_spv, sizeof(conv_spv), 4, 56)) return 1;
    if (mkpipe(1, elementwise_spv, sizeof(elementwise_spv), 3, 28)) return 1;
    if (mkpipe(2, copy_spv, sizeof(copy_spv), 2, 4)) return 1;
    printf("pipelines ok\n"); fflush(stdout);

    typedef struct { uint32_t type, silu, stride, pad, oc, ic, kh, kw, in0, out, extra, w_off, b_off; } Layer;
    Layer* layers = calloc(nlayers, sizeof(Layer));
    uint32_t* vsize = calloc(nvals, sizeof(uint32_t));
    vsize[in_vid] = (uint32_t)(isz / 4);
    uint32_t* concat_n = calloc(nlayers, sizeof(uint32_t));
    uint32_t** concat_ins = calloc(nlayers, sizeof(uint32_t*));

    uint32_t off = 20;
    for (uint32_t i = 0; i < nlayers; ++i) {
        uint8_t* p = md + off;
        uint32_t t = p[0];
        layers[i].type = t;
        if (t == 1) {
            layers[i].silu = p[1]; layers[i].stride = p[2]; layers[i].pad = p[3];
            memcpy(&layers[i].oc, p+4, 2); memcpy(&layers[i].ic, p+6, 2);
            layers[i].kh = p[8]; layers[i].kw = p[9];
            memcpy(&layers[i].in0, p+10, 2); memcpy(&layers[i].out, p+12, 2);
            memcpy(&layers[i].w_off, p+14, 4); memcpy(&layers[i].b_off, p+18, 4);
            uint32_t in_sz = vsize[layers[i].in0];
            uint32_t spatial = layers[i].ic > 0 ? in_sz / layers[i].ic : 1;
            uint32_t ih = (uint32_t)sqrt((double)spatial); uint32_t iw = ih > 0 ? spatial / ih : 1;
            uint32_t oh = layers[i].stride == 2 ? (ih+1)/2 : ih;
            uint32_t ow = layers[i].stride == 2 ? (iw+1)/2 : iw;
            vsize[layers[i].out] = layers[i].oc * oh * ow;
            off += 22;
        } else if (t == 2) {
            uint8_t n = p[1];
            concat_n[i] = n;
            concat_ins[i] = malloc(n * 4);
            uint32_t total = 0;
            for (uint8_t j = 0; j < n; ++j) {
                memcpy(&concat_ins[i][j], p+4+j*2, 2);
                total += vsize[concat_ins[i][j]];
            }
            memcpy(&layers[i].out, p+4+n*2, 2);
            vsize[layers[i].out] = total;
            off += 4 + n*2 + 2;
        } else {
            memcpy(&layers[i].in0, p+10, 2); memcpy(&layers[i].out, p+12, 2);
            if (t == 5) memcpy(&layers[i].extra, p+14, 2);
            vsize[layers[i].out] = (t == 3) ? vsize[layers[i].in0] / 4
                               : (t == 4) ? vsize[layers[i].in0] * 4
                               : vsize[layers[i].in0];
            off += 22;
        }
    }
    uint32_t effective_out = out_vid;
    if (vsize[out_vid] == 0) {
        for (int j = (int)nlayers - 1; j >= 0; --j) {
            if (layers[j].out < nvals && vsize[layers[j].out] > 0) {
                effective_out = layers[j].out; break;
            }
        }
        printf("out has no size, fallback to v%u (%u elems)\n", effective_out, vsize[effective_out]);
    }
    out_vid = effective_out;
    printf("sizes: out=%u elems\n", vsize[out_vid]); fflush(stdout);

    Buf* vb = calloc(nvals, sizeof(Buf));
    for (uint32_t i = 0; i < nvals; ++i) {
        if (vsize[i] == 0) continue;
        if (mkbuf((VkDeviceSize)vsize[i] * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &vb[i])) {
            printf("buf alloc fail v[%u] (%u elems)\n", i, vsize[i]); return 1;
        }
    }
    printf("buffers ok\n"); fflush(stdout);
    memcpy(vb[in_vid].map, inp, isz);

    VkCommandBufferBeginInfo beg = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHK(vkBeginCommandBuffer(cmd, &beg));

    for (uint32_t i = 0; i < nlayers; ++i) {
        Layer* L = &layers[i];
        if (L->out >= nvals || vsize[L->out] == 0) continue;
        if (L->in0 < nvals && vsize[L->in0] == 0) continue;
        if (L->type == 1) {
            uint32_t w_elems = L->oc * L->ic * L->kh * L->kw;
            uint32_t b_elems = L->oc;
            Buf wb, bb;
            mkbuf((VkDeviceSize)w_elems*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &wb);
            mkbuf((VkDeviceSize)b_elems*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bb);
            memcpy(wb.map, md + wt_off + L->w_off, (size_t)w_elems*4);
            memcpy(bb.map, md + wt_off + L->b_off, (size_t)b_elems*4);
            uint32_t in_sz = vsize[L->in0];
            uint32_t spatial = in_sz / L->ic;
            uint32_t ih = (uint32_t)sqrt((double)spatial);
            uint32_t iw = spatial / ih;
            uint32_t oh = L->stride==2 ? (ih+1)/2 : ih;
            uint32_t ow = L->stride==2 ? (iw+1)/2 : iw;
            uint32_t push[14] = {1, L->ic, ih, iw, L->oc, oh, ow,
                                  L->kh, L->kw, L->stride, L->stride, L->pad, L->pad, L->silu};
            Buf* bufs[4] = {&vb[L->in0], &wb, &bb, &vb[L->out]};
            uint32_t total = ((L->oc+3)/4) * oh * ow;
            dispatch(0, bufs, 4, push, sizeof(push), (total + 127) / 128);
        } else if (L->type == 3) {
            uint32_t push[7] = {1, vsize[L->in0], 0, 0, 0, 0, 0};
            Buf* bufs[3] = {&vb[L->in0], &vb[L->in0], &vb[L->out]};
            dispatch(1, bufs, 3, push, sizeof(push), (vsize[L->in0]+127)/128);
        } else if (L->type == 5) {
            uint32_t push[7] = {3, vsize[L->in0], 0, 0, 0, 0, 0};
            Buf* bufs[3] = {&vb[L->in0], &vb[L->extra], &vb[L->out]};
            dispatch(1, bufs, 3, push, sizeof(push), (vsize[L->in0]+127)/128);
        } else if (L->type == 6) {
            uint32_t push[7] = {4, vsize[L->in0], 0, 0, 0, 0, 0};
            Buf* bufs[3] = {&vb[L->in0], &vb[L->in0], &vb[L->out]};
            dispatch(1, bufs, 3, push, sizeof(push), (vsize[L->in0]+127)/128);
        } else if (L->type == 4) {
            uint32_t push[7] = {2, vsize[L->in0], 0, 0, 0, 0, 0};
            Buf* bufs[3] = {&vb[L->in0], &vb[L->in0], &vb[L->out]};
            dispatch(1, bufs, 3, push, sizeof(push), (vsize[L->in0]+127)/128);
        } else if (L->type == 2) {
            CHK(vkEndCommandBuffer(cmd));
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount=1; si.pCommandBuffers=&cmd;
            CHK(vkQueueSubmit(queue,1,&si,fence));
            CHK(vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX));
            vkResetFences(dev,1,&fence);
            vkResetCommandBuffer(cmd,0);
            CHK(vkBeginCommandBuffer(cmd, &beg));
            float* dst = (float*)vb[L->out].map;
            uint64_t doff = 0;
            for (uint32_t j = 0; j < concat_n[i]; ++j) {
                uint32_t vid2 = concat_ins[i][j];
                memcpy((uint8_t*)dst + doff*4, vb[vid2].map, (size_t)vsize[vid2]*4);
                doff += vsize[vid2];
            }
        }
    }
    CHK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount=1; si.pCommandBuffers=&cmd;
    CHK(vkQueueSubmit(queue,1,&si,fence));
    CHK(vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX));
    printf("GPU done\n"); fflush(stdout);

    float* out = (float*)vb[out_vid].map;
    uint32_t out_elems = vsize[out_vid];
    FILE* fo = fopen(argv[3], "wb");
    fwrite(out, 4, out_elems, fo);
    fclose(fo);
    printf("output: %u elems\n", out_elems);
    float mx=-1e30f, mn=1e30f; for (uint32_t i=0;i<out_elems;++i) {if(out[i]>mx)mx=out[i];if(out[i]<mn)mn=out[i];}
    printf("range: [%.4f, %.4f]\n", mn, mx);
    printf("AEXRT-M mobile: PASS\n");
    return 0;
}
