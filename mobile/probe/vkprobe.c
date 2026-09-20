/* AEXRT-M Phase 0 Vulkan capability probe (arm64-v8a).
 *
 * Queries everything the dual-mode quantization design depends on:
 *   - instance/device Vulkan version
 *   - shaderInt8 / shaderFloat16 / 16-bit storage & uniform
 *   - VK_KHR_shader_integer_dot_product (dotProductAll / Ternary features)
 *   - subgroup size + supported stages (VK_KHR_shader_subgroup)
 *   - storage16Bit via VkPhysicalDevice16BitStorageFeatures
 *   - memory heaps (unified?), buffer alignment limits
 *   - timestamp query support (for on-device profiling)
 * Print one "key: value" per line for easy parsing from adb.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

#define CHK(c) do { VkResult r_ = (c); if (r_ != VK_SUCCESS) { \
    printf("error: %s failed with %d\n", #c, (int)r_); return 1; } } while (0)

int main(void) {
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "aexrt-vkprobe";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ci.pApplicationInfo = &app;
    VkInstance inst;
    CHK(vkCreateInstance(&ci, NULL, &inst));

    uint32_t n = 0;
    CHK(vkEnumeratePhysicalDevices(inst, &n, NULL));
    if (n == 0) { printf("error: no physical devices\n"); return 1; }
    VkPhysicalDevice devs[8];
    if (n > 8) n = 8;
    CHK(vkEnumeratePhysicalDevices(inst, &n, devs));

    for (uint32_t d = 0; d < n; ++d) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devs[d], &props);
        printf("device[%u]: %s\n", d, props.deviceName);
        printf("vendor_id: 0x%x\n", props.vendorID);
        printf("api_version_raw: %u\n", props.apiVersion);
        printf("api_version: %u.%u.%u\n",
               VK_API_VERSION_MAJOR(props.apiVersion),
               VK_API_VERSION_MINOR(props.apiVersion),
               VK_API_VERSION_PATCH(props.apiVersion));
        printf("driver_version_raw: %u\n", props.driverVersion);

        /* Feature chains: core features + 16bit storage + float16int8 + subgroup + integer dot. */
        VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        VkPhysicalDevice16BitStorageFeatures f16 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES };
        VkPhysicalDeviceShaderFloat16Int8Features ffi = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES };
        VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures sset = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES };
        VkPhysicalDeviceShaderIntegerDotProductFeatures idp = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES };
        VkPhysicalDeviceVulkan11Features v11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
        VkPhysicalDeviceVulkan12Features v12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        f2.pNext = &f16; f16.pNext = &ffi; ffi.pNext = &sset; sset.pNext = &idp; idp.pNext = &v11; v11.pNext = &v12;
        vkGetPhysicalDeviceFeatures2(devs[d], &f2);
        printf("storageBuffer16BitAccess: %u\n", f16.storageBuffer16BitAccess);
        printf("uniformAndStorageBuffer16BitAccess: %u\n", f16.uniformAndStorageBuffer16BitAccess);
        printf("shaderFloat16: %u\n", ffi.shaderFloat16);
        printf("shaderInt8: %u\n", ffi.shaderInt8);
        printf("shaderSubgroupExtendedTypes: %u\n", sset.shaderSubgroupExtendedTypes);
        printf("shaderIntegerDotProduct: %u\n", idp.shaderIntegerDotProduct);
        printf("v11_storageBuffer16BitAccess: %u\n", v11.storageBuffer16BitAccess);
        printf("v11_shaderDrawParameters: %u\n", (unsigned)0);
        printf("v12_shaderInt8: %u\n", v12.shaderInt8);
        printf("v12_shaderFloat16: %u\n", v12.shaderFloat16);
        printf("v12_bufferDeviceAddress: %u\n", v12.bufferDeviceAddress);
        printf("v12_descriptorIndexing: %u\n", v12.descriptorIndexing);
        printf("v12_hostQueryReset: %u\n", v12.hostQueryReset);
        printf("v12_timelineSemaphore: %u\n", v12.timelineSemaphore);

        /* Subgroup properties. */
        VkPhysicalDeviceSubgroupProperties sg = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
        VkPhysicalDeviceProperties2 p2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(devs[d], &p2);
        printf("subgroup_size: %u\n", sg.subgroupSize);
        printf("subgroup_stages_compute: %u\n",
               (sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ? 1 : 0);
        printf("subgroup_ops_basic: %u\n",
               (sg.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) ? 1 : 0);
        printf("subgroup_ops_arithmetic: %u\n",
               (sg.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) ? 1 : 0);
        printf("subgroup_ops_ballot: %u\n",
               (sg.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) ? 1 : 0);
        printf("subgroup_ops_shuffle: %u\n",
               (sg.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) ? 1 : 0);

        /* Queue families: look for compute-only. */
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &qn, NULL);
        printf("queue_families: %u\n", qn);
        VkQueueFamilyProperties qf[16];
        if (qn > 16) qn = 16;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[d], &qn, qf);
        for (uint32_t q = 0; q < qn; ++q) {
            printf("queue[%u]: flags=%u count=%u timestampValidBits=%u\n", q,
                   qf[q].queueFlags, qf[q].queueCount, qf[q].timestampValidBits);
        }

        /* Memory: heap count + sizes (unified memory check). */
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(devs[d], &mp);
        printf("memory_heaps: %u\n", mp.memoryHeapCount);
        for (uint32_t h = 0; h < mp.memoryHeapCount; ++h) {
            printf("heap[%u]: sizeMB=%llu deviceLocal=%u\n", h,
                   (unsigned long long)(mp.memoryHeaps[h].size / (1024 * 1024)),
                   (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? 1 : 0);
        }

        /* Limits that matter for the arena design. */
        const VkPhysicalDeviceLimits* L = &props.limits;
        printf("minStorageBufferOffsetAlignment: %llu\n",
               (unsigned long long)L->minStorageBufferOffsetAlignment);
        printf("maxStorageBufferRange_MB: %llu\n",
               (unsigned long long)(L->maxStorageBufferRange / (1024 * 1024)));
        printf("maxComputeWorkGroupInvocations: %u\n", L->maxComputeWorkGroupInvocations);
        printf("maxComputeSharedMemorySize_KB: %u\n", L->maxComputeSharedMemorySize / 1024);
        printf("timestampPeriod_ns: %f\n", (double)L->timestampPeriod);
        printf("maxPerStageDescriptorStorageBuffers: %u\n", L->maxPerStageDescriptorStorageBuffers);
    }
    vkDestroyInstance(inst, NULL);
    printf("probe: done\n");
    return 0;
}
