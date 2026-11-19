# AEXRT-M Android 集成指南

本文件说明如何把 AEXRT-M 作为**进程内库**集成到 Android 应用（无 JVM 依赖的
C API），以及已验证的构建/冒烟测试步骤。推理核心与 CLI 是同一份源码
（`mobile/runtime/yolo_flat.c`），量化/策略/内核行为完全一致。

## 1. 构建产物

```bash
py mobile/tools/build.py lib-android
```

产出：

| 文件 | 说明 |
|---|---|
| `build/mobile/libaexrt_m.so` | aarch64 共享库，导出 `aexrt_m_*`（无 `main`） |
| `build/mobile/dl_test` | dlopen 冒烟测试程序 |

头文件：`mobile/runtime/aexrt_m.h`。API：

```c
typedef struct { float x1, y1, x2, y2, score; int cls; } AexrtMDet;
typedef struct AexrtMSession AexrtMSession;

AexrtMSession *aexrt_m_open(const char *model_path, const char *policy_path,
                            int frames_in_flight, float conf_threshold);
int  aexrt_m_input_size(AexrtMSession *s);              /* 3*w*h */
void aexrt_m_shape(AexrtMSession *s, int *nc, int *w, int *h);
int  aexrt_m_run(AexrtMSession *s, const float *frame_nchw,
                 AexrtMDet *dets, int max_dets);
void aexrt_m_close(AexrtMSession *s);
```

## 2. 真机冒烟测试（已验证）

```bash
adb push build/mobile/libaexrt_m.so /data/local/tmp/
adb push build/mobile/dl_test        /data/local/tmp/
adb push build/tmp/cs2v8_full.bin    /data/local/tmp/
adb push build/tmp/cs2_input.f32     /data/local/tmp/
adb shell "cd /data/local/tmp && chmod +x dl_test && \
           ./dl_test ./libaexrt_m.so cs2v8_full.bin cs2_input.f32 0.25"
```

Mali-G925 实测输出（与 CLI 结果一致）：

```
dl: input 307200 elems 320x320 classes 4
dl: 11 detections
  c0 0.911 [254.5 65.1 320.1 241.6]
  c1 0.851 [136.9 56.7 161.6 87.2]
  ...
```

## 3. App 接入步骤

1. 把 `libaexrt_m.so` 放进 `app/src/main/jniLibs/arm64-v8a/`；
2. `System.loadLibrary("aexrt_m")`（或 `dlopen` 自己的 wrapper）；
3. 通过 JNI 调用四个函数即可。JNI wrapper 的最小形式：

```c
// app/src/main/cpp/aexrt_jni.c
#include <jni.h>
#include "aexrt_m.h"

JNIEXPORT jlong JNICALL
Java_com_example_aexrt_Engine_nativeOpen(JNIEnv *env, jclass c,
                                         jstring model, jstring policy, jfloat conf) {
    const char *m = (*env)->GetStringUTFChars(env, model, NULL);
    const char *p = policy ? (*env)->GetStringUTFChars(env, policy, NULL) : NULL;
    AexrtMSession *s = aexrt_m_open(m, p, 2, conf);
    if (p) (*env)->ReleaseStringUTFChars(env, policy, p);
    (*env)->ReleaseStringUTFChars(env, model, m);
    return (jlong)(intptr_t)s;
}

JNIEXPORT jint JNICALL
Java_com_example_aexrt_Engine_nativeRun(JNIEnv *env, jclass c, jlong h,
                                        jfloatArray frame, jfloatArray out) {
    AexrtMDet dets[64];
    jfloat *f = (*env)->GetFloatArrayElements(env, frame, NULL);
    int n = aexrt_m_run((AexrtMSession *)(intptr_t)h, f, dets, 64);
    (*env)->ReleaseFloatArrayElements(env, frame, f, JNI_ABORT);
    if (n > 0) {
        jfloat tmp[64 * 6];
        for (int i = 0; i < n; ++i) {
            tmp[i*6+0] = dets[i].x1; tmp[i*6+1] = dets[i].y1;
            tmp[i*6+2] = dets[i].x2; tmp[i*6+3] = dets[i].y2;
            tmp[i*6+4] = dets[i].score; tmp[i*6+5] = (float)dets[i].cls;
        }
        (*env)->SetFloatArrayRegion(env, out, 0, n * 6, tmp);
    }
    return n;
}
```

（这段 JNI 代码未在本仓库构建——本机没有 JDK；C API 路径已用 `dl_test`
在真机验证。）

## 4. 调用契约

- **单进程单会话**：`aexrt_m_open` 只能成功一次；Vulkan 设备、pipeline、
  录制好的命令图常驻，`aexrt_m_close` 只释放句柄（进程生命周期内不销毁
  设备，符合 app engine 用法）。
- **同步**：`aexrt_m_run` 写 cached staging → 提交 → 等待 → per-class NMS，
  返回该帧结果；不要多线程并发调用同一会话。
- **输入**：`float[3*w*h]`，NCHW，RGB，范围与训练一致（通常 `[0,1]`）；
  尺寸由 `aexrt_m_input_size` 给出。
- **阈值**：`conf_threshold <= 0` 使用内置 0.25；`policy_path = NULL` 使用
  逐层 auto 规则（建议设备端用 `--emit-policy` 固化的 map）。
- **预热**：`aexrt_m_open` 内含一次零帧推理预热，首帧不会出现 2× 冷启动。

## 5. 性能参考（OnePlus / Mali-G925，cs2V8_320）

| 模式 | 延迟 | 吞吐 |
|---|---|---|
| 库 API（同步单帧，含 NMS） | ~8.2 ms 中位（best 6.4） | ~122 fps |
| CLI `--pipeline 2`（吞吐基准） | — | 5.37 ms/帧，186 fps |

同步 API 的延迟包含提交/栅栏/NMS；需要更高吞吐时应在应用层做双缓冲
（或后续的异步 submit/poll API）。
