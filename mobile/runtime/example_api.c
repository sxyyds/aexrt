/* Minimal client of the AEXRT-M library API.
 *
 *   example_api <model.bin> <dump.bin> <frame0.f32> [frame1.f32 ...]
 *
 * Each frame is fp32 NCHW. dump.bin receives the last frame's detections in
 * the CLI's format (u32 count + 6 floats per detection), so a gate can
 * compare the API path against the CLI byte-for-byte.
 */
#include "aexrt_m.h"

#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
#endif

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.bin> <dump.bin> <frame0.f32> [frame1 ...]\n", argv[0]);
        return 2;
    }
    const char *cs = getenv("AEXRT_API_CONF");
    float conf = cs ? (float)atof(cs) : 0.0f;
    AexrtMSession *s = aexrt_m_open(argv[1], NULL, 2, conf);
    if (!s) { fprintf(stderr, "aexrt_m_open failed\n"); return 1; }
    const int elems = aexrt_m_input_size(s);
    int classes = 0, w = 0, h = 0;
    aexrt_m_shape(s, &classes, &w, &h);
    printf("api: input %d elems, %dx%d, %d classes\n", elems, w, h, classes);
    float *frame = malloc((size_t)elems * sizeof(float));
    AexrtMDet dets[256];
    int last = 0;
    for (int i = 3; i < argc; ++i) {
        FILE *f = fopen(argv[i], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[i]); return 1; }
        if (fread(frame, sizeof(float), (size_t)elems, f) != (size_t)elems) {
            fprintf(stderr, "short frame %s\n", argv[i]);
            return 1;
        }
        fclose(f);
        const double t0 = now_ms();
        last = aexrt_m_run(s, frame, dets, 256);
        const double t1 = now_ms();
        if (last < 0) { fprintf(stderr, "run failed\n"); return 1; }
        printf("frame %d: %d detections in %.2f ms\n", i - 3, last, t1 - t0);
        for (int j = 0; j < last && j < 5; ++j)
            printf("  cls %d conf %.4f [%.1f %.1f %.1f %.1f]\n", dets[j].cls,
                   dets[j].score, dets[j].x1, dets[j].y1, dets[j].x2, dets[j].y2);
    }
    FILE *d = fopen(argv[2], "wb");
    if (!d) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    unsigned int n = (unsigned int)(last > 0 ? last : 0);
    fwrite(&n, 4, 1, d);
    for (unsigned int i = 0; i < n; ++i) {
        const float rec[6] = {dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2,
                              dets[i].score, (float)dets[i].cls};
        fwrite(rec, 4, 6, d);
    }
    fclose(d);
    aexrt_m_close(s);
    return 0;
}
