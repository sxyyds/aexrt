/* dlopen smoke test for libaexrt_m.so: verifies the shared library exports
 * the C API and runs one frame on the device without an app/JVM.
 *
 *   dl_test <libaexrt_m.so> <model.bin> <frame.f32> <conf>
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { float x1, y1, x2, y2, score; int cls; } Det;
typedef struct AexrtMSession AexrtMSession;

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <lib.so> <model.bin> <frame.f32> <conf>\n", argv[0]);
        return 2;
    }
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    AexrtMSession *(*op)(const char *, const char *, int, float) =
        (AexrtMSession * (*)(const char *, const char *, int, float)) dlsym(h, "aexrt_m_open");
    int (*isz)(AexrtMSession *) = (int (*)(AexrtMSession *))dlsym(h, "aexrt_m_input_size");
    void (*shp)(AexrtMSession *, int *, int *, int *) =
        (void (*)(AexrtMSession *, int *, int *, int *))dlsym(h, "aexrt_m_shape");
    int (*run)(AexrtMSession *, const float *, Det *, int) =
        (int (*)(AexrtMSession *, const float *, Det *, int))dlsym(h, "aexrt_m_run");
    void (*cls)(AexrtMSession *) = (void (*)(AexrtMSession *))dlsym(h, "aexrt_m_close");
    if (!op || !isz || !run || !cls) { fprintf(stderr, "missing symbols\n"); return 1; }
    AexrtMSession *s = op(argv[2], NULL, 2, (float)atof(argv[4]));
    if (!s) { fprintf(stderr, "open failed\n"); return 1; }
    int n = isz(s), classes = 0, w = 0, hh = 0;
    if (shp) shp(s, &classes, &w, &hh);
    printf("dl: input %d elems %dx%d classes %d\n", n, w, hh, classes);
    float *f = malloc((size_t)n * sizeof(float));
    FILE *fp = fopen(argv[3], "rb");
    if (!fp || fread(f, sizeof(float), (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "frame read failed\n");
        return 1;
    }
    fclose(fp);
    Det d[64];
    int k = run(s, f, d, 64);
    printf("dl: %d detections\n", k);
    for (int i = 0; i < k && i < 5; ++i)
        printf("  c%d %.3f [%.1f %.1f %.1f %.1f]\n", d[i].cls, d[i].score,
               d[i].x1, d[i].y1, d[i].x2, d[i].y2);
    cls(s);
    dlclose(h);
    return k < 0 ? 1 : 0;
}
