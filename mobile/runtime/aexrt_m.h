/* AEXRT-M in-process inference API.
 *
 * The single-file runtime (yolo_flat.c) is normally built as a CLI. Defining
 * AEXRT_M_NO_MAIN and linking that translation unit turns it into a library:
 *
 *   AexrtMSession *s = aexrt_m_open("model.bin", "policy.txt", 2);
 *   float frame[3*320*320];                  // fp32 NCHW, [0,1] or normalized
 *   AexrtMDet dets[64];
 *   int n = aexrt_m_run(s, frame, dets, 64);
 *   aexrt_m_close(s);
 *
 * Notes:
 *  - one session per process (the runtime owns one Vulkan device/context);
 *  - aexrt_m_run is synchronous: it writes the frame into a cached staging
 *    buffer, submits the pre-recorded graph, waits, and runs per-class NMS;
 *  - pass policy_path = NULL to use the built-in per-layer auto rule;
 *  - conf_threshold <= 0 keeps the built-in 0.25 decode threshold;
 *  - frames_in_flight > 1 allocates independent value arenas (future
 *    asynchronous mode); the synchronous call still waits per frame.
 */
#ifndef AEXRT_M_H
#define AEXRT_M_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x1, y1, x2, y2;
    float score;
    int cls;
} AexrtMDet;

typedef struct AexrtMSession AexrtMSession;

/* Returns NULL if the model cannot be loaded or the GPU lacks the required
 * features (fp16 storage buffers; shaderIntegerDotProduct for AXM3 models). */
AexrtMSession *aexrt_m_open(const char *model_path, const char *policy_path,
                            int frames_in_flight, float conf_threshold);

/* Number of fp32 elements the input expects (3*w*h), or 0 if the session is
 * invalid. */
int aexrt_m_input_size(AexrtMSession *session);

/* Class count and input width/height; any output pointer may be NULL. */
void aexrt_m_shape(AexrtMSession *session, int *classes, int *width, int *height);

/* Runs one frame and writes up to max_dets detections (score-descending,
 * per-class NMS applied). Returns the number of detections, or -1 on error. */
int aexrt_m_run(AexrtMSession *session, const float *frame_nchw,
                AexrtMDet *dets, int max_dets);

void aexrt_m_close(AexrtMSession *session);

#ifdef __cplusplus
}
#endif

#endif /* AEXRT_M_H */
