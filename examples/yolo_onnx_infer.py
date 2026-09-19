import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aexrt import InferenceSession, OnnxInferenceSession, load_onnx, yolo_postprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="models/cs2V8_320.onnx")
    parser.add_argument("--mode", choices=["onnx", "ir"], default="onnx")
    parser.add_argument("--backend", default="torch")
    parser.add_argument("--device", default="dml")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--compare-ort", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--max-detections", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "onnx":
        session = OnnxInferenceSession(args.model, device=args.device)
        info = session.info()
        input_info = session.inputs[0]
        input_name = input_info.name
        shape = tuple(int(x) for x in input_info.shape)
        output_name = session.outputs[0].name
    else:
        graph = load_onnx(args.model)
        input_name, spec = next(iter(graph.inputs.items()))
        shape = tuple(int(x) for x in spec.shape)
        output_name = graph.outputs[0]
        session = InferenceSession(graph, backend=args.backend, device=args.device, optimize=False)
        info = {"providers": [f"{args.backend}:{args.device}"]}

    x = np.random.default_rng(0).random(shape, dtype=np.float32)
    use_gpu_yolo = args.mode == "ir" and args.backend in {"native_d3d12", "d3d12"}

    if use_gpu_yolo:
        for _ in range(3):
            detections = session.run_yolo_gpu(
                {input_name: x},
                output_name=output_name,
                max_candidates=args.max_candidates,
                max_detections=args.max_detections,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
            )

        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            detections = session.run_yolo_gpu(
                {input_name: x},
                output_name=output_name,
                max_candidates=args.max_candidates,
                max_detections=args.max_detections,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
            )
            times.append((time.perf_counter() - t0) * 1000.0)

        print(f"AEXRT mode={args.mode} providers={info['providers']} gpu_yolo_avg_ms={sum(times)/len(times):.3f} min_ms={min(times):.3f}")
        print(f"gpu detections conf>={args.conf}: {len(detections)}")
        for det in detections[:10]:
            print(f"  cls={int(det[5])} score={float(det[4]):.4f} xyxy={[round(float(v), 2) for v in det[:4]]}")
        if not args.compare_ort:
            return

    for _ in range(3):
        out = session.run({input_name: x})

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        out = session.run({input_name: x})
        times.append((time.perf_counter() - t0) * 1000.0)

    y = out[output_name]
    print(f"AEXRT mode={args.mode} providers={info['providers']} avg_ms={sum(times)/len(times):.3f} min_ms={min(times):.3f}")
    print(f"output {output_name} shape={y.shape} dtype={y.dtype} min={float(y.min()):.6g} max={float(y.max()):.6g} mean={float(y.mean()):.6g}")
    detections = yolo_postprocess(y, conf_threshold=args.conf, iou_threshold=args.iou)
    print(f"detections conf>={args.conf}: {len(detections)}")
    for det in detections[:10]:
        print(f"  cls={det.class_id} score={det.score:.4f} xyxy={[round(x, 2) for x in det.xyxy]}")

    if args.compare_ort:
        import onnxruntime as ort

        ort_session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        y_ort = ort_session.run(None, {input_name: x})[0]
        diff = np.abs(y - y_ort)
        print(f"ORT diff max={float(diff.max()):.6g} mean={float(diff.mean()):.6g} p99={float(np.quantile(diff, 0.99)):.6g}")


if __name__ == "__main__":
    main()
