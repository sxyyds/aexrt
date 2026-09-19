import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aexrt import save_aexrt_engine_from_onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", help="output .aexrt engine")
    parser.add_argument("--classes", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    save_aexrt_engine_from_onnx(
        args.model,
        args.output,
        classes=args.classes,
        max_candidates=args.max_candidates,
        max_detections=args.max_detections,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
