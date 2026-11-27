"""Generate one tracking cache from an ordered directory of frames and YOLOv5 weights."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from acrtrack import CONFIG
from evaluation.annotations import load_ignore_regions, filter_ignored_boxes, sanitize_detections
from tools.yolo_detector import YOLOv5Detector


def frame_number(path):
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"An image filename must end in its frame number: {path.name}")
    return int(match[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "weights", "yolov5-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dataset", choices=["ua", "balanced141"], required=True)
    parser.add_argument("--ignore", type=Path, help="Per-frame ignore-region CSV for UA-DETRAC")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.dataset == "ua" and args.ignore is None:
        parser.error("UA-DETRAC requires an ignore CSV, with a header even when empty")
    if args.output.suffix != ".npz":
        parser.error("Use an .npz output filename")
    import cv2
    images = sorted((p for p in args.images.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}),
                    key=frame_number)
    if not images or [frame_number(p) for p in images] != list(range(1,len(images)+1)):
        parser.error("Provide consecutive images numbered from 1")
    cfg = json.loads((CONFIG / "detectors.json").read_text())[args.dataset]
    model = YOLOv5Detector(args.yolov5_root.resolve(), args.weights.resolve(), args.device,
                           cfg["imgsz"], cfg["confidence"], cfg["nms_iou"], cfg["max_det"])
    ignore = load_ignore_regions(args.ignore, len(images)) if args.ignore else {}
    frames, boxes = [], []
    shape = None
    for i, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        if image is None or (shape is not None and image.shape[:2] != shape):
            raise ValueError(f"Unreadable frame or inconsistent dimensions: {path}")
        shape = image.shape[:2]
        rows = sanitize_detections(model.detect(image), cfg["min_box_area"], cfg["class_id"])
        rows, _ = filter_ignored_boxes(rows, ignore.get(i), cfg["ignore_overlap_threshold"])
        frames.extend([i] * len(rows))
        boxes.extend(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, frame_indices=np.asarray(frames, dtype=np.int32),
                        detections=np.asarray(boxes, dtype=np.float32).reshape(-1, 5),
                        frame_count=np.array([len(images)], dtype=np.int32),
                        image_shape=np.array(shape, dtype=np.int32))


if __name__ == "__main__":
    main()
