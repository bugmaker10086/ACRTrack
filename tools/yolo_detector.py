"""Adapter for a separately installed YOLOv5 source checkout and detector weights."""
from pathlib import Path
import sys
import numpy as np


class YOLOv5Detector:
    def __init__(self, yolov5_root: Path, weights: Path, device: str,
                 imgsz: int, conf: float, nms_iou: float,
                 max_det: int) -> None:
        if str(yolov5_root) not in sys.path:
            sys.path.insert(0, str(yolov5_root))
        try:
            import torch
            from models.common import DetectMultiBackend
            from utils.dataloaders import letterbox
            from utils.general import check_img_size, non_max_suppression, scale_boxes
            from utils.torch_utils import select_device
        except ImportError as exc:
            raise RuntimeError(
                f"Failed to import YOLOv5 from {yolov5_root}: {exc}"
            ) from exc

        self.torch = torch
        self.letterbox = letterbox
        self.non_max_suppression = non_max_suppression
        self.scale_boxes = scale_boxes
        self.device = select_device(device)
        self.model = DetectMultiBackend(str(weights), device=self.device)
        self.stride = self.model.stride
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.conf = conf
        self.nms_iou = nms_iou
        self.max_det = max_det
        warm_size = self.imgsz if isinstance(self.imgsz, (list, tuple)) else (
            self.imgsz, self.imgsz
        )
        self.model.warmup(imgsz=(1, 3, int(warm_size[0]), int(warm_size[1])))

    def detect(self, image: np.ndarray) -> np.ndarray:
        resized = self.letterbox(
            image, self.imgsz, stride=self.stride, auto=True
        )[0]
        tensor_image = resized.transpose((2, 0, 1))[::-1]
        tensor_image = np.ascontiguousarray(tensor_image)
        tensor = self.torch.from_numpy(tensor_image).to(self.device)
        tensor = tensor.float() / 255.0
        if tensor.ndim == 3:
            tensor = tensor[None]
        with self.torch.inference_mode():
            prediction = self.model(tensor)
            prediction = self.non_max_suppression(
                prediction,
                self.conf,
                self.nms_iou,
                max_det=self.max_det,
            )
            rows: list[list[float]] = []
            for detections in prediction:
                if len(detections):
                    detections[:, :4] = self.scale_boxes(
                        tensor.shape[2:], detections[:, :4], image.shape
                    )
                    for x1, y1, x2, y2, score, class_index in detections.tolist():
                        rows.append([x1, y1, x2, y2, score, class_index])
        if not rows:
            return np.empty((0, 6), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)
