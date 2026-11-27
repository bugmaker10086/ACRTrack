"""Ground truth and ignore-region geometry used in the tracking experiments."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import Sequence
import csv
import math
import numpy as np


def load_ground_truth(path: Path, frame_count: int) -> dict[int, np.ndarray]:
    rows: dict[int, list[list[float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(",")
            if len(parts) < 6:
                raise ValueError(f"Malformed GT row {path}:{line_number}")
            frame = int(float(parts[0]))
            if frame < 1 or frame > frame_count:
                continue
            track_id = int(float(parts[1]))
            x, y, width, height = (float(value) for value in parts[2:6])
            if width <= 0 or height <= 0 or not all(
                math.isfinite(value) for value in (x, y, width, height)
            ):
                raise ValueError(f"Invalid GT box {path}:{line_number}")
            rows[frame].append([x, y, x + width, y + height, track_id])
    return {
        frame: np.asarray(frame_rows, dtype=np.float64).reshape(-1, 5)
        for frame, frame_rows in rows.items()
    }


def load_ignore_regions(path: Path, frame_count: int) -> dict[int, np.ndarray]:
    rows: dict[int, list[list[float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "xmin", "ymin", "width", "height"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected ignore-region columns in {path}")
        for line_number, row in enumerate(reader, 2):
            frame = int(float(row["frame"]))
            if frame < 1 or frame > frame_count:
                continue
            x = float(row["xmin"])
            y = float(row["ymin"])
            width = float(row["width"])
            height = float(row["height"])
            if width <= 0 or height <= 0 or not all(
                math.isfinite(value) for value in (x, y, width, height)
            ):
                raise ValueError(f"Invalid ignore box {path}:{line_number}")
            rows[frame].append([x, y, x + width, y + height])
    return {
        frame: np.asarray(frame_rows, dtype=np.float64).reshape(-1, 4)
        for frame, frame_rows in rows.items()
    }


def union_intersection_area(box: Sequence[float],
                            regions: np.ndarray) -> float:
    """Exact continuous union area of ignore rectangles inside one box."""
    x1, y1, x2, y2 = (float(value) for value in box[:4])
    clipped: list[tuple[float, float, float, float]] = []
    for rx1, ry1, rx2, ry2 in np.asarray(regions, dtype=np.float64):
        cx1, cy1 = max(x1, rx1), max(y1, ry1)
        cx2, cy2 = min(x2, rx2), min(y2, ry2)
        if cx2 > cx1 and cy2 > cy1:
            clipped.append((cx1, cy1, cx2, cy2))
    if not clipped:
        return 0.0

    x_edges = sorted({value for rect in clipped for value in (rect[0], rect[2])})
    area = 0.0
    for left, right in zip(x_edges, x_edges[1:]):
        if right <= left:
            continue
        y_intervals = [
            (cy1, cy2) for cx1, cy1, cx2, cy2 in clipped
            if cx1 < right and cx2 > left
        ]
        if not y_intervals:
            continue
        y_intervals.sort()
        start, end = y_intervals[0]
        union_y = 0.0
        for next_start, next_end in y_intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                union_y += end - start
                start, end = next_start, next_end
        union_y += end - start
        area += (right - left) * union_y
    return area


def filter_ignored_boxes(boxes: np.ndarray, regions: np.ndarray | None,
                         threshold: float) -> tuple[np.ndarray, int]:
    if boxes.size == 0 or regions is None or regions.size == 0:
        return boxes, 0
    keep: list[int] = []
    dropped = 0
    for index, box in enumerate(boxes):
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        area = width * height
        ratio = 0.0 if area <= 0 else union_intersection_area(box[:4], regions) / area
        if ratio >= threshold:
            dropped += 1
        else:
            keep.append(index)
    if not keep:
        return np.empty((0, boxes.shape[1]), dtype=boxes.dtype), dropped
    return boxes[np.asarray(keep, dtype=int)], dropped


def sanitize_detections(detections: np.ndarray, min_box_area: float,
                        class_id: int) -> np.ndarray:
    array = np.asarray(detections, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 5), dtype=np.float64)
    array = array.reshape(-1, array.shape[-1])
    if array.shape[1] < 5:
        raise ValueError("Detector output must have at least five columns")
    if array.shape[1] >= 6:
        array = array[array[:, 5].astype(int) == class_id]
    if array.size == 0:
        return np.empty((0, 5), dtype=np.float64)
    array = array[:, :5]
    finite = np.all(np.isfinite(array), axis=1)
    widths = array[:, 2] - array[:, 0]
    heights = array[:, 3] - array[:, 1]
    valid = finite & (widths > 0) & (heights > 0) & (
        widths * heights >= min_box_area
    )
    return array[valid]
