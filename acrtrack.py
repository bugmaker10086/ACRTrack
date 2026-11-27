"""Paper presets and frame-by-frame replay for one independent video sequence."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bytetrack.basetrack import BaseTrack
from bytetrack.byte_tracker import BYTETracker
from bytetrack.kalman_filter import KalmanFilter
from bytetrack.modules.lost_low_score_core import CoreLostLowScoreRecovery

CONFIG = Path(__file__).resolve().parent / "configs"
VARIANTS = json.loads((CONFIG / "variants.json").read_text(encoding="utf-8"))
FULL = "M1_M4_M5"


class PaperKalmanFilter(KalmanFilter):
    def __init__(self, preset):
        super().__init__(enable_aspect_adaptive=preset["M4"], aspect_scale_low=1.0,
                         aspect_scale_high=preset["high"], aspect_gate_thresh=preset["tau"],
                         aspect_ema_alpha=preset["alpha"])
        self.fixed_scale = None if preset["M4"] else preset["fixed_scale"]

    def _motion_covariance(self, mean, aspect_scale=1.0):
        scale = aspect_scale if self.fixed_scale is None else self.fixed_scale
        return super()._motion_covariance(mean, scale)


def make_tracker(variant=FULL, fps=25):
    """Start a sequence. IDs use the original process-wide ByteTrack counter."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    preset = VARIANTS[variant]
    if (preset["center_gate"], preset["center_min_iou"], preset["center_size"]) != (5.9915, 0.05, 3.0):
        raise ValueError("The frozen CAA implementation uses the paper gates")
    BaseTrack._count = 0
    tracker = BYTETracker(frame_rate=fps, center_weight=preset["center_weight"],
                          center_distance_mode="gated_rescue", enable_aspect_adaptive=preset["M4"],
                          enable_score_weight=False, enable_gap_recovery=False,
                          enable_lost_low_score=preset["M5"], kalman_filter=PaperKalmanFilter(preset))
    tracker.low_score_recovery = CoreLostLowScoreRecovery() if preset["M5"] else None
    return tracker


def replay(cache, variant=FULL, fps=25, *, metadata=None):
    """Yield one (1-based frame, N x 5 [x1,y1,x2,y2,id]) result per frame."""
    with np.load(cache, allow_pickle=False) as data:
        frames = data["frame_indices"]
        detections = data["detections"]
        if "frame_count" in data and "image_shape" in data:
            count, = data["frame_count"].reshape(-1)
            height, width = data["image_shape"].reshape(-1)
            if metadata is not None and tuple(metadata) != (count, height, width):
                raise ValueError("Sequence metadata disagrees with the cache")
        elif metadata is not None:
            count, height, width = metadata
        else:
            raise ValueError("Supply frame_count and image_shape in the cache")
    if any(int(v) != v or v < 1 for v in (count, height, width)):
        raise ValueError("Invalid frame count or image dimensions")
    if frames.ndim != 1 or detections.shape != (len(frames), 5):
        raise ValueError("Expected one frame index per 5-column detection")
    if (not np.issubdtype(frames.dtype, np.integer) or (frames < 1).any()
            or (frames > count).any() or (frames[1:] < frames[:-1]).any()):
        raise ValueError("Frame indices must be sorted integers within the sequence")
    if (not np.isfinite(detections).all() or (detections[:, 2:4] <= detections[:, :2]).any()
            or (detections[:, 4] < 0).any() or (detections[:, 4] > 1).any()):
        raise ValueError("Invalid box coordinates or detection confidence")
    tracker = make_tracker(variant, fps)
    boundaries = np.searchsorted(frames, np.arange(1, count + 2))
    for index in range(int(count)):
        boxes = detections[boundaries[index]:boundaries[index+1]]
        tracks = tracker.update(boxes, (int(height), int(width)))
        rows = np.asarray([list(t.tlbr) + [t.track_id] for t in tracks],
                          dtype=np.float64).reshape(-1, 5)
        areas = (rows[:, 2] - rows[:, 0]) * (rows[:, 3] - rows[:, 1])
        yield index + 1, rows[areas >= 10.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default=FULL)
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.cache.resolve():
        parser.error("Use separate input and output files")
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for frame, rows in replay(args.cache, args.variant, args.fps):
            for x1, y1, x2, y2, track_id in rows:
                writer.writerow([frame, int(track_id), x1, y1, x2-x1, y2-y1, 1, -1, -1, -1])


if __name__ == "__main__":
    main()
