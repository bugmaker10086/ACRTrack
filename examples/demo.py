"""Generate an artificial one-target sequence and run the full tracker on it."""
import argparse
from pathlib import Path

import numpy as np

from acrtrack import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("runs/demo.npz"))
    args = parser.parse_args()
    if args.cache.suffix != ".npz":
        parser.error("Use an .npz filename")
    frames = np.array([1, 2, 3, 4, 6, 7, 8], dtype=np.int32)
    boxes = np.array([[20+f, 24, 60+f, 50, 0.32 if f == 6 else 0.92]
                      for f in frames], dtype=np.float32)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, frame_indices=frames, detections=boxes,
                        frame_count=np.array([8], dtype=np.int32),
                        image_shape=np.array([96, 160], dtype=np.int32))
    seen = {}
    for frame, rows in replay(args.cache):
        seen[frame] = [int(v) for v in rows[:, 4]]
        print(f"frame {frame}: {seen[frame]}")
    if seen[1] != [1] or seen[5] or seen[6] != [1] or seen[8] != [1]:
        raise RuntimeError("Synthetic identity recovery check failed")


if __name__ == "__main__":
    main()
