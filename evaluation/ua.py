"""Evaluate ACRTrack on the UA-DETRAC test sequences with the paper's MOT protocol."""
import argparse
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np

from acrtrack import CONFIG, FULL, VARIANTS, replay
from .annotations import load_ground_truth, load_ignore_regions, filter_ignored_boxes


def get_metrics(folder):
    """Load the six metric source files from the reference OC-SORT TrackEval copy."""
    hashes = json.loads((CONFIG / "metric_files.json").read_text())
    for name, digest in hashes.items():
        data = (folder / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Metric implementation differs from the paper: {name}")
    for name, root in (("acr_metrics", folder), ("acr_metrics.metrics", folder / "metrics")):
        module = types.ModuleType(name)
        module.__path__ = [str(root.resolve())]
        sys.modules[name] = module
    # These legacy TrackEval files predate NumPy 2.
    for name, builtin in (("float", float), ("int", int), ("bool", bool)):
        if name not in vars(np):
            setattr(np, name, builtin)
    classes = [(importlib.import_module("acr_metrics.metrics." + name), cls)
               for name, cls in (("hota", "HOTA"), ("clear", "CLEAR"), ("identity", "Identity"))]
    hota, clear, identity = [getattr(module, cls) for module, cls in classes]
    options = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    return [hota(), clear(options), identity(options)]


def similarities(gt, tracks):
    corner = np.maximum(gt[:, None, :2], tracks[None, :, :2])
    size = np.maximum(0, np.minimum(gt[:, None, 2:4], tracks[None, :, 2:4]) - corner)
    intersection = size.prod(axis=2)
    gt_area = (gt[:, 2:4]-gt[:, :2]).prod(axis=1)
    tr_area = (tracks[:, 2:4]-tracks[:, :2]).prod(axis=1)
    return intersection / np.maximum(gt_area[:, None] + tr_area[None, :] - intersection, 1e-12)


def prepare_for_metrics(sequence, gt, predictions):
    excluded = {f for gap in sequence.get("gaps", [])
                for f in range(gap["first_frame"], gap["last_frame"]+1)}
    frames = [f for f in range(1, sequence["frames"]+1) if f not in excluded]
    ids_gt = sorted({int(r[4]) for f in frames for r in gt.get(f, ())})
    ids_pr = sorted({int(r[4]) for f in frames for r in predictions[f]})
    gm = {v: i for i, v in enumerate(ids_gt)}
    pm = {v: i for i, v in enumerate(ids_pr)}
    data = dict(seq=sequence["name"], num_timesteps=len(frames), num_gt_ids=len(gm),
                num_tracker_ids=len(pm), num_gt_dets=0, num_tracker_dets=0,
                gt_ids=[], tracker_ids=[], similarity_scores=[])
    for frame in frames:
        truth = gt.get(frame, np.empty((0, 5)))
        rows = predictions[frame]
        gids = np.array([gm[int(v)] for v in truth[:, 4]], dtype=int)
        tids = np.array([pm[int(v)] for v in rows[:, 4]], dtype=int)
        if len(tids) != len(set(tids)) or len(gids) != len(set(gids)):
            raise ValueError(f"Duplicate identity at frame {frame}")
        data["gt_ids"].append(gids)
        data["tracker_ids"].append(tids)
        data["num_gt_dets"] += len(truth)
        data["num_tracker_dets"] += len(rows)
        data["similarity_scores"].append(similarities(truth, rows))
    return data


def score(seq, cache, ground_truth, ignore_file, variant, metrics):
    gt = load_ground_truth(ground_truth, seq["frames"])
    ignore = load_ignore_regions(ignore_file, seq["frames"]) if ignore_file is not None else {}
    predictions = {}
    dimensions = (seq["frames"], seq["height"], seq["width"])
    for frame, rows in replay(cache, variant, seq["fps"], metadata=dimensions):
        predictions[frame], _ = filter_ignored_boxes(rows, ignore.get(frame), 0.5)
    data = prepare_for_metrics(seq, gt, predictions)
    return [metric.eval_sequence(data) for metric in metrics]


def summarize(results, metrics):
    hota, clear, identity = [m.combine_sequences({name: r[j] for name, r in results.items()})
                             for j, m in enumerate(metrics)]
    return {"MOTA": float(100*clear["MOTA"]), "HOTA": float(100*np.mean(hota["HOTA"])),
            "DetA": float(100*np.mean(hota["DetA"])), "AssA": float(100*np.mean(hota["AssA"])),
            "IDF1": float(100*identity["IDF1"]), "FP": int(clear["CLR_FP"]),
            "FN": int(clear["CLR_FN"]), "IDSw": int(clear["IDSW"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for opt in ("metrics", "caches", "gt"):
        ap.add_argument("--" + opt, required=True, type=Path)
    ap.add_argument("--ignore", type=Path, help="Per-frame ignore CSVs; required for UA test40")
    ap.add_argument("--manifest", type=Path, help="Custom sequence list; defaults to UA test40")
    ap.add_argument("--variant", default=FULL, choices=VARIANTS)
    ap.add_argument("--sequence", help="One sequence name from configs/ua_test40.json")
    ap.add_argument("--verify-inputs", action="store_true", help="Require the original file hashes")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    metrics = get_metrics(args.metrics)
    if args.manifest is None and args.ignore is None:
        ap.error("UA test40 requires --ignore")
    sequences = json.loads((args.manifest or CONFIG / "ua_test40.json").read_text(encoding="utf-8-sig"))
    if not sequences or len({s["name"] for s in sequences}) != len(sequences):
        ap.error("The manifest must contain uniquely named sequences")
    if args.sequence:
        sequences = [s for s in sequences if s["name"] == args.sequence]
        if not sequences:
            ap.error("Unknown sequence")
    results = {}
    for seq in sequences:
        name = seq["name"]
        files = {"cache": args.caches / seq.get("cache_file", name + ".npz"),
                 "gt": args.gt / seq.get("gt_file", name + ".txt")}
        if args.ignore is not None:
            files["ignore"] = args.ignore / seq.get("ignore_file", name + ".csv")
        if args.verify_inputs:
            for kind, path in files.items():
                if hashlib.sha256(path.read_bytes()).hexdigest() != seq[kind+"_sha256"]:
                    raise ValueError(f"Input digest mismatch: {path}")
        results[name] = score(seq, files["cache"], files["gt"], files.get("ignore"), args.variant, metrics)
    summary = {"variant": args.variant, "sequences": len(sequences), **summarize(results, metrics)}
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
