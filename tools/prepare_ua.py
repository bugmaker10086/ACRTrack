#!/usr/bin/env python3
"""Prepare UA-DETRAC for one-class YOLOv5 training and MOT evaluation.

The script is deterministic and resumable. It keeps the downloaded archives
unchanged, extracts only JPEG images and the source CSV annotations, performs a
sequence-level train/validation split, and generates YOLO and MOT-format labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


EXPECTED_ARCHIVES = {
    "train": {
        "name": "ua_detrac_training_set.zip",
        "size": 5_630_247_258,
        "sha256": "7045ffc28bb247248e2c16f8329c7131751fe1135bc96e327aef8816508ffaf9",
    },
    "test": {
        "name": "ua_detrac_test_set.zip",
        "size": 4_252_838_102,
        "sha256": "ab018f74b2ed84bee47593871771887b771c7d79b6fb4fea0be5a89bfba77edd",
    },
}


@dataclass(frozen=True)
class SequenceSource:
    name: str
    image_members: tuple[zipfile.ZipInfo, ...]
    annotation_member: zipfile.ZipInfo


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_archive(archive_path: Path) -> tuple[dict[str, SequenceSource], int]:
    grouped_images: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    annotations: dict[str, zipfile.ZipInfo] = {}
    ignored = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            if len(parts) != 2:
                ignored += 1
                continue
            sequence, filename = parts
            lowered = filename.lower()
            if lowered.endswith((".jpg", ".jpeg")):
                grouped_images[sequence].append(member)
            elif lowered == "annotations.csv":
                annotations[sequence] = member
            else:
                ignored += 1

    missing_csv = sorted(set(grouped_images) - set(annotations))
    extra_csv = sorted(set(annotations) - set(grouped_images))
    if missing_csv or extra_csv:
        raise RuntimeError(
            f"archive structure mismatch: missing_csv={missing_csv}, extra_csv={extra_csv}"
        )

    sequences = {
        name: SequenceSource(
            name=name,
            image_members=tuple(
                sorted(grouped_images[name], key=lambda item: item.filename.lower())
            ),
            annotation_member=annotations[name],
        )
        for name in sorted(grouped_images)
    }
    return sequences, ignored


def choose_validation_sequences(
    sequences: dict[str, SequenceSource], count: int, seed: int
) -> set[str]:
    if not 0 < count < len(sequences):
        raise ValueError("validation sequence count must be between 1 and N-1")

    ranked = sorted(
        sequences.values(), key=lambda item: (len(item.image_members), item.name)
    )
    rng = random.Random(seed)
    selected: set[str] = set()
    total = len(ranked)
    for stratum in range(count):
        start = math.floor(stratum * total / count)
        end = math.floor((stratum + 1) * total / count)
        candidates = ranked[start:end]
        if not candidates:
            raise RuntimeError(f"empty validation stratum {stratum}")
        selected.add(rng.choice(candidates).name)
    if len(selected) != count:
        raise RuntimeError("validation selection did not produce the requested count")
    return selected


def safe_extract_member(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, destination: Path
) -> bool:
    """Extract one member atomically. Return True when a file was written."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == member.file_size:
        return False

    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    with archive.open(member, "r") as source, partial.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    if partial.stat().st_size != member.file_size:
        raise IOError(
            f"extracted size mismatch for {member.filename}: "
            f"{partial.stat().st_size} != {member.file_size}"
        )
    partial.replace(destination)
    return True


def parse_frame_id(filename: str) -> int:
    match = re.search(r"(\d+)$", Path(filename).stem)
    if not match:
        raise ValueError(f"cannot parse frame number from {filename}")
    return int(match.group(1))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(text, encoding="utf-8", newline="\n")
    partial.replace(path)


def load_annotations(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict]:
    raw = archive.read(member)
    text = raw.decode("utf-8-sig", errors="strict")
    rows_by_image: dict[str, list[dict]] = defaultdict(list)
    ignored_by_image: dict[str, list[dict]] = defaultdict(list)
    stats = {
        "rows": 0,
        "target_rows": 0,
        "ignored_rows": 0,
        "invalid_boxes": 0,
        "clamped_boxes": 0,
        "dimension_conflicts": 0,
        "classes": Counter(),
    }
    dimensions: dict[str, tuple[int, int]] = {}

    reader = csv.DictReader(io.StringIO(text))
    required = {
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "confidence",
        "target_id",
    }
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        raise ValueError(
            f"unexpected annotation columns in {member.filename}: {reader.fieldnames}"
        )

    for row in reader:
        stats["rows"] += 1
        filename = row["filename"].strip()
        width = int(float(row["width"]))
        height = int(float(row["height"]))
        if width <= 0 or height <= 0:
            raise ValueError(f"non-positive dimensions in {member.filename}: {row}")

        previous_dims = dimensions.setdefault(filename, (width, height))
        if previous_dims != (width, height):
            stats["dimension_conflicts"] += 1

        original = tuple(float(row[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
        xmin = min(max(original[0], 0.0), float(width))
        ymin = min(max(original[1], 0.0), float(height))
        xmax = min(max(original[2], 0.0), float(width))
        ymax = min(max(original[3], 0.0), float(height))
        clamped = (xmin, ymin, xmax, ymax)
        if any(abs(a - b) > 1e-9 for a, b in zip(original, clamped)):
            stats["clamped_boxes"] += 1
        if xmax <= xmin or ymax <= ymin:
            stats["invalid_boxes"] += 1
            continue

        class_name = (row["class"].strip() or "vehicle").lower()
        stats["classes"][class_name] += 1
        parsed = {
            "width": width,
            "height": height,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "confidence": float(row["confidence"]),
            "target_id": int(float(row["target_id"])),
            "class": class_name,
        }
        if class_name == "ignored" or parsed["target_id"] < 0:
            ignored_by_image[filename].append(parsed)
            stats["ignored_rows"] += 1
        elif class_name == "vehicle":
            rows_by_image[filename].append(parsed)
            stats["target_rows"] += 1
        else:
            raise ValueError(
                f"unsupported target class {class_name!r} in {member.filename}"
            )

    stats["classes"] = dict(sorted(stats["classes"].items()))
    return rows_by_image, ignored_by_image, stats


def prepare_sequence(
    archive: zipfile.ZipFile,
    source_split: str,
    role: str,
    sequence: SequenceSource,
    prepared_root: Path,
) -> dict:
    image_dir = prepared_root / "images" / source_split / sequence.name
    label_dir = prepared_root / "labels" / source_split / sequence.name
    source_csv_path = (
        prepared_root / "source_annotations" / source_split / f"{sequence.name}.csv"
    )
    mot_path = prepared_root / "tracking_gt" / source_split / f"{sequence.name}.txt"
    ignore_path = (
        prepared_root / "tracking_ignore" / source_split / f"{sequence.name}.csv"
    )

    raw_csv = archive.read(sequence.annotation_member)
    source_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not source_csv_path.exists() or source_csv_path.read_bytes() != raw_csv:
        partial_csv = source_csv_path.with_name(source_csv_path.name + ".part")
        partial_csv.write_bytes(raw_csv)
        partial_csv.replace(source_csv_path)

    rows_by_image, ignored_by_image, annotation_stats = load_annotations(
        archive, sequence.annotation_member
    )
    image_names = {PurePosixPath(item.filename).name for item in sequence.image_members}
    orphan_annotation_frames = sorted(set(rows_by_image) - image_names)
    orphan_ignore_frames = sorted(set(ignored_by_image) - image_names)
    if orphan_annotation_frames or orphan_ignore_frames:
        raise RuntimeError(
            f"{sequence.name} annotations reference missing images: "
            f"targets={orphan_annotation_frames[:5]}, ignored={orphan_ignore_frames[:5]}"
        )

    extracted = 0
    image_paths: list[Path] = []
    mot_rows: list[tuple] = []
    ignore_rows: list[tuple] = []
    empty_frames = 0
    boxes_written = 0

    for member in sequence.image_members:
        filename = PurePosixPath(member.filename).name
        image_path = image_dir / filename
        if safe_extract_member(archive, member, image_path):
            extracted += 1
        image_paths.append(image_path.resolve())

        labels: list[str] = []
        annotations = rows_by_image.get(filename, [])
        if not annotations:
            empty_frames += 1
        frame_id = parse_frame_id(filename)
        for item in ignored_by_image.get(filename, []):
            ignore_rows.append(
                (
                    frame_id,
                    item["xmin"],
                    item["ymin"],
                    item["xmax"] - item["xmin"],
                    item["ymax"] - item["ymin"],
                )
            )
        for item in annotations:
            box_width = item["xmax"] - item["xmin"]
            box_height = item["ymax"] - item["ymin"]
            center_x = item["xmin"] + box_width / 2.0
            center_y = item["ymin"] + box_height / 2.0
            normalized = (
                center_x / item["width"],
                center_y / item["height"],
                box_width / item["width"],
                box_height / item["height"],
            )
            if not all(0.0 <= value <= 1.0 for value in normalized):
                raise ValueError(
                    f"normalized label outside [0,1] in {sequence.name}/{filename}: "
                    f"{normalized}"
                )
            labels.append("0 " + " ".join(f"{value:.8f}" for value in normalized))
            mot_rows.append(
                (
                    frame_id,
                    item["target_id"],
                    item["xmin"],
                    item["ymin"],
                    box_width,
                    box_height,
                    1,
                    1,
                    1,
                )
            )
            boxes_written += 1

        label_path = label_dir / f"{Path(filename).stem}.txt"
        atomic_write_text(label_path, "\n".join(labels) + ("\n" if labels else ""))

    mot_rows.sort(key=lambda row: (row[0], row[1]))
    mot_text = "".join(
        f"{frame},{target},{xmin:.6f},{ymin:.6f},{width:.6f},{height:.6f},"
        f"{confidence},{class_id},{visibility}\n"
        for (
            frame,
            target,
            xmin,
            ymin,
            width,
            height,
            confidence,
            class_id,
            visibility,
        ) in mot_rows
    )
    atomic_write_text(mot_path, mot_text)

    ignore_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    ignore_text = "frame,xmin,ymin,width,height\n" + "".join(
        f"{frame},{xmin:.6f},{ymin:.6f},{width:.6f},{height:.6f}\n"
        for frame, xmin, ymin, width, height in ignore_rows
    )
    atomic_write_text(ignore_path, ignore_text)

    return {
        "sequence": sequence.name,
        "source_split": source_split,
        "detector_role": role,
        "frames": len(sequence.image_members),
        "boxes": boxes_written,
        "ignored_boxes": len(ignore_rows),
        "empty_frames": empty_frames,
        "extracted_files": extracted,
        "annotation_rows": annotation_stats["rows"],
        "invalid_boxes": annotation_stats["invalid_boxes"],
        "clamped_boxes": annotation_stats["clamped_boxes"],
        "dimension_conflicts": annotation_stats["dimension_conflicts"],
        "classes": annotation_stats["classes"],
        "image_paths": [path.as_posix() for path in image_paths],
    }


def write_outputs(
    prepared_root: Path,
    sequence_results: list[dict],
    seed: int,
    archive_records: dict,
    elapsed_seconds: float,
) -> None:
    split_dir = prepared_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    paths_by_role: dict[str, list[str]] = defaultdict(list)
    sequences_by_role: dict[str, list[str]] = defaultdict(list)
    for result in sequence_results:
        role = result["detector_role"]
        paths_by_role[role].extend(result["image_paths"])
        sequences_by_role[role].append(result["sequence"])

    for role in ("train", "val", "test"):
        atomic_write_text(
            split_dir / f"{role}.txt",
            "\n".join(paths_by_role[role]) + "\n",
        )
        atomic_write_text(
            split_dir / f"{role}_sequences.txt",
            "\n".join(sorted(sequences_by_role[role])) + "\n",
        )

    manifest_path = split_dir / "sequence_manifest.csv"
    manifest_lines = io.StringIO(newline="")
    writer = csv.DictWriter(
        manifest_lines,
        fieldnames=[
            "sequence",
            "source_split",
            "detector_role",
            "frames",
            "boxes",
            "empty_frames",
            "invalid_boxes",
            "clamped_boxes",
            "ignored_boxes",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for result in sorted(sequence_results, key=lambda item: item["sequence"]):
        writer.writerow({key: result[key] for key in writer.fieldnames})
    atomic_write_text(manifest_path, manifest_lines.getvalue())

    root_posix = prepared_root.resolve().as_posix()
    yaml_text = (
        "# Generated by experiments/prepare_ua_detrac.py\n"
        f"path: {root_posix}\n"
        "train: splits/train.txt\n"
        "val: splits/val.txt\n"
        "test: splits/test.txt\n"
        "nc: 1\n"
        "names: ['vehicle']\n"
    )
    atomic_write_text(prepared_root / "ua_detrac.yaml", yaml_text)

    totals = {}
    for role in ("train", "val", "test"):
        items = [item for item in sequence_results if item["detector_role"] == role]
        totals[role] = {
            "sequences": len(items),
            "frames": sum(item["frames"] for item in items),
            "boxes": sum(item["boxes"] for item in items),
            "ignored_boxes": sum(item["ignored_boxes"] for item in items),
            "empty_frames": sum(item["empty_frames"] for item in items),
        }

    checks = {
        "sequence_roles_disjoint": not (
            set(sequences_by_role["train"]) & set(sequences_by_role["val"])
            or set(sequences_by_role["train"]) & set(sequences_by_role["test"])
            or set(sequences_by_role["val"]) & set(sequences_by_role["test"])
        ),
        "invalid_boxes": sum(item["invalid_boxes"] for item in sequence_results),
        "dimension_conflicts": sum(
            item["dimension_conflicts"] for item in sequence_results
        ),
        "classes": sorted(
            {
                class_name
                for item in sequence_results
                for class_name in item["classes"].keys()
            }
        ),
    }
    report = {
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": seed,
        "split_method": "sequence-level, frame-count-stratified, one sequence per stratum",
        "class_mapping": {"vehicle": 0},
        "excluded_from_training": ["ignored"],
        "archives": archive_records,
        "totals": totals,
        "checks": checks,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "sequences": [
            {key: value for key, value in item.items() if key != "image_paths"}
            for item in sequence_results
        ],
    }
    atomic_write_text(
        prepared_root / "preparation_report.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--val-sequences", type=int, default=12)
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip SHA-256 verification when archives were already independently verified.",
    )
    args = parser.parse_args()

    started = time.time()
    archive_dir = args.root / "archives"
    prepared_root = args.root / "prepared"
    prepared_root.mkdir(parents=True, exist_ok=True)

    archive_paths: dict[str, Path] = {}
    archive_records: dict[str, dict] = {}
    sources: dict[str, dict[str, SequenceSource]] = {}
    for source_split, expected in EXPECTED_ARCHIVES.items():
        archive_path = archive_dir / expected["name"]
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        actual_size = archive_path.stat().st_size
        if actual_size != expected["size"]:
            raise IOError(
                f"archive size mismatch for {archive_path}: "
                f"{actual_size} != {expected['size']}"
            )
        actual_hash = None
        if not args.skip_hash_check:
            print(f"[hash] {archive_path.name}", flush=True)
            actual_hash = sha256_file(archive_path)
            if actual_hash != expected["sha256"]:
                raise IOError(
                    f"archive SHA-256 mismatch for {archive_path}: {actual_hash}"
                )
        sequence_map, ignored = inspect_archive(archive_path)
        archive_paths[source_split] = archive_path
        sources[source_split] = sequence_map
        archive_records[source_split] = {
            "path": str(archive_path.resolve()),
            "size": actual_size,
            "sha256": actual_hash or expected["sha256"],
            "sha256_status": "verified_now" if actual_hash else "verified_before_run",
            "sequences": len(sequence_map),
            "images": sum(len(item.image_members) for item in sequence_map.values()),
            "ignored_members": ignored,
        }

    validation_sequences = choose_validation_sequences(
        sources["train"], args.val_sequences, args.seed
    )
    print(
        f"[split] train={len(sources['train']) - len(validation_sequences)} "
        f"val={len(validation_sequences)} test={len(sources['test'])}",
        flush=True,
    )

    results: list[dict] = []
    sequence_total = len(sources["train"]) + len(sources["test"])
    completed = 0
    for source_split in ("train", "test"):
        with zipfile.ZipFile(archive_paths[source_split]) as archive:
            for sequence_name, sequence in sources[source_split].items():
                role = (
                    "test"
                    if source_split == "test"
                    else ("val" if sequence_name in validation_sequences else "train")
                )
                result = prepare_sequence(
                    archive,
                    source_split,
                    role,
                    sequence,
                    prepared_root,
                )
                results.append(result)
                completed += 1
                print(
                    f"[prepare] {completed}/{sequence_total} {sequence_name} "
                    f"role={role} frames={result['frames']} boxes={result['boxes']} "
                    f"new_images={result['extracted_files']}",
                    flush=True,
                )

    elapsed = time.time() - started
    write_outputs(prepared_root, results, args.seed, archive_records, elapsed)
    print(f"[complete] {prepared_root} elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
