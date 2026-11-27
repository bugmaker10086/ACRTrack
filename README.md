# ACRTrack

ACRTrack tracks vehicles across frames from detection boxes. It extends ByteTrack with adaptive aspect-ratio process noise (ANI), center-assisted association (CAA), and low-score recovery (LSR). This repository contains the tracker used for the ACRTrack row of the tracking comparison and the eight configurations of the ablation study.

## Run the example

Use Python 3.12. From the repository root, in a fresh Python environment:

```sh
python -m pip install -r requirements.txt
python -m examples.demo
python acrtrack.py runs/demo.npz --output runs/demo_tracks.txt
```

The example generates eight synthetic frames. Frame 5 has no detection. Frame 6 has a low-confidence detection. The full tracker outputs ID 1 on frames 1–4 and 6–8. The example checks this automatically. The tracker and this example run on CPU with NumPy, SciPy and LAP.

## Track a sequence

```sh
python acrtrack.py my_sequence.npz --fps 25 --variant M1_M4_M5 --output runs/tracks.txt
```

One NPZ file represents one video. Store arrays without pickle.

| Array | Shape | Meaning |
|---|---|---|
| `frame_indices` | `(N,)` | One-based integer frame indices in nondecreasing order, one per detection |
| `detections` | `(N, 5)` | Original-image pixel coordinates `x1, y1, x2, y2` and confidence in [0, 1] |
| `frame_count` | `(1,)` | Total number of frames, including empty frames at the end |
| `image_shape` | `(2,)` | Image height, then width, in pixels |

The runner advances the tracker once per frame, including frames without detections. Output uses ten MOT-style columns: frame, track ID, left, top, width, height, and the fixed values `1,-1,-1,-1`. Coordinates are in original image pixels. Output boxes smaller than 10 square pixels are discarded. Each invocation starts a new ID sequence. For independent videos in parallel, use separate processes.

## Paper configurations

The configuration names match the keys in [configs/variants.json](configs/variants.json). M1 is CAA, M4 is ANI (called ANN in the original experiment configuration), and M5 is LSR.

| Preset | Enabled modules |
|---|---|
| `baseline_fixed10` | All three off |
| `M1`, `M4`, `M5` | CAA only, ANI only, LSR only, respectively |
| `M1_M4`, `M1_M5`, `M4_M5` | The corresponding pairs |
| `M1_M4_M5` | Full ACRTrack |

- **ANI.** The third component of the Cholesky-whitened innovation updates a per-track scale for the next prediction. The multiplier scales the process-noise standard deviations of the aspect ratio and its velocity. The paper uses a range of 1–15, a gate of 0.03 and an EMA coefficient of 0.5. With ANI off, all Table 5 presets use a fixed multiplier of 10.
- **CAA.** The high-score association stage uses a center weight of 0.05 with squared-distance gate 5.9915, minimum IoU 0.05 and maximum symmetric size ratio 3.0. The low-score association stage retains its original distance.
- **LSR.** The full tracker uses `CoreLostLowScoreRecovery` for eligible lost tracks and unmatched low-score detections. Its gates are defined in [the recovery module](bytetrack/modules/lost_low_score_core.py). The extra score-noise and gap-state switches inherited from the source are disabled in every paper preset.

**Table 4 and Table 5 have different baselines.** Table 4 uses the upstream ByteTrack implementation with native motion noise. Table 5 starts from the modular implementation with fixed ×10 aspect-noise standard deviations. Use `baseline_fixed10` to reproduce the ablation baseline. The other trackers in Table 4 are separate implementations.

## Evaluate on UA-DETRAC

The paper evaluates 40 test sequences (56,340 frames) with a fixed YOLOv5n detection input. The file [ua_test40.json](configs/ua_test40.json) records each sequence name, dimensions, frame count, frame rate and input file digests. This is a fixed-detector MOT evaluation. The official UA-DETRAC PR-MOTA sweep uses a different protocol.

1. Obtain the UA-DETRAC data under its distribution terms. For the CSV-annotated archives used in the experiments, place `ua_detrac_training_set.zip` and `ua_detrac_test_set.zip` in `data/ua/archives/`, then run `python -m tools.prepare_ua --root data/ua`. The converter checks the archive digests and writes sequence images, ground truth and ignore-region files under `data/ua/prepared/`. If your distribution is in another format, supply ground truth and ignore regions in the formats below.
2. Place one detection cache per sequence in `data/ua/caches/`, named exactly as in the sequence manifest, with extension `.npz`. Alternatively generate caches with the detector adapter described below.
3. Obtain the external metric source used by the experiments. From the repository root run `git clone https://github.com/noahcao/OC_SORT.git external/OC_SORT`. The evaluator reads six TrackEval files under its `trackeval/` directory and verifies their SHA-256 digests before scoring. The digests in [metric_files.json](configs/metric_files.json) identify the reference files independently of checkout line endings.
4. Run the full evaluation:

```sh
python -m evaluation.ua --metrics external/OC_SORT/trackeval --caches data/ua/caches --gt data/ua/prepared/tracking_gt/test --ignore data/ua/prepared/tracking_ignore/test --variant M1_M4_M5 --output runs/ua_full.json
```

Use `--sequence detrac_61_MVI_39031` for a single-sequence check. Add `--verify-inputs` if you have the original input files. It checks byte-level hashes, so an NPZ written with a different compressor can fail the check even when its arrays match.

Ground truth is a headerless MOT-style CSV (first six columns: frame, identity, left, top, width, height). The per-sequence ignore CSV has a header `frame,xmin,ymin,width,height`. UA caches already exclude detections with at least 50% of their area covered by the union of ignore regions. The evaluator also removes predicted boxes by the same rule before scoring. HOTA, CLEAR and Identity results are combined across sequences by TrackEval. The five accuracy scores are reported on a 0–100 scale.

| Full ACRTrack, original UA inputs | MOTA | HOTA | DetA | AssA | IDF1 | ID switches |
|---|---:|---:|---:|---:|---:|---:|
| 40 sequences | 64.228 | 64.194 | 57.036 | 72.761 | 75.155 | 728 |

Reference results for all eight presets are in [ua_reference.json](configs/ua_reference.json). To evaluate a different dataset, use `--manifest your_sequences.json` with a list of records containing `name, frames, width, height, fps`. The loader accepts per-sequence `cache_file` and `gt_file` paths relative to the respective input directories. Omit `--ignore` for a custom dataset without ignore regions. Optional `gaps` entries with inclusive `first_frame` and `last_frame` exclude those frames from evaluation while keeping the tracker running.

## Generate detections from images (optional)

Supply a separately installed [YOLOv5](https://github.com/ultralytics/yolov5) source checkout, its Python dependencies (including PyTorch and OpenCV), and compatible trained weights. The adapter expects the YOLOv5 `DetectMultiBackend` and `scale_boxes` APIs and consecutive, numerically suffixed image filenames starting at 1. Run this command for each UA test sequence, changing its name in all three paths:

```sh
python -m tools.detect --dataset ua --images data/ua/prepared/images/test/detrac_61_MVI_39031 --ignore data/ua/prepared/tracking_ignore/test/detrac_61_MVI_39031.csv --yolov5-root external/yolov5 --weights data/ua_weights.pt --device cpu --output data/ua/caches/detrac_61_MVI_39031.npz
```

The repository includes synthetic inputs through the demo generator, configuration files, and input fingerprints. The industrial footage, detector checkpoints and original detection caches are outside this distribution. Exact agreement with the reference metrics requires the same detection arrays, annotations, ignore regions and sequence boundaries. [detectors.json](configs/detectors.json) records the checkpoint digests and inference settings for the two datasets. Another checkpoint can be used to evaluate a new detector–tracker combination.

## Credits

ACRTrack builds on [ByteTrack](https://github.com/FoundationVision/ByteTrack) by Zhang et al. (ECCV 2022). The upstream license is retained in [licenses/ByteTrack-MIT.txt](licenses/ByteTrack-MIT.txt). The evaluation uses [TrackEval](https://github.com/JonathonLuiten/TrackEval) via the OC-SORT distribution. External software and datasets retain their own licenses and terms.
