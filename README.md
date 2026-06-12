# Retrieval Evaluation

This repository standardizes image retrieval benchmarking with a clear two-stage flow:
1. Run retrieval and generate `pairs.txt`.
2. Run distance-based evaluation and generate JSON metrics.

## 1) Evaluation Scope
- Task: retrieve top-K map images for each query image.
- Minimal inputs: map/query images and image lists.
- Distance-based labeling: a pair is positive when distance <= `M`.

## 2) Dataset Layout
```text
datasets/
  my_dataset_scene/
    map/
      images/
      sfm/                       # optional map poses from SfM model
      images.txt
      poses.txt                  # optional if poses come from sfm/
      timestamps.txt             # optional, required for sequence timing
    query_day/
      images/
      images.txt
      query_names.txt            # optional; fallback to images.txt
      timestamps.txt             # optional; fallback timeline if missing
      query_sequences.json       # optional; sequence groups for sequence methods
      poses.txt                  # optional; needed for distance-based eval
      other_sensors_data.txt     # optional
    query_night/
      images/
      images.txt
      query_names.txt
      timestamps.txt
      poses.txt
```

## 3) Data File Conventions

### 3.1 `images.txt` (required)
- One relative image path per line (relative to `images/`).
- No header.

Example:
```text
frame_000001.jpg
frame_000002.jpg
subdir/frame_000003.jpg
```

### 3.2 `query_names.txt` (optional)
- One query image name per line.
- If present, evaluation uses only these query names.
- If missing, evaluation uses all names from `query/images.txt`.

### 3.3 `poses.txt` (optional, required for distance evaluation)
- No header.
- Line format:
`image_name tx ty tz qx qy qz qw`
- `image_name` must exist in `images.txt`.
- Position unit is meters in a shared map/query coordinate frame.

Example:
```text
frame_000001.jpg 12.30 5.02 1.44 0.01 0.02 0.03 0.99
frame_000002.jpg 12.81 5.40 1.46 0.01 0.02 0.04 0.99
```

### 3.4 `timestamps.txt` (optional, used by sequence retrieval)
- Used by `seqprob` and `seqfg`.
- No header.
- Line format:
`image_name timestamp_value`
- `image_name` must exist in `images.txt`.

Example:
```text
frame_000001.jpg 1717392000.12
frame_000002.jpg 1717392000.46
frame_000003.jpg 1717392000.81
```

Timestamp unit handling:
- Sequence retrieval converts timestamps to seconds before computing `dt`.
- Configure the unit with `retrieval.sequence.timestamp_unit` in:
  `s | ms | us | ns`
- Default is `us` (microseconds), matching common LaMAR/Capture data.

Conversion to seconds:
- `s`: `dt_s = dt_raw`
- `ms`: `dt_s = dt_raw / 1e3`
- `us`: `dt_s = dt_raw / 1e6`
- `ns`: `dt_s = dt_raw / 1e9`

Fallback when timestamps are missing or invalid:
- Use a uniform timeline with `retrieval.sequence.dt_s_default` seconds per frame.
- Recommended default: `10.0` seconds per frame.

### 3.5 `query_sequences` annotation (optional, used by sequence retrieval)
- Used only by `seqprob` and `seqfg`.
- Supported file formats:
  - `query_sequences.txt`: each line is `image_name sequence_id`
  - `query_sequences.json`: sequence groups as lists
- If present, sequence optimization is run independently per group.
- If a query image is not annotated, it is processed as a singleton sequence.
- If the file is missing, all query images are treated as one global sequence.

JSON example (no manual sequence names needed):
```json
[
  ["frame_000001.jpg", "frame_000002.jpg", "frame_000003.jpg"],
  ["frame_000101.jpg", "frame_000102.jpg"]
]
```

TXT example:
```text
frame_000001.jpg seq_a
frame_000002.jpg seq_a
frame_000003.jpg seq_b
```

## 4) Retrieval Output: `pairs.txt`
- No header.
- One line per ranked pair:
`query_name map_name score rank`
- `rank` starts at 1 per query.
- `score` is sorted descending.

Example:
```text
frame_q_0001.jpg frame_m_1021.jpg 0.9132 1
frame_q_0001.jpg frame_m_0908.jpg 0.9011 2
frame_q_0001.jpg frame_m_1103.jpg 0.8870 3
frame_q_0002.jpg frame_m_0220.jpg 0.9321 1
```

## 5) Evaluation Protocol

### 5.1 Positive/Negative Labeling
For each pair `(query_name, map_name)`:
1. Load query pose `p_q` from `query_*/poses.txt`.
2. Load map pose `p_m` from `map/poses.txt` or `map/sfm/`.
3. Compute Euclidean distance `d = ||p_q - p_m||_2`.
4. Label pair:
   - positive if `d <= M`
   - negative if `d > M`

### 5.2 Recall@K
- A query is a hit if at least one positive pair appears in top-K.
- `Recall@K = hits / evaluated_queries`.

## 6) Standard Output Layout
All outputs are written to:
`outputs/<dataset>/<scene>/<query_set>/<retrieval_method>/<run_id>/`

Supported retrieval methods:
- `hloc-netvlad`
- `hloc-megaloc`
- `hloc-dir`
- `seqprob`
- `seqfg`
- `temporal-smooth`

Example:
```text
outputs/
  my_dataset/my_dataset_scene/query_day/hloc-netvlad/exp_001/
    retrieval/
      pairs.txt
    eval/
      evaluation.json
      per_query.jsonl
      pairs_scored.csv
    meta/
      manifest.yaml
      stage_status.json
      input_hashes.json
```

## 7) Main Artifacts

### 7.1 `evaluation.json`
Primary benchmark artifact with:
- dataset/run metadata
- config metadata
- recall values
- distance statistics
- artifact paths

### 7.2 `per_query.jsonl`
Per-query debug records.

### 7.3 `pairs_scored.csv`
Scored pair table with distance and binary label.

## 8) Resume/Skip Behavior
- Stages support resume/skip based on input hashes.
- If relevant inputs change, the stage is recomputed.
- Stage metadata is stored in `meta/stage_status.json`.

## 9) Sequence Retrieval Config
Use a retrieval config that contains `retrieval.sequence`.

Example:
```yaml
retrieval:
  global_model: megaloc
  top_k: 10
  overwrite_features: false
  sequence:
    sequence_file: query_sequences.json
    num_candidates: 50
    tau: 0.07
    v_max: 3.0
    eps: 0.25
    use_query_displacement: false
    lambda_dist: 1.0
    sigma_d: 1.0
    transition_weight: 1.0
    fusion: logprob_add
    fusion_weight: 0.05
    max_dt_s: 30.0
    timestamp_unit: us
    dt_s_default: 10.0
```

Rules:
- If `query_*/timestamps.txt` exists, real timestamps are used.
- If missing/invalid, a uniform fallback timeline is used.
- If `query_*/query_sequences.txt` or `query_*/query_sequences.json` exists, optimization runs per sequence group.
- Unannotated query images are processed as singleton sequences.
- Set `use_query_displacement: false` for timestamp-only retrieval. This is the recommended main setting when query poses are ground-truth labels used only for evaluation.
- Set `use_query_displacement: true` with `lambda_dist > 0` only for odometry-assisted experiments or upper-bound analysis.
- `transition_weight` is the pairwise strength beta in the factor graph.
- `fusion: logprob_add` uses `similarity + fusion_weight * log posterior`; `fusion: prob_add` reproduces the older `similarity + posterior` behavior.

### 9.1 Sequential Baselines and Analysis

`temporal-smooth` is a lightweight sequential baseline that averages global descriptor scores inside each sequence group. It uses:

```yaml
retrieval:
  global_model: megaloc
  top_k: 10
  sequence:
    sequence_file: query_sequences.json
    smooth_window_radius: 2
    smooth_weight: 1.0
```

Useful config files:
- `configs/retrieval.sequence.yaml`: timestamp-only SeqFG main setting.
- `configs/retrieval.sequence.pose_assisted.yaml`: query-displacement/odometry-assisted setting.
- `configs/retrieval.sequence.legacy.yaml`: older `similarity + posterior` behavior.
- `configs/retrieval.temporal_smooth.yaml`: moving-average sequential baseline.
- `configs/retrieval.hloc.top100.yaml` and `configs/eval.top100.yaml`: top-100 baseline run for oracle candidate-pool analysis.

Analyze promotions, losses, and oracle candidate-pool recall:

```bash
python3 tools/analyze_rerank_effects.py \
  --outputs-root ../outputs2 \
  --run-id <RUN_ID> \
  --baseline-method hloc-megaloc \
  --sequence-method seqfg \
  --oracle-k 100 \
  --out-json ../outputs2/<RUN_ID>/rerank_effects.json \
  --out-csv ../outputs2/<RUN_ID>/rerank_effects.csv
```

For oracle@100, the baseline run must write at least 100 retrieved pairs per query.

## 10) Environment Setup
```bash
cd src/retrieval-evaluation
python3 -m pip install -r requirements.txt

git clone --recursive -b v1.4 https://github.com/cvg/Hierarchical-Localization.git hloc --depth=1
python3 -m pip install -e ./hloc

# Optional for the Tkinter annotation app
python3 -m pip install pillow
```

## 10.1) Sequence Annotation App (Tkinter)
You can annotate query sequences interactively and export `query_sequences.json`.

Script:
- `tools/annotate_sequences_tk.py`

Features:
- Shows query images in timestamp order.
- Displays timestamp in raw value and human-readable UTC.
- `Add To Sequence`: add current image to active sequence and advance.
- `Save And Create Seq`: save active sequence and start a new one.
- Supports continue annotation: if output JSON already exists, annotated images are skipped.

Run:
```bash
python3 tools/annotate_sequences_tk.py \
  --query-root datasets/<dataset>/<scene>/<query_set> \
  --output-json datasets/<dataset>/<scene>/<query_set>/query_sequences.json \
  --timestamp-unit us \
  --dt-s-default 10.0
```

Output JSON format (`list of lists`):
```json
[
  ["frame_000001.jpg", "frame_000002.jpg", "frame_000003.jpg"],
  ["frame_000101.jpg", "frame_000102.jpg"]
]
```

## 11) Minimal CLI Workflow
```bash
# 1) Retrieval
python3 -m retrieval_eval.cli retrieve \
  --config configs/retrieval.hloc.yaml \
  --dataset-root datasets \
  --dataset my_dataset \
  --scene my_dataset_scene \
  --query-set query_day \
  --outputs-root outputs \
  --retrieval-method hloc-netvlad \
  --run-id exp_001

# 2) Distance evaluation
python3 -m retrieval_eval.cli eval-distance \
  --config configs/eval.yaml \
  --dataset-root datasets \
  --dataset my_dataset \
  --scene my_dataset_scene \
  --query-set query_day \
  --outputs-root outputs \
  --retrieval-method hloc-netvlad \
  --run-id exp_001 \
  --threshold-m 5.0

# 3) Full pipeline
python3 -m retrieval_eval.cli run \
  --retrieval-config configs/retrieval.hloc.yaml \
  --eval-config configs/eval.yaml \
  --dataset-root datasets \
  --dataset my_dataset \
  --scene my_dataset_scene \
  --query-set query_day \
  --outputs-root outputs \
  --retrieval-method hloc-netvlad \
  --run-id exp_001 \
  --threshold-m 5.0
```

## 12) Done Criteria
A run is valid when:
- `retrieval/pairs.txt` exists with correct format.
- `eval/evaluation.json` is valid and contains Recall@K.
- `eval/per_query.jsonl` exists for per-query debugging.
- `eval/pairs_scored.csv` exists for analysis.
- `meta/stage_status.json` exists for reproducibility.
