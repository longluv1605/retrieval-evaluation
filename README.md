# Retrieval Evaluation

Repo này chuẩn hóa đánh giá retrieval theo hướng:
- Chạy retrieval trước để tạo `pairs.txt`.
- Từ `pairs.txt` + pose (nếu có), gán nhãn positive/negative theo ngưỡng khoảng cách `M`.
- Tính Recall@K và lưu kết quả ra JSON.

## 1) Phạm vi đánh giá
- Bài toán: với mỗi ảnh query, retrieve top-K ảnh map.
- Dữ liệu đầu vào tối thiểu: ảnh map/query + danh sách ảnh.
- Đánh giá theo khoảng cách vị trí giữa query và ảnh map retrieve được.
- Nếu distance <= `M` thì pair là positive, ngược lại negative.

## 2) Cấu trúc dataset chuẩn
```text
datasets/
  my_dataset_scene/
    map/
      images/
      sfm/                       # optional: pose map có thể đọc từ sfm model
      images.txt
      poses.txt                  # optional nếu đã có sfm
    query_day/
      images/
      images.txt
      query_names.txt            # optional: nếu thiếu thì dùng toàn bộ images.txt
      poses.txt                  # optional: cần cho evaluation theo distance
      other_sensors_data.txt     # optional
    query_night/
      images/
      images.txt
      query_names.txt
      poses.txt
```

## 3) Quy ước file dữ liệu

### 3.1 `images.txt` (bắt buộc)
- Mỗi dòng là một tên ảnh hoặc đường dẫn tương đối so với thư mục `images/`.
- Không có header.

Ví dụ:
```text
frame_000001.jpg
frame_000002.jpg
subdir/frame_000003.jpg
```

### 3.2 `query_names.txt` (tùy chọn)
- Mỗi dòng là một query name cần đánh giá.
- Nếu file này tồn tại, chỉ evaluate các ảnh nằm trong file này.
- Nếu không tồn tại, evaluate toàn bộ `query_*/images.txt`.

### 3.3 `poses.txt` (tùy chọn nhưng cần cho distance-based eval)
- Không header.
- Định dạng mỗi dòng:
`image_name tx ty tz qx qy qz qw`
- `image_name` phải khớp với tên trong `images.txt`.
- Đơn vị tọa độ: mét (m), cùng hệ trục giữa map và query.

Ví dụ:
```text
frame_000001.jpg 12.30 5.02 1.44 0.01 0.02 0.03 0.99
frame_000002.jpg 12.81 5.40 1.46 0.01 0.02 0.04 0.99
```

Ghi chú:
- Với map: có thể lấy pose từ `map/poses.txt` hoặc từ `map/sfm/`.
- Với query: nếu không có `query_*/poses.txt` thì không thể tính recall-distance chuẩn.

## 4) Output retrieval: `pairs.txt`
`pairs.txt` là output chuẩn của bước retrieval cho từng query set.

Định dạng:
- Không header.
- Mỗi dòng:
`query_name map_name score rank`
- `rank` bắt đầu từ 1 và tăng liên tục theo từng query.
- `score` càng cao càng giống (quy ước descending ranking).

Ví dụ:
```text
frame_q_0001.jpg frame_m_1021.jpg 0.9132 1
frame_q_0001.jpg frame_m_0908.jpg 0.9011 2
frame_q_0001.jpg frame_m_1103.jpg 0.8870 3
frame_q_0002.jpg frame_m_0220.jpg 0.9321 1
```

## 5) Protocol evaluation từ `pairs.txt`

### 5.1 Gán nhãn positive/negative
Cho mỗi pair `(query_name, map_name)`:
1. Lấy pose query `p_q` từ `query_*/poses.txt`.
2. Lấy pose map `p_m` từ `map/poses.txt` hoặc `map/sfm/`.
3. Tính khoảng cách Euclidean:
`d = ||p_q - p_m||_2`
4. Gán nhãn:
- positive nếu `d <= M`
- negative nếu `d > M`

### 5.2 Recall@K
- Với mỗi query, xét top-K candidates trong `pairs.txt`.
- Query được tính hit nếu tồn tại ít nhất 1 positive trong top-K.
- `Recall@K = (#query hit) / (#query được đánh giá)`.

Ghi chú:
- Nên báo cáo nhiều mốc K: ví dụ `K = [1, 5, 10, 20]`.
- Có thể báo cáo thêm distribution khoảng cách cho top-1/top-K.

## 6) Cấu trúc output thống nhất
Mọi kết quả lưu theo đường dẫn:
`outputs/<dataset>/<scene>/<query_set>/<retrieval_method>/<run_id>/`

Trong đó:
- `retrieval_method`: tên phương pháp retrieval (hiện có: `hloc-netvlad`, `hloc-megaloc`, `hloc-dir`).
- `run_id`: lần chạy cụ thể của cùng method (ví dụ: `exp_001`, `ablation_m5`).

Ví dụ:
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

Ý nghĩa chính:
- `retrieval/pairs.txt`: ranking thô từ model retrieval.
- `eval/evaluation.json`: metric tổng để so sánh run.
- `eval/per_query.jsonl`: kết quả từng query để debug.
- `eval/pairs_scored.csv`: pair + distance + label để visualize.
- `meta/stage_status.json`: hỗ trợ resume/skip.
- `meta/input_hashes.json`: quyết định có cần chạy lại không.

## 7) Schema `evaluation.json`
File này là artifact chính để tổng hợp benchmark.

Ví dụ:
```json
{
  "dataset": "my_dataset",
  "scene": "my_dataset_scene",
  "query_set": "query_day",
  "retrieval_method": "hloc-netvlad",
  "run_id": "exp_001",
  "retrieval_config": {
    "config_path": "configs/retrieval.yaml",
    "config_hash": "sha256:9d8f...",
    "checkpoint": "weights/dino_v2_base.pth",
    "descriptor_dim": 768,
    "similarity": "cosine",
    "index_type": "faiss_ivfpq",
    "top_k": 100
  },
  "distance_threshold_m": 5.0,
  "num_queries_total": 1200,
  "num_queries_evaluated": 1187,
  "num_queries_missing_pose": 13,
  "k_values": [1, 5, 10, 20],
  "recall": {
    "R@1": 0.41,
    "R@5": 0.62,
    "R@10": 0.70,
    "R@20": 0.77
  },
  "distance_stats_top1": {
    "mean": 8.31,
    "median": 4.92,
    "p90": 18.40
  },
  "files": {
    "pairs": "retrieval/pairs.txt",
    "per_query": "eval/per_query.jsonl",
    "pairs_scored": "eval/pairs_scored.csv"
  }
}
```

## 8) Schema `per_query.jsonl` (để debug và show pairs)
Mỗi dòng là 1 query:
```json
{"query_name":"frame_q_0001.jpg","evaluated":true,"best_rank_positive":2,"hit_at":{"1":false,"5":true,"10":true},"top1_distance_m":6.7}
{"query_name":"frame_q_0002.jpg","evaluated":false,"reason":"missing_query_pose"}
```

## 9) Schema `pairs_scored.csv` (để visualize)
- Có header.
- Cột đề xuất:
`query_name,map_name,score,rank,distance_m,is_positive,within_topk`

Ví dụ:
```csv
query_name,map_name,score,rank,distance_m,is_positive,within_topk
frame_q_0001.jpg,frame_m_1021.jpg,0.9132,1,6.70,0,1
frame_q_0001.jpg,frame_m_0908.jpg,0.9011,2,3.91,1,1
```

## 10) Resume/skip khi chạy lại
Mỗi stage (`retrieve`, `eval`) cần hỗ trợ `--resume` hoặc `--skip-existing`:
- Nếu output stage tồn tại và `input_hashes.json` không đổi, stage được skip.
- Nếu thay đổi model/config/data threshold `M`, stage tương ứng phải chạy lại.
- `stage_status.json` lưu: trạng thái, thời gian, input hash, output hash, lỗi (nếu có).

## 11) Các trường hợp không evaluate được
- Thiếu pose query hoặc pose map cho query/pair tương ứng.
- Pair tham chiếu tới ảnh không có trong `images.txt`.
- Hệ tọa độ map/query không đồng nhất.

Trong các trường hợp này:
- Không làm hỏng run toàn cục.
- Ghi rõ số lượng và lý do vào `evaluation.json` + `per_query.jsonl`.

## 12) Cài môi trường (đơn giản, không cần cài editable repo này)
```bash
cd src/retrieval-evaluation
python3 -m pip install -r requirements.txt

# Cài HLOC giống cách LaMAR dùng
git clone --recursive -b v1.4 https://github.com/cvg/Hierarchical-Localization.git hloc --depth=1
python3 -m pip install -e ./hloc
```

Ghi chú:
- Bạn có thể chạy trực tiếp bằng `python3 -m retrieval_eval.cli ...`.
- Không bắt buộc `pip install -e .` cho repo `retrieval-evaluation`.

## 13) CLI workflow tối thiểu
```bash
# 1) Chạy retrieval, sinh pairs.txt
python3 -m retrieval_eval.cli retrieve \
  --config configs/retrieval.hloc.yaml \
  --dataset-root datasets \
  --dataset my_dataset \
  --scene my_dataset_scene \
  --query-set query_day \
  --outputs-root outputs \
  --retrieval-method hloc-netvlad \
  --run-id exp_001

# 2) Evaluate theo ngưỡng M, sinh evaluation.json
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

# 3) Chạy 1 lệnh cả retrieval + eval
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

### Chạy nhanh bằng `run.sh` (style LaMAR)
```bash
cd src/retrieval-evaluation
chmod +x run.sh

# Mặc định chạy MODE=run cho QUERY_SETS="query_day"
./run.sh

# Ví dụ đổi method + nhiều query set
RETRIEVAL_METHOD=hloc-megaloc \
QUERY_SETS="query_day query_night" \
RUN_ID=exp_megaloc_001 \
THRESHOLD_M=5.0 \
./run.sh

# Chỉ chạy retrieval hoặc chỉ chạy eval
MODE=retrieve ./run.sh
MODE=eval ./run.sh
```

## 14) Definition of Done
Một run hợp lệ khi:
- Có `retrieval/pairs.txt` đúng format.
- Có `eval/evaluation.json` parse được và có Recall@K.
- Có `eval/per_query.jsonl` để truy vết query fail/hit.
- Có `eval/pairs_scored.csv` để show pair và kiểm chứng trực quan.
- Có `meta/manifest.yaml` + `meta/stage_status.json` để tái lập run.
