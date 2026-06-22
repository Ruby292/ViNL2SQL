# NL2SQL

Repo này có 2 pipeline Text-to-SQL:

1. `zero_shot` — baseline chạy trực tiếp trên schema đầy đủ.
2. `schema_linking/dict_list` — pipeline lọc schema bằng dictionary list trước khi gọi model.

Cả hai pipeline đều dùng chung Spider evaluation wrapper qua `shared/spider_eval.py`, và phần scoring gốc vẫn nằm trong `spider_repo/`.

## Repository layout

```text
E:\NL2SQL
├── data/
│   ├── vispider_data/         # ViSpider JSON, gold SQL, tables.json
│   └── spider_db/             # SQLite databases của Spider
├── shared/
│   └── spider_eval.py         # Wrapper eval dùng chung cho mọi pipeline
├── spider_repo/               # Spider evaluator gốc, vendored
├── zero_shot/
│   ├── run_zero_shot.py       # Baseline pipeline
│   └── prompts.py             # Prompt/schema formatting utilities
├── schema_linking/
│   └── dict_list/
│       ├── to_dict_list.py    # Build dict_list.json từ spider_db
│       ├── candidate_filter.py
│       ├── prompt_builder.py
│       ├── qwen_client.py
│       └── run_pipeline.py    # End-to-end schema-linking pipeline
├── tests/
└── requirements.txt
```

## Data requirements

Cần có sẵn:

- `data/vispider_data/vispider_dev.json`
- `data/vispider_data/vispider_train.json`
- `data/vispider_data/vispider_test.json`
- `data/vispider_data/dev_gold.sql`
- `data/vispider_data/train_gold.sql`
- `data/vispider_data/test_gold.sql`
- `data/vispider_data/tables.json`
- `data/spider_db/` với đầy đủ SQLite databases

## Environment

- Python 3.10+
- PyTorch compatible với CUDA trên máy chạy inference
- `vllm` nếu muốn chạy model thật

Cài dependencies:

```bash
pip install -r requirements.txt
```

Nếu chạy inference bằng vLLM, model mặc định là `Qwen/Qwen2.5-Coder-7B-Instruct`.

## Shared evaluation

Cả hai pipeline đều ghi ra cùng một contract:

- `predictions.txt` — mỗi dòng một SQL
- `gold.txt` — mỗi dòng `sql\tdb_id`
- `output.json` — kết quả đánh giá

Wrapper dùng chung ở `shared/spider_eval.py` gọi Spider evaluator gốc trong `spider_repo/`, nên không cần copy file evaluation sang từng pipeline.

## Pipeline 1 — zero_shot

Baseline pipeline chạy trực tiếp trên schema đầy đủ.

### Run

```bash
python -m zero_shot.run_zero_shot \
  --dataset vispider \
  --split dev \
  --model Qwen/Qwen2.5-Coder-7B-Instruct
```

### Useful flags

- `--limit N` — chạy nhanh trên N example đầu
- `--disable-exec` — chỉ tính EM, bỏ EX
- `--predictions-input path/to/predictions.txt` — bỏ qua inference, chỉ evaluation
- `--max-model-len 4096` — context length
- `--gpu-memory-utilization 0.7` — mức dùng VRAM cho vLLM

### Outputs

- `zero_shot/results/{dataset}_{split}_predictions.txt`
- `zero_shot/results/{dataset}_{split}_gold.txt`
- `zero_shot/results/{dataset}_{split}_zeroshot.json`

## Pipeline 2 — schema_linking/dict_list

Pipeline này build một schema dictionary offline, lọc candidate schema theo câu hỏi, rồi mới tạo prompt cho model.

### Step 1: Build dictionary list

```bash
python -m schema_linking.dict_list.to_dict_list \
  --db_root data/spider_db \
  --output schema_linking/dict_list/results/dict_list.json
```

### Step 2: Run pipeline

```bash
python -m schema_linking.dict_list.run_pipeline \
  --dev data/vispider_data/vispider_dev.json \
  --dict_list schema_linking/dict_list/results/dict_list.json \
  --output_dir schema_linking/dict_list/results \
  --backend vllm
```

### Notes

- Pipeline hiện tại dùng `Qwen/Qwen2.5-Coder-7B-Instruct` qua vLLM.
- `candidate_filter.py` chỉ giữ schema đủ nhỏ để prompt gọn hơn, nhưng vẫn có fallback an toàn khi signal yếu.
- Output vẫn đi qua `shared/spider_eval.py` để so sánh công bằng với `zero_shot`.

### Outputs

- `schema_linking/dict_list/results/dict_list.json`
- `schema_linking/dict_list/results/predictions.txt`
- `schema_linking/dict_list/results/gold.txt`
- `schema_linking/dict_list/results/output.json`

## Verify the repo

Chạy test mock và kiểm tra contract cơ bản:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Expected workflow

1. Build `dict_list.json` từ Spider DB.
2. Chạy `zero_shot` để lấy baseline.
3. Chạy `schema_linking/dict_list` để so sánh với baseline.
4. Dùng cùng `output.json` format để đối chiếu EM/EX.
