# ViNL2SQL

Pipeline zero-shot Text-to-SQL cho **ViSpider**: chạy Qwen2.5-Coder trên câu hỏi tiếng Việt, có thể bật contextual augmentation để thêm schema hints, sinh SQL, rồi chấm **Exact Match (EM)** và **Execution Accuracy (EX)** bằng Spider evaluator.

Pipeline zero-shot được tách thành 2 phase độc lập:

```text
Phase 1: inference + EM
  - load model
  - nếu bật augmentation: thêm schema hints vào prompt
  - sinh SQL
  - lưu prediction/raw output/gold
  - tính EM only
  - không chạy SQL trên SQLite

Phase 2: EX
  - không load model
  - đọc predictions.txt + gold.txt đã lưu
  - chạy pred_sql và gold_sql trên SQLite
  - có timeout per query để tránh treo evaluator
```

Contextual augmentation là bước tùy chọn: tách danh từ/cụm danh từ tiếng Việt, so semantic với table/column bằng `intfloat/multilingual-e5-large-instruct`, rồi thêm hint dạng `"danh từ" -> Table.column` vào prompt.

Mục tiêu: nếu server Vast.ai bị mất kết nối hoặc evaluator bị kẹt ở EX, kết quả inference của model vẫn đã được lưu an toàn từ Phase 1.

## Repository layout

```text
NL2SQL/
├── data/
│   ├── vispider_data/          # ViSpider json/gold/tables files
│   └── spider_db/              # <db_id>/<db_id>.sqlite
├── shared/
│   └── spider_eval.py          # wrapper EM/EX, timeout-safe execution
├── spider_repo/                # Spider evaluator gốc, không chỉnh sửa
├── zero_shot/
│   ├── run_zero_shot.py        # pipeline 2 phase, hỗ trợ --augment/--hints-input
│   ├── prompts.py              # baseline + augmented prompt, SQL extraction
│   ├── common.py               # load dataset/tables + write_json dùng chung
│   └── results/                # output local, ignored by git
├── augmentation/
│   ├── run_augment.py          # tạo hints.json + augment_stats.json riêng
│   ├── pipeline.py             # POS filter + schema similarity pipeline
│   ├── pos_filter.py           # tách danh từ/cụm danh từ tiếng Việt
│   ├── schema_nouns.py         # chuẩn hóa table/column thành schema nouns
│   ├── similarity.py           # E5 embedding + cosine similarity
│   ├── stats.py                # thống kê coverage của hints
│   └── results/                # output local, ignored by git
├── scripts/
│   └── run_qwen_compare.sh     # chạy baseline và augmented Qwen tuần tự
├── tests/
└── requirements.txt
```

## Data layout

Dữ liệu không nằm trong git. Sau khi copy/tải data, repo cần có cấu trúc:

```text
data/vispider_data/
├── vispider_dev.json
├── vispider_train.json
├── vispider_test.json
├── dev_gold.sql
├── train_gold.sql
├── test_gold.sql
├── tables.json
└── test_tables.json

data/spider_db/
├── concert_singer/
│   └── concert_singer.sqlite
├── singer/
│   └── singer.sqlite
└── <db_id>/
    └── <db_id>.sqlite
```

Pipeline resolve path theo file code, không phụ thuộc thư mục chạy lệnh:

```python
BASE_DIR = Path(__file__).parent.parent
SPIDER_DB = BASE_DIR / "data/spider_db"
VISPIDER_DIR = BASE_DIR / "data/vispider_data"
```

Nếu repo nằm ở `/workspace/NL2SQL` trên Vast.ai, path tương ứng là:

```text
/workspace/NL2SQL/data/spider_db
/workspace/NL2SQL/data/vispider_data
```

## Cài đặt môi trường

Khuyến nghị chạy trên Vast.ai hoặc GPU Linux có CUDA.

```bash
cd /workspace
git clone https://github.com/Ruby292/ViNL2SQL.git NL2SQL
cd NL2SQL

pip install -r requirements.txt
pip install -U vllm huggingface_hub
```

Nếu copy data thủ công từ máy local lên server bằng `scp`, cần đặt đúng vào:

```text
/workspace/NL2SQL/data/vispider_data
/workspace/NL2SQL/data/spider_db
```

Ví dụ kiểm tra nhanh trên server:

```bash
ls data/vispider_data/tables.json
ls data/vispider_data/vispider_dev.json
ls data/spider_db/concert_singer/concert_singer.sqlite
```

## Contextual augmentation pipeline

Augmentation tạo schema hints cho từng câu hỏi tiếng Việt trước khi sinh SQL:

```text
question_vi
  -> underthesea POS tag
  -> danh từ/cụm danh từ tiếng Việt
  -> encode bằng intfloat/multilingual-e5-large-instruct
  -> so với table/column đã normalize từ tables.json
  -> giữ match có similarity >= threshold
  -> thêm vào prompt dạng "danh từ" -> Table.column
```

Chạy augmentation riêng để tạo `hints.json` và `augment_stats.json`:

```bash
python -m augmentation.run_augment \
  --dataset vispider \
  --split dev \
  --limit 20 \
  --threshold 0.8 \
  --output augmentation/results/smoke/hints.json \
  --stats-output augmentation/results/smoke/augment_stats.json
```

`hints.json` có dạng:

```json
[
  {
    "index": 0,
    "db_id": "concert_singer",
    "question_vi": "...",
    "hints": [
      {
        "vi_noun": "ca sĩ",
        "schema_key": "singer.Name",
        "similarity": 0.8123
      }
    ]
  }
]
```

`augment_stats.json` lưu config, số example có hint, phân phối similarity, và top danh từ chưa match.

Có 2 cách dùng augmentation trong Phase 1:

1. Tính hints trực tiếp trong lúc inference:

```bash
python -m zero_shot.run_zero_shot \
  --mode inference \
  --dataset vispider \
  --split dev \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --augment \
  --augment-threshold 0.8 \
  --output zero_shot/results/qwen_compare_aug/7B/eval_em_only.json \
  --predictions-output zero_shot/results/qwen_compare_aug/7B/predictions.txt \
  --gold-output zero_shot/results/qwen_compare_aug/7B/gold.txt
```

2. Dùng lại hints đã tạo trước đó:

```bash
python -m zero_shot.run_zero_shot \
  --mode inference \
  --dataset vispider \
  --split dev \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --hints-input augmentation/results/smoke/hints.json \
  --output zero_shot/results/qwen_compare_aug/7B/eval_em_only.json \
  --predictions-output zero_shot/results/qwen_compare_aug/7B/predictions.txt \
  --gold-output zero_shot/results/qwen_compare_aug/7B/gold.txt
```

Khi dùng augmentation, `eval_em_only.json` vẫn lưu `hints` theo từng example để trace lại prompt context đã dùng.

## Chạy 4 model Qwen tuần tự

Script chính:

```bash
bash scripts/run_qwen_compare.sh
```

Mặc định chạy cả baseline và augmented cho 4 model:

```text
Qwen/Qwen2.5-Coder-0.5B-Instruct
Qwen/Qwen2.5-Coder-1.5B-Instruct
Qwen/Qwen2.5-Coder-3B-Instruct
Qwen/Qwen2.5-Coder-7B-Instruct
```

Với augmented run, script dùng `hints.json` đã tạo sẵn theo threshold:

```text
augmentation/results/<split>_t080/hints.json
augmentation/results/<split>_t085/hints.json
```

Tức là `run_qwen_compare.sh` truyền `--hints-input` vào Phase 1, không encode schema/question lại. Nếu muốn chỉ định file hints thủ công, dùng `AUGMENT_HINTS_INPUT=/path/to/hints.json`.

Mỗi biến thể/model chạy theo thứ tự:

```text
baseline Phase 1: inference + EM
baseline Phase 2: EX từ artifact Phase 1
augmented Phase 1: inference + EM với --hints-input
augmented Phase 2: EX từ artifact Phase 1
```

Nếu Phase 2 fail, các file Phase 1 vẫn được giữ nguyên.

Chỉ chạy baseline:

```bash
RUN_AUG=0 bash scripts/run_qwen_compare.sh
```

Chỉ chạy augmented:

```bash
RUN_BASELINE=0 bash scripts/run_qwen_compare.sh
```

Đổi threshold cho augmentation:

```bash
AUGMENT_THRESHOLD=0.85 bash scripts/run_qwen_compare.sh
```

Lệnh trên sẽ tìm `augmentation/results/dev_t085/hints.json` khi `SPLIT=dev`.

### Smoke test

Chạy nhanh vài câu trước khi full benchmark:

```bash
LIMIT=20 bash scripts/run_qwen_compare.sh
```

Chỉ chạy một vài model:

```bash
SIZES="0_5B 7B" bash scripts/run_qwen_compare.sh
```

Tăng timeout cho EX:

```bash
TIMEOUT_SECONDS=60 bash scripts/run_qwen_compare.sh
```

Đổi split:

```bash
SPLIT=dev bash scripts/run_qwen_compare.sh
```

## Output structure

Mỗi model có folder riêng trong baseline hoặc augmented result tree:

```text
zero_shot/results/qwen_compare/       # baseline
├── 0_5B/
├── 1_5B/
├── 3B/
└── 7B/

zero_shot/results/qwen_compare_aug/   # augmented
├── 0_5B/
├── 1_5B/
├── 3B/
└── 7B/
```

Trong mỗi folder model:

```text
zero_shot/results/qwen_compare/<size>/
├── predictions.txt       # 1 pred_sql / dòng
├── gold.txt              # {gold_sql}\t{db_id} / dòng
├── eval_em_only.json     # summary EM + per-example + raw_output
├── exec_details.json     # per-example EX/error/timeout
├── eval_ex.json          # summary EX + EM merged từ Phase 1
└── run.log               # log cả 2 phase
```

Cuối script có thêm:

```text
zero_shot/results/qwen_compare/summary.log
```

Ví dụ format summary:

```text
=== Baseline ===
   model      N        EM        EX
    0_5B   1034    0.xxxx    0.xxxx
    1_5B   1034    0.xxxx    0.xxxx
      3B   1034    0.xxxx    0.xxxx
      7B   1034    0.xxxx    0.xxxx

=== Augmented (threshold=0.8) ===
   model      N        EM        EX       dEM       dEX
    0_5B   1034    0.xxxx    0.xxxx   +0.xxxx   +0.xxxx
```

Nếu model chỉ chạy xong Phase 1, EX sẽ hiện `n/a` nhưng EM vẫn có.

## Chạy Phase 1 riêng: inference + EM

```bash
python -m zero_shot.run_zero_shot \
  --mode inference \
  --dataset vispider \
  --split dev \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output zero_shot/results/qwen_compare/7B/eval_em_only.json \
  --predictions-output zero_shot/results/qwen_compare/7B/predictions.txt \
  --gold-output zero_shot/results/qwen_compare/7B/gold.txt \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

Phase 1 sẽ lưu:

```text
predictions.txt
gold.txt
eval_em_only.json
```

`eval_em_only.json` gồm:

```json
{
  "summary": {
    "count": 1034,
    "exact_match": 0.0,
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "dataset": "vispider_dev",
    "timestamp": "..."
  },
  "by_difficulty": {},
  "examples": [
    {
      "id": 0,
      "example_id": "...",
      "db_id": "concert_singer",
      "question": "...",
      "gold_sql": "...",
      "pred_sql": "...",
      "raw_output": "...",
      "hardness": "easy",
      "exact_match": false
    }
  ]
}
```

## Chạy Phase 2 riêng: EX từ predictions đã lưu

```bash
python -m zero_shot.run_zero_shot \
  --mode exec \
  --predictions-input zero_shot/results/qwen_compare/7B/predictions.txt \
  --gold-input zero_shot/results/qwen_compare/7B/gold.txt \
  --em-input zero_shot/results/qwen_compare/7B/eval_em_only.json \
  --output zero_shot/results/qwen_compare/7B/eval_ex.json \
  --exec-details-output zero_shot/results/qwen_compare/7B/exec_details.json \
  --timeout-seconds 30
```

Phase 2 không load model và không cần GPU. Nó chỉ đọc artifact đã có, mở SQLite tương ứng với từng `db_id`, rồi chạy:

```text
pred_sql -> pred_result
gold_sql -> gold_result
```

Sau đó so sánh kết quả để tính EX.

`exec_details.json` có dạng:

```json
[
  {
    "id": 0,
    "db_id": "concert_singer",
    "pred_sql": "...",
    "gold_sql": "...",
    "exec_match": true,
    "error": null,
    "timeout": false
  }
]
```

Quy tắc EX:

```text
SQL chạy được, kết quả khớp          -> exec_match=true
SQL chạy được, kết quả không khớp    -> exec_match=false
SQL lỗi cú pháp/cột/table không có   -> exec_match=false, error != null
SQL timeout                          -> exec_match=false, timeout=true
```

`eval_ex.json` có dạng:

```json
{
  "summary": {
    "count": 1034,
    "execution_accuracy": 0.0,
    "exact_match": 0.0,
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "dataset": "vispider_dev",
    "timeout_seconds": 30,
    "exec_stats": {
      "matches": 0,
      "errors": 0,
      "timeouts": 0,
      "total": 1034
    }
  },
  "em_summary": {}
}
```

## EM và EX khác nhau thế nào?

### EM — Exact Match

EM dùng Spider evaluator gốc. Trong [spider_repo/evaluation.py](spider_repo/evaluation.py), evaluator có:

```python
DISABLE_VALUE = True
DISABLE_DISTINCT = True
```

Vì vậy EM so cấu trúc SQL sau khi normalize, không so literal value. Ví dụ:

```sql
SELECT name FROM singer WHERE country = 'France'
SELECT name FROM singer WHERE country = 'United States'
```

Hai câu này có thể EM=1 vì cùng cấu trúc, nhưng EX=0 nếu kết quả khác.

### EX — Execution Accuracy

EX chạy `pred_sql` và `gold_sql` trên cùng database SQLite rồi so output. EX quan trọng vì model có thể viết SQL khác gold nhưng vẫn trả đúng kết quả.

Ví dụ:

```sql
gold: SELECT COUNT(*) FROM singer
pred: SELECT COUNT(Singer_ID) FROM singer
```

Hai câu có thể EM=0 nhưng EX=1 nếu kết quả giống nhau.

## Kiểm tra trước khi chạy full benchmark

Compile:

```bash
python -m compileall zero_shot shared tests
```

Unit tests:

```bash
python -m unittest discover -s tests -v
```

Smoke bằng script:

```bash
LIMIT=20 SIZES="0_5B" bash scripts/run_qwen_compare.sh
```

## Vast.ai workflow khuyến nghị

Dùng `tmux` để tránh mất SSH làm chết process:

```bash
cd /workspace/NL2SQL
tmux new -s qwen_eval
bash scripts/run_qwen_compare.sh
```

Detach:

```text
Ctrl+B rồi D
```

Reconnect:

```bash
tmux attach -t qwen_eval
```

Theo dõi output:

```bash
tail -f zero_shot/results/qwen_compare/7B/run.log
```

## Tải kết quả về máy local

Từ PowerShell trên Windows:

```powershell
scp -P <PORT> -r root@<HOST>:/workspace/NL2SQL/zero_shot/results/qwen_compare E:\NL2SQL\vast_results\qwen_compare
scp -P <PORT> -r root@<HOST>:/workspace/NL2SQL/zero_shot/results/qwen_compare_aug E:\NL2SQL\vast_results\qwen_compare_aug
```

## Notes

- `data/`, `zero_shot/results/`, `augmentation/results/`, và `vast_results/` bị ignore bởi git.
- `spider_repo/evaluation.py` là evaluator gốc, không chỉnh sửa trực tiếp.
- Timeout EX nằm trong wrapper [shared/spider_eval.py](shared/spider_eval.py), không nằm trong evaluator gốc.
- Phase 2 có thể chạy lại nhiều lần từ cùng `predictions.txt` và `gold.txt` với timeout khác nhau.
