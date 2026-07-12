# ViNL2SQL

Pipeline Text-to-SQL trên bộ **ViSpider** (Spider bản tiếng Việt). Pipeline chính hiện tại là `zero_shot`: chạy trực tiếp mô hình Qwen2.5-Coder trên full schema và chấm điểm bằng Spider evaluator gốc.

Code trên GitHub chỉ chứa pipeline và scripts. Dữ liệu ViSpider và các file SQLite của Spider được kéo về từ Hugging Face lúc chạy.

## Repository layout

```text
NL2SQL/
├── data/                    # data không đưa lên git — tải riêng từ Hugging Face
│   ├── vispider_data/       # vispider_dev.json, tables.json, dev_gold.sql, ...
│   └── spider_db/           # <db_id>/<db_id>.sqlite cho mọi db_id trong Spider
├── shared/
│   └── spider_eval.py       # Wrapper eval, dùng chung cho mọi pipeline
├── spider_repo/             # Spider evaluator gốc, vendored
├── zero_shot/
│   ├── run_zero_shot.py     # Orchestrator vLLM + evaluator
│   └── prompts.py           # Prompt/schema formatting utilities
├── scripts/
│   └── run_qwen_compare.sh  # So sánh nhiều Qwen2.5-Coder size trên Vast.ai
├── tests/
└── requirements.txt
```

## Data sources (Hugging Face)

- ViSpider JSON, gold SQL, tables: [`hoadm/vispider`](https://huggingface.co/datasets/hoadm/vispider)
- Spider SQLite databases: [`rubypham292/spider_db`](https://huggingface.co/datasets/rubypham292/spider_db) (private)

Cấu trúc sau khi tải về đúng như pipeline expect:

```text
data/vispider_data/vispider_dev.json
data/vispider_data/vispider_train.json
data/vispider_data/vispider_test.json
data/vispider_data/dev_gold.sql
data/vispider_data/train_gold.sql
data/vispider_data/test_gold.sql
data/vispider_data/tables.json

data/spider_db/<db_id>/<db_id>.sqlite
```

## Environment

- Python 3.10+
- CUDA-capable GPU (khuyến nghị VRAM ≥ 24GB cho Qwen 7B)
- vLLM cho phần inference

```bash
pip install -r requirements.txt
pip install -U vllm huggingface_hub
```

## Quickstart — chạy trên Vast.ai

Giả sử đã thuê được GPU instance trên Vast.ai (RTX 3090/4090/A5000/RTX 5090, disk ≥ 100GB, SSH enabled) và SSH vào được.

### 1. Clone code + tải data

```bash
cd /workspace
git clone https://github.com/Ruby292/ViNL2SQL.git NL2SQL
cd NL2SQL

pip install -r requirements.txt
pip install -U vllm huggingface_hub

# spider_db là dataset private — cần đăng nhập
hf auth login

mkdir -p data/vispider_data data/spider_db

hf download hoadm/vispider \
  --repo-type dataset \
  --local-dir data/vispider_data

hf download rubypham292/spider_db \
  --repo-type dataset \
  --local-dir data/spider_db
```

Verify path (tùy chọn):

```bash
ls data/vispider_data | head
ls data/spider_db/concert_singer
```

Phải thấy `vispider_dev.json`, `dev_gold.sql`, `tables.json`, và `data/spider_db/concert_singer/concert_singer.sqlite`.

### 2. Smoke test 20 câu

```bash
LIMIT=20 bash scripts/run_qwen_compare.sh
```

### 3. Chạy full so sánh 4 model

```bash
tmux new -s qwen_eval
bash scripts/run_qwen_compare.sh
```

Detach: `Ctrl+B` rồi `D`. Reconnect:

```bash
tmux attach -t qwen_eval
```

### 4. Tải kết quả về máy

Từ PowerShell:

```powershell
scp -P <PORT> -r root@<HOST>:/workspace/NL2SQL/zero_shot/results/qwen_compare E:\NL2SQL\vast_results
```

## Chạy 1 model đơn lẻ

```bash
python -m zero_shot.run_zero_shot \
  --dataset vispider \
  --split dev \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output zero_shot/results/qwen_7B/eval.json \
  --predictions-output zero_shot/results/qwen_7B/predictions.txt \
  --gold-output zero_shot/results/qwen_7B/gold.txt \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

Các flag hữu ích:

- `--limit N` — chạy nhanh N example đầu (smoke test)
- `--disable-exec` — chỉ tính EM, bỏ EX
- `--predictions-input path` — bỏ qua inference, chỉ eval từ file predictions có sẵn

## Script so sánh nhiều model (`scripts/run_qwen_compare.sh`)

Mặc định chạy 4 size:

```text
Qwen/Qwen2.5-Coder-0.5B-Instruct
Qwen/Qwen2.5-Coder-1.5B-Instruct
Qwen/Qwen2.5-Coder-3B-Instruct
Qwen/Qwen2.5-Coder-7B-Instruct
```

Override bằng env var:

```bash
SIZES="1_5B 7B" bash scripts/run_qwen_compare.sh        # chọn size cụ thể
LIMIT=50 bash scripts/run_qwen_compare.sh               # smoke 50 câu
SPLIT=test bash scripts/run_qwen_compare.sh             # đổi split
```

## Output structure

Mỗi model có folder riêng, không đè lên nhau:

```text
zero_shot/results/qwen_compare/
├── 0_5B/
│   ├── eval.json         # summary EM/EX + per-example (question, gold, pred, raw_output)
│   ├── predictions.txt   # 1 SQL / dòng, cùng thứ tự với dev set
│   ├── gold.txt          # "{sql}\t{db_id}" / dòng
│   └── run.log           # log terminal của model này
├── 1_5B/
├── 3B/
├── 7B/
└── summary.log           # bảng tổng hợp EM/EX của các model đã chạy xong
```

`eval.json` có dạng:

```json
{
  "summary": {
    "count": 1034,
    "exact_match": 0.xxxx,
    "execution_accuracy": 0.xxxx,
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "dataset": "vispider_dev"
  },
  "by_difficulty": {
    "easy":   { "exact_match": ..., "execution_accuracy": ... },
    "medium": { "exact_match": ..., "execution_accuracy": ... },
    "hard":   { "exact_match": ..., "execution_accuracy": ... },
    "extra":  { "exact_match": ..., "execution_accuracy": ... }
  },
  "predictions": [
    {
      "question": "...",
      "gold_sql": "...",
      "pred_sql": "...",
      "raw_output": "..."
    }
  ]
}
```

## Shared evaluation

Cả pipeline và mọi lệnh chấm điểm đều đi qua `shared/spider_eval.py`, wrap Spider evaluator gốc trong `spider_repo/evaluation.py`. Contract file rất đơn giản:

```text
predictions.txt   # 1 SQL / dòng
gold.txt          # "{sql}\t{db_id}" / dòng
```

`etype="all"` mới lấy cả EM lẫn EX, `"match"` chỉ lấy EM. Chi tiết trong docstring của `run_evaluation()`.

## Testing

Chạy test suite cơ bản:

```bash
python -m unittest discover -s tests -p "test_*.py"
```
