# Zero-Shot Text-to-SQL with ViSpider

Pipeline đánh giá zero-shot Text-to-SQL cho tiếng Việt, sử dụng dataset ViSpider và Spider evaluation metrics.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Dataset](#dataset)
- [Hướng dẫn chạy trên Vast.ai](#hướng-dẫn-chạy-trên-vastai)
- [Sử dụng](#sử-dụng)
- [Output format](#output-format)
- [Troubleshooting](#troubleshooting)

## 🎯 Tổng quan

Project này implement pipeline đầy đủ cho zero-shot Text-to-SQL evaluation:

1. **Load data**: Dataset ViSpider (Vietnamese translation của Spider)
2. **Build prompts**: Format schema + question thành prompt cho LLM
3. **Inference**: Generate SQL predictions với vLLM (batched inference)
4. **Evaluation**: Tính Exact Match (EM) và Execution Accuracy (EX) metrics

**Điểm mạnh:**
- Batched inference với vLLM (nhanh, tiết kiệm VRAM)
- Hỗ trợ evaluation-only mode (skip inference nếu đã có predictions)
- Output format chi tiết với per-example breakdown
- Spider evaluation chuẩn (unmodified từ taoyds/spider)

## 📁 Cấu trúc thư mục

```
zero_shot/
├── text_to_sql/                    # Main package
│   ├── __init__.py
│   ├── prompts.py                  # Schema formatting + prompt building
│   ├── spider_eval.py              # Spider evaluation wrapper
│   ├── run_zero_shot.py           # Pipeline orchestrator
│   └── results/                    # Output directory (auto-created)
│       ├── *_predictions.txt       # Raw predictions (1 SQL per line)
│       ├── *_gold.txt              # Gold SQL (format: sql\tdb_id)
│       └── *_zeroshot.json         # Full results with metrics
│
├── data/
│   ├── vispider_data/              # ViSpider dataset
│   │   ├── vispider_dev.json       # 1,034 examples (Vietnamese)
│   │   ├── vispider_train.json     # 7,000 examples
│   │   ├── vispider_test.json      # 2,147 examples
│   │   ├── dev_gold.sql            # Gold SQL for dev set
│   │   ├── train_gold.sql
│   │   ├── test_gold.sql
│   │   └── tables.json             # 166 database schemas
│   │
│   └── spider_db/                  # SQLite databases (166 databases)
│       ├── concert_singer/
│       ├── world_1/
│       └── ...
│
├── spider_repo/                    # Spider evaluation code (vendored)
│   ├── evaluation.py               # Original Spider evaluation
│   └── process_sql.py              # SQL parsing utilities
│
├── requirements.txt                # Python dependencies
└── test_prompts.py                 # Unit tests cho prompts module
```

### Luồng dữ liệu

```
vispider_dev.json  ──┐
                     ├──> prompts.py ──> vLLM inference ──> predictions.txt
tables.json        ──┘                                             │
                                                                   │
dev_gold.sql ─────────────────────────────────────────────────────┤
                                                                   │
spider_db/* + tables.json ────────────────────────────────────────┤
                                                                   │
                                                                   v
                                                          spider_eval.py
                                                                   │
                                                                   v
                                                         *_zeroshot.json
                                                    (EM + EX metrics + details)
```

## 💻 Yêu cầu hệ thống

### Phần mềm

- **Python**: 3.10 - 3.12
- **CUDA**: 11.8+ (cho vLLM)
- **Git**: Để clone repo

### Phần cứng (cho inference)

Model size và VRAM requirements:

| Model | Params | FP16 VRAM | Recommended GPU | Vast.ai price/hr |
|-------|--------|-----------|-----------------|------------------|
| Qwen2.5-Coder-7B | 7B | ~16GB | RTX 3090/4090 (24GB) | $0.15 - $0.30 |
| Qwen2.5-Coder-14B | 14B | ~32GB | A6000 (48GB), A100 (40GB) | $0.40 - $0.80 |
| Qwen2.5-Coder-32B | 32B | ~70GB | A100 (80GB) | $1.20 - $2.00 |

**Recommendation cho dev set (1,034 examples)**:
- GPU: **RTX 4090 24GB** hoặc **RTX 3090 24GB**
- RAM: 32GB+
- Storage: 50GB+ (cho model weights + dataset)
- Expected time: ~10-15 phút (với Qwen2.5-Coder-7B)

## 🔧 Cài đặt

### Bước 1: Clone repository

```bash
git clone <repo-url>
cd zero_shot
```

### Bước 2: Tạo virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
# Install PyTorch với CUDA support trước
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install các package còn lại
pip install -r requirements.txt
```

**Note**: vLLM cần ~5-10 phút để compile. Nếu gặp lỗi compile:
```bash
# Fallback: Install pre-built wheel
pip install vllm --extra-index-url https://pypi.nvidia.com
```

### Bước 4: Verify installation

```bash
# Test basic components (không cần GPU)
python test_prompts.py
```

Kết quả mong đợi: Schema formatting và prompt building hoạt động bình thường.

## 📊 Dataset

### Setup Dataset

**Lưu ý**: Dataset files (~40MB) không được commit vào repo. Bạn cần tải về thủ công.

#### 1. Tải ViSpider data từ HuggingFace

```bash
# Tạo thư mục data
mkdir -p data/vispider_data

# Tải files từ HuggingFace
# Truy cập: https://huggingface.co/datasets/hoadm/vispider/tree/main

# Download các files sau:
# - vispider_dev.json (467 KB)
# - vispider_train.json (4.04 MB)
# - vispider_test.json (980 KB)
# - dev_gold.sql (124 KB)
# - train_gold.sql (1.18 MB)
# - test_gold.sql (275 KB)
# - tables.json (811 KB)
# - dev.json, train.json, test.json (optional - English versions)
```

**Hoặc dùng `huggingface-cli`:**

```bash
pip install huggingface-hub

# Clone toàn bộ dataset
huggingface-cli download hoadm/vispider --repo-type dataset --local-dir data/vispider_data
```

#### 2. Tải Spider databases từ repo gốc

```bash
# Clone Spider repo (chỉ cần database folder)
cd data
git clone https://github.com/taoyds/spider.git spider_temp

# Copy databases
cp -r spider_temp/database spider_db

# Xóa folder tạm
rm -rf spider_temp
```

**Hoặc download trực tiếp:**
- Truy cập: https://github.com/taoyds/spider
- Download `database.zip` từ releases
- Extract vào `data/spider_db/`

### ViSpider

Vietnamese translation của Spider dataset:

- **Dev set**: 1,034 examples (dùng để evaluate)
- **Train set**: 7,000 examples (dùng cho few-shot hoặc fine-tuning)
- **Test set**: 2,147 examples (hidden gold labels)

Mỗi example có:
```json
{
  "id": 0,
  "db_id": "concert_singer",
  "question": "How many singers do we have?",           // English
  "question_vi": "Có bao nhiêu ca sĩ?",                 // Vietnamese
  "query": "SELECT count(*) FROM singer",               // Gold SQL
  "sql": {...}                                          // Parsed SQL structure
}
```

### Database schemas

166 SQLite databases trong `data/spider_db/`:
- Music, flights, concerts, restaurants, etc.
- Schema được format từ `tables.json`

### Verify dataset setup

Sau khi tải xong, verify cấu trúc thư mục:

```bash
data/
├── vispider_data/
│   ├── vispider_dev.json
│   ├── vispider_train.json
│   ├── vispider_test.json
│   ├── dev_gold.sql
│   ├── train_gold.sql
│   ├── test_gold.sql
│   └── tables.json
└── spider_db/
    ├── concert_singer/
    ├── world_1/
    └── ... (166 databases total)
```

Check bằng Python:

```bash
python -c "
import json
from pathlib import Path

# Check ViSpider data
vispider = Path('data/vispider_data/vispider_dev.json')
print(f'ViSpider dev: {\"✓\" if vispider.exists() else \"✗\"}')

# Check databases
db_dir = Path('data/spider_db')
db_count = len([d for d in db_dir.iterdir() if d.is_dir()]) if db_dir.exists() else 0
print(f'Spider databases: {db_count}/166')
"
```

## 🚀 Hướng dẫn chạy trên Vast.ai

### Bước 1: Chọn server phù hợp

Truy cập [vast.ai](https://vast.ai) và search với filters:

#### Recommended filters:
```
GPU Model: RTX 4090, RTX 3090, A5000, A6000
VRAM: >= 24 GB
RAM: >= 32 GB
Disk Space: >= 100 GB
CUDA: >= 11.8
Docker: Yes
```

#### Sort by: `$/hr` (thấp nhất trước)

**GPU recommendations:**

1. **RTX 4090 24GB** (Best value)
   - Price: ~$0.20 - $0.35/hr
   - Speed: Rất nhanh (Ada Lovelace architecture)
   - VRAM: Đủ cho model 7B-14B
   
2. **RTX 3090 24GB** (Budget option)
   - Price: ~$0.15 - $0.25/hr
   - Speed: Nhanh
   - VRAM: Đủ cho model 7B-14B
   
3. **A6000 48GB** (For larger models)
   - Price: ~$0.50 - $0.80/hr
   - VRAM: Đủ cho model 32B
   - Enterprise-grade

### Bước 2: Rent instance

1. Click **RENT** trên server đã chọn
2. Select **On-Demand** (pay-as-you-go)
3. Choose image: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel`
4. Click **CREATE & START**

### Bước 3: Connect SSH

Copy SSH command từ Vast.ai dashboard:
```bash
ssh -p <port> root@<ip> -L 8080:localhost:8080
```

### Bước 4: Setup trên server

```bash
# 1. Update system
apt update
apt install -y git vim

# 2. Clone repo
cd /workspace
git clone <repo-url>
cd zero_shot

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 4. Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Bước 5: Download model weights (optional)

vLLM sẽ tự động download, nhưng có thể pre-download để tiết kiệm thời gian:

```bash
# Install huggingface-cli
pip install huggingface-hub

# Download model
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir ./models/qwen2.5-7b
```

## 📖 Sử dụng

### Mode 1: Full pipeline (Inference + Evaluation)

Chạy inference và evaluation end-to-end:

```bash
python -m text_to_sql.run_zero_shot \
  --dataset vispider \
  --split dev \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8
```

**Parameters:**
- `--dataset`: Dataset name (`vispider`)
- `--split`: Dataset split (`dev`, `train`, `test`)
- `--model`: HuggingFace model ID
- `--max-model-len`: Max context length (default: 4096)
- `--gpu-memory-utilization`: GPU memory fraction (default: 0.7, tăng lên 0.8-0.9 nếu GPU idle)
- `--limit`: Limit examples for smoke test (e.g., `--limit 10`)
- `--disable-exec`: Only compute Exact Match, skip Execution Accuracy

#### Test nhanh với 10 examples:

```bash
python -m text_to_sql.run_zero_shot \
  --dataset vispider \
  --split dev \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --limit 10
```

### Mode 2: Evaluation-only (skip inference)

Nếu đã có predictions từ run trước:

```bash
python -m text_to_sql.run_zero_shot \
  --dataset vispider \
  --split dev \
  --predictions-input text_to_sql/results/vispider_dev_predictions.txt
```

**Use case**: 
- Re-evaluate với different metrics
- Debug evaluation issues
- Compare multiple prediction files

### Mode 3: Chỉ generate predictions (không evaluate)

```bash
python -m text_to_sql.run_zero_shot \
  --dataset vispider \
  --split dev \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --disable-exec  # Faster, chỉ tính EM
```

## 📤 Output format

### Predictions file (`*_predictions.txt`)

Format: 1 SQL query per line
```
SELECT count(*) FROM singer
SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id WHERE T2.year = 2014
...
```

### Gold file (`*_gold.txt`)

Format: `{sql}\t{db_id}` per line
```
SELECT count(*) FROM singer	concert_singer
SELECT name FROM singer WHERE age > 20	concert_singer
...
```

### Results file (`*_zeroshot.json`)

Cấu trúc đầy đủ:

```json
{
  "summary": {
    "count": 1034,
    "exact_match": 0.68,              // EM: SQL string match
    "execution_accuracy": 0.72,       // EX: Query results match
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "dataset": "vispider_dev",
    "timestamp": "2026-06-18T15:30:00"
  },
  "by_difficulty": {
    "easy": {
      "count": 248,
      "exact_match": 0.85,
      "execution_accuracy": 0.88
    },
    "medium": {
      "count": 446,
      "exact_match": 0.70,
      "execution_accuracy": 0.75
    },
    "hard": {
      "count": 174,
      "exact_match": 0.48,
      "execution_accuracy": 0.52
    },
    "extra": {
      "count": 166,
      "exact_match": 0.35,
      "execution_accuracy": 0.40
    }
  },
  "predictions": [
    {
      "id": 0,
      "example_id": "concert_singer-0",
      "db_id": "concert_singer",
      "question": "Có bao nhiêu ca sĩ?",
      "gold_sql": "SELECT count(*) FROM singer",
      "pred_sql": "SELECT count(*) FROM singer"
    },
    ...
  ]
}
```

## 🎯 Expected results

Baseline performance cho Qwen2.5-Coder-7B trên ViSpider dev:

| Metric | Easy | Medium | Hard | Extra | Overall |
|--------|------|--------|------|-------|---------|
| **EM** | ~85% | ~70% | ~48% | ~35% | **~68%** |
| **EX** | ~88% | ~75% | ~52% | ~40% | **~72%** |

**Note**: Results có thể khác nhau tùy model và hyperparameters.

## 🐛 Troubleshooting

### 1. vLLM out of memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
- Giảm `--max-model-len` (4096 → 2048)
- Giảm `--gpu-memory-utilization` (0.8 → 0.6)
- Dùng model nhỏ hơn (14B → 7B)
- Upgrade GPU (24GB → 48GB)

### 2. vLLM không compile được

```
ERROR: Failed building wheel for vllm
```

**Solutions:**
```bash
# Option 1: Use pre-built wheel
pip install vllm --extra-index-url https://pypi.nvidia.com

# Option 2: Install ninja for faster compilation
pip install ninja
pip install vllm

# Option 3: Use Docker image
docker pull vllm/vllm-openai:latest
```

### 3. CUDA version mismatch

```
RuntimeError: CUDA version mismatch
```

**Solution:**
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Install matching PyTorch
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Evaluation fails

```
FileNotFoundError: evaluation.py not found
```

**Solution:**
```bash
# Verify spider_repo/evaluation.py exists
ls spider_repo/evaluation.py

# If missing, download from Spider repo:
wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py -O spider_repo/evaluation.py
```

### 5. Unicode encoding errors (Windows)

```
UnicodeEncodeError: 'charmap' codec can't encode character
```

**Solution:**
```bash
# Set environment variable before running
set PYTHONIOENCODING=utf-8
python -m text_to_sql.run_zero_shot ...
```

## 📝 Citation

Nếu sử dụng pipeline này, vui lòng cite:

```bibtex
@inproceedings{yu2018spider,
  title={Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and others},
  booktitle={EMNLP},
  year={2018}
}

@inproceedings{vispider,
  title={ViSpider: Vietnamese Text-to-SQL Dataset},
  author={...},
  year={2024}
}
```

## 📧 Support

- **Issues**: Mở issue trên GitHub repo
- **Questions**: Liên hệ qua email hoặc discussion

---

**Happy experimenting! 🚀**
