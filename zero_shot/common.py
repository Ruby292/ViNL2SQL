import json
from pathlib import Path
from typing import Dict, List


BASE_DIR = Path(__file__).parent.parent
DATA_ROOT = BASE_DIR / "data"
SPIDER_DB = DATA_ROOT / "spider_db"
VISPIDER_DIR = DATA_ROOT / "vispider_data"
RESULTS_DIR = BASE_DIR / "zero_shot" / "results"
TABLE_FILE = VISPIDER_DIR / "tables.json"

DATA_FILES = {
    ("vispider", "dev"): VISPIDER_DIR / "vispider_dev.json",
    ("vispider", "train"): VISPIDER_DIR / "vispider_train.json",
    ("vispider", "test"): VISPIDER_DIR / "vispider_test.json",
}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(dataset: str, split: str) -> List[Dict]:
    path = DATA_FILES.get((dataset, split))
    if path is None:
        raise ValueError(f"Unknown dataset/split combination: {dataset}/{split}")
    return load_json(path)


def load_tables() -> Dict:
    return {db["db_id"]: db for db in load_json(TABLE_FILE)}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
