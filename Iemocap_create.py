# iemocap_splitter.py

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# === CONFIG ===
source_root = Path("/home/oochi/data_dl/oochi/IEMOCAP/IEMOCAP_full_release")  # 元のIEMOCAPデータパス
output_root = Path("./iemocap_data")
modalities = ["text", "audio", "video"]
exts = {"text": ".txt", "audio": ".wav", "video": ".mp4"}  # 必要に応じて変更
split_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}

# === 1. 全モーダルのファイル収集 ===
data_dict = defaultdict(dict)  # {data_id: {"text": path, "audio": path, "video": path}}
for modality in modalities:
    for file_path in source_root.rglob(f"*{exts[modality]}"):
        data_id = file_path.stem  # 例: Ses01F_impro01_F000
        data_dict[data_id][modality] = file_path

# === 2. 3モーダル揃っているデータのみ抽出 ===
valid_data = [data_id for data_id, files in data_dict.items() if len(files) == 3]

# === 3. データ分割 ===
random.seed(42)
random.shuffle(valid_data)
total = len(valid_data)
train_end = int(split_ratio["train"] * total)
val_end = train_end + int(split_ratio["val"] * total)

splits = {
    "train": valid_data[:train_end],
    "val": valid_data[train_end:val_end],
    "test": valid_data[val_end:]
}

# === 4. フォルダ作成とコピー ===
for split in splits:
    for modality in modalities:
        (output_root / split / modality).mkdir(parents=True, exist_ok=True)

    for data_id in splits[split]:
        for modality in modalities:
            src = data_dict[data_id][modality]
            dst = output_root / split / modality / src.name
            shutil.copy2(src, dst)

print("✅ IEMOCAP データの分割と格納が完了しました。")