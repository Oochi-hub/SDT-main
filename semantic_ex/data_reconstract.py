import pandas as pd
import re
import os

# ========= パラメータ =========
csv_path = "../iemocap_data/csv/iemocap_test.csv"           # 元のCSV
base_folder = "/home/oochi/data_dl/oochi/IEMOCAP/IEMOCAP_full_release/IEMOCAP_full_release"
output_csv = "iemo_test_with_vad.csv"

# ========= CSV読み込み =========
df = pd.read_csv(csv_path)

vad_list = []

# ========= 各行処理 =========
for _, row in df.iterrows():
    vid = row["vid"]
    utter_id = row["utter_id_x"]

    # Ses01 → Session1, Ses02 → Session2, ...
    session_id = int(vid[3:5])   # "01" → 1
    session_folder = f"Session{session_id}/dialog/EmoEvaluation"
    txt_file = os.path.join(base_folder, session_folder, f"{vid}.txt")

    vad_value = [None, None, None]

    if os.path.exists(txt_file):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        # utter_id_x に一致する行から [V, A, D] を抽出
        # 例: [6.2901 - 8.2357] Ses01F_impro01_F000 neu [2.5000, 2.5000, 2.5000]
        pattern = rf"\[.*?\]\s+{re.escape(utter_id)}\s+\w+\s+\[(.*?)\]"
        match = re.search(pattern, text)

        if match:
            vad_str = match.group(1)  # "2.5000, 2.5000, 2.5000"
            vad_value = [float(x.strip()) for x in vad_str.split(",")]

    vad_list.append(vad_value)

# ========= DataFrameに追加 =========
df[["Valence", "Arousal", "Dominance"]] = pd.DataFrame(vad_list, index=df.index)

# ========= 保存 =========
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("✅ VADを追加して保存しました:", output_csv)