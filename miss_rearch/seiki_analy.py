import pandas as pd
from collections import Counter

# データ読み込み＆ソート
df = pd.read_csv("misclassified_dialogues_with_flags_by_meld.csv")

print(df["Dialogue_ID"].nunique())

df = df.sort_values(by=["Dialogue_ID", "Utterance_ID"])

pattern_counter = Counter()

# ダイアログごとに処理
for _, group in df.groupby("Dialogue_ID"):
    emotions = group["Emotion"].tolist()
    miss_flags = group["miss"].tolist()

    i = 0
    while i < len(emotions) - 1:
        base_emotion = emotions[i]
        run_length = 1

        # 連続する base_emotion をカウント
        for j in range(i+1, len(emotions)):
            if emotions[j] == base_emotion:
                run_length += 1
            else:
                # 異なる感情が出現
                next_emotion = emotions[j]
                if miss_flags[j]:  # 誤分類されていたら記録
                    pattern_key = f"{base_emotion}×{run_length} → {next_emotion}"
                    pattern_counter[pattern_key] += 1
                break

        # 次の開始位置を更新
        i += run_length

# 上位パターンを表示
print("頻出する誤分類感情遷移パターン（長さ自由）:")
for pattern, count in pattern_counter.most_common(20):
    print(f"{pattern}: {count} 回")
