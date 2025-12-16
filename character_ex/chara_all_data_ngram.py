import pandas as pd
from collections import defaultdict, Counter
import os

def analyze_emotion_ngrams(df, target_emotion="surprise", n=3, position="start",
                           output_csv_patterns="ngrams_patterns.csv", output_folder="demo"):
    """
    人物ごとに target_emotion を含む n-gram を分析し、2種類のCSVを出力する
    - n-gramは1人の人物の発話のみに基づいて作成する
    - patterns: 各n-gramパターンの分布
    """

    results = defaultdict(Counter)
    total_ngrams_per_speaker = Counter()  # キャラごとの全n-gram数

    for dialog_id, group in df.groupby("Dialogue_ID"):
        group = group.sort_values("Utterance_ID").reset_index(drop=True)
        # さらに Speaker ごとに発話を追う
        for speaker, spk_group in group.groupby("Speaker"):
            emotions = spk_group["Emotion"].tolist()

            for i in range(len(emotions) - n + 1):
                ngram = tuple(emotions[i:i+n])
                total_ngrams_per_speaker[speaker] += 1

                # 基準感情の位置を判定
                if position == "start":
                    match = (ngram[0] == target_emotion)
                elif position == "end":
                    match = (ngram[-1] == target_emotion)
                else:
                    raise ValueError("position must be 'start' or 'end'")

                if match:
                    results[speaker][ngram] += 1



    # --- CSV1: パターンごとの分布 ---
    rows_patterns = []
    for speaker, counter in results.items():
        total = sum(counter.values())  # 基準感情を含むn-gram数
        for ng, cnt in counter.items():
            rows_patterns.append({
                "Speaker": speaker,
                "Ngram": " -> ".join(ng),
                "Count": cnt,
                "Ratio_within_target": cnt / total if total > 0 else 0,
                "Total_Ngrams_with_target": total
            })
    df_patterns = pd.DataFrame(rows_patterns)
    out_path = os.path.join(output_folder, output_csv_patterns)
    df_patterns.to_csv(out_path, index=False)

    return df_patterns

def calc_emotion_in_ngrams(csv1_path, output_csv, output_folder):
    # CSV1 を読み込み
    df = pd.read_csv(csv1_path)

    results = []
    speakers = df["Speaker"].unique()
    emotions = set()
    for ngram in df["Ngram"]:
        emotions.update(ngram.split(" -> "))

    # 各 Speaker × Emotion で割合を計算
    for spk in speakers:
        sub = df[df["Speaker"] == spk]
        total = sub["Total_Ngrams_with_target"].iloc[0] if not sub.empty else 0

        for emo in emotions:
            count_with_emo = sub[sub["Ngram"].str.contains(emo)]["Count"].sum()
            ratio = count_with_emo / total if total > 0 else 0
            results.append({
                "Speaker": spk,
                "Emotion": emo,
                "Ngrams_with_emotion": count_with_emo,
                "Total_Ngrams_for_Speaker": total,
                "Ratio": ratio
            })

    df_out = pd.DataFrame(results)
    out_path = os.path.join(output_folder, output_csv)
    df_out.to_csv(out_path, index=False)

    return df_out



# --- 使用例 ---
df = pd.read_csv("../meld_data/csv/all_sent_emo.csv", sep=",")

# キャラ辞書（MELD論文準拠）
main_characters = {"Chandler", "Ross", "Phoebe", "Monica", "Joey", "Rachel"}
# Speaker正規化（主要6キャラ以外をother）
df["Speaker"] = df["Speaker"].apply(lambda x: x if x in main_characters else "other")

#パラメータ指定
# traget_emmotion = "disgust"
# n = 3
# position = "start"

emotion_list = {
    0: "neutral",
    1: "surprise",
    2: "fear",
    3: "sadness",
    4: "joy",
    5: "disgust",
    6: "anger"
}
emotion_order = list(emotion_list.values())

ns = [2]

for n in ns:
    for traget_emotion in emotion_order:
        for position in ["start", "end"]:

            output_folder = f"0907_ngram/{n}grams/{traget_emotion}/{position}"
            os.makedirs(output_folder, exist_ok=True)

            df_patterns = analyze_emotion_ngrams(
                df,\
                target_emotion=traget_emotion,\
                n=n,\
                position=position,\
                output_csv_patterns=f"{n}grams_{traget_emotion}_{position}_patterns.csv",\
                output_folder=output_folder
            )

            # print("パターン分布サンプル:\n", df_patterns.head())

            df_out = calc_emotion_in_ngrams(os.path.join(output_folder, f"{n}grams_{traget_emotion}_{position}_patterns.csv"), f"{n}grams_{traget_emotion}_{position}_prob.csv", output_folder=output_folder)
            # print(df_out.head())
