import pandas as pd
from collections import defaultdict

# データ読み込み（例: TSV形式なら sep="\t"）
df = pd.read_csv("../meld_data/csv/all_sent_emo.csv", sep=",")

# 発話順にソート
df = df.sort_values(by=["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)

# キャラ辞書（MELD論文準拠）
main_characters = {"Chandler", "Ross", "Phoebe", "Monica", "Joey", "Rachel"}
# Speaker正規化（主要6キャラ以外をother）
df["Speaker"] = df["Speaker"].apply(lambda x: x if x in main_characters else "other")

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

# =====================================================
# 分布計算関数
# =====================================================
def compute_context(df, by_speaker=True):
    context_dict = defaultdict(lambda: defaultdict(lambda: {"prev": [], "next": []}))

    if by_speaker:
        groups = df.groupby("Speaker")
    else:
        groups = [("ALL", df)]

    for speaker, group in groups:
        group = group.sort_values(by=["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)

        for i, row in group.iterrows():
            emotion = row["Emotion"]

            # 直前（同じスピーカー or 全体）
            if i > 0:
                prev_emotion = group.iloc[i-1]["Emotion"]
                context_dict[speaker][emotion]["prev"].append(prev_emotion)

            # 直後
            if i < len(group)-1:
                next_emotion = group.iloc[i+1]["Emotion"]
                context_dict[speaker][emotion]["next"].append(next_emotion)

    return context_dict

# 個別スピーカー
context_dict = compute_context(df, by_speaker=True)
# 全体
context_all = compute_context(df, by_speaker=False)

# =====================================================
# CSV用の表にまとめる
# =====================================================
records = []

def add_records(context_dict):
    for speaker, emo_dict in context_dict.items():
        for emotion, ctx in emo_dict.items():
            for pos in ["prev", "next"]:
                if ctx[pos]:
                    counts = pd.Series(ctx[pos]).value_counts(normalize=True)
                    for emo2, prop in counts.items():
                        records.append([speaker, emotion, pos, emo2, round(prop, 4)])

add_records(context_dict)
add_records(context_all)

# DataFrame化
result_df = pd.DataFrame(records, columns=["Speaker", "Emotion", "Context", "NextEmotion", "Proportion"])

# 出力CSVも感情順でソート
result_df["Emotion"] = pd.Categorical(result_df["Emotion"], categories=emotion_order, ordered=True)
result_df["NextEmotion"] = pd.Categorical(result_df["NextEmotion"], categories=emotion_order, ordered=True)

# CSVに保存
result_df.to_csv("emotion_context_distribution.csv", index=False)

print("✅ 集計完了: emotion_context_distribution.csv に保存しました")