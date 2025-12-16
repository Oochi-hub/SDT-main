import pandas as pd
# CSV読み込み
df = pd.read_csv("../experience_results/0826_ex/MELD_characters/all_pred.csv")

# 感情マッピング
emotion_list = {
    0:"neutral", 
    1:"surprise", 
    2:"fear", 
    3:"sadness", 
    4:"joy", 
    5:"disgust", 
    6:"anger"
}
emotion_order = list(emotion_list.values())

# キャラマッピング（主要6キャラ＋other）
speaker2id = {
    0:"Chandler", 
    1:"Ross", 
    2:"Phoebe", 
    3:"Monica", 
    4:"Joey", 
    5:"Rachel", 
    6:"other"
}
speaker_order = list(speaker2id.values())

# ラベル変換
df["true_label"] = df["true"].map(emotion_list)
df["pred_label"] = df["pred"].map(emotion_list)
df["character_name"] = df["character"].map(speaker2id)

# 誤分類抽出
misclassified = df[df["true"] != df["pred"]]

contexts = []

for idx, row in misclassified.iterrows():
    vid, utt_index, char = row["vid"], row["utt_index"], row["character"]
    
    # 同じ vid & 同じキャラ の発話だけを取る
    group = df[(df["vid"] == vid) & (df["character"] == char)].sort_values("utt_index")
    
    prev_row = group[group["utt_index"] == utt_index - 1]
    next_row = group[group["utt_index"] == utt_index + 1]
    
    contexts.append({
        "vid": vid,
        "character": speaker2id[char],
        "center_emotion": emotion_list[row["true"]],
        "pred_emotion": emotion_list[row["pred"]],
        "prev_true": emotion_list[prev_row["true"].values[0]] if not prev_row.empty else None,
        "next_true": emotion_list[next_row["true"].values[0]] if not next_row.empty else None,
    })

contexts_df = pd.DataFrame(contexts)

# ===== 集計（キャラ × 中心感情ごとに、前後の分布をカウント） =====
rows = []

for speaker in speaker_order:
    for center in emotion_order:
        subset = contexts_df[(contexts_df["character"] == speaker) &
                             (contexts_df["center_emotion"] == center)]
        if len(subset) == 0:
            continue
        
        prev_counts = subset["prev_true"].value_counts().reindex(emotion_order, fill_value=0)
        prev_ratio = prev_counts / prev_counts.sum() if prev_counts.sum() > 0 else prev_counts
        
        next_counts = subset["next_true"].value_counts().reindex(emotion_order, fill_value=0)
        next_ratio = next_counts / next_counts.sum() if next_counts.sum() > 0 else next_counts
        
        for e in emotion_order:
            rows.append({
                "character": speaker,
                "center_emotion": center,
                "position": "prev",
                "context_emotion": e,
                "count": prev_counts[e],
                "ratio": prev_ratio[e]
            })
            rows.append({
                "character": speaker,
                "center_emotion": center,
                "position": "next",
                "context_emotion": e,
                "count": next_counts[e],
                "ratio": next_ratio[e]
            })

summary_char_df = pd.DataFrame(rows)
summary_char_df.to_csv("misclassification_context_by_character_own_prevnext.csv", index=False)
