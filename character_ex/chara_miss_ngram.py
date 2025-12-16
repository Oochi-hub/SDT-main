import pandas as pd
from collections import Counter

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

# n-gram長
n = 3

# 誤分類だけ抽出
misclassified = df[df["true"] != df["pred"]]

records = []

for idx, row in misclassified.iterrows():
    vid, utt_index, char = row["vid"], row["utt_index"], row["character"]
    
    # 同じキャラだけの発話系列
    group = df[(df["vid"] == vid) & (df["character"] == char)].sort_values("utt_index")
    
    # 該当位置までの直前 (n-1) 個を取る
    prev_rows = group[group["utt_index"] < utt_index].tail(n-1)
    
    if len(prev_rows) < n-1:
        continue  # 前文脈が足りない場合はスキップ
    
    ngram = list(prev_rows["true"].map(emotion_list)) + [emotion_list[row["true"]]]
    
    records.append({
        "character": speaker2id[char],
        "center_emotion": emotion_list[row["true"]],
        "pred_emotion": emotion_list[row["pred"]],
        "ngram": tuple(ngram)
    })

ngrams_df = pd.DataFrame(records)

# ===== 集計 =====
results = []

for char in speaker_order:
    for center in emotion_order:
        subset = ngrams_df[(ngrams_df["character"] == char) &
                           (ngrams_df["center_emotion"] == center)]
        if len(subset) == 0:
            continue
        
        counts = Counter(subset["ngram"])
        total = sum(counts.values())
        
        for ng, c in counts.items():
            results.append({
                "character": char,
                "center_emotion": center,
                "ngram": " → ".join(ng),
                "count": c,
                "ratio": c/total
            })

summary_ngram_df = pd.DataFrame(results).sort_values(
    ["character", "center_emotion", "count"], ascending=[True, True, False]
)

summary_ngram_df.to_csv(f"misclassification_{n}gram_patterns.csv", index=False)

# 結果を DataFrame から出力
for char in summary_ngram_df["character"].unique():
    print(f"=== {char} ===")
    char_df = summary_ngram_df[summary_ngram_df["character"] == char]
    
    for emo in char_df["center_emotion"].unique():
        emo_df = char_df[char_df["center_emotion"] == emo]
        
        print(f"  --- {emo} ---")
        for _, row in emo_df.head(10).iterrows():  # 上位10件だけ表示
            print(f"    {row['ngram']} : count={row['count']} ratio={row['ratio']:.3f}")
    print()
