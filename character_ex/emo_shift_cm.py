import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

emotion_order = ["neutral","surprise","fear","sadness","joy","disgust","anger"]
main_characters = {"Chandler","Ross","Phoebe","Monica","Joey","Rachel"}

# データ読み込み
df = pd.read_csv("../meld_data/csv/all_sent_emo.csv", sep=",")

# Speaker正規化
df["Speaker"] = df["Speaker"].apply(lambda x: x if x in main_characters else "other")
df["Emotion"] = pd.Categorical(df["Emotion"], categories=emotion_order, ordered=True)

# ソート
df = df.sort_values(["Dialogue_ID","Utterance_ID"]).reset_index(drop=True)

def build_transition_matrix(group, emotions):
    size = len(emotions)
    mat = np.zeros((size, size), dtype=int)
    for i in range(len(group)-1):
        e1 = group.iloc[i]["Emotion"]
        e2 = group.iloc[i+1]["Emotion"]
        if pd.isna(e1) or pd.isna(e2):
            continue
        i1 = emotions.index(e1)
        i2 = emotions.index(e2)
        mat[i1, i2] += 1
    return mat

# 保存フォルダ
os.makedirs("transition_plots_globalnorm", exist_ok=True)

for spk, g in df.groupby("Speaker"):
    g = g.sort_values(["Dialogue_ID","Utterance_ID"]).reset_index(drop=True)
    mat = build_transition_matrix(g, emotion_order)
    
    # === 全件数で正規化 ===
    total = mat.sum()
    if total > 0:
        mat_prob = mat.astype(float) / total
    else:
        mat_prob = mat.astype(float)
    
    df_mat = pd.DataFrame(mat_prob, index=emotion_order, columns=emotion_order)
    
    plt.figure(figsize=(7,6))
    sns.heatmap(df_mat, annot=True, fmt=".3f", cmap="Blues", cbar=True,
                xticklabels=emotion_order, yticklabels=emotion_order)
    plt.title(f"Emotion Transition Distribution (global norm) - {spk}")
    plt.xlabel("Next Emotion")
    plt.ylabel("Current Emotion")
    
    plt.tight_layout()
    plt.savefig(f"transition_plots_globalnorm/{spk}.png")
    plt.close()
