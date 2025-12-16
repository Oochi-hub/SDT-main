from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def vad_emotion_similarity(vad_df, emotion_df, save_path="vad_emotion_similarity.csv"):
    """
    VAD高群と感情クラスの発火確率パターンの類似度をコサイン類似度で計算
    """

    # --- 列名を統一的に整形 ---
    def normalize_colname(name):
        # "dim_0" → "dim0" に統一
        name = str(name).strip().lower().replace("dim_", "dim")
        return name

    vad_df.columns = [normalize_colname(c) for c in vad_df.columns]
    emotion_df.columns = [normalize_colname(c) for c in emotion_df.columns]

    # 数値列のみ残す
    vad_df = vad_df.select_dtypes(include=["float", "int"])
    emotion_df = emotion_df.select_dtypes(include=["float", "int"])

    # 共通の列のみ抽出
    common_cols = sorted(set(vad_df.columns) & set(emotion_df.columns))
    if len(common_cols) == 0:
        raise ValueError("⚠️ 共通の特徴次元が見つかりません。列名の形式を確認してください。")

    vad_df = vad_df[common_cols]
    emotion_df = emotion_df[common_cols]

    # --- コサイン類似度を計算 ---
    sim_matrix = cosine_similarity(vad_df.values, emotion_df.values)
    sim_df = pd.DataFrame(
        sim_matrix,
        index=vad_df.index,
        columns=emotion_df.index
    )

    sim_df.to_csv(save_path)
    print(f"✅ 類似度行列を保存しました: {save_path}")

    return sim_df

def vad_emotion_mae(vad_df, emotion_df, save_path="vad_emotion_mae.csv"):
    """
    VAD高群と感情クラスの発火確率パターンの「誤差（MAE）」を計算

    Parameters
    ----------
    vad_df : pd.DataFrame
        各VAD成分ごとの発火確率（例: Valence, Arousal, Dominance）
    emotion_df : pd.DataFrame
        各感情クラスごとの発火確率（例: happy, sad, angry...）
    save_path : str
        出力CSVの保存先

    Returns
    -------
    mae_df : pd.DataFrame
        行: VAD, 列: 感情クラス のMAE行列
    """

    # --- 列名の形式を統一 ---
    def normalize_colname(name):
        return str(name).strip().lower().replace("dim_", "dim")
    
    vad_df.columns = [normalize_colname(c) for c in vad_df.columns]
    emotion_df.columns = [normalize_colname(c) for c in emotion_df.columns]

    # 数値列だけ残す
    vad_df = vad_df.select_dtypes(include=["float", "int"])
    emotion_df = emotion_df.select_dtypes(include=["float", "int"])

    # 共通の列（特徴次元）を抽出
    common_cols = sorted(set(vad_df.columns) & set(emotion_df.columns))
    if len(common_cols) == 0:
        raise ValueError("⚠️ 共通の特徴次元が見つかりません。列名を確認してください。")

    vad_df = vad_df[common_cols]
    emotion_df = emotion_df[common_cols]

    # --- MAE（平均絶対誤差）を計算 ---
    mae_matrix = np.zeros((len(vad_df), len(emotion_df)))

    for i, (_, vad_vec) in enumerate(vad_df.iterrows()):
        for j, (_, emo_vec) in enumerate(emotion_df.iterrows()):
            mae = np.mean(np.abs(vad_vec.values - emo_vec.values))
            mae_matrix[i, j] = mae

    mae_df = pd.DataFrame(mae_matrix, index=vad_df.index, columns=emotion_df.index)

    mae_df.to_csv(save_path)
    print(f"✅ 平均絶対誤差（MAE）行列を保存しました: {save_path}")

    return mae_df

for which_modal in ["text", "audio", "visual"]:
    for train_flag in ["train", "test"]:

        inequality = "larger"

        vad_df = pd.read_csv(f"results/1006_result/autoencoder/vad_prob/{which_modal}/{train_flag}/vad_threshold_{inequality}_prob.csv", index_col=0)
        emotion_df = pd.read_csv(f"results/1003_result/autoencoder/prob/{which_modal}/{train_flag}/dim_label_prob.csv", index_col=0)


        sim_df = vad_emotion_similarity(vad_df, emotion_df)

        output_folder_grah = f"results/1006_result/autoencoder/vad_label_sim_{inequality}/{which_modal}"
        ##保存先
        if not os.path.exists(output_folder_grah):
            os.makedirs(output_folder_grah)

        plt.figure(figsize=(8, 5))
        sns.heatmap(sim_df, annot=True, cmap="coolwarm", vmin=0, vmax=1)
        plt.title("Similarity between VAD High Activation and Emotion Classes")
        plt.xlabel("Emotion Class")
        plt.ylabel("VAD Component")
        plt.tight_layout()
        plt.savefig(f"{output_folder_grah}/vad_emotion_similarity_heatmap_{which_modal}_{train_flag}.png", dpi=300)
        plt.close()

        mae_df = vad_emotion_mae(vad_df, emotion_df, save_path=f"{output_folder_grah}/vad_emotion_mae_{which_modal}_{train_flag}.csv")

        plt.figure(figsize=(8, 5))
        sns.heatmap(mae_df, annot=True, cmap="coolwarm", vmin=0, vmax=1)
        plt.title("VAD–Emotion Activation Similarity (MAE: lower = more similar)")
        plt.xlabel("Emotion Class")
        plt.ylabel("VAD Component")
        plt.tight_layout()
        plt.savefig(f"{output_folder_grah}/vad_emotion_eam_heatmap_{which_modal}_{train_flag}.png", dpi=300)
        plt.close()
