
#各モダリティの正答事例ごとにバイナリ表現を調査
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns
import numpy as np
from collections import defaultdict
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #1階層上のモジュールも呼び出し
from dataloader import IEMOCAPDataset, MELDDataset
from model import BinaryActivation, binary_ste


#seed値の設定
#default 42
seed = 42
torch.manual_seed(seed)

# モデルから融合表現を取得する関数 SDTモデル用
def get_fused_features(model, textf, visuf, acouf, umask, qmask, lengths, modal):
    # モデルを評価モードに設定
    model.eval()
    
    # フックを登録して表現を取得
    features = []


    def get_attention_hook():
        def hook(module, input, output):
            features.append(binary_ste(output).detach().cpu().numpy().copy()) #ndarray (1, 1, 発話数, 1024)
        return hook
    
    # フックを追加
    hook = model.encoder.register_forward_hook(get_attention_hook())


    # 入力データをモデルに渡す
    with torch.no_grad():
        target_pred, _, z, h_hat, binary_prob, kl_log_prob = model(textf, visuf, acouf, umask, qmask, lengths, modal)

    # フックを解除
    hook.remove

    # """
    #デモ IEMOCAP
    # print(np.shape(features))
    # print(features)
    # exit()

    #(1, 16, 91, 256)
    #(1, バッチサイズ, 発話数, 256)
    # """

    #print(binary_prob.size()) #(バッチサイズ, 発話数, 6)
    pred_np = torch.argmax(binary_prob, dim=-1).cpu().numpy()  # ← 予測ラベルを追加 (B, 発話数)
  
    return features[-1], pred_np#(バッチサイズ, 発話数, 256) (バッチサイズ, 発話数)

def model_run(model, loader, modal, df_lookup):

    #保存用
    all_fused = []
    all_labels = []
    all_vads = []
    all_preds = []


    with torch.no_grad():
        for data in loader:
            """
            入力デモ IEMOCAP
            textf:  torch.Size([30, 16, 1024])
            visuf:  torch.Size([30, 16, 342])
            acouf:  torch.Size([30, 16, 1582])
            qmask:  torch.Size([16, 30, 2])
            umask:  torch.Size([16, 30])
            label:  torch.Size([16, 30])
            """
            textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-2]]

            qmask = qmask.permute(1, 0, 2) #qmask: 人物の区別のための行列   torch.Size([21, 8, 9])
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))] #各会話の発話数をumaskから逆算 list 長さ=バッチサイズ

            vids = data[-1]  # 会話ID

            fused_vec, pred_np= get_fused_features(model, textf, visuf, acouf, umask, qmask, lengths, modal)#(バッチサイズ, 発話数, 256)

            #np配列に変換
            umask_np = umask.cpu().numpy()
            label_np = label.cpu().numpy()

            for b, vid in enumerate(vids):  # バッチ内ループ
                num_utts = lengths[b]
                vad_list = []

                for utt_id in range(num_utts):
                    try:
                        vad = df_lookup.loc[(vid, utt_id), ["Valence", "Arousal", "Dominance"]].values
                    except KeyError:
                        vad = [np.nan, np.nan, np.nan]
                    vad_list.append(vad)
                    # vad = df_lookup.loc[(vid, utt_id), ["Valence", "Arousal", "Dominance"]].values
                    # vad_list.append(vad)

                vad_array = np.array(vad_list, dtype=float)  # (num_utts, 3)

                valid_idx = umask_np[b] == 1
                all_fused.append(fused_vec[b][valid_idx])    # (L_i, D)
                all_labels.append(label_np[b][valid_idx])    # (L_i,)
                all_vads.append(vad_array)                   # (L_i, 3)
                all_preds.append(pred_np[b][valid_idx])      # (L_i,)

    #1つのnp配列にまとめる
    fused_vectors = np.concatenate(all_fused, axis=0)   # shape = (N_total, D)
    labels        = np.concatenate(all_labels, axis=0)  # shape = (N_total,)
    vad_np        = np.concatenate(all_vads, axis=0)    # shape = (N_total,3(v,a,d))
    preds         = np.concatenate(all_preds, axis=0)   # shape = (N_total,)

    # print(preds.shape)  # (N_total,)
    # print(np.unique(preds))
    # print(np.mean(preds == labels))  # 正答率の確認

    return fused_vectors, labels, vad_np, preds

def save_dim_label_prob(fused_vectors, labels, label_mapping, correct,
                    csv_path="dim_label_prob.csv",
                    fig_path="dim_label_prob.png"):
    """
    各次元の発火確率 (P(x_d=1 | y=c)) を計算し，
    CSVとヒートマップとして保存する
    
    Parameters
    ----------
    fused_vectors : np.ndarray
        shape = (N, D) のバイナリ特徴量 (0/1)
    labels : np.ndarray
        shape = (N,) のラベルID
    label_mapping : dict
        {label_id: label_name}
    csv_path : str
        発火確率結果を保存するCSVファイルのパス
    fig_path : str
        ヒートマップ画像を保存するファイルのパス
    """
    
    N, D = fused_vectors.shape
    class_ids = sorted(label_mapping.keys())
    class_names = [label_mapping[c] for c in class_ids]

    # 各クラスごとの発火確率を計算
    prob_matrix = np.zeros((len(class_ids), D))
    for i, c in enumerate(class_ids):
        idx = (labels == c) & correct
        # print(f"\n--- idx: total samples = {idx.sum()} ---")

        if np.sum(idx) > 0:
            prob_matrix[i, :] = fused_vectors[idx].mean(axis=0)
        else:
            prob_matrix[i, :] = 0.0  # データがない場合は0にする

    # DataFrame化
    df_prob = pd.DataFrame(prob_matrix, index=class_names,
                        columns=[f"dim{d}" for d in range(D)])

    # CSV保存
    df_prob.to_csv(csv_path, encoding="utf-8")

    # ヒートマップ描画
    plt.figure(figsize=(20, 6))
    sns.heatmap(df_prob, cmap="Blues", vmin=0, vmax=1,
                xticklabels=False, yticklabels=True,
                cbar_kws={'label': 'Firing Probability'})
    plt.title("Dimension-Class Firing Probability Heatmap", fontsize=14)
    plt.ylabel("Emotion Class")
    plt.xlabel("Feature Dimension")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return df_prob




###============== 実装 ==============###

#データセット
which_data = "IEMOCAP" #"IEMOCAP" or "MELD"

for train_flag in [False, True]:

    #モダリティ対応辞書
    modal_dict = {"text":"t", "audio":"a", "visual":"v"}

    #保存先
    if train_flag:
        mode = "train"
    else:
        mode = "test"

    #データローダ準備
    if which_data == "IEMOCAP":
        dataset = IEMOCAPDataset(train=train_flag, path='../data/iemocap_multimodal_features.pkl')
        loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=dataset.collate_fn)
        label_mapping = {
        0:"happy",
        1:"sad",
        2:"neutral",
        3:"angry",
        4:"excited",
        5:"frustrated",
    }


    ##学習済みオートエンコーダの指定
    weight_folder = f"../experience_results/0929_autoencoder/not_fine_tune/100epoch/symmetry"

    # # --- CSV 読み込み ---
    if train_flag:
        df_vad = pd.read_csv("vad_csv/iemo_train_with_vad.csv")
    else:
        df_vad = pd.read_csv("vad_csv/iemo_test_with_vad.csv")

    # DataFrame を検索しやすいようにインデックス化
    df_lookup = df_vad.set_index(["vid", "utter_id_y"])
    

    ##テキスト
    model_t = torch.load(f"{weight_folder}/text/weights/model_weights_last.pth")
    #モデル初期化
    model_t.eval()
    fused_vectors_t, labels, vad_np, preds_t = model_run(model=model_t, loader=loader, modal="t", df_lookup=df_lookup)

    ##音声
    model_a = torch.load(f"{weight_folder}/audio/weights/model_weights_last.pth")
    #モデル初期化
    model_a.eval()
    fused_vectors_a, _, _, preds_a = model_run(model=model_a, loader=loader, modal="a", df_lookup=df_lookup)

    ##視覚
    model_v = torch.load(f"{weight_folder}/visual/weights/model_weights_last.pth")
    #モデル初期化
    model_v.eval()
    fused_vectors_v, _, _, preds_v = model_run(model=model_v, loader=loader, modal="v", df_lookup=df_lookup)


    # #各モダリティの正答事例を収集
    correct_t = (preds_t == labels)
    correct_a = (preds_a == labels)
    correct_v = (preds_v == labels)

    # #各モダリティの誤分類事例を収集
    # correct_t = (preds_t != labels)
    # correct_a = (preds_a != labels)
    # correct_v = (preds_v != labels)





    # csv_path_t = f"{dir}/text_correct_dim_label_prob.csv"
    # fig_path_t = f"{dir}/text_correct_dim_label_prob.png"
    # save_dim_label_prob(fused_vectors_t, labels, label_mapping, correct_t, csv_path_t, fig_path_t)
    
    # csv_path_a = f"{dir}/audio_correct_dim_label_prob.csv"
    # fig_path_a = f"{dir}/audio_correct_dim_label_prob.png"
    # save_dim_label_prob(fused_vectors_a, labels, label_mapping, correct_a, csv_path_a, fig_path_a)
    
    # csv_path_v = f"{dir}/visual_correct_dim_label_prob.csv"
    # fig_path_v = f"{dir}/visual_correct_dim_label_prob.png"
    # save_dim_label_prob(fused_vectors_v, labels, label_mapping, correct_v, csv_path_v, fig_path_v)

    # csv_path_t = f"{dir}/text_miss_dim_label_prob.csv"
    # fig_path_t = f"{dir}/text_miss_dim_label_prob.png"
    # save_dim_label_prob(fused_vectors_t, labels, label_mapping, correct_t, csv_path_t, fig_path_t)
    
    # csv_path_a = f"{dir}/audio_miss_dim_label_prob.csv"
    # fig_path_a = f"{dir}/audio_miss_dim_label_prob.png"
    # save_dim_label_prob(fused_vectors_a, labels, label_mapping, correct_a, csv_path_a, fig_path_a)
    
    # csv_path_v = f"{dir}/visual_miss_dim_label_prob.csv"
    # fig_path_v = f"{dir}/visual_miss_dim_label_prob.png"
    # save_dim_label_prob(fused_vectors_v, labels, label_mapping, correct_v, csv_path_v, fig_path_v)

    ### --- 各モダリティの正答・誤答別にVAD分布を分析 --- ###

    dir = "results/1007_result"
    output_folder = f"{dir}/vad_stats/{mode}"
    os.makedirs(output_folder, exist_ok=True)

    modal_list = ["text", "audio", "visual"]
    correct_dict = {
        "text": correct_t,
        "audio": correct_a,
        "visual": correct_v,
    }

    for modal in modal_list:
        print(f"\n==== {modal.upper()} モダリティ ====")

        correct_mask = correct_dict[modal]
        incorrect_mask = ~correct_mask  # 誤答 = NOT 正答

        # --- DataFrame作成 ---
        vad_df = pd.DataFrame(vad_np, columns=["Valence", "Arousal", "Dominance"])
        vad_df["Label_id"] = labels
        vad_df["Label"] = vad_df["Label_id"].map(label_mapping)
        vad_df["Correct"] = np.where(correct_mask, "Correct", "Incorrect")

        # --- 欠損除外 ---
        vad_df = vad_df.dropna(subset=["Valence", "Arousal", "Dominance"])

        # --- サブセット ---
        vad_df_correct = vad_df[vad_df["Correct"] == "Correct"]
        vad_df_incorrect = vad_df[vad_df["Correct"] == "Incorrect"]

        print(f"正答サンプル数: {len(vad_df_correct)}, 誤答サンプル数: {len(vad_df_incorrect)}")

        # --- ラベル × 正答誤答 ごとの統計 (mean, std, count) ---
        group_stats = (
            vad_df
            .groupby(["Label", "Correct"])[["Valence", "Arousal", "Dominance"]]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        # --- CSV保存 ---
        csv_path = f"{output_folder}/{modal}_by_label_correct_incorrect.csv"
        group_stats.to_csv(csv_path, index=False)
        print(f"✅ CSV保存完了: {csv_path}")

        # --- バイオリンプロット ---
        emotion_order = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        for i, col in enumerate(["Valence", "Arousal", "Dominance"]):
            sns.violinplot(
                data=vad_df,
                x="Label",
                y=col,
                hue="Correct",
                split=True,
                order=emotion_order,
                ax=axes[i],
                palette={"Correct": "skyblue", "Incorrect": "lightcoral"}
            )
            axes[i].set_title(f"{col} by Emotion ({modal})")
            axes[i].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{modal}_violin_correct_vs_incorrect.png"))
        plt.close()
        print(f"✅ バイオリン保存: {modal}_violin_correct_vs_incorrect.png")

        # --- 箱ひげ図 (正答 vs 誤答) ---
        emotion_order = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]
        hue_order = ["Correct", "Incorrect"]  # ← 表示順を指定

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

        for i, col in enumerate(["Valence", "Arousal", "Dominance"]):
            sns.boxplot(
                data=vad_df,
                x="Label",
                y=col,
                hue="Correct",
                order=emotion_order,
                hue_order=hue_order,  # ✅ ここで順序を指定
                ax=axes[i],
                palette={"Correct": "skyblue", "Incorrect": "lightcoral"},
                width=0.6,
                fliersize=2,
            )
            axes[i].set_title(f"{col} by Emotion ({modal})")
            axes[i].set_xlabel("")
            axes[i].set_ylabel(col)
            axes[i].legend(title="", loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{modal}_box_correct_vs_incorrect.png"))
        plt.close()
        print(f"✅ 箱ひげ図保存: {modal}_box_correct_vs_incorrect.png")