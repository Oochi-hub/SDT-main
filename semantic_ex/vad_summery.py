#VAD指標の分布を調べる


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #1階層上のモジュールも呼び出し
from dataloader import IEMOCAPDataset


#seed値の設定
#default 42
seed = 42
torch.manual_seed(seed)

###============== 実装 ==============###

#データセット
which_data = "IEMOCAP" #"IEMOCAP" or "MELD"

for train_flag in [False, True]:
    
    #保存先
    if train_flag:
        mode = "train"
    else:
        mode = "test"

    output_folder = f"results/1003_result/vad_summery/{mode}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

        
    # # --- CSV 読み込み ---
    if train_flag:
        df_vad = pd.read_csv("vad_csv/iemo_train_with_vad.csv")
    else:
        df_vad = pd.read_csv("vad_csv/iemo_test_with_vad.csv")

    # DataFrame を検索しやすいようにインデックス化
    df_lookup = df_vad.set_index(["vid", "utter_id_y"])
    


    #保存用
    all_fused = []
    all_labels = []
    all_vads = []


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
            texts = data[-2]  # 発話テキスト

            #np配列に変換
            umask_np = umask.cpu().numpy()
            label_np = label.cpu().numpy()

            valid_fused = []
            valid_labels = []

            vad_batch = []  # バッチ全体の VAD を格納

            for b, vid in enumerate(vids):  # バッチ内ループ
                num_utts = lengths[b]
                vad_list = []

                for utt_id in range(num_utts):
                    try:
                        vad = df_lookup.loc[(vid, utt_id), ["Valence", "Arousal", "Dominance"]].values
                    except KeyError:
                        vad = [np.nan, np.nan, np.nan]
                    vad_list.append(vad)

                vad_array = np.array(vad_list, dtype=float)  # (num_utts, 3)

                valid_idx = umask_np[b] == 1
                all_labels.append(label_np[b][valid_idx])    # (L_i,)
                all_vads.append(vad_array)                   # (L_i, 3)

    #1つのnp配列にまとめる
    labels        = np.concatenate(all_labels, axis=0)  # shape = (N_total,)
    vad_np        = np.concatenate(all_vads, axis=0)  # shape = (N_total,3(v,a,d))

    # --- 連結 ---
    vads_concat = np.vstack(all_vads)  # (N, 3)
    labels_concat = np.concatenate(all_labels)  # (N,)

    # --- DataFrame 作成 ---
    vad_df = pd.DataFrame(vads_concat, columns=["Valence", "Arousal", "Dominance"])
    vad_df["Label_id"] = labels_concat

    # --- 欠損値除外 ---
    vad_df = vad_df.dropna()

    # --- ラベルを文字列に変換 ---
    vad_df["Label"] = vad_df["Label_id"].map(label_mapping)

    # --- 全体の記述統計 ---
    summary_all = vad_df[["Valence", "Arousal", "Dominance"]].describe()
    summary_all.to_csv(f"{output_folder}/vad_distribution_all.csv")

    # --- ラベルごとの平均・標準偏差 ---
    group_stats = vad_df.groupby("Label")[["Valence", "Arousal", "Dominance"]].agg(["mean", "std"])
    group_stats.to_csv(f"{output_folder}/vad_distribution_by_label.csv")

    print("✅ CSV 保存完了: vad_distribution_all.csv, vad_distribution_by_label.csv")

    # --- 全体のヒストグラム ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, col in enumerate(["Valence", "Arousal", "Dominance"]):
        sns.histplot(vad_df[col], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {col} (All)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "histogram_all.png"))
    plt.close()
    print(f"✅ ヒストグラム保存: {output_folder}/histogram_all.png")

    # --- バイオリンプロット（1枚にまとめる） ---
    emotion_order = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for i, col in enumerate(["Valence", "Arousal", "Dominance"]):
        sns.violinplot(
            data=vad_df,
            x="Label",
            y=col,
            order=emotion_order,  # ← これを追加
            ax=axes[i]
        )
        axes[i].set_title(f"{col} per Emotion Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "violin_all.png"))
    plt.close()
    print(f"✅ バイオリン保存: {output_folder}/violin_all.png")
