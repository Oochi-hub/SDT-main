#T-SNE可視化
#バイナリオートエンコーダのバイナリ表現を可視化
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
        model(textf, visuf, acouf, umask, qmask, lengths, modal)

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
    
    return features[-1]#(バッチサイズ, 発話数, 256)

def compute_dim_label_correlation(fused_vectors, labels, label_mapping,
                               csv_path="dim_label_corr.csv",
                               fig_path="dim_label_corr.png"):
    """
    各次元と各ラベルの相関を計算し，CSVとヒートマップとして保存する
    
    Parameters
    ----------
    fused_vectors : np.ndarray
        shape = (N, D) のバイナリ特徴量
    labels : np.ndarray
        shape = (N,) のラベルID
    label_mapping : dict
        {label_id: label_name}
    csv_path : str
        相関結果を保存するCSVファイルのパス
    fig_path : str
        ヒートマップ画像を保存するファイルのパス
    """
    #発話数と次元数を取得
    N, D = fused_vectors.shape
    #クラスラベルを取得
    class_ids = sorted(label_mapping.keys())
    #クラス名を取得
    class_names = [label_mapping[c] for c in class_ids]

    #labelsをone-hot化
    one_hot = np.zeros((N, len(class_ids)))
    one_hot[np.arange(N), labels] = 1

    # 相関計算 (次元ごとに各クラスとの相関を取る)
    corr_matrix = np.zeros((len(class_ids), D))
    for i, c in enumerate(class_ids):
        for d in range(D):
            #全データのi次元目の要素
            x = fused_vectors[:, d]
            y = one_hot[:, i]
            if np.std(x) == 0 or np.std(y) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(x, y)[0, 1]
            corr_matrix[i, d] = corr

    # DataFrame化
    df_corr = pd.DataFrame(corr_matrix, index=class_names,
                           columns=[f"dim{d}" for d in range(D)])

    # CSV保存
    df_corr.to_csv(csv_path, encoding="utf-8")

    # ヒートマップ描画
    plt.figure(figsize=(20, 6))
    sns.heatmap(df_corr, cmap="coolwarm", center=0,
                xticklabels=False, yticklabels=True,
                cbar_kws={'label': 'Correlation'})
    plt.title("Dimension-Class Correlation Heatmap", fontsize=14)
    plt.ylabel("Emotion Class")
    plt.xlabel("Feature Dimension")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return df_corr

def save_dim_vad_correlation(fused_vectors, vad_np,
                             csv_path="dim_vad_corr.csv",
                             fig_path="dim_vad_corr.png"):


    N, D = fused_vectors.shape
    vad_names = ["Valence", "Arousal", "Dominance"]

    # --- NaN を含むサンプルを除外 ---
    mask = ~np.isnan(vad_np).any(axis=1)
    fused_vectors = fused_vectors[mask]
    vad_np = vad_np[mask]

    print(f"除外後のサンプル数: {fused_vectors.shape[0]} / {N}")

    corr_matrix = np.zeros((len(vad_names), D))

    for i in range(len(vad_names)):
        y = vad_np[:, i]
        for d in range(D):
            x = fused_vectors[:, d]
            if np.std(x) == 0 or np.std(y) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            corr_matrix[i, d] = corr

    # --- DataFrame として保存 ---
    df_corr = pd.DataFrame(corr_matrix, index=vad_names,
                           columns=[f"dim{d}" for d in range(D)])
    df_corr.to_csv(csv_path, encoding="utf-8")

    # --- ヒートマップ出力 ---
    plt.figure(figsize=(20, 4))
    sns.heatmap(df_corr, cmap="coolwarm", center=0,
                xticklabels=False, yticklabels=True,
                cbar_kws={'label': 'Correlation'})
    plt.title("Dimension-VAD Correlation Heatmap", fontsize=14)
    plt.ylabel("VAD Dimension")
    plt.xlabel("Feature Dimension")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return df_corr

def save_dim_label_prob(fused_vectors, labels, label_mapping,
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
        idx = (labels == c)
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

def vad_threshold_activation(fused_vectors, vad_np, threshold={"Valence":3.5, "Arousal":3.5, "Dominance":4.0},save_path="dim_vad_threshold_prob.png", inequality="larger"):
    """
    各VAD項目ごとに，指定した閾値以上のサンプルにおける発火確率を計算する

    Parameters
    ----------
    fused_vectors : np.ndarray, shape (N, D)
        各発話のバイナリ特徴量（0/1）
    vad_np : np.ndarray, shape (N, 3)
        各サンプルの Valence, Arousal, Dominance 値
    threshold : float
        閾値（例：4.0）
    vad_names : list
        VAD項目のラベル名

    Returns
    -------
    result_df : pd.DataFrame
        各VAD項目ごとの発火確率ベクトル（平均値）
        index: vad_names, columns: 次元インデックス
    """

    # 出力を格納する辞書
    result_dict = {}
    vad_names=["Valence", "Arousal", "Dominance"]
    for i, name in enumerate(vad_names):
        valid_mask = ~np.isnan(vad_np[:, i])  # NaN除外
        vad_values = vad_np[valid_mask, i]
        fused_valid = fused_vectors[valid_mask]


        # 閾値以上のサンプルを抽出
        if inequality == "larger":
            high_mask = vad_values >= threshold[name]
        else:
            high_mask = vad_values <= threshold[name]
        high_samples = fused_valid[high_mask]

        if len(high_samples) == 0:
            print(f"[Warning] {name}: 閾値を超えるサンプルがありません (threshold={threshold})")
            result_dict[name] = np.full(fused_vectors.shape[1], np.nan)
            continue

        # 発火確率 = 各次元の平均（0/1の平均 = 発火確率）
        probs = high_samples.mean(axis=0)
        result_dict[name] = probs

        print(f"{name}: 閾値以上 {len(high_samples)} 件 (全体 {len(vad_values)} 件)")

    # DataFrame化（行=VAD項目，列=次元）
    result_df = pd.DataFrame(result_dict).T
    result_df.index.name = "VAD_type"
    result_df.columns = [f"dim_{i}" for i in range(result_df.shape[1])]

    plt.figure(figsize=(20, 6))
    sns.heatmap(
        result_df,
        cmap="Reds",
        cbar=True,
        linewidths=0.3,
        xticklabels=False,  # 256次元すべて表示すると読みにくいため非表示
        yticklabels=result_df.index,
    )
    plt.title("Activation Probability (VAD ≥ threshold)", fontsize=14)
    plt.xlabel("Feature Dimension (0–255)")
    plt.ylabel("VAD Component")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return result_df


###============== 実装 ==============###



#データセット
which_data = "IEMOCAP" #"IEMOCAP" or "MELD"
train_flag = True #True or False 学習データかテストデータか

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



# exit()
for which_modal in ["text", "audio", "visual"]:
    for train_flag in [False, True]:

        #モダリティ対応辞書
        modal_dict = {"text":"t", "audio":"a", "visual":"v"}
    
        #保存先
        if train_flag:
            mode = "train"
        else:
            mode = "test"


        ##学習済みオートエンコーダの指定
        weight_folder = f"../experience_results/0929_autoencoder/not_fine_tune/100epoch/symmetry/{which_modal}"
        model_path = f"{weight_folder}/weights/model_weights_last.pth"
        model = torch.load(model_path)
        modal = modal_dict[which_modal] #モダリティ指定 t, a, v


        #モデル初期化
        model.eval()

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
        else:
            dataset = MELDDataset('../data/meld_multimodal_features.pkl', train=train_flag)
            loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=dataset.collate_fn)
            label_mapping = {
            0:"neutral",
            1:"surprise",
            2:"fear",
            3:"sadness",
            4:"joy",
            5:"disgust",
            6:"anger"
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
                texts = data[-2]  # 発話テキスト


                fused_vec = get_fused_features(model, textf, visuf, acouf, umask, qmask, lengths, modal)#(バッチサイズ, 発話数, 256)

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
                    all_fused.append(fused_vec[b][valid_idx])    # (L_i, D)
                    all_labels.append(label_np[b][valid_idx])    # (L_i,)
                    all_vads.append(vad_array)                   # (L_i, 3)

        #1つのnp配列にまとめる
        fused_vectors = np.concatenate(all_fused, axis=0)   # shape = (N_total, D)
        labels        = np.concatenate(all_labels, axis=0)  # shape = (N_total,)
        vad_np        = np.concatenate(all_vads, axis=0)  # shape = (N_total,3(v,a,d))


        # threshold= {"Valence":3.5, "Arousal":3.5, "Dominance":4.0}
        # inequality = "larger"
        threshold= {"Valence":2.0, "Arousal":2.5, "Dominance":2.5}
        inequality = "smaller" 

        output_folder_grah = f"results/1006_result/autoencoder/vad_prob/{which_modal}/{mode}"
        ##保存先
        if not os.path.exists(output_folder_grah):
            os.makedirs(output_folder_grah)
        save_path = output_folder_grah + f"/dim_vad_threshold_{inequality}_prob.png"
        result_df = vad_threshold_activation(fused_vectors, vad_np, threshold, save_path=save_path, inequality=inequality)
        # CSV保存
        result_df.to_csv(f"{output_folder_grah}/vad_threshold_{inequality}_prob.csv", index=True)

        # output_folder_grah = f"results/1003_result/autoencoder/correlation/{which_modal}/{mode}"
        # ##保存先
        # if not os.path.exists(output_folder_grah):
        #     os.makedirs(output_folder_grah)

        # csv_path = output_folder_grah + "/dim_label_correlatino.csv"
        # fig_path = output_folder_grah + "/dim_label_correlation.png"
        # df_corr = compute_dim_label_correlation(fused_vectors, labels, label_mapping, csv_path, fig_path)

        # output_folder_grah = f"results/1003_result/autoencoder/prob/{which_modal}/{mode}"
        # ##保存先
        # if not os.path.exists(output_folder_grah):
        #     os.makedirs(output_folder_grah)
        # csv_path = output_folder_grah + "/dim_label_prob.csv"
        # fig_path = output_folder_grah + "/dim_label_prob.png"
        # df_corr = save_dim_label_prob(fused_vectors, labels, label_mapping, csv_path, fig_path)

        # output_folder_grah = f"results/1003_result/autoencoder/vad/{which_modal}/{mode}"
        # ##保存先
        # if not os.path.exists(output_folder_grah):
        #     os.makedirs(output_folder_grah)
        # csv_path = output_folder_grah + "/dim_label_vad.csv"
        # fig_path = output_folder_grah + "/dim_label_vad.png"
        # res_vad = save_dim_vad_correlation(fused_vectors, vad_np, csv_path, fig_path)
        # print(df_corr.head())  # 各次元×各感情クラスの相関表

