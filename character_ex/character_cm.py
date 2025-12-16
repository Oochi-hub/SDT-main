# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

# #MELD論文を基に，主要6キャラを設定
# speaker2id = {0:"Chandler", 1:"Ross", 2:"Phoebe", 3:"Monica", 4:"Joey", 5:"Rachel", 6:"other"}

# # CSV読み込み
# df = pd.read_csv("../experience_results/0826_ex/MELD_characters/all_pred.csv")

# # 欠損を除外
# df = df.dropna(subset=["pred", "true"])
# df["pred"] = df["pred"].astype(int)
# df["true"] = df["true"].astype(int)

# # クラス名
# name_class = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
# num_classes = len(name_class)


# def plot_cm(cm, title, filename):
#     # 行ごとに割合
#     cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
#     cm_normalized = np.nan_to_num(cm_normalized)

#     # 件数＋割合の文字列
#     annot = np.empty_like(cm).astype(str)
#     for i in range(num_classes):
#         for j in range(num_classes):
#             count = cm[i, j]
#             ratio = cm_normalized[i, j]
#             annot[i, j] = f"{count}\n{ratio:.2f}"

#     # プロット
#     plt.figure(figsize=(7,6))
#     sns.heatmap(cm_normalized, annot=annot, fmt="", cmap="Blues",
#                 xticklabels=name_class, yticklabels=name_class,
#                 vmin=0, vmax=1, annot_kws={"size": 9})

#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title(title)
#     plt.savefig(filename, dpi=300, bbox_inches="tight")
#     plt.close()

# # ---- 全体の混同行列 ----
# cm_all = confusion_matrix(df["true"], df["pred"], labels=range(num_classes))
# plot_cm(cm_all, "Confusion Matrix (All characters)", "confusion_matrix_all.png")

# # ---- キャラクタごとの混同行列 ----
# for char_id, group in df.groupby("character"):
#     cm = confusion_matrix(group["true"], group["pred"], labels=range(num_classes))
#     plot_cm(cm,
#             f"Confusion Matrix - Character {speaker2id[char_id]}",
#             f"confusion_matrix_character_{speaker2id[char_id]}_count_ratio.png")
    
# char_acc = {}  # 結果を格納する辞書

# for char_id, group in df.groupby("character"):
#     cm = confusion_matrix(group["true"], group["pred"], labels=range(num_classes))
#     acc = np.trace(cm) / cm.sum()  # 正解率
#     char_acc[char_id] = acc

# # 結果表示
# for char_id, acc in char_acc.items():
#     print(f"Character {char_id} ({speaker2id.get(char_id, char_id)}): Accuracy = {acc:.2f}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

#MELD論文を基に，主要6キャラを設定
speaker2id = {0:"Chandler", 1:"Ross", 2:"Phoebe", 3:"Monica", 4:"Joey", 5:"Rachel", 6:"other"}

# CSV読み込み
df = pd.read_csv("../experience_results/0826_ex/MELD_characters/all_pred.csv")
#df = pd.read_csv("../experience_results/0905_ex/MELD_charaID_modal/all_pred.csv")

# 欠損を除外
df = df.dropna(subset=["pred", "true"])
df["pred"] = df["pred"].astype(int)
df["true"] = df["true"].astype(int)

# クラス名
name_class = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
num_classes = len(name_class)


def plot_cm(cm, title, filename):
    # 行ごとに割合
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    # 件数＋割合の文字列
    annot = np.empty_like(cm).astype(str)
    for i in range(num_classes):
        for j in range(num_classes):
            count = cm[i, j]
            ratio = cm_normalized[i, j]
            annot[i, j] = f"{count}\n{ratio:.2f}"

    # プロット
    plt.figure(figsize=(7,6))
    sns.heatmap(cm_normalized, annot=annot, fmt="", cmap="Blues",
                xticklabels=name_class, yticklabels=name_class,
                vmin=0, vmax=1, annot_kws={"size": 9})

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_cm_for_group(df, char_ids, group_name, filename):
    """
    任意の人物グループの混同行列を作成して保存する
    char_ids: list[int] -> キャラIDのリスト
    group_name: str     -> グループ名（タイトル/ファイル名用）
    filename: str       -> 出力ファイル名
    """
    # 指定した人物だけ抽出
    group_df = df[df["character"].isin(char_ids)]
    if group_df.empty:
        print(f"⚠️ グループ {group_name} はデータがありません")
        return

    # 混同行列
    cm = confusion_matrix(group_df["true"], group_df["pred"], labels=range(num_classes))

    # プロット
    plot_cm(cm,
            f"Confusion Matrix - Group: {group_name}",
            filename)

    # Accuracy
    acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    print(f"Group {group_name}: Accuracy = {acc:.3f}")

# ---- 全体の混同行列 ----
cm_all = confusion_matrix(df["true"], df["pred"], labels=range(num_classes))
plot_cm(cm_all, "Confusion Matrix (All characters)", "confusion_matrix_all.png")

# ---- キャラクタごとの混同行列と精度 ----
char_results = {}

for char_id, group in df.groupby("character"):
    cm = confusion_matrix(group["true"], group["pred"], labels=range(num_classes))
    plot_cm(cm,
            f"Confusion Matrix - Character {speaker2id[char_id]}",
            f"confusion_matrix_character_{speaker2id[char_id]}_count_ratio.png")
    
    # Accuracy
    acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    
    # Weighted avg（不均衡考慮）
    report = classification_report(group["true"], group["pred"], labels=range(num_classes),
                                   target_names=name_class, output_dict=True, zero_division=0)
    weighted_avg = report["weighted avg"]
    
    char_results[char_id] = {
        "accuracy": acc,
        "weighted_precision": weighted_avg["precision"],
        "weighted_recall": weighted_avg["recall"],
        "weighted_f1": weighted_avg["f1-score"]
    }

# 結果表示
for char_id, metrics in char_results.items():
    print(f"Character {char_id} ({speaker2id.get(char_id, char_id)}):")
    print(f"  Accuracy          = {metrics['accuracy']:.3f}")
    print(f"  Weighted Precision= {metrics['weighted_precision']:.3f}")
    print(f"  Weighted Recall   = {metrics['weighted_recall']:.3f}")
    print(f"  Weighted F1-score = {metrics['weighted_f1']:.3f}")
    print("-"*50)

# 男子まとめ
plot_cm_for_group(df, [0,1,4], "Man with modal_charaID", "confusion_matrix_men_modal_charaID.png")

# 女子まとめ
plot_cm_for_group(df, [2,3,5], "Women with modal_charaID", "confusion_matrix_women_modal_charaID.png")

