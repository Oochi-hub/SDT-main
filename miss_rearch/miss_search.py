import pandas as pd
import ast
import numpy as np
from sklearn.metrics import classification_report

# 3つのCSVファイルを読み込み
file1 = pd.read_csv("../experience_results/0916_ex/text/IEMOCAP/all_pred.csv")
file2 = pd.read_csv("../experience_results/0916_ex/audio/IEMOCAP/all_pred.csv")
file3 = pd.read_csv("../experience_results/0916_ex/visual/IEMOCAP/all_pred.csv")

file1 = pd.read_csv("../experience_results/0926_ex/text/IEMOCAP/all_pred.csv")
file2 = pd.read_csv("../experience_results/0926_ex/audio/IEMOCAP/all_pred.csv")
file3 = pd.read_csv("../experience_results/0926_ex/visual/IEMOCAP/all_pred.csv")

# file1 = pd.read_csv("../experience_results/0925_ex/text/IEMOCAP/all_pred.csv")
# file2 = pd.read_csv("../experience_results/0925_ex/audio/IEMOCAP/all_pred.csv")
# file3 = pd.read_csv("../experience_results/0925_ex/visual/IEMOCAP/all_pred.csv")

# file1 = pd.read_csv("../experience_results/0925_evaldata_training_ex/valid_test/text/IEMOCAP/all_pred.csv")
# file2 = pd.read_csv("../experience_results/0925_evaldata_training_ex/valid_test/audio/IEMOCAP/all_pred.csv")
# file3 = pd.read_csv("../experience_results/0925_evaldata_training_ex/valid_test/visual/IEMOCAP/all_pred.csv")


# pred列を区別する
file1 = file1[["vid", "utt_index", "true", "pred"]].rename(columns={"pred": "pred_file1"})
file2 = file2[["vid", "utt_index", "pred"]].rename(columns={"pred": "pred_file2"})
file3 = file3[["vid", "utt_index", "pred"]].rename(columns={"pred": "pred_file3"})

# マージ（vid と utt_index をキーにする）
merged = file1.merge(file2, on=["vid", "utt_index"]).merge(file3, on=["vid", "utt_index"])

# ===== 正解判定列を追加 =====
merged["m1_correct"] = merged["pred_file1"] == merged["true"]
merged["m2_correct"] = merged["pred_file2"] == merged["true"]
merged["m3_correct"] = merged["pred_file3"] == merged["true"]

def count_correct_combinations(df):
    """
    各モデルの正解/不正解の組み合わせごとに事例数を数える
    df: DataFrame (列に m1_correct, m2_correct, m3_correct が含まれる想定)
    return: DataFrame（組み合わせごとの件数）
    """
    # 各サンプルの正誤パターンをタプル化
    df["pattern"] = list(zip(df["m1_correct"], df["m2_correct"], df["m3_correct"]))
    
    # 出現回数をカウント
    counts = df["pattern"].value_counts().reset_index()
    counts.columns = ["pattern", "count"]

    # 可読性のために列分解
    counts[["m1", "m2", "m3"]] = pd.DataFrame(counts["pattern"].tolist(), index=counts.index)

    # 並べ替え（多い順）
    counts = counts[["m1", "m2", "m3", "count"]].sort_values("count", ascending=False)

    return counts

def count_by_class(df, combo=(True, True, False)):
    """
    指定した正解/不正解の組み合わせに一致する事例を
    感情クラスごとにカウントする関数
    combo: (m1_correct, m2_correct, m3_correct) のタプル
    """
    # パターン列を作成
    df = df.copy()
    df["pattern"] = list(zip(df["m1_correct"], df["m2_correct"], df["m3_correct"]))
    
    # 指定した組み合わせだけ抽出
    subset = df[df["pattern"] == combo]
    
    # 感情クラスごとに件数をカウント
    return subset["true"].value_counts().sort_index()

# 例: file1とfile2が正解、file3が不正解
counts = count_by_class(merged, combo=(True, False, False))
print(counts)
counts2 = count_by_class(merged, combo=(False, True, False))
print(counts2)
counts3 = count_by_class(merged, combo=(False, False, True))
print(counts3)

# ✅ 使い方（例）
# merged は既に m1_correct, m2_correct, m3_correct を持っている前提
combo_counts = count_correct_combinations(merged)
print(combo_counts)

exit()

# # ===== 設定 =====
# target_label = 2   # <-- ここを変えて検索したいtrueラベルを指定

# # pred列を区別する
# file1 = file1[["vid", "utt_index", "true", "pred"]].rename(columns={"pred": "pred_file1"})
# file2 = file2[["vid", "utt_index", "pred"]].rename(columns={"pred": "pred_file2"})
# file3 = file3[["vid", "utt_index", "pred"]].rename(columns={"pred": "pred_file3"})

# pred列を区別する
file1 = file1[["vid", "utt_index", "true", "pred"]].rename(columns={"pred": "pred_file1"})
file2 = file2[["vid", "utt_index", "pred"]].rename(columns={"pred": "pred_file2"})
file3 = file3[["vid", "utt_index", "pred"]].rename(columns={"pred": "pred_file3"})

# マージ
merged = file1.merge(file2, on=["vid", "utt_index"]).merge(file3, on=["vid", "utt_index"])

# 正答フラグ
merged["m1_correct"] = merged["pred_file1"] == merged["true"]
merged["m2_correct"] = merged["pred_file2"] == merged["true"]
merged["m3_correct"] = merged["pred_file3"] == merged["true"]

# ensemble予測: 1つでも正解ならtrueラベルをそのまま予測として使う
merged["ensemble_pred"] = merged["true"]  # 仮に全部正解とみなす
mask = ~(merged["m1_correct"] | merged["m2_correct"] | merged["m3_correct"])  # 誰も正解してない場合
merged.loc[mask, "ensemble_pred"] = merged.loc[mask, "pred_file1"]  # 代表としてモデル1の予測を採用

# classification_report
report = classification_report(merged["true"], merged["ensemble_pred"], digits=4)
print("=== Ensemble Classification Report ===")
print(report)

# 1. ensemble_correct 列を追加
merged["ensemble_correct"] = merged[["m1_correct", "m2_correct", "m3_correct"]].any(axis=1)

# 2. ensemble_pred のロジック再確認
merged["ensemble_pred"] = merged["true"]  # 正解が1つでもある場合は true
mask = ~merged["ensemble_correct"]        # 誰も正解してない場合
merged.loc[mask, "ensemble_pred"] = merged.loc[mask, "pred_file1"]

# 3. サンプルを確認（最初の10件）
print("=== サンプル確認 ===")
print(merged[["true", "pred_file1", "pred_file2", "pred_file3", 
              "m1_correct", "m2_correct", "m3_correct", 
              "ensemble_correct", "ensemble_pred"]].head(10))

# 4. 正解数カウント
print("\n=== 集計確認 ===")
print("全サンプル数:", len(merged))
print("ensemble 正解数:", merged["ensemble_correct"].sum())
print("ensemble Accuracy:", merged["ensemble_correct"].mean())

exit()

# pred列を区別する
file1 = file1[["vid", "utt_index", "true", "pred"]].rename(columns={"pred": "pred_file1"})
file2 = file2[["vid", "utt_index", "pred"]].rename(columns={"pred": "pred_file2"})
file3 = file3[["vid", "utt_index", "pred"]].rename(columns={"pred": "pred_file3"})

# マージ（vid と utt_index をキーにする）
merged = file1.merge(file2, on=["vid", "utt_index"]).merge(file3, on=["vid", "utt_index"])

# 正答/誤答フラグ
merged["m1_correct"] = merged["pred_file1"] == merged["true"]
merged["m2_correct"] = merged["pred_file2"] == merged["true"]
merged["m3_correct"] = merged["pred_file3"] == merged["true"]

# モデルごとのみ正答
only_m1 = merged[(merged["m1_correct"]) & (~merged["m2_correct"]) & (~merged["m3_correct"])]
only_m2 = merged[(~merged["m1_correct"]) & (merged["m2_correct"]) & (~merged["m3_correct"])]
only_m3 = merged[(~merged["m1_correct"]) & (~merged["m2_correct"]) & (merged["m3_correct"])]

# 件数を感情クラスごとに集計
count_m1 = only_m1.groupby("true").size()
count_m2 = only_m2.groupby("true").size()
count_m3 = only_m3.groupby("true").size()

# 結果をまとめる
summary = pd.DataFrame({
    "model1_only": count_m1,
    "model2_only": count_m2,
    "model3_only": count_m3
}).fillna(0).astype(int)

print("=== 各感情クラスごとの唯一正答数 ===")
print(summary)



# マージ（vid と utt_index をキーにする）
merged = file1.merge(file2, on=["vid", "utt_index"]).merge(file3, on=["vid", "utt_index"])

# 条件:
# true == target_label
# file1 で正解（true == pred_file1）
# 他ファイルでは不一致がある
condition = (
    (merged["true"] == target_label) &
    (merged["true"] == merged["pred_file2"]) &
    (
        (merged["pred_file2"] != merged["true"]) |
        (merged["pred_file3"] != merged["true"])
    )
)

result = merged[condition]

# 結果をprint
# if result.empty:
#     print(f"true={target_label} の条件に一致するサンプルはありません。")
# else:
#     print(result.to_string(index=False))

def load_and_process(file_path):
    # CSV読み込み
    df = pd.read_csv(file_path)
    # all_prob を文字列からリストに変換
    df["all_prob"] = df["all_prob"].apply(lambda x: ast.literal_eval(x))
    return df

def compute_mean_distribution(df, file_label):
    # trueラベルごとに平均分布を計算
    records = []
    for label, group in df.groupby("true"):
        arr = np.array(group["all_prob"].tolist())
        mean_prob = arr.mean(axis=0)
        # 行データを作成（file名, trueラベル, 各クラス確率…）
        row = {"file": file_label, "true_label": label}
        for i, p in enumerate(mean_prob):
            row[f"class_{i}"] = p
        records.append(row)
    return records

# ===== 実行例 =====
files = ["../experience_results/0916_ex/text/IEMOCAP/all_pred.csv", 
         "../experience_results/0916_ex/audio/IEMOCAP/all_pred.csv", 
         "../experience_results/0916_ex/visual/IEMOCAP/all_pred.csv"]
all_results = []

for f in files:
    df = load_and_process(f)
    file_label = f.split(".")[0]  # 拡張子抜きの名前を使用
    all_results.extend(compute_mean_distribution(df, file_label))

# DataFrame化
result_df = pd.DataFrame(all_results)

# CSVに保存
result_df.to_csv("mean_prob_distribution.csv", index=False)

print("mean_prob_distribution.csv に保存しました。")
print(result_df.head())