#取得した誤分類発話を含む会話を元データから抽出したい
#vidとutter_idから元データを特定
import pandas as pd

####MELD ver##########################################


# CSVファイルを読み込み
df1 = pd.read_csv("../520_MELD/misclassified.csv") #誤分類
df2 = pd.read_csv("../meld_data/csv/test_sent_emo.csv") #テストデータ

#vidを0始まりに
df1["vid"] = df1["vid"] - 1153

# #両方のデータで一致している行を取得
# matched_rows = pd.merge(
#     df1,
#     df2,
#     left_on=["vid", "utt_index"],   # df1側の列名
#     right_on=["Dialogue_ID", "Utterance_ID"],           # df2側の列名
#     how="inner"
# )

# #確認
# # print(len(matched_rows))

# #誤分類を含む会話の確認
# df_check = df2
# df_check =
# 誤分類データと元のテストデータを読み込み
misclassified_df = pd.read_csv("../520_MELD/misclassified.csv")
test_df = pd.read_csv("../meld_data/csv/test_sent_emo.csv")

# vidを0始まりに調整
misclassified_df["vid"] = misclassified_df["vid"] - 1153

# 該当するDialogue_IDとUtterance_IDの組をセットに
misclassified_keys = set(zip(misclassified_df["vid"], misclassified_df["utt_index"]))

# 対象となるDialogue_ID一覧を取得
target_dialogue_ids = misclassified_df["vid"].unique()

# test_dfから該当する会話を抽出
subset_df = test_df[test_df["Dialogue_ID"].isin(target_dialogue_ids)].copy()

# miss列を作成
subset_df["miss"] = subset_df.apply(
    lambda row: (row["Dialogue_ID"], row["Utterance_ID"]) in misclassified_keys,
    axis=1
)

# actual列を付与するためにマージ
actual_map = misclassified_df.set_index(["vid", "utt_index"])["actual"]
subset_df["actual"] = subset_df.apply(
    lambda row: actual_map.get((row["Dialogue_ID"], row["Utterance_ID"]), None),
    axis=1
)

# actual列の数値をクラス名に変換
name_class = {0:'neutral', 1:'surprise', 2:'fear', 3:'sadness', 4:'joy', 5:'disgust', 6:'anger'}
subset_df["actual"] = subset_df["actual"].map(name_class)

# miss列をUtterance列の隣に移動
if "Utterance" in subset_df.columns:
    cols = list(subset_df.columns)
    cols.insert(cols.index("Utterance") + 1, cols.pop(cols.index("miss")))
    subset_df = subset_df[cols]
    # 次に"actual"を移動
    cols.insert(cols.index("Utterance") + 2, cols.pop(cols.index("actual")))
    subset_df = subset_df[cols]


# 結果出力（例: CSVに保存）
# subset_df.to_csv("misclassified_dialogues_with_flags_by_meld.csv", index=False)

# 表示用
target_id = 11
print(subset_df[subset_df["Dialogue_ID"]==target_id])