#誤分類例調査用
#vidとutter_idから元データを特定
import pandas as pd

# CSVファイルを読み込み
df1 = pd.read_csv("../IEMO_test/misclassified.csv") #誤分類
#df1 = pd.read_csv("../iemo_test_data_info.csv") #誤分類
df2 = pd.read_csv("../iemocap_data/csv/iemocap_test.csv") #テストデータ

# 一致させたい列（複数）
key_columns = ["vid", "utt_index"]

# df1["vid"] = df1["vid"] - 1153

#両方のデータで一致している行を取得
# matched_rows = pd.merge(df1, df2, on=key_columns, how="inner")
matched_rows = pd.merge(
    df1,
    df2,
    left_on=["vid", "utt_index"],   # df1側の列名
    right_on=["vid", "utter_id_y"],           # df2側の列名
    how="inner"
)

#列名とマッピング辞書を指定
class_column = "emotion"
label_column = "actual"
label_mapping = {
    "neutral": 0,
    "surprise": 1,
    "fear": 2,
    "sadness":3,
    "joy":4,
    "disgust":5,
    "anger":6
}

# 対応が正しい行をチェック
matched_rows["expected_label"] = matched_rows[class_column].map(label_mapping)
matched_rows["is_correct"] = matched_rows[label_column] == matched_rows["expected_label"]

# matched_rows["is_correct"] = matched_rows["actual"] == matched_rows["label"]

# さらにフィルタ条件を指定（例：status列が"active"の行）
# filtered_rows = matched_rows[matched_rows["emotion"] == "hap"]
# filtered_rows = matched_rows[
#     (matched_rows["status"] == "active") & (matched_rows["amount"] >= 10000)
# ]

# matched_rows = matched_rows[~matched_rows["is_correct"]]

# 結果の表示
print(matched_rows.shape[0])

# 必要であれば保存
matched_rows.to_csv("filtered_result2.csv", index=False)

#列名とマッピング辞書を指定
# class_column = "emotion"
# label_column = "label"
# label_mapping = {
#     "hap": 0,
#     "sad": 1,
#     "neu": 2,
#     "ang":3,
#     "exc":4,
#     "fru":5,
# }

# # 対応が正しい行をチェック
# df2["expected_label"] = df2[class_column].map(label_mapping)
# df2["is_correct"] = df2[label_column] == df2["expected_label"]

# # 正しくない行を抽出
# incorrect_rows = df2[~df2["is_correct"]]

# # 結果出力
# if incorrect_rows.empty:
#     print("✅ すべての行でクラス名とラベルの対応が正しいです。")
# else:
#     print("⚠️ クラス名とラベルの対応が正しくない行があります:")
#     print(incorrect_rows[[class_column, label_column, "expected_label"]])

# print(incorrect_rows.shape[0])