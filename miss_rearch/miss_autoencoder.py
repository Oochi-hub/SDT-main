#autoencoderモデル(バイナリ表現)の分類精度調査
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import classification_report

# CSVを読み込み（区切り文字は適宜変更）
# df = pd.read_csv("../experience_results/0929_train_for_autoencoder/text/all_pred.csv") #テキスト
# df = pd.read_csv("../experience_results/0929_train_for_autoencoder/audio/all_pred.csv") #音声
# df = pd.read_csv("../experience_results/0929_train_for_autoencoder/visual/all_pred.csv") #視覚

# df = pd.read_csv("../experience_results/0930_evaldata_training_ex/model/text/all_pred.csv") #テキスト
# df = pd.read_csv("../experience_results/0930_evaldata_training_ex/model/audio/all_pred.csv") #音声
df = pd.read_csv("../experience_results/0930_evaldata_training_ex/model/visual/all_pred.csv") #視覚

# 複数のtrueラベルを指定
target_labels = [1,2]
# target_labels = [0, 3, 4]

# 抽出
# subset = df[df["true"].isin(target_labels)][["vid", "utt_index"]]
subset = df[(df["true"] == df["pred"])][["vid", "utt_index"]] #テキスト 
# subset = df[(df["true"].isin(target_labels)) & (df["true"] == df["pred"])][["vid", "utt_index"]] #音声 視覚


# df2_t = pd.read_csv("../experience_results/0929_autoencoder/not_fine_tune_last/temp2/100epoch/symmetry/text/all_pred.csv") #テキスト
# df2_a = pd.read_csv("../experience_results/0929_autoencoder/not_fine_tune_last/temp2/100epoch/symmetry/audio/all_pred.csv") #音声
# df2 = pd.read_csv("../experience_results/0929_autoencoder/not_fine_tune_last/temp2/100epoch/symmetry/visual/all_pred.csv") #視覚

df2_t = pd.read_csv("../experience_results/0930_evaldata_training_ex/autoencoder/last/temp1/text/all_pred.csv") #テキスト
df2_a = pd.read_csv("../experience_results/0930_evaldata_training_ex/autoencoder/last/temp1/audio/all_pred.csv") #音声
df2 = pd.read_csv("../experience_results/0930_evaldata_training_ex/autoencoder/last/temp1/visual/all_pred.csv") #視覚

# df2_t = pd.read_csv("../experience_results/0930_evaldata_training_ex/autoencoder/last/temp2/text/all_pred.csv") #テキスト
# df2_a = pd.read_csv("../experience_results/0930_evaldata_training_ex/autoencoder/last/temp2/audio/all_pred.csv") #音声
# df2 = pd.read_csv("../experience_results/0930_evaldata_training_ex/autoencoder/last/temp2/visual/all_pred.csv") #視覚

def acc(df2, subset):
    # df2からsubsetと一致する行を取り出す
    result = df2.merge(subset, on=["vid", "utt_index"], how="inner")

    # result に pred, true が含まれている前提
    y_true = result["true"]
    y_pred = result["pred"]

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)


def report(df2, subset):
    # df2からsubsetと一致する行を取り出す
    result = df2.merge(subset, on=["vid", "utt_index"], how="inner")
    # result に pred, true が含まれている前提
    y_true = result["true"]
    y_pred = result["pred"]

    print(classification_report(y_true, y_pred, digits=4))

print("text")
acc(df2_t, subset)
print("audio")
acc(df2_a, subset)
print("visual")
acc(df2, subset)

print("text")
report(df2_t, subset)
print("audio")
report(df2_a, subset)
print("visual")
report(df2, subset)