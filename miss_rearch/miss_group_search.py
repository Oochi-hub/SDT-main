#MELDのキャラクター別誤分類を調査
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix

def speaker_emo(df, output_dir):

    # クラス名の順番を定義（Emotion列はこの名前で構成されている前提）
    emotion_order = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']

    # 出現数（発話数）が多い上位 N 人を抽出
    N = 6
    top_speakers = df['Speaker'].value_counts().nlargest(N).index

    # フォルダがなければ作成
    os.makedirs(output_dir, exist_ok=True)  


    # 上位スピーカーの感情分布をプロットして保存
    for speaker in top_speakers:
        group = df[df['Speaker'] == speaker]
        emotion_counts = group['Emotion'].value_counts().reindex(emotion_order, fill_value=0)

        plt.figure(figsize=(8, 4))
        plt.bar(emotion_order, emotion_counts, color='skyblue')
        plt.title(f'Emotion Distribution for Speaker: {speaker}')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()

        safe_speaker = str(speaker).replace(" ", "_").replace("/", "_")
        output_path = os.path.join(output_dir, f"{safe_speaker}_emotion_distribution.png")
        plt.savefig(output_path)
        plt.close()

#人物ごとの棒グラフを一つにまとめる
def speaker_emo_same_grah(df, output_dir):

    # クラス名の順番を定義（Emotion列は文字列）
    emotion_order = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']

    # 上位 N 人のスピーカー
    N = 6  # ← ここを変更して表示人数を調整
    top_speakers = df['Speaker'].value_counts().nlargest(N).index

    # フィルタリング
    filtered_df = df[df['Speaker'].isin(top_speakers)]

    # クロステーブル（Emotion × Speaker）
    pivot_table = pd.crosstab(filtered_df['Emotion'], filtered_df['Speaker']).reindex(index=emotion_order, fill_value=0)

    # 描画準備
    fig, ax = plt.subplots(figsize=(15, 6))

    # プロット：EmotionごとにSpeakerの棒を並べる
    pivot_table.plot(kind='bar', ax=ax, colormap='tab10')

    # ラベル・スタイル設定
    ax.set_title(f'Emotion-wise Distribution by Top {N} Speakers')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Speaker', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 保存
    output_path = output_dir + '/emotion_distribution_top_speakers.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"1枚のグラフとして {output_path} に保存しました。")

def speakers_matrix(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # クラス名の順番
    emotion_map = {
    0: 'neutral',
    1: 'surprise',
    2: 'fear',
    3: 'sadness',
    4: 'joy',
    5: 'disgust',
    6: 'anger'
    }
    
    # 感情ラベルの順序
    emotion_labels = list(emotion_map.values())

    # 上位 N 人のスピーカー
    N = 6  # ← ここを変更して表示人数を調整
    top_speakers = df['Speaker'].value_counts().nlargest(N).index

    # フィルタリング
    filtered_df = df[df['Speaker'].isin(top_speakers)]
    # スピーカーごとにループ
    for speaker in filtered_df['Speaker'].unique():
        sub_df = filtered_df[filtered_df['Speaker'] == speaker]

        y_true = sub_df['actual'].map(emotion_map)
        y_pred = sub_df['predicted'].map(emotion_map)

        cm = confusion_matrix(y_true, y_pred, labels=emotion_labels)

        # 可視化
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=emotion_labels, yticklabels=emotion_labels)
        plt.title(f'Confusion Matrix - Speaker: {speaker}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        # 保存
        filename = os.path.join(output_dir, f'confusion_{speaker}.png')
        plt.savefig(filename)
        plt.close()

        print(f"✅ {speaker} の混同行列を保存しました: {filename}")

# 誤分類データと元のテストデータを読み込み
misclassified_df = pd.read_csv("../520_MELD/misclassified.csv")
test_df = pd.read_csv("../meld_data/csv/test_sent_emo.csv")


# #誤分類のSpeaker取得のため
# vidを0始まりに調整
misclassified_df["vid"] = misclassified_df["vid"] - 1153

matched_rows = pd.merge(
    misclassified_df,
    test_df,
    left_on=["vid", "utt_index"],   # df1側の列名
    right_on=["Dialogue_ID", "Utterance_ID"],           # df2側の列名
    how="inner"
)

# # カテゴリごとの件数をカウント
# miss_grouped_counts = matched_rows.groupby('Speaker').size().sort_values(ascending=False)
# test_grouped_counts = test_df.groupby('Speaker').size().sort_values(ascending=False)

# print("キャラごとの誤分類件数")
# print(miss_grouped_counts.head(10))

# print()
# print("テストデータのキャラごとの件数")
# print(test_grouped_counts.head(10))

#話者ごとの感情件数
output_dir = "meld_speaker_emo/matrix/test"

miss_flag = False

# speaker_emo(test_df, output_dir)
# speaker_emo_same_grah(test_df, output_dir)
speakers_matrix(matched_rows, output_dir)