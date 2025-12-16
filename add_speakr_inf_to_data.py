import pickle
import pandas as pd
import numpy as np

#MELD論文を基に，主要6キャラを設定
speaker2id = {"Chandler":0, "Ross":1, "Phoebe":2, "Monica":3, "Joey":4, "Rachel":5}

# pkl読み込み
##データセットの読み込み
pkl_path = "data/meld_multimodal_features.pkl"
with open(pkl_path, "rb") as f:
    videoIDs, videoSpeakers, videoLabels, videoText, \
    roberta2, roberta3, roberta4, \
    videoAudio, videoVisual, videoSentence, trainVid, \
    testVid, _ = pickle.load(f)

"""
videoSpeakers:話者埋め込みのための情報
type: videoCharacter
size: 1432
内容: [発話数, 話者を区別する配列(9)]
"""

# CSV読み込み
#datasetに含まれていないキャラクターIDがある
df = pd.read_csv("meld_data/csv/all_sent_emo.csv", sep=",")


#キャラクター情報収納用
# videoCharacter = {i: [] for i in range(1433)}

# print(df[df["Dialogue_ID"]==0]["Speaker"].tolist())

#欠損確認用 vid=60が欠損している
# keys = sorted(videoSpeakers.keys())  # キーをソート
# expected = list(range(min(keys), max(keys)+1))  # 途切れない連番

# if keys == expected:
#     print("キーは途切れなく揃っています")
# else:
#     print("欠けているキーがあります:", set(expected) - set(keys))

#確認用
count = 0

# for i in range(1433):
videoCharacter = {}
for k in videoSpeakers.keys():

    # if i == 60:
    #     continue

    #会話におけるキャラ情報を発話順に取得
    name_list = df[df["Dialogue_ID"]==k]["Speaker"].tolist()

    speaker_list = []

    for t in range(len(name_list)):
        
        #キャラ情報保存配列
        id_list = [0] * 7

        #この発話におけるキャラ名を取得
        speaker = name_list[t]

        if speaker in speaker2id:
            id_list[speaker2id[speaker]] = 1
        else:
            id_list[-1] = 1
        #各発話のキャラ情報を会話単位で保存
        speaker_list.append(id_list)
    videoCharacter[k] = speaker_list

    #確認用
    dataset_num = np.shape(videoSpeakers[k])[0]
    videoCharacter_num = np.shape(videoCharacter[k])[0]

    if dataset_num != videoCharacter_num:
        count += 1

print("不一致数:", count)

#欠損しているvid=60を削除
# del videoCharacter[60]

print("確認用スクリプト")
print()
print("###type###")
print(f"videoSpeaker  :{type(videoSpeakers)}")
print(f"videoCharacter:{type(videoCharacter)}")
print()
print("###size###")
print(f"videoSpeaker  :{len(list(videoSpeakers))}")
print(f"videoCharacter:{len(list(videoCharacter))}")
print()
print("###fist idx shape###")
print(f"videoSpeaker  :{np.shape(videoSpeakers[0])}")
print(f"videoCharacter:{np.shape(videoCharacter[0])}")

print(videoCharacter[0])

"""
確認用スクリプト 出力結果

###type###
videoSpeaker  :<class 'videoCharacter'>
videoCharacter:<class 'videoCharacter'>

###size###
videoSpeaker  :1432
videoCharacter:1432

###fist idx shape###
videoSpeaker  :(14, 9)
videoCharacter:(14, 6)
"""

#新しいデータセットとして保存
new_data = (
    videoIDs, videoSpeakers, videoLabels, videoText,
    roberta2, roberta3, roberta4,
    videoAudio, videoVisual, videoSentence, trainVid,
    testVid, videoCharacter
)

with open("data/meld_with_characterID.pkl", "wb") as f:
    pickle.dump(new_data, f)