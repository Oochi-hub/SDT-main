#iemocapにvad評価を追加する
import pickle
import pandas as pd
import numpy as np



# pkl読み込み
##データセットの読み込み
videoIDs, videoSpeakers, videoLabels, videoText,\
roberta2, roberta3, roberta4, \
videoAudio, videoVisual, videoSentence, trainVid,\
testVid = pickle.load(open('data/iemocap_multimodal_features.pkl', 'rb'), encoding='latin1')



exit()
print("確認用スクリプト")
print()
print("###type###")
print(f"videoSpeaker  :{type(videoSpeakers)}")
print(f"videoCharacter:{type(videoVAD1)}")
print()
print("###size###")
print(f"videoSpeaker  :{len(list(videoSpeakers))}")
print(f"videoCharacter:{len(list(videoVAD1))}")
print()
print("###fist idx shape###")
print(f"videoSpeaker  :{np.shape(videoSpeakers[0])}")
print(f"videoCharacter:{np.shape(videoVAD1[0])}")

print(videoVAD1[0])


#新しいデータセットとして保存
new_data = (
    videoIDs, videoSpeakers, videoLabels, videoText,
    roberta2, roberta3, roberta4,
    videoAudio, videoVisual, videoSentence, trainVid,
    testVid, videoVAD1, videoVAD2
)

with open("data/iemocap_with_VAD.pkl", "wb") as f:
    pickle.dump(new_data, f)