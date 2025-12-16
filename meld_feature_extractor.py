#MELDデータセット用　SDTモデルデータローダー生成

import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

import opensmile
import cv2
from torchvision import models, transforms
import time

# ========== SETTINGS ==========
DATA_DIR = './meld_data/csv/'
WAV_DIR = '.meld_data/wav/'
VIDEO_DIR = '.meld_data/video/'
SAVE_PATH = './meld_features_demo.pkl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FRAMES = 15  #videoから切り取るフレーム数
MAX_SPEAKER_NUM = 9 #一つの会話における登場人物の最大値 MELDは9

# ========== INIT ==========
videoIDs, videoSpeakers, videoLabels = defaultdict(list), defaultdict(list), defaultdict(list)
videoText, roberta2, roberta3, roberta4 = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
videoAudio, videoVisual, videoSentence = defaultdict(list), defaultdict(list), defaultdict(list)
trainVid, testVid = [], []


# ========== EMOTION LABELS ==========
EMOTION_LABELS = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}

# ========== CSV ==========
# PARTITIONS = {
#     'train': 'train_sent_emo.csv',
#     'dev': 'dev_sent_emo.csv',
#     'test': 'test_sent_emo.csv'
# }
PARTITIONS = {
    'dev': 'dev_sent_emo.csv'
}

# ========== EXTRACTORS ==========

#テキスト特徴量抽出 RoBERTa
class TextFeatureExtractor:
    def __init__(self, device='cpu'):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = RobertaModel.from_pretrained('roberta-large').to(device).eval()
        self.device = device

    def extract(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

#オーディオ特徴量抽出 openSMILE
class AudioFeatureExtractor:
    def __init__(self, device='cpu'):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals
        )
        self.linear = nn.Linear(6373, 300).to(device)  #線形層で300次元に変換
        self.device = device

    def extract(self, wav_path):
        if not os.path.exists(wav_path):
            return np.zeros(300)
        features = self.smile.process_file(wav_path).values[0]
        x = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            x = self.linear(x)
        return x.cpu().numpy()

#ビジュアル特徴量抽出 DenseNet
class VisualFeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = torch.nn.Identity()
        self.model = self.model.to(device).eval()
        self.proj = nn.Linear(1024, 342).to(device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, video_path):
        if not os.path.exists(video_path):
            return np.zeros(342)

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            cap.release()
            return np.zeros(342)

        indices = np.linspace(0, frame_count - 1, NUM_FRAMES, dtype=int)
        frame_feats = []
        current_idx = 0

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i == indices[current_idx]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.model(frame).squeeze(0)
                    feat = self.proj(feat)
                frame_feats.append(feat.cpu().numpy())
                current_idx += 1
                if current_idx >= len(indices):
                    break
        cap.release()

        return np.mean(frame_feats, axis=0) if frame_feats else np.zeros(342)
    
# ========== TEST MODE ==========
DEBUG = True  # ← True にするとテスト用の最初の N 件だけ処理
MAX_DIALOGUES = 3

# ========== PROCESS ==========
text_extractor = TextFeatureExtractor(DEVICE)
audio_extractor = AudioFeatureExtractor(DEVICE)
visual_extractor = VisualFeatureExtractor(DEVICE)

for split, filename in PARTITIONS.items():
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    for vid, group in df.groupby('Dialogue_ID'):
        # 処理前の時刻
        # t1 = time.time() 

        group = group.sort_values('Utterance_ID')
        vid = str(vid)

        speaker_index = {}
        next_index = 0

        for _, row in group.iterrows():
            uid = row['Utterance_ID']
            spk = row['Speaker']
            emo = row['Emotion']
            utt = row['Utterance']



            if spk not in speaker_index:
                speaker_index[spk] = next_index
                next_index += 1

            one_hot = [0] * MAX_SPEAKER_NUM
            one_hot[speaker_index[spk]] = 1

            wav_path = os.path.join(WAV_DIR, f"dia{vid}_utt{uid}.wav")
            video_path = os.path.join(VIDEO_DIR, f"dia{vid}_utt{uid}.mp4")

            videoIDs[vid].append(uid)
            videoSpeakers[vid].append(one_hot)
            videoLabels[vid].append(EMOTION_LABELS[emo])
            videoSentence[vid].append(utt)

            videoText[vid].append(text_extractor.extract(utt))
            roberta2[vid].append(text_extractor.extract(utt))
            roberta3[vid].append(text_extractor.extract(utt))
            roberta4[vid].append(text_extractor.extract(utt))
            videoAudio[vid].append(audio_extractor.extract(wav_path))
            videoVisual[vid].append(visual_extractor.extract(video_path))

        (trainVid if split != 'test' else testVid).append(vid)
        # # 処理後の時刻
        # t2 = time.time()
        
        # # 経過時間を表示
        # elapsed_time = t2-t1
        # print(f"経過時間：{elapsed_time}")
        # if vid == 0:
        #     print("==== DEBUG OUTPUT ====")
        #     for i, vid in enumerate(videoIDs.keys()):
        #         print(f"Video: {vid}")
        #         print(f"  Speakers  : {videoSpeakers[vid]}")
        #         print(f"  Labels    : {videoLabels[vid]}")
        #         print(f"  Sentences : {videoSentence[vid]}")
        #         print(f"Audio Feature : {np.shape(videoAudio[vid])}")
        #         print(f"Visual Feature : {np.shape(videoVisual[vid])}")
        #         print(f" Feature : {np.shape(videoText[vid])}")
        #     exit()

print("==== DEBUG OUTPUT ====")
for i, vid in enumerate(videoIDs.keys()):
    print(f"Video: {vid}")
    print(f"  Speakers  : {videoSpeakers[vid]}")
    print(f"  Labels    : {videoLabels[vid]}")
    print(f"  Sentences : {videoSentence[vid]}")
    print(f"Audio Feature : {np.shape(videoAudio[vid])}")
    print(f"Visual Feature : {np.shape(videoVisual[vid])}")
    print(f" Feature : {np.shape(videoText[vid])}")
    if DEBUG and i >= MAX_DIALOGUES:
        break


# ========== SAVE ==========
# data_tuple = (
#     dict(videoIDs), dict(videoSpeakers), dict(videoLabels), dict(videoText),
#     dict(roberta2), dict(roberta3), dict(roberta4),
#     dict(videoAudio), dict(videoVisual), dict(videoSentence),
#     trainVid, testVid, None
# )

# with open(SAVE_PATH, 'wb') as f:
#     pickle.dump(data_tuple, f)

# print(f"Saved features to {SAVE_PATH}")
