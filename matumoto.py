#openSMILE 1582次元 確認用
#MELD test data 
"""
Sr No.	Utterance	Speaker	Emotion	Sentiment	Dialogue_ID	Utterance_ID	Season	Episode	StartTime	EndTime
1	Why do all youre coffee mugs have numbers on the bottom?	Mark	surprise	positive	0	0	3	19	00:14:38,127	00:14:40,378
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

# --- データセットを選択 ---
from dataloader import IEMOCAPDataset, MELDDataset
dataset = IEMOCAPDataset(train=False)
#dataset = MELDDataset("data/meld_multimodal_features.pkl", train=False)

sample = dataset[0]
textf, visualf, audiof, qmask, u_mask, _, sentence, vid = sample

qmask_np = qmask.numpy()            # shape: (seq_len, n_speakers)
u_mask_np = u_mask.numpy().reshape(-1)  # shape: (seq_len,)
sentence_list = sentence
print(sentence[0])
print(audiof[0].shape)
print(vid)
torch.set_printoptions(edgeitems=2000)
print(audiof[0])