import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
from torch.utils.data import DataLoader

class IEMOCAPDataset(Dataset):
    def __init__(self, train=True, path='data/iemocap_multimodal_features.pkl'):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)
    #5/12 確認のためにvideoTextを追加
    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               self.videoSentence[vid],\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


class MELDDataset(Dataset):
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        #self.videoSentence, vidはモデルに入力しない
        #5/12 確認のためにvideoTextを追加
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               self.videoSentence[vid],\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]
    
class MELDDataset_c(Dataset):
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, self.videoCharacter = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        #self.videoSentence, vidはモデルに入力しない
        #5/12 確認のためにvideoTextを追加
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               torch.FloatTensor(self.videoCharacter[vid]),\
               self.videoSentence[vid],\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        #返り値の内 i<4,i==6 テンソル 4<i<6 パディングありテンソル else リスト
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6  else pad_sequence(dat[i]) if i==6 else dat[i].tolist() for i in dat]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # --- データセットを選択 ---
    from dataloader import IEMOCAPDataset, MELDDataset, MELDDataset_c
    
    dataset = IEMOCAPDataset(train=True)


    train_loader = DataLoader(dataset,
                            batch_size=16,
                            shuffle=False,
                            collate_fn=dataset.collate_fn,
                            num_workers=0,
                            pin_memory=False)

    
    for data in train_loader:

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-2]]

        texts = data[-2]  # 発話テキスト
        vids = data[-1]  # 会話ID

        print("textf: ",textf.shape)
        print("visuf: ", visuf.shape)
        print("acouf: ", acouf.shape)
        print("qmask: ", qmask.shape)
        print("umask: ", umask.shape)
        print("label: ", label.shape)
        print("vid:", vids)
        print("texts:", np.shape(texts[0]))

        print(texts[0])

        break