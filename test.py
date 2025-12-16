#学習済みの重みを用いて，テストのみを行う
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset, MELDDataset_c
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model, WassersteinLoss, LogWassersteinLoss, Single_Modal_Transformer_Based_Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score
import pickle as pk
import datetime
import pandas as pd
import csv
import ast

# 学習率スケジューラー
from torch.optim import lr_scheduler

import random

#seed値の設定
#default 42
seed = 42
torch.manual_seed(seed)
random.seed(seed)


#パラメータ設定記録
def params_save_to_csv(params, output_folder):
    path = f"{output_folder}/arguemts.csv"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    df = pd.DataFrame([params])  # 引数をDataFrameに変換
    df.to_csv(path, index=False)  # CSVに保存


def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def get_MELD_c_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset_c('data/meld_with_characterID.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset_c('data/meld_with_characterID.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, kl_loss, ws_loss, dataloader, epoch, optimizer=None, train=False, data_flag="IEMOCAP", loss_func="kl"):

    #表示用の入れ子
    losses, preds, labels, masks, \
    ce_losses, kl_losses, ws_losses, task_losses, \
    ce_losses_t,ce_losses_v,ce_losses_a, \
    kl_losses_t,kl_losses_v,kl_losses_a, \
    ws_losses_t,ws_losses_v,ws_losses_a, \
    preds_t, preds_a, preds_v= [], [], [], [], \
                                         [], [], [], [], \
                                         [], [], [], \
                                         [], [], [], \
                                         [], [], [], \
                                         [], [], []
    misclassified = []
    all_pred = []

    #学習 or テスト
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    #最適化手法の初期化
    for data in dataloader:
        if train:
            optimizer.zero_grad()
    
        if data_flag == "MELD_c":
            textf, visuf, acouf, qmask, umask, label, cmatrix= [d.cuda() for d in data[:-2]] if cuda else data[:-2]
        
            # cmatrix = data[-3] # キャラクター情報
            texts = data[-2]  # 発話テキスト
            vids = data[-1]  # 会話ID

            cmatrix = cmatrix.permute(1, 0, 2)
            cmatrix_np = cmatrix.cpu().numpy()
        else:
            textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-2]] if cuda else data[:-2]
        
            texts = data[-2]  # 発話テキスト
            vids = data[-1]  # 会話ID

        qmask = qmask.permute(1, 0, 2) #qmask: 人物の区別のための行列   torch.Size([21, 8, 9])
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))] #各会話の発話数をumaskから逆算 list 長さ=バッチサイズ

        # print(texts)
        # exit()

        # print("textf: ",textf.shape)
        # print("visuf: ", visuf.shape)
        # print("acouf: ", acouf.shape)
        # print("qmask: ", qmask.shape)
        # print("umask: ", umask.shape)
        # print("label: ", label.shape)
        # print("texts:", np.shape(texts[0]))
        # print("cmatrix:", cmatrix.shape)
        # print('vids:', vids)

        """
        各入力の説明とデモ(21:バッチ内最大発話数, 16:バッチサイズ 9:MELDの最大話者 IEMOは2 7:MELDのキャラクタID数)
        textf: テキスト特徴量           torch.Size([21, 16, 1024])
        visuf: ビジュアル特徴量         torch.Size([21, 16, 342])
        acouf: オーディオ特徴量         torch.Size([21, 16, 300])
        qmask: 発話の区別のための行列   torch.Size([16, 21, 9])
        umask: 発話のパディングマスク   torch.Size([16, 21])
        label: 教師ラベル               torch.Size([16, 21])
        cmatrix: MELDのキャラクタ固有ID torch.Size([21, 16, 7])
        """

        """
        texts: (3,)
        vids: list [16(バッチサイズ)] 
        """
        #モダリティごと埋め込み用
        demo_charaID_flag = False

        if demo_charaID_flag:
            #prob1:t prob2:a prob3v
            log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
            kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob, \
            ws_t_prob, ws_a_prob, ws_v_prob, ws_all_prob, \
            t_prob, a_prob, v_prob = model(textf, visuf, acouf, umask, cmatrix, lengths, demo_charaID_flag)
        else:
            #prob1:t prob2:a prob3v
            log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
            kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob, \
            ws_t_prob, ws_a_prob, ws_v_prob, ws_all_prob, \
            t_prob, a_prob, v_prob = model(textf, visuf, acouf, umask, qmask, lengths, demo_charaID_flag)
        
        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[2])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[2])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[2])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])

        #自己蒸留損失項のモダリティごとの係数
        global DECAY_CE_T, DECAY_CE_A, DECAY_CE_V, DECAY_KL_T, DECAY_KL_A, DECAY_KL_V, GAMMA_1, GAMMA_2, GAMMA_3

        #親表現の分類損失
        L_task = GAMMA_1 * loss_function(lp_all, labels_, umask)

        #子表現の分類損失
        L_ce_t = loss_function(lp_1, labels_, umask)
        L_ce_a = loss_function(lp_2, labels_, umask)
        L_ce_v = loss_function(lp_3, labels_, umask)
        L_ce   = GAMMA_2 *(DECAY_CE_T * L_ce_t + DECAY_CE_V * L_ce_v + DECAY_CE_A * L_ce_a)

        #親表現と子表現のKL情報量誤差用
        L_kl_t = kl_loss(kl_lp_1, kl_p_all, umask)
        L_kl_a = kl_loss(kl_lp_2, kl_p_all, umask)
        L_kl_v = kl_loss(kl_lp_3, kl_p_all, umask)

        # print("gannma_3:", GAMMA_3)

        
        L_kl   = GAMMA_3 *(DECAY_KL_T * L_kl_t + DECAY_KL_V * L_kl_v + DECAY_KL_A * L_kl_a)


        if loss_func == "ws":
            #親表現と子表現のWS損失誤差
            L_ws_t = ws_loss(ws_t_prob, ws_all_prob, umask)
            L_ws_a = ws_loss(ws_a_prob, ws_all_prob, umask)
            L_ws_v = ws_loss(ws_v_prob, ws_all_prob, umask)

            L_ws   = GAMMA_3 *(DECAY_KL_T * L_ws_t + DECAY_KL_V * L_ws_v + DECAY_KL_A * L_ws_a)

            #モデル全体の損失関数 WS損失採用
            loss = L_task + L_ce + L_ws
        else:
            #モデル全体の損失関数 KL損失を採用
            # loss = L_task + L_ce + L_kl
            loss = L_task + L_kl

        #誤分類事例の回収
        lp_ = all_prob.view(-1, all_prob.size()[2])
        pred_ = torch.argmax(lp_,1)

        #単一モダリティの予測結果
        lp_t = t_prob.view(-1, t_prob.size()[2])
        lp_a = a_prob.view(-1, a_prob.size()[2])
        lp_v = v_prob.view(-1, v_prob.size()[2])
        pred_t = torch.argmax(lp_t,1)
        pred_a = torch.argmax(lp_a,1)
        pred_v = torch.argmax(lp_v,1)

        # print("pred_", pred_.size())
        # print("pred_t", pred_t.size())
        # print("pred_a", pred_a.size())
        # print("pred_v", pred_v.size())

        """
        pred_ torch.Size([1504])
        pred_t torch.Size([1504])
        pred_a torch.Size([1504])
        pred_v torch.Size([1504])

        """

    
        #誤分類事例の回収
        #all_prob torch.Size([batch, seq_len, num_class])
        all_probmax = torch.argmax(all_prob, dim=-1)  # shape: (batch, seq_len)

        #バッチサイズと最大発話数を記録
        x = umask.size()[0]
        y = umask.size()[1]

        #np配列に変換
        umask_np = umask.cpu().numpy()
        label_np = label.cpu().numpy()
        all_probmax_np = all_probmax.cpu().numpy()

        ###確率分布保存用　np配列に変換
        all_prob_np = all_prob.detach().cpu().numpy()
        t_prob_np = t_prob.detach().cpu().numpy()
        a_prob_np = a_prob.detach().cpu().numpy()
        v_prob_np = v_prob.detach().cpu().numpy()

        #テストにおける予測結果を記録
        for xi in range(x):
            vid_ = vids[xi]
            for yi in range(y):
                mask_value = umask_np[xi][yi]
                if mask_value == 1:
                    if data_flag == 'MELD_c':
                        #MELDでは各人物ごとにまとめる
                        character = list(cmatrix_np[xi][yi]).index(1)
                        all_pred.append({"vid": vid_,
                                    "utt_index": yi,
                                    "character": character,
                                    "pred": all_probmax_np[xi][yi],
                                    "true": label_np[xi][yi],
                                    "text": texts[xi][yi], 
                                    "all_prob": str(all_prob_np[xi][yi].tolist()), #csvに保存するので文字列に
                                    "t_prob": str(t_prob_np[xi][yi].tolist()),
                                    "a_prob": str(a_prob_np[xi][yi].tolist()),
                                    "v_prob": str(v_prob_np[xi][yi].tolist())
                                    })
                        #誤分類を収集
                        if all_probmax_np[xi][yi] != label_np[xi][yi]:
                            misclassified.append({"vid": vid_,
                                    "utt_index": yi,
                                    "character": character,
                                    "pred": all_probmax_np[xi][yi],
                                    "true": label_np[xi][yi],
                                    "text": texts[xi][yi], 
                                    "all_prob": str(all_prob_np[xi][yi].tolist()), #csvに保存するので文字列に
                                    "t_prob": str(t_prob_np[xi][yi].tolist()),
                                    "a_prob": str(a_prob_np[xi][yi].tolist()),
                                    "v_prob": str(v_prob_np[xi][yi].tolist())
                                    })
                    else:
                        #MELD_c以外はキャラ情報なしで記録
                        all_pred.append({"vid": vid_,
                                    "utt_index": yi,
                                    "pred": all_probmax_np[xi][yi],
                                    "true": label_np[xi][yi],
                                    "text": texts[xi][yi], 
                                    "all_prob": str(all_prob_np[xi][yi].tolist()), #csvに保存するので文字列に
                                    "t_prob": str(t_prob_np[xi][yi].tolist()),
                                    "a_prob": str(a_prob_np[xi][yi].tolist()),
                                    "v_prob": str(v_prob_np[xi][yi].tolist()) 
                                    })
                        #誤分類を収集
                        if all_probmax_np[xi][yi] != label_np[xi][yi]:
                            misclassified.append({"vid": vid_,
                                    "utt_index": yi,
                                    "pred": all_probmax_np[xi][yi],
                                    "true": label_np[xi][yi],
                                    "text": texts[xi][yi], 
                                    "all_prob": str(all_prob_np[xi][yi].tolist()),
                                    "t_prob": str(t_prob_np[xi][yi].tolist()),
                                    "a_prob": str(a_prob_np[xi][yi].tolist()),
                                    "v_prob": str(v_prob_np[xi][yi].tolist()) 
                                    })


        preds.append(pred_.data.cpu().numpy())

        #各モダリティの予測
        preds_t.append(pred_t.data.cpu().numpy())
        preds_a.append(pred_a.data.cpu().numpy())
        preds_v.append(pred_v.data.cpu().numpy())

        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
  

        #誤差逆伝搬
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    #配列を結合
    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
        preds_t = np.concatenate(preds_t)
        preds_a = np.concatenate(preds_a)
        preds_v = np.concatenate(preds_v)
    else:
        #avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, misclassified, L_ce, L_kl
        return float('nan'), float('nan'), [], [], [], float('nan'), [], [],\
               [], [], [] #for preds_m m in {t,a,v}


    #avg_ 表示用　有効サンプル数(発話数)で割る
    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)


    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, misclassified, all_pred,\
           preds_t, preds_a, preds_v

def uni_train_or_eval_model(model, loss_function, dataloader, optimizer=None, train=False, data_flag="IEMOCAP", modal="t"):

    #表示用の入れ子
    losses, preds, labels, masks, = [], [], [], []
    misclassified = []
    all_pred = []

    #学習 or テスト
    assert not train or optimizer!=None

    model.eval()
    #最適化手法の初期化
    for data in dataloader:
    
        if data_flag == "MELD_c":
            textf, visuf, acouf, qmask, umask, label, cmatrix= [d.cuda() for d in data[:-2]] if cuda else data[:-2]
        
            # cmatrix = data[-3] # キャラクター情報
            texts = data[-2]  # 発話テキスト
            vids = data[-1]  # 会話ID

            cmatrix = cmatrix.permute(1, 0, 2)
            cmatrix_np = cmatrix.cpu().numpy()
        else:
            textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-2]] if cuda else data[:-2]
        
            texts = data[-2]  # 発話テキスト
            vids = data[-1]  # 会話ID

        qmask = qmask.permute(1, 0, 2) #qmask: 人物の区別のための行列   torch.Size([21, 8, 9])
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))] #各会話の発話数をumaskから逆算 list 長さ=バッチサイズ

        # print(texts)
        # exit()

        # print("textf: ",textf.shape)
        # print("visuf: ", visuf.shape)
        # print("acouf: ", acouf.shape)
        # print("qmask: ", qmask.shape)
        # print("umask: ", umask.shape)
        # print("label: ", label.shape)
        # print("texts:", np.shape(texts[0]))
        # print("cmatrix:", cmatrix.shape)
        # print('vids:', vids)

        """
        各入力の説明とデモ(21:バッチ内最大発話数, 16:バッチサイズ 9:MELDの最大話者 IEMOは2 7:MELDのキャラクタID数)
        textf: テキスト特徴量           torch.Size([21, 16, 1024])
        visuf: ビジュアル特徴量         torch.Size([21, 16, 342])
        acouf: オーディオ特徴量         torch.Size([21, 16, 300])
        qmask: 発話の区別のための行列   torch.Size([16, 21, 9])
        umask: 発話のパディングマスク   torch.Size([16, 21])
        label: 教師ラベル               torch.Size([16, 21])
        cmatrix: MELDのキャラクタ固有ID torch.Size([21, 16, 7])
        """

        """
        texts: (3,)
        vids: list [16(バッチサイズ)] 
        """
        #モダリティごと埋め込み用
        demo_charaID_flag = False

        with torch.no_grad():
            if demo_charaID_flag:
                #prob1:t prob2:a prob3v
                log_prob, prob, _, _ = model(textf, visuf, acouf, umask, cmatrix, lengths, modal)
            else:
                #prob1:t prob2:a prob3v
                log_prob, prob, _, _ = model(textf, visuf, acouf, umask, qmask, lengths, modal)
        

        lp_all = log_prob.view(-1, log_prob.size()[2])
        labels_ = label.view(-1)

        #親表現の分類損失
        L_task = loss_function(lp_all, labels_, umask)

        #誤分類事例の回収
        lp_ = prob.view(-1, prob.size()[2])
        pred_ = torch.argmax(lp_,1)



        # print("pred_", pred_.size())
        """
        pred_ torch.Size([1504])
        """

        #誤分類事例の回収
        #all_prob torch.Size([batch, seq_len, num_class])
        all_probmax = torch.argmax(prob, dim=-1)  # shape: (batch, seq_len)

        #バッチサイズと最大発話数を記録
        x = umask.size()[0]
        y = umask.size()[1]

        #np配列に変換
        umask_np = umask.cpu().numpy()
        label_np = label.cpu().numpy()
        all_probmax_np = all_probmax.cpu().numpy()

        ###確率分布保存用　np配列に変換
        all_prob_np = prob.detach().cpu().numpy()

        #テストにおける予測結果を記録
        for xi in range(x):
            vid_ = vids[xi]
            for yi in range(y):
                mask_value = umask_np[xi][yi]
                if mask_value == 1:
                    if data_flag == 'MELD_c':
                        #MELDでは各人物ごとにまとめる
                        character = list(cmatrix_np[xi][yi]).index(1)
                        all_pred.append({"vid": vid_,
                                    "utt_index": yi,
                                    "character": character,
                                    "pred": all_probmax_np[xi][yi],
                                    "true": label_np[xi][yi],
                                    "text": texts[xi][yi], 
                                    "all_prob": str(all_prob_np[xi][yi].tolist()) #csvに保存するので文字列に
                                    })
                        #誤分類を収集
                        if all_probmax_np[xi][yi] != label_np[xi][yi]:
                            misclassified.append({"vid": vid_,
                                    "utt_index": yi,
                                    "character": character,
                                    "pred": all_probmax_np[xi][yi],
                                    "true": label_np[xi][yi],
                                    "text": texts[xi][yi], 
                                    "all_prob": str(all_prob_np[xi][yi].tolist()) #csvに保存するので文字列に
                                    })
                    else:
                        #MELD_c以外はキャラ情報なしで記録
                        all_pred.append({"vid": vid_,
                                    "utt_index": yi,
                                    "pred": all_probmax_np[xi][yi],
                                    "true": label_np[xi][yi],
                                    "text": texts[xi][yi], 
                                    "all_prob": str(all_prob_np[xi][yi].tolist()) #csvに保存するので文字列に
                                    })
                        #誤分類を収集
                        if all_probmax_np[xi][yi] != label_np[xi][yi]:
                            misclassified.append({"vid": vid_,
                                    "utt_index": yi,
                                    "pred": all_probmax_np[xi][yi],
                                    "true": label_np[xi][yi],
                                    "text": texts[xi][yi], 
                                    "all_prob": str(all_prob_np[xi][yi].tolist())
                                    })


        preds.append(pred_.data.cpu().numpy())



        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(L_task.item()*masks[-1].sum())
        
  


    #配列を結合
    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        #avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, misclassified, L_ce, L_kl
        return float('nan'), float('nan'), [], [], [], float('nan'), [], []


    #avg_ 表示用　有効サンプル数(発話数)で割る
    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)


    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, misclassified, all_pred





def make_cm(best_label, best_pred, best_mask, name_class, cm_name, folder="/cm"):
    #混同行列の作成
    cm = confusion_matrix(best_label,best_pred,sample_weight=best_mask)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=name_class)
    disp.plot(cmap=plt.cm.Blues)

    for text in disp.text_.flat:
        try:
            value = int(float(text.get_text()))
            text.set_text(f"{value:,}")  # カンマ区切りの整数表示（例：1,100）
        except:
            pass 

    output_folder = args.out_path + folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ラベルのフォントサイズと位置調整
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()  # レイアウト調整

    plt.savefig(output_folder+f'/{cm_name}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--out_path', default='demo')
    parser.add_argument('--dist_coefficients', default="[1.00,1.00,1.00 ,1.00,1.00,1.00]")
    parser.add_argument('--change_epoch', default=50, type=int) #動的な自己蒸留係数用 変更を検討するタイミング
    parser.add_argument('--add_kl', type=float, default=0.5)
    parser.add_argument('--schedular', type=float, default=1.0) #ExponentialLR用 epochごとに何倍するか
    parser.add_argument('--dynamic', action='store_true', default=False)#動的な係数の有無
    parser.add_argument('--loss_func', default='kl', help='which use loss_func, kl or ws') #損失関数にKLかWSのどちらを使うか
    parser.add_argument('--model', type=str, default="multi", help='multi or text or audio or visual')
    parser.add_argument('--weight', type=str, help='Specify the path to the trained weights') #テストに使用する学習済み重み



    args = parser.parse_args()
    today = datetime.datetime.now()
    #引数設定の保存
    params_save_to_csv(vars(args), args.out_path)
    print(args)

    #6/19　自己蒸留によるL_CE, L_KLの係数調整実験
    ##文字列からリストに変換
    lst = ast.literal_eval(args.dist_coefficients)
    #global変数に設定
    DECAY_CE_T, DECAY_CE_A, DECAY_CE_V, DECAY_KL_T, DECAY_KL_A, DECAY_KL_V = lst
    # print(DECAY_CE_T, DECAY_CE_A, DECAY_CE_V, DECAY_KL_T, DECAY_KL_A, DECAY_KL_V)
    

    GAMMA_1 = 1.0
    GAMMA_2 = 1.0
    GAMMA_3 = 1.0

    # exit()

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda = args.cuda
    batch_size = args.batch_size
    feat2dim = {'IS10':1582, 'denseface':342, 'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    n_speakers = 2 if args.Dataset=='IEMOCAP' else 9
    n_classes = 6 if args.Dataset=='IEMOCAP' else 7 


    



    print('temp {}'.format(args.temp))

    #キャラID demo
    demo_charaID_flag = False

    if args.model == "multi":
        # model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
        #                                     n_classes=n_classes,
        #                                     hidden_dim=args.hidden_dim,
        #                                     n_speakers=n_speakers,
        #                                     dropout=args.dropout,
        #                                     demo_charaID_flag = demo_charaID_flag)

        model_path = args.weight
        model = torch.load(model_path)

        total_params = sum(p.numel() for p in model.parameters())
        print('total parameters: {}'.format(total_params))
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('training parameters: {}'.format(total_trainable_params))
    else:
        # model = Single_Modal_Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
        #                                     n_classes=n_classes,
        #                                     hidden_dim=args.hidden_dim,
        #                                     n_speakers=n_speakers,
        #                                     dropout=args.dropout,
        #                                     demo_charaID_flag = demo_charaID_flag)
        model_path = args.weight
        model = torch.load(model_path)

        total_params = sum(p.numel() for p in model.parameters())
        print('total parameters: {}'.format(total_params))
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('training parameters: {}'.format(total_trainable_params))

    if cuda:
        model.cuda()


    #KL情報量損 
    kl_loss = MaskedKLDivLoss()

    #wasserstein損失
    # クラス間距離（0/1)
    M = torch.ones(n_classes, n_classes) - torch.eye(n_classes)
    ws_loss = LogWassersteinLoss(M)


    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        #人物情報付きMELD
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_c_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)

    best_acc, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    best_pred_t, best_pred_a, best_pred_v = None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    best_misclassified = None
    best_all_pred = None
    best_epoch = None

    start_time = time.time()

    e = 1

    if args.model == "multi":
        #検証，テスト(テストデータを使用)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, misclassified, all_pred,\
        test_ce_loss, test_kl_loss, test_ws_loss, test_task_loss, \
        test_ce_loss_t, test_ce_loss_v, test_ce_loss_a, \
        test_kl_loss_t, test_kl_loss_v, test_kl_loss_a, \
        test_ws_loss_t, test_ws_loss_v, test_ws_loss_a, \
        test_pred_t, test_pred_a, test_pred_v= train_or_eval_model(model, loss_function, kl_loss, ws_loss, train_loader, e, data_flag=args.Dataset)

        #最良モデルの更新
        if best_acc == None or best_acc < test_acc:
            best_acc = test_acc
            best_label, best_pred, best_mask = test_label, test_pred, test_mask
            best_misclassified = misclassified
            best_all_pred = all_pred
            best_pred_t, best_pred_a, best_pred_v = test_pred_t, test_pred_a, test_pred_v

            #best epochの更新
            best_epoch = e + 1



        print('test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))


    #単一モダリティでの学習
    else:
        #検証，テスト(テストデータを使用)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, misclassified, all_pred,\
            = uni_train_or_eval_model(model, loss_function, train_loader, data_flag=args.Dataset, modal=args.model)

        #最良モデルの更新
        if best_acc == None or best_acc < test_acc:
            best_acc = test_acc
            best_label, best_pred, best_mask = test_label, test_pred, test_mask
            best_misclassified = misclassified
            best_all_pred = all_pred
            # best_pred_t, best_pred_a, best_pred_v = test_pred_t, test_pred_a, test_pred_v
            # モデルの重みの保存 Best Weight
            output_folder = args.out_path+'/weights'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            torch.save(model, output_folder+"/model_weights_best.pth")

            #best epochの更新
            best_epoch = e + 1

        print('test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
        format(test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))


    #各データセットのラベル名
    if args.Dataset=='IEMOCAP':
        name_class = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
    else:
        name_class = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
    
    #分類精度結果
    report_dict = classification_report(best_label, best_pred, sample_weight=best_mask, target_names=name_class, digits=4, output_dict=True)
    ##DataFrameに変換
    df_dict = pd.DataFrame(report_dict).transpose()

    ##Best Acc.のエポックを記録
    df_dict.loc["best_epoch"] = best_epoch


    ##CSVに出力
    df_dict.to_csv(args.out_path+'/classification_report.csv')

    #混同行列の作成
    # print("best_label", np.shape(best_label))
    # print("best_pred", np.shape(best_pred))
    # print("best_pred_t", np.shape(best_pred_t))
    # print("best_pred_a", np.shape(best_pred_a))
    # print("best_pred_v", np.shape(best_pred_v))
    # exit()


    make_cm(best_label, best_pred, best_mask, name_class, "cm")
    if args.model == "multi":
        make_cm(best_label, best_pred_t, best_mask, name_class, "cm_t")
        make_cm(best_label, best_pred_a, best_mask, name_class, "cm_a")
        make_cm(best_label, best_pred_v, best_mask, name_class, "cm_v")


    with open(args.out_path+"/all_pred.csv", "w", newline='') as f:
        if args.model == "multi":
            fieldnames2 = ["vid", "utt_index", "character","pred", "true", "text", "all_prob", "t_prob", "a_prob", "v_prob"]
        else:
            fieldnames2 = ["vid", "utt_index", "character","pred", "true", "text", "all_prob"]
        writer = csv.DictWriter(f, fieldnames=fieldnames2)
        writer.writeheader()
        for row in best_all_pred:
            filtered_row = {key: row.get(key, "") for key in fieldnames2}
            writer.writerow(filtered_row)
        print("Saved all Pred examples to all_pred.csv")

