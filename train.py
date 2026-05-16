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



#train_loss train_acc val_loss val_accの保存
def csv_history(dict, output_folder):
    # DataFrameに変換
    df = pd.DataFrame(dict)


    # 最初の行にキーを設定
    df.insert(0, 'Metric', df.index)

    output_folder = output_folder + "/train_results/"

    path = f"{output_folder}history.csv"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df.to_csv(path, index=False)

#パラメータ設定記録
def params_save_to_csv(params, output_folder):
    path = f"{output_folder}/arguemts.csv"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    df = pd.DataFrame([params])  # 引数をDataFrameに変換
    df.to_csv(path, index=False)  # CSVに保存

# 学習曲線のプロット
def show_history(history, out_folder, data, modal):
    output_folder_grah = out_folder + '/train_results'
    if not os.path.exists(output_folder_grah):
                os.makedirs(output_folder_grah)

    output_folder_grah2 = out_folder + '/reproduction/'
    if not os.path.exists(output_folder_grah2):
                os.makedirs(output_folder_grah2)

    #横軸の表示幅 50epochずつ
    xticks_range = range(0, len(history["train_loss"]) + 1, 50)

    #親表現による分類損失
    plt.figure()   #新しいウィンドウを描画
    plt.plot(history["train_loss"], label = "train loss")
    plt.plot(history["val_loss"], label = "val loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(output_folder_grah+'/grah_loss.png')

    if modal == "multi":
        #子表現による分類損失
        plt.figure()   #新しいウィンドウを描画
        plt.plot(history["train_ce_loss"], label = "train CE loss")
        plt.plot(history["val_ce_loss"], label = "val CE loss")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(output_folder_grah+'/grah_ce_loss.png')

        #子表現と親表現のKL情報量誤差
        plt.figure()   #新しいウィンドウを描画
        plt.plot(history["train_kl_loss"], label = "train KL loss")
        plt.plot(history["val_kl_loss"], label = "val KL loss")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(output_folder_grah+'/grah_kl_loss.png')

        #子表現と親表現のKL情報量誤差
        plt.figure()   #新しいウィンドウを描画
        plt.plot(history["train_ws_loss"], label = "train WS loss")
        plt.plot(history["val_ws_loss"], label = "val WS loss")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(output_folder_grah+'/grah_ws_loss.png')

        #L_Task, L_CE_m 学習データ
        plt.figure()   #新しいウィンドウを描画
        plt.plot(history["train_task_loss"], label = "Task loss", color="black")
        plt.plot(history["train_ce_loss_t"], label = "CE text loss", color='green')
        plt.plot(history["train_ce_loss_v"], label = "CE visual loss", color='blue')
        plt.plot(history["train_ce_loss_a"], label = "CE audio loss", color='red')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        if data == "IEMOCAP":
            plt.ylim(0.00, 2.50)  # 縦軸の範囲を制限
        else:
            plt.ylim(0.00, 1.80)  # 縦軸の範囲を制限
        plt.xticks(xticks_range)
        plt.savefig(output_folder_grah2+'/train_task_ce_modal_loss.png')

        #L_Task, L_CE_m テストデータ
        plt.figure()   #新しいウィンドウを描画
        plt.plot(history["val_task_loss"], label = "Task loss", color='black')
        plt.plot(history["val_ce_loss_t"], label = "CE text loss", color='green')
        plt.plot(history["val_ce_loss_v"], label = "CE visual loss", color='blue')
        plt.plot(history["val_ce_loss_a"], label = "CE audio loss", color='red')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        if data == "IEMOCAP":
            plt.ylim(0.00, 2.50)  # 縦軸の範囲を制限
        else:
            plt.ylim(0.00, 1.80)  # 縦軸の範囲を制限
        plt.xticks(xticks_range)
        plt.savefig(output_folder_grah2+'/test_task_ce_modal_loss.png')

        #L_KL_m　学習データ
        plt.figure()   #新しいウィンドウを描画
        plt.plot(history["train_kl_loss_t"], label = "KL text loss", color='green')
        plt.plot(history["train_kl_loss_v"], label = "KL visual loss", color='blue')
        plt.plot(history["train_kl_loss_a"], label = "KL audio loss", color='red')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # plt.legend()
        if data == "IEMOCAP":
            plt.ylim(0.00, 0.50)  # 縦軸の範囲を制限
            #plt.ylim(0.00, 0.18)
        else:
            plt.ylim(0.00, 0.015)  # 縦軸の範囲を制限
        plt.xticks(xticks_range)
        plt.savefig(output_folder_grah2+'/train_kl_modal_loss.png')

        #L_ws_m　学習データ
        plt.figure()   #新しいウィンドウを描画
        plt.plot(history["train_ws_loss_t"], label = "WS text loss", color='green')
        plt.plot(history["train_ws_loss_v"], label = "WS visual loss", color='blue')
        plt.plot(history["train_ws_loss_a"], label = "WS audio loss", color='red')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.ylim(0.00, 0.30)  # 縦軸の範囲を制限
        plt.xticks(xticks_range)
        plt.savefig(output_folder_grah2+'/train_ws_modal_loss.png')

        #L_KL_m　テストデータ
        plt.figure()   #新しいウィンドウを描画
        plt.plot(history["val_kl_loss_t"], label = "KL text loss", color='green')
        plt.plot(history["val_kl_loss_v"], label = "KL visual loss", color='blue')
        plt.plot(history["val_kl_loss_a"], label = "KL audio loss", color='red')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # plt.legend()
        if data == "IEMOCAP":
            plt.ylim(0.00, 0.50)  # 縦軸の範囲を制限
        else:
            plt.ylim(0.00, 0.015)  # 縦軸の範囲を制限
        plt.xticks(xticks_range)
        plt.savefig(output_folder_grah2+'/test_kl_modal_loss.png')

        #L_ws_m　テストデータ
        plt.figure()   #新しいウィンドウを描画
        plt.plot(history["val_ws_loss_t"], label = "WS text loss", color='green')
        plt.plot(history["val_ws_loss_v"], label = "WS visual loss", color='blue')
        plt.plot(history["val_ws_loss_a"], label = "WS audio loss", color='red')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.ylim(0.00, 0.30)  # 縦軸の範囲を制限
        plt.xticks(xticks_range)
        plt.savefig(output_folder_grah2+'/test_ws_modal_loss.png')

    #親表現による分類のAcc.
    plt.figure()   #新しいウィンドウを描画
    plt.plot(history["train_acc"], label = "train acc")
    plt.plot(history["val_acc"], label = "val acc")
    plt.xlabel('epoch')
    plt.ylabel('acc.')
    plt.legend()
    plt.ylim(0, 100)
    #plt.ylim(60,100)
    plt.savefig(output_folder_grah+'/grah_acc.png')

    # グラフの描画
    plt.figure(figsize=(8, 4))
    plt.plot(range(1,len(history["lr"])+1),history["lr"], marker='o')
    plt.title("ExponentialLR Learning Rate Schedule")
    plt.xlabel("epoch")
    plt.ylabel("Learning Rate")
    plt.savefig(output_folder_grah+'/grah_lr.png')

def get_train_valid_sampler(trainset, valid=0.1, valid_num=0):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)

    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])
    #9/26 交差検証用
    # size = len(trainset)
    # idx = list(range(size))
    # fold_size = int(valid*size)

    # train_set = []
    # valid_set = []
    # count = 0
    # for i in range(0, size, fold_size):
    #     if count == valid_num:
    #         valid_set += idx[i:i+fold_size]
    #     else:
    #         train_set += idx[i:i+fold_size]
    #     count += 1
    # return SubsetRandomSampler(train_set), SubsetRandomSampler(valid_set)


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

def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, valid_num=0):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, valid_num)
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
            loss = L_task + L_ce + L_kl


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
        if train == False:
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
        
        ce_losses.append(L_ce.item()*masks[-1].sum())
        kl_losses.append(L_kl.item()*masks[-1].sum())

        task_losses.append(L_task.item()*masks[-1].sum())

        ce_losses_t.append(L_ce_t.item()*masks[-1].sum())
        ce_losses_a.append(L_ce_a.item()*masks[-1].sum())
        ce_losses_v.append(L_ce_v.item()*masks[-1].sum())


        kl_losses_t.append(L_kl_t.item()*masks[-1].sum())
        kl_losses_a.append(L_kl_a.item()*masks[-1].sum())
        kl_losses_v.append(L_kl_v.item()*masks[-1].sum())

        if loss_func == "ws":
            ws_losses.append(L_ws.item()*masks[-1].sum())
            ws_losses_t.append(L_ws_t.item()*masks[-1].sum())
            ws_losses_a.append(L_ws_a.item()*masks[-1].sum())
            ws_losses_v.append(L_ws_v.item()*masks[-1].sum())
  

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
               float('nan'), float('nan'), float('nan'), float('nan'), \
               float('nan'), float('nan'), float('nan'), \
               float('nan'), float('nan'), float('nan'), \
               float('nan'), float('nan'), float('nan'), \
               [], [], [] #for preds_m m in {t,a,v}


    #avg_ 表示用　有効サンプル数(発話数)で割る
    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)

    avg_ce_loss = round(np.sum(ce_losses)/np.sum(masks), 4)
    avg_kl_loss = round(np.sum(kl_losses)/np.sum(masks), 4)
    avg_ws_loss = round(np.sum(ws_losses)/np.sum(masks), 4)
    avg_task_loss = round(np.sum(task_losses)/np.sum(masks), 4)

    avg_ce_loss_t = round(np.sum(ce_losses_t)/np.sum(masks), 4)
    avg_ce_loss_v = round(np.sum(ce_losses_v)/np.sum(masks), 4)
    avg_ce_loss_a = round(np.sum(ce_losses_a)/np.sum(masks), 4)

    avg_kl_loss_t = round(np.sum(kl_losses_t)/np.sum(masks), 4)
    avg_kl_loss_v = round(np.sum(kl_losses_v)/np.sum(masks), 4)
    avg_kl_loss_a = round(np.sum(kl_losses_a)/np.sum(masks), 4)

    if loss_func == "ws":
        avg_ws_loss_t = round(np.sum(ws_losses_t)/np.sum(masks), 4)
        avg_ws_loss_v = round(np.sum(ws_losses_v)/np.sum(masks), 4)
        avg_ws_loss_a = round(np.sum(ws_losses_a)/np.sum(masks), 4)
    else:
        avg_ws_loss_t = float('nan')
        avg_ws_loss_v = float('nan')
        avg_ws_loss_a = float('nan')

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, misclassified, all_pred,\
           avg_ce_loss, avg_kl_loss, avg_ws_loss, avg_task_loss, \
           avg_ce_loss_t, avg_ce_loss_v, avg_ce_loss_a, \
           avg_kl_loss_t, avg_kl_loss_v, avg_kl_loss_a, \
           avg_ws_loss_t, avg_ws_loss_v, avg_ws_loss_a, \
           preds_t, preds_a, preds_v

def uni_train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False, data_flag="IEMOCAP", modal="t"):

    #表示用の入れ子
    losses, preds, labels, masks, = [], [], [], []
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

        #prob1:t prob2:a prob3v
        if modal=="t":
            log_prob, prob, _, _ = model(textf, umask, qmask, lengths, modal)
        elif modal=="a":
            log_prob, prob, _, _ = model(acouf, umask, qmask, lengths, modal)  
        else:     
            log_prob, prob, _, _ = model(visuf, umask, qmask, lengths, modal)  

        #正解ラベル
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
        if train == False:
            for xi in range(x):
                vid_ = vids[xi]
                for yi in range(y):
                    mask_value = umask_np[xi][yi]
                    if mask_value == 1:
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
        
        #誤差逆伝搬
        if train:
            L_task.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

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


#動的な自己蒸留係数の導入
#計算のための関数
##モダリティ間のKL損失の比率
def rate_kl(kl_t, kl_a, kl_v):
    if (kl_t + kl_a + kl_v) == 0:
        return 1,1,1
    
    rate_t = kl_t/(kl_t + kl_a + kl_v)
    rate_a = kl_a/(kl_t + kl_a + kl_v)
    rate_v = kl_v/(kl_t + kl_a + kl_v)

    return rate_t, rate_a, rate_v

##各モダリティのKL損失の平均変化率を計算
def ave_rate_of_change(kl_t_1, kl_a_1, kl_v_1, kl_t_2, kl_a_2, kl_v_2, t):
    #1:現在の損失
    #2:過去の損失

    ave_chan_t = (kl_t_1 - kl_t_2)/t
    ave_chan_a = (kl_a_1 - kl_a_2)/t
    ave_chan_v = (kl_v_1 - kl_v_2)/t

    return ave_chan_t, ave_chan_a, ave_chan_v

##モーメンタム速度による各モダリティのKL損失の変化を計算
def moment(loss_kl_t_1, loss_kl_a_1, loss_kl_v_1, loss_kl_t_2, loss_kl_a_2, loss_kl_v_2, velocity, beta=0.95):
    #各モダリティの速度 = 前回の速度 + 直前の変化量
    velocity_t = beta * velocity["t"] + (1 - beta) * (loss_kl_t_1 - loss_kl_t_2)
    velocity_a = beta * velocity["a"] + (1 - beta) * (loss_kl_a_1 - loss_kl_a_2)
    velocity_v = beta * velocity["v"] + (1 - beta) * (loss_kl_v_1 - loss_kl_v_2)

    return velocity_t, velocity_a, velocity_v

def csv_history_coeff(dict, path):
    # DataFrameに変換
    df = pd.DataFrame(dict)


    # 最初の行にキーを設定
    df.insert(0, 'Metric', df.index)

    output_folder = path + "/"

    file = f"{output_folder}/history_coeff.csv"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df.to_csv(file, index=False)

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
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=200, metavar='E', help='number of epochs')
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
    parser.add_argument('--modal', type=str, default="multi", help='multi or text or audio or visual')
    parser.add_argument('--valid_num', type=int, default=0) #交差検証用



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
    #学習エポック数
    n_epochs = args.epochs
    #バッチサイズ
    batch_size = args.batch_size
    #各モダリティの次元数
    feat2dim = {'IS10':1582, 'denseface':342, 'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024
    ##単一入力モデル用 keyで入力次元を指定
    D_input_dic = {"t":D_text, "a":D_audio, "v":D_visual}


    n_speakers = 2 if args.Dataset=='IEMOCAP' else 9
    n_classes = 6 if args.Dataset=='IEMOCAP' else 7 

    if args.modal == "multi":
        #エポックごとの学習記録
        history = {"train_loss": [], "train_acc": [],
                "train_ce_loss": [],"train_kl_loss": [], "train_ws_loss":[],"train_task_loss":[],
                "val_loss": [], "val_acc": [],
                "val_ce_loss": [],"val_kl_loss": [], "val_ws_loss":[], "val_task_loss": [],
                "val_ce_loss_t": [],"val_ce_loss_v": [],"val_ce_loss_a": [],
                "val_kl_loss_t": [],"val_kl_loss_v": [],"val_kl_loss_a": [],
                "val_ws_loss_t": [],"val_ws_loss_v": [],"val_ws_loss_a": [],
                "train_ce_loss_t": [],"train_ce_loss_v": [],"train_ce_loss_a": [],
                "train_kl_loss_t": [],"train_kl_loss_v": [],"train_kl_loss_a": [],
                "train_ws_loss_t": [],"train_ws_loss_v": [],"train_ws_loss_a": [],
                "lr": []
                }
        
        #動的な係数のための保存
        history_coeff = {"rate_t":[], "rate_a":[], "rate_v":[],
                "ave_chan_t":[], "ave_chan_a":[], "ave_chan_v":[],
                "velocity_t":[], "velocity_a":[], "velocity_v":[],
                "coeff_t":[], "coeff_a":[], "coeff_v":[]
                }
    else:
        #エポックごとの学習記録
        history = {"train_loss": [], "train_acc": [],
                "val_loss": [], "val_acc": [],
                "lr": []
                }
    
    #モーメンタムの記録
    velocity = {"t":0.0, "a":0.0, "v":0.0}

    #更新を検討する間隔
    change_epochs = args.change_epoch


    print('temp {}'.format(args.temp))

    #キャラID demo
    demo_charaID_flag = False

    if args.modal == "multi":
        model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
                                            n_classes=n_classes,
                                            hidden_dim=args.hidden_dim,
                                            n_speakers=n_speakers,
                                            dropout=args.dropout,
                                            demo_charaID_flag = demo_charaID_flag)
    else:
        #入力特徴次元を指定
        D_input = D_input_dic[args.modal]

        model = Single_Modal_Transformer_Based_Model(args.Dataset, args.temp, D_input, args.n_head,
                                            n_classes=n_classes,
                                            hidden_dim=args.hidden_dim,
                                            n_speakers=n_speakers,
                                            dropout=args.dropout,
                                            demo_charaID_flag = demo_charaID_flag)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    if cuda:
        model.cuda()

    #学習率
    lr=args.lr

    #KL情報量損 
    kl_loss = MaskedKLDivLoss()

    #wasserstein損失
    # クラス間距離（0/1)
    M = torch.ones(n_classes, n_classes) - torch.eye(n_classes)
    ws_loss = LogWassersteinLoss(M)

    #最適化手法
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=args.l2)

    #LambdaLR
    #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: args.schedular ** e)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.schedular)

    #StepLR
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    #CosineAnnealingLR
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)

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
                                                                      num_workers=0, valid_num=args.valid_num)
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

    for e in range(n_epochs):
        start_time = time.time()

        if args.modal == "multi":
            #学習
            train_loss, train_acc, _, _, _, train_fscore, hoge, _,\
            train_ce_loss, train_kl_loss, train_ws_loss, train_task_loss, \
            train_ce_loss_t, train_ce_loss_v, train_ce_loss_a, \
            train_kl_loss_t, train_kl_loss_v, train_kl_loss_a, \
            train_ws_loss_t, train_ws_loss_v, train_ws_loss_a, \
            _, _, _= train_or_eval_model(model, loss_function, kl_loss, ws_loss, train_loader, e, optimizer, True, data_flag=args.Dataset)

            valid_loss, valid_acc, _, _, _, valid_fscore, hoge, _,\
            valid_ce_loss, valid_kl_loss, _, _, \
            _, _, _, \
            _, _, _, \
            _, _, _, \
            _, _, _= train_or_eval_model(model, loss_function, kl_loss, ws_loss, valid_loader, e, data_flag=args.Dataset)

            #検証，テスト(テストデータを使用)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, misclassified, all_pred,\
            test_ce_loss, test_kl_loss, test_ws_loss, test_task_loss, \
            test_ce_loss_t, test_ce_loss_v, test_ce_loss_a, \
            test_kl_loss_t, test_kl_loss_v, test_kl_loss_a, \
            test_ws_loss_t, test_ws_loss_v, test_ws_loss_a, \
            test_pred_t, test_pred_a, test_pred_v= train_or_eval_model(model, loss_function, kl_loss, ws_loss, test_loader, e, data_flag=args.Dataset)

            all_acc.append(test_acc)
            all_fscore.append(test_fscore)

            #最良モデルの更新
            if best_acc == None or best_acc < test_acc:
                best_acc = test_acc
                best_label, best_pred, best_mask = test_label, test_pred, test_mask
                best_misclassified = misclassified
                best_all_pred = all_pred
                best_pred_t, best_pred_a, best_pred_v = test_pred_t, test_pred_a, test_pred_v
                # モデルの重みの保存 Best Weight
                output_folder = args.out_path+'/weights'
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                torch.save(model, output_folder+"/model_weights_best.pth")

                #best epochの更新
                best_epoch = e + 1

            if args.tensorboard:
                writer.add_scalar('test: accuracy', test_acc, e)
                writer.add_scalar('test: fscore', test_fscore, e)
                writer.add_scalar('train: accuracy', train_acc, e)
                writer.add_scalar('train: fscore', train_fscore, e)

            #学習経過の記録
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(test_loss)
            history["val_acc"].append(test_acc)

            history["train_ce_loss"].append(train_ce_loss)
            history["train_kl_loss"].append(train_kl_loss)
            history["train_ws_loss"].append(train_ws_loss)
            history["train_task_loss"].append(train_task_loss)

            history["val_ce_loss"].append(test_ce_loss)
            history["val_kl_loss"].append(test_kl_loss)
            history["val_ws_loss"].append(test_ws_loss)
            history["val_task_loss"].append(test_task_loss)

            history["val_ce_loss_t"].append(test_ce_loss_t)
            history["val_ce_loss_v"].append(test_ce_loss_v)
            history["val_ce_loss_a"].append(test_ce_loss_a)

            history["val_kl_loss_t"].append(test_kl_loss_t)
            history["val_kl_loss_v"].append(test_kl_loss_v)
            history["val_kl_loss_a"].append(test_kl_loss_a)

            history["val_ws_loss_t"].append(test_ws_loss_t)
            history["val_ws_loss_v"].append(test_ws_loss_v)
            history["val_ws_loss_a"].append(test_ws_loss_a)

            history["train_ce_loss_t"].append(train_ce_loss_t)
            history["train_ce_loss_v"].append(train_ce_loss_v)
            history["train_ce_loss_a"].append(train_ce_loss_a)

            history["train_kl_loss_t"].append(train_kl_loss_t)
            history["train_kl_loss_v"].append(train_kl_loss_v)
            history["train_kl_loss_a"].append(train_kl_loss_a)

            history["train_ws_loss_t"].append(train_ws_loss_t)
            history["train_ws_loss_v"].append(train_ws_loss_v)
            history["train_ws_loss_a"].append(train_ws_loss_a)
            
            history["lr"].append(optimizer.param_groups[0]['lr'])  # 現在の学習率を記録

            #動的な係数
            #係数の更新
            if args.dynamic:
                if args.loss_func == 'kl':
                    # ##各モダリティの学習データに対するkl損失(list)
                    loss_kl_t = history["train_kl_loss_t"]
                    loss_kl_a = history["train_kl_loss_a"]
                    loss_kl_v = history["train_kl_loss_v"]
                elif args.loss_func == 'ws':
                    ##各モダリティの学習データに対するws損失(list)
                    loss_kl_t = history["train_ws_loss_t"]
                    loss_kl_a = history["train_ws_loss_a"]
                    loss_kl_v = history["train_ws_loss_v"]

                #最初の20エポックは切り捨て
                if (e+1) > 20:

                    ####入力を用意################
                    """
                    loss_kl_{modal}_1  : eエポック時(現在)のkl損失
                    loss_kl_{modal}_1_2: e-1エポック時のkl損失
                    """
                    loss_kl_t_1, loss_kl_a_1, loss_kl_v_1 = loss_kl_t[e], loss_kl_a[e], loss_kl_v[e]
                    loss_kl_t_1_2, loss_kl_a_1_2, loss_kl_v_1_2 = loss_kl_t[e-1], loss_kl_a[e-1], loss_kl_v[e-1]

                    #モーメントの計算
                    velocity_t, velocity_a, velocity_v = moment(loss_kl_t_1, loss_kl_a_1, loss_kl_v_1, loss_kl_t_1_2, loss_kl_a_1_2, loss_kl_v_1_2, velocity)

                    ####モーメントを保存
                    velocity["t"] = velocity_t
                    velocity["a"] = velocity_a
                    velocity["v"] = velocity_v

                    #移動平均のための更新幅
                    t=change_epochs

                    #change_epochsごとに更新を検討
                    if (e+1)%change_epochs == 0:

                        if (e+1) == change_epochs:
                            t = change_epochs - 20

                        ####入力を用意2################
                        """
                        loss_kl_{modal}_2:tエポック前の損失
                        """
                        loss_kl_t_2, loss_kl_a_2, loss_kl_v_2 = loss_kl_t[e-(t-1)], loss_kl_a[e-(t-1)], loss_kl_v[e-(t-1)]

                        ##モダリティ間のKL損失の比率を計算
                        rate_t, rate_a, rate_v = rate_kl(loss_kl_t_1, loss_kl_a_1, loss_kl_v_1)
                        
                        ####モダリティ間のKL損失の比率を保存
                        history_coeff["rate_t"].append(rate_t)
                        history_coeff["rate_a"].append(rate_a)
                        history_coeff["rate_v"].append(rate_v)

                        ##モダリティのKL損失の平均変化率を計算
                        ave_chan_t, ave_chan_a, ave_chan_v = ave_rate_of_change(loss_kl_t_1, loss_kl_a_1, loss_kl_v_1, 
                                                                                loss_kl_t_2, loss_kl_a_2, loss_kl_v_2, t)

                        history_coeff["ave_chan_t"].append(ave_chan_t)
                        history_coeff["ave_chan_a"].append(ave_chan_a)
                        history_coeff["ave_chan_v"].append(ave_chan_v)


                        ##現在のモーメンタムを保存
                        # print(velocity_t, velocity_a, velocity_v)
                        history_coeff["velocity_t"].append(velocity_t)
                        history_coeff["velocity_a"].append(velocity_a)
                        history_coeff["velocity_v"].append(velocity_v)

                        #DECAY_CE_T, DECAY_CE_A, DECAY_CE_V, DECAY_KL_T, DECAY_KL_A, DECAY_KL_Vの更新
                        # if rate_t > 0.4 and ave_chan_t > 0:
                        if rate_t > 0.4 and velocity_t > 0:
                            # history_coeff["coeff_t"].append(True)
                            DECAY_CE_T += args.add_kl
                            DECAY_KL_T += args.add_kl
                            history_coeff["coeff_t"].append((DECAY_CE_T, DECAY_KL_T))
                        else:
                            # history_coeff["coeff_t"].append(False)
                            history_coeff["coeff_t"].append((DECAY_CE_T, DECAY_KL_T))

                        # if rate_a > 0.4 and ave_chan_a > 0:
                        if rate_a > 0.4 and velocity_a > 0:
                            # history_coeff["coeff_a"].append(True)
                            DECAY_CE_A += args.add_kl
                            DECAY_KL_A += args.add_kl
                            history_coeff["coeff_a"].append((DECAY_CE_A, DECAY_KL_A))
                        else:
                            history_coeff["coeff_a"].append((DECAY_CE_A, DECAY_KL_A))
                            # history_coeff["coeff_a"].append(False)

                        # if rate_v > 0.4 and ave_chan_v > 0:
                        if rate_v > 0.4 and velocity_v > 0:
                            DECAY_CE_V += args.add_kl
                            DECAY_KL_V += args.add_kl
                            history_coeff["coeff_v"].append((DECAY_CE_V, DECAY_KL_V))
                        else:
                            history_coeff["coeff_v"].append((DECAY_CE_V, DECAY_KL_V))

            print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                    format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
            #10epochごとに途中経過をまとめる
            # if (e+1)%10 == 0:
            #     # 学習途中のモデルの重みの保存
            #     output_folder = args.out_path+'/weights'
            #     if not os.path.exists(output_folder):
            #         os.makedirs(output_folder)
            #     torch.save(model, output_folder+f"/model_weights_{e+1}epoch.pth")
            #     # print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
            #     # print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))

                #各データセットのラベル名
            if args.Dataset=='IEMOCAP':
                name_class = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
            else:
                name_class = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']

            if (e+1) == 5:
                make_cm(best_label, best_pred, best_mask, name_class, "cm", folder="/cm/epoch5")
                make_cm(best_label, best_pred_t, best_mask, name_class, "cm_t", folder="/cm/epoch5")
                make_cm(best_label, best_pred_a, best_mask, name_class, "cm_a", folder="/cm/epoch5")
                make_cm(best_label, best_pred_v, best_mask, name_class, "cm_v", folder="/cm/epoch5")

                with open(args.out_path+"/all_pred_epoch5.csv", "w", newline='') as f:
                    fieldnames2_epoch5 = ["vid", "utt_index", "character","pred", "true", "text", "all_prob", "t_prob", "a_prob", "v_prob"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames2_epoch5)
                    writer.writeheader()
                    for row in best_all_pred:
                        filtered_row = {key: row.get(key, "") for key in fieldnames2_epoch5}
                        writer.writerow(filtered_row)
 


            #減衰の導入
            scheduler.step()

        #単一モダリティでの学習
        else:
            #学習
            train_loss, train_acc, _, _, _, train_fscore, hoge, _,\
                = uni_train_or_eval_model(model, loss_function, train_loader, e, optimizer, True, data_flag=args.Dataset, modal=args.modal) #9/25 削減した学習データでの学習

            valid_loss, valid_acc, _, _, _, valid_fscore, hoge, _,\
                = uni_train_or_eval_model(model, loss_function, valid_loader, e, data_flag=args.Dataset, modal=args.modal)

            #検証，テスト(テストデータを使用)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, misclassified, all_pred,\
                = uni_train_or_eval_model(model, loss_function, test_loader, e, data_flag=args.Dataset, modal=args.modal)

            all_acc.append(test_acc)
            all_fscore.append(test_fscore)

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

            if args.tensorboard:
                writer.add_scalar('test: accuracy', test_acc, e)
                writer.add_scalar('test: fscore', test_fscore, e)
                writer.add_scalar('train: accuracy', train_acc, e)
                writer.add_scalar('train: fscore', train_fscore, e)

            #学習経過の記録
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(test_loss)
            history["val_acc"].append(test_acc)
            
            history["lr"].append(optimizer.param_groups[0]['lr'])  # 現在の学習率を記録

            print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
            format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))

            #減衰の導入
            scheduler.step()

        if args.tensorboard:
            writer.close()

    # モデルの重みの保存 Last Weight
    output_folder = args.out_path+'/weights'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    torch.save(model, output_folder+"/model_weights_last.pth")

    print('Test performance..')
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))


    #各データセットのラベル名
    if args.Dataset=='IEMOCAP':
        name_class = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
    else:
        name_class = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']

    with open(args.out_path+"/all_pred.csv", "w", newline='') as f:
        if args.modal == "multi":
            fieldnames2 = ["vid", "utt_index", "character","pred", "true", "text", "all_prob", "t_prob", "a_prob", "v_prob"]
        else:
            fieldnames2 = ["vid", "utt_index", "character","pred", "true", "text", "all_prob"]
        writer = csv.DictWriter(f, fieldnames=fieldnames2)
        writer.writeheader()
        for row in best_all_pred:
            filtered_row = {key: row.get(key, "") for key in fieldnames2}
            writer.writerow(filtered_row)
        print("Saved all Pred examples to all_pred.csv")
    
    #分類精度結果
    # labels = list(range(len(name_class)))
    report_dict = classification_report(best_label, best_pred,  sample_weight=best_mask, target_names=name_class, digits=4, output_dict=True)
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
    if args.modal == "multi":
        make_cm(best_label, best_pred_t, best_mask, name_class, "cm_t")
        make_cm(best_label, best_pred_a, best_mask, name_class, "cm_a")
        make_cm(best_label, best_pred_v, best_mask, name_class, "cm_v")

    #学習の記録
    show_history(history, args.out_path, args.Dataset, args.modal)
    csv_history(history, args.out_path)

    if args.modal == "multi":
        csv_history_coeff(history_coeff, args.out_path)