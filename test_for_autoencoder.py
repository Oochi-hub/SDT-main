#オートエンコーダ検証用コード
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset, MELDDataset_c
from model import MaskedNLLLoss, MaskedKLDivLoss, Single_Modal_Transformer_Based_Model, AutoEncoder, Symmetry_AutoEncoder, MaskedL2Loss
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

#パラメータ設定記録
def params_save_to_csv(params, output_folder):
    path = f"{output_folder}/arguemts.csv"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    df = pd.DataFrame([params])  # 引数をDataFrameに変換
    df.to_csv(path, index=False)  # CSVに保存


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


def uni_train_or_eval_model(model, loss_function, dataloader, data_flag="IEMOCAP", modal="t", kl_loss=None):

    #表示用の入れ子
    losses, preds, labels, masks, = [], [], [], []
    kl_losses, l2_losses = [], []
    misclassified = []
    all_pred = []

    #検証モード
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

        """
        traget_pred:単一表現モデルのSoftmax予測
        features:単一表現モデルの特徴量
        z:バイナリ表現
        h_hat:デコードした特徴量(L2損失用)
        binary_prob:バイナリ表現によるSoftmax予測
        kl_log_prob:バイナリ表現による温度係数付きlog_softmax
        log_prob:バイナリ表現によるlog_softmax
        """

        if demo_charaID_flag:
            #prob1:t prob2:a prob3v
            target_pred, features, z, h_hat, binary_prob, kl_log_prob = model(textf, visuf, acouf, umask, cmatrix, lengths, modal)
        else:
            #prob1:t prob2:a prob3v
            target_pred, features, z, h_hat, binary_prob, kl_log_prob = model(textf, visuf, acouf, umask, qmask, lengths, modal)
        

        labels_ = label.view(-1)

        #損失計算
        alpha = 0.5

        ##L2損失計算
        L_l2 = loss_function(h_hat, features, umask)

        #KL損失計算
        target = target_pred.view(-1, target_pred.size()[2])
        kl_p = kl_log_prob.view(-1, kl_log_prob.size()[2])
        L_kl = kl_loss(kl_p, target, umask)

        ##モデル全体の損失
        L_task = alpha * L_l2 + (1-alpha) * L_kl

        #誤分類事例の回収
        lp_ = binary_prob.view(-1, binary_prob.size()[2])
        pred_ = torch.argmax(lp_,1)



        # print("pred_", pred_.size())
        """
        pred_ torch.Size([1504])
        """

        #誤分類事例の回収
        #all_prob torch.Size([batch, seq_len, num_class])
        all_probmax = torch.argmax(binary_prob, dim=-1)  # shape: (batch, seq_len)

        #バッチサイズと最大発話数を記録
        x = umask.size()[0]
        y = umask.size()[1]

        #np配列に変換
        umask_np = umask.cpu().numpy()
        label_np = label.cpu().numpy()
        all_probmax_np = all_probmax.cpu().numpy()
        z_np = z.detach().cpu().numpy()

        ###確率分布保存用　np配列に変換
        all_prob_np = binary_prob.detach().cpu().numpy()

        #テストにおける予測結果を記録
        for xi in range(x):
            vid_ = vids[xi]
            for yi in range(y):
                mask_value = umask_np[xi][yi]
                if mask_value == 1:
                    # if data_flag == 'MELD_c':
                    #     #MELDでは各人物ごとにまとめる
                    #     character = list(cmatrix_np[xi][yi]).index(1)
                    #     all_pred.append({"vid": vid_,
                    #                 "utt_index": yi,
                    #                 "character": character,
                    #                 "pred": all_probmax_np[xi][yi],
                    #                 "true": label_np[xi][yi],
                    #                 "text": texts[xi][yi], 
                    #                 "all_prob": str(all_prob_np[xi][yi].tolist()) #csvに保存するので文字列に
                    #                 })
                    #     #誤分類を収集
                    #     if all_probmax_np[xi][yi] != label_np[xi][yi]:
                    #         misclassified.append({"vid": vid_,
                    #                 "utt_index": yi,
                    #                 "character": character,
                    #                 "pred": all_probmax_np[xi][yi],
                    #                 "true": label_np[xi][yi],
                    #                 "text": texts[xi][yi], 
                    #                 "all_prob": str(all_prob_np[xi][yi].tolist()) #csvに保存するので文字列に
                    #                 })
                    # else:
                    #     #MELD_c以外はキャラ情報なしで記録
                    #     all_pred.append({"vid": vid_,
                    #                 "utt_index": yi,
                    #                 "pred": all_probmax_np[xi][yi],
                    #                 "true": label_np[xi][yi],
                    #                 "text": texts[xi][yi], 
                    #                 "all_prob": str(all_prob_np[xi][yi].tolist()) #csvに保存するので文字列に
                    #                 })
                    #     #誤分類を収集
                    #     if all_probmax_np[xi][yi] != label_np[xi][yi]:
                    #         misclassified.append({"vid": vid_,
                    #                 "utt_index": yi,
                    #                 "pred": all_probmax_np[xi][yi],
                    #                 "true": label_np[xi][yi],
                    #                 "text": texts[xi][yi], 
                    #                 "all_prob": str(all_prob_np[xi][yi].tolist())
                    #                 })
                    if data_flag == 'MELD_c':
                        #MELDでは各人物ごとにまとめる
                        character = list(cmatrix_np[xi][yi]).index(1)
                        all_pred.append({"vid": vid_,
                                    "utt_index": yi,
                                    "character": character,
                                    "pred": all_probmax_np[xi][yi],
                                    "true": label_np[xi][yi],
                                    "text": texts[xi][yi], 
                                    "all_prob": str(z_np[xi][yi].tolist()) #csvに保存するので文字列に
                                    })
                        #誤分類を収集
                        if all_probmax_np[xi][yi] != label_np[xi][yi]:
                            misclassified.append({"vid": vid_,
                                    "utt_index": yi,
                                    "character": character,
                                    "pred": all_probmax_np[xi][yi],
                                    "true": label_np[xi][yi],
                                    "text": texts[xi][yi], 
                                    "all_prob": str(z_np[xi][yi].tolist()) #csvに保存するので文字列に
                                    })
                    else:
                        #MELD_c以外はキャラ情報なしで記録
                        all_pred.append({"vid": vid_,
                                    "utt_index": yi,
                                    "pred": all_probmax_np[xi][yi],
                                    "true": label_np[xi][yi],
                                    "text": texts[xi][yi], 
                                    "all_prob": str(z_np[xi][yi].tolist()) #csvに保存するので文字列に
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

        #予測結果の保存
        preds.append(pred_.data.cpu().numpy())
        #入力ラベルの保存
        labels.append(labels_.data.cpu().numpy())
        #マスクの保存(レポート計算用)
        masks.append(umask.view(-1).cpu().numpy())

        #各損失の保存
        losses.append(L_task.item()*masks[-1].sum())
        kl_losses.append(L_kl.item()*masks[-1].sum())
        l2_losses.append(L_l2.item()*masks[-1].sum())

    #配列を結合
    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        #avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, misclassified, all_pred
        return float('nan'), float('nan'), [], [], [], float('nan'), [], [],\
               float('nan'), float('nan') #avg_kl, avg_l2


    #avg_ 表示用　有効サンプル数(発話数)で割る
    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    avg_kl_loss = round(np.sum(kl_losses)/np.sum(masks), 4)
    avg_l2_loss = round(np.sum(l2_losses)/np.sum(masks), 4)


    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, misclassified, all_pred,\
           avg_kl_loss, avg_l2_loss

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

#学習可能なパラメータ数
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    parser.add_argument('--out_path', default='demo_autoencoder')
    parser.add_argument('--autoencoder', type=str, default="symmetry", help='choose the autoencoder model')
    parser.add_argument('--learned_model', type=str, help='the path of learned unimodal model')
    parser.add_argument('--model', type=str, default="multi", help='multi or text or audio or visual')
    parser.add_argument('--valid_num', type=int, default=0) #交差検証用
    parser.add_argument('--weight', type=str, help='Specify the path to the trained weights') #テストに使用する学習済み重み


    args = parser.parse_args()
    today = datetime.datetime.now()
    #引数設定の保存
    params_save_to_csv(vars(args), args.out_path)
    print(args)

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


    #学習済みモデルの呼び出し
    model = torch.load(args.weight)

    print(model)
    exit()

    print("Trainable parameters:", count_trainable_params(model))

    if cuda:
        model.cuda()

    #KL情報量損 
    kl_loss = MaskedKLDivLoss()


    if args.Dataset == 'MELD':
        loss_function = MaskedL2Loss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        loss_function = MaskedL2Loss()
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0, valid_num=args.valid_num)
    else:
        #人物情報付きMELD
        loss_function = MaskedL2Loss()
        train_loader, valid_loader, test_loader = get_MELD_c_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)

    best_acc, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    best_misclassified = None
    best_all_pred = None
    best_epoch = None

    start_time = time.time()


    #検証，テスト
    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, misclassified, all_pred,\
    test_kl, test_l2,\
        = uni_train_or_eval_model(model, loss_function, test_loader, data_flag=args.Dataset, modal=args.model, kl_loss=kl_loss)

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


        print('test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
        format(test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))

  
    print('Test performance..')
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))


    #各データセットのラベル名
    if args.Dataset=='IEMOCAP':
        name_class = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
    else:
        name_class = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']

    with open(args.out_path+"/all_pred.csv", "w", newline='') as f:
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

    ##CSVに出力
    df_dict.to_csv(args.out_path+'/classification_report.csv')

    #混同行列の作成
    make_cm(best_label, best_pred, best_mask, name_class, "cm")


