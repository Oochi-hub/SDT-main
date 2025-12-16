#改良案学習用コード
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset, MELDDataset_c
from model import MaskedNLLLoss, MaskedNLLLoss_2, autoencoder_and_multimodal_Model_with_flag_binary
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score
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
def show_history(history, out_folder):
    output_folder_grah = out_folder + '/train_results'
    if not os.path.exists(output_folder_grah):
                os.makedirs(output_folder_grah)

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

    #L_bi
    plt.figure()   #新しいウィンドウを描画
    plt.plot(history["train_bi_loss"], label = "train bi loss")
    plt.plot(history["val_bi_loss"], label = "val bi loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.xticks(xticks_range)
    plt.savefig(output_folder_grah+'/bi_loss.png')

    #ce
    plt.figure()   #新しいウィンドウを描画
    plt.plot(history["train_ce_loss"], label = "train ce loss")
    plt.plot(history["val_ce_loss"], label = "val ce loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.xticks(xticks_range)
    plt.savefig(output_folder_grah+'/ce_loss.png')

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

    plt.figure()   #新しいウィンドウを描画
    plt.plot(history["train_bi_loss_t"], label = "text loss", color='green')
    plt.plot(history["train_bi_loss_v"], label = "visual loss", color='blue')
    plt.plot(history["train_bi_loss_a"], label = "audio loss", color='red')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.xticks(xticks_range)
    plt.savefig(output_folder_grah+'/train_binary_modal_loss.png')

    # # グラフの描画
    # plt.figure(figsize=(8, 4))
    # plt.plot(range(1,len(history["lr"])+1),history["lr"], marker='o')
    # plt.title("ExponentialLR Learning Rate Schedule")
    # plt.xlabel("epoch")
    # plt.ylabel("Learning Rate")
    # plt.savefig(output_folder_grah+'/grah_lr.png')

def get_train_valid_sampler(trainset, valid=0.1, valid_num=0):
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

# --- パディング関数定義 ---
def pad_to_max(arr_list, max_len):
    padded_list = []
    for arr in arr_list:
        length = len(arr)
        if length < max_len:
            pad_width = ((0, max_len - length), (0, 0))  # [(時間次元のパディング), (特徴次元)]
            arr_padded = np.pad(arr, pad_width, mode="constant", constant_values=0)
        else:
            arr_padded = arr
        padded_list.append(arr_padded)
    return np.stack(padded_list)  # shape = (バッチ, max_len, 特徴次元)


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False, data_flag="IEMOCAP", bi_loss=None, merged_df=None):

    #表示用の入れ子
    losses, preds, labels, masks, = [], [], [], []
    bi_losses, ce_losses = [], []
    bi_losses_t, bi_losses_a, bi_losses_v = [], [], []
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

        batch_allprob_text = []
        batch_allprob_audio = []
        batch_allprob_visual = []
        batch_modal_flag = []

            
        max_len = max(lengths)

        for vid in vids:
            # 該当するvidの行を抽出
            vid_df = merged_df[merged_df["vid"] == vid].sort_values("utt_index")

            # all_prob系をnumpy → tensor化
            allprob_t = np.stack(vid_df["all_prob_text"].to_numpy())
            allprob_a = np.stack(vid_df["all_prob_audio"].to_numpy())
            allprob_v = np.stack(vid_df["all_prob_visual"].to_numpy())

            # modal_flag（各行ごとに取得、numpy配列のリスト）
            modal_flags = np.stack(vid_df["modal_flag"].to_numpy())

            # バッチリストに追加
            batch_allprob_text.append(allprob_t)
            batch_allprob_audio.append(allprob_a)
            batch_allprob_visual.append(allprob_v)
            batch_modal_flag.append(modal_flags)

        # --- 各モダリティをパディング ---
        batch_allprob_t = pad_to_max(batch_allprob_text, max_len)
        batch_allprob_a = pad_to_max(batch_allprob_audio, max_len)
        batch_allprob_v = pad_to_max(batch_allprob_visual, max_len)

        # --- modal_flagも同様に（特徴数3） ---
        batch_modal_flag = pad_to_max(batch_modal_flag, max_len)

        #tensorに変換
        batch_allprob_t = torch.from_numpy(batch_allprob_t.astype(np.float32)).clone().cuda()
        batch_allprob_a = torch.from_numpy(batch_allprob_a.astype(np.float32)).clone().cuda()
        batch_allprob_v = torch.from_numpy(batch_allprob_v.astype(np.float32)).clone().cuda()
        batch_modal_flag = torch.from_numpy(batch_modal_flag.astype(np.float32)).clone().cuda()

        if demo_charaID_flag:
            #prob1:t prob2:a prob3v
            final_prob, final_log_prob, binary_t_mask, binary_a_mask, binary_v_mask = model(textf, visuf, acouf, umask, qmask, lengths, modality_flags=batch_modal_flag)
            
        else:
            #prob1:t prob2:a prob3v
            final_prob, final_log_prob, binary_t_mask, binary_a_mask, binary_v_mask = model(textf, visuf, acouf, umask, qmask, lengths, modality_flags=batch_modal_flag)
        
        #損失計算
        mask_t = batch_modal_flag[:, :, 0].unsqueeze(-1)  # [batch, utt, 1]
        mask_a = batch_modal_flag[:, :, 1].unsqueeze(-1)
        mask_v = batch_modal_flag[:, :, 2].unsqueeze(-1)

        #ラベルの準備
        labels_ = label.view(-1)

        #損失項の係数
        global ALPHA

        ##CE損失計算
        ce_log_final_prob = final_log_prob.view(-1, final_log_prob.size()[2])

        L_ce = loss_function(ce_log_final_prob, labels_, umask)

        #bi損失計算
        target_t = batch_allprob_t.view(-1, batch_allprob_t.size()[2])
        target_a = batch_allprob_a.view(-1, batch_allprob_a.size()[2])
        target_v = batch_allprob_v.view(-1, batch_allprob_v.size()[2])

        bi_t = binary_t_mask.view(-1, binary_t_mask.size()[2])
        bi_a = binary_a_mask.view(-1, binary_a_mask.size()[2])
        bi_v = binary_v_mask.view(-1, binary_v_mask.size()[2])

        #親表現と子表現のbi情報量誤差用
        L_bi_t = bi_loss(bi_t, target_t, mask_t)
        L_bi_a = bi_loss(bi_a, target_a, mask_a)
        L_bi_v = bi_loss(bi_v, target_v, mask_v)

        DECAY_BI_T, DECAY_BI_A, DECAY_BI_V = 1.0, 1.0, 1.0

        L_bi   = DECAY_BI_T * L_bi_t + DECAY_BI_A * L_bi_a + DECAY_BI_V * L_bi_v

        ##モデル全体の損失
        L_task = ALPHA * L_ce + (1-ALPHA) * L_bi
        #誤分類事例の回収
        lp_ = final_prob.view(-1, final_prob.size()[2])
        pred_ = torch.argmax(lp_,1)



        # print("pred_", pred_.size())
        """
        pred_ torch.Size([1504])
        """

        #誤分類事例の回収
        #all_prob torch.Size([batch, seq_len, num_class])
        all_probmax = torch.argmax(final_prob, dim=-1)  # shape: (batch, seq_len)

        #バッチサイズと最大発話数を記録
        x = umask.size()[0]
        y = umask.size()[1]

        #np配列に変換
        umask_np = umask.cpu().numpy()
        label_np = label.cpu().numpy()
        all_probmax_np = all_probmax.cpu().numpy()

        ###確率分布保存用　np配列に変換
        all_prob_np = final_prob.detach().cpu().numpy()

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

        #予測結果の保存
        preds.append(pred_.data.cpu().numpy())
        #入力ラベルの保存
        labels.append(labels_.data.cpu().numpy())
        #マスクの保存(レポート計算用)
        masks.append(umask.view(-1).cpu().numpy())

        #各損失の保存
        losses.append(L_task.item()*masks[-1].sum())
        bi_losses.append(L_bi.item()*masks[-1].sum())
        ce_losses.append(L_ce.item()*masks[-1].sum())

        bi_losses_t.append(L_bi_t.item()*masks[-1].sum())
        bi_losses_a.append(L_bi_a.item()*masks[-1].sum())
        bi_losses_v.append(L_bi_v.item()*masks[-1].sum())
  

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
        #avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, misclassified, all_pred
        #avg_bi, avg_ce
        #avg_bi_t, avg_bi_a, avg_bi_v
        return float('nan'), float('nan'), [], [], [], float('nan'), [], [],\
               float('nan'), float('nan'),\
               float('nan'), float('nan'), float('nan')

    #avg_ 表示用　有効サンプル数(発話数)で割る
    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    avg_bi_loss = round(np.sum(bi_losses)/np.sum(masks), 4)
    avg_ce_loss = round(np.sum(ce_losses)/np.sum(masks), 4)

    avg_bi_loss_t = round(np.sum(bi_losses_t)/np.sum(masks), 4)
    avg_bi_loss_a = round(np.sum(bi_losses_a)/np.sum(masks), 4)
    avg_bi_loss_v = round(np.sum(bi_losses_v)/np.sum(masks), 4)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, misclassified, all_pred,\
           avg_bi_loss, avg_ce_loss,\
           avg_bi_loss_t, avg_bi_loss_a, avg_bi_loss_v

def make_cm(best_label, best_pred, best_mask, name_class, cm_name, folder="/cm"):
    #混同行列の作成
    #cm = confusion_matrix(best_label,best_pred,sample_weight=best_mask)
    cm = confusion_matrix(best_label,best_pred)

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

def for_merged_df(merged_df):
    # all_prob列とmodal_flag列を安全にPythonリスト→numpy配列に変換
    for col in ["all_prob_text", "all_prob_audio", "all_prob_visual", "modal_flag"]:
        merged_df[col] = merged_df[col].apply(
            lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x)
        )
    return merged_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--epochs', type=int, default=1, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--out_path', default='demo')
    parser.add_argument('--schedular', type=float, default=1.0) #ExponentialLR用 epochごとに何倍するか
    parser.add_argument('--valid_num', type=int, default=0) #交差検証用
    parser.add_argument('--modal_mask_type', type=str, help="modal mask flag")
    parser.add_argument('--alpha', type=float, default=0.90)


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
    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10':1582, 'denseface':342, 'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    n_speakers = 2 if args.Dataset=='IEMOCAP' else 9
    n_classes = 6 if args.Dataset=='IEMOCAP' else 7 


    #損失項の係数
    ALPHA = args.alpha

    #エポックごとの学習記録
    history = {"train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "train_bi_loss": [], "train_ce_loss": [],
            "val_bi_loss": [], "val_ce_loss": [],
            "lr": [],
            "train_bi_loss_t": [], "train_bi_loss_a": [], "train_bi_loss_v": []
            }

    print('temp {}'.format(args.temp))

    #キャラID demo
    demo_charaID_flag = False

    #モデル呼び出し

    #学習済みモデルのパス
    hidden_dim=args.hidden_dim
    dropout=args.dropout
    #バイナリ表現の次元
    latent_dim = 256
    #温度係数　IEMOCAP 1 MELD 8
    temp = args.temp
    finetune_flag = True #ファインチューニングの有無
    #学習済みモデルの重み
    auto_t = "experience_results/1009_autoencoder/text/weights/model_weights_last.pth"
    auto_a = "experience_results/1009_autoencoder/audio/weights/model_weights_last.pth"
    auto_v = "experience_results/1009_autoencoder/visual/weights/model_weights_last.pth"
    model = autoencoder_and_multimodal_Model_with_flag_binary(auto_t, auto_a, auto_v, hidden_dim, latent_dim, n_classes, finetune_flag, temp)


    # total_params = sum(p.numel() for p in model.parameters())
    # print('total parameters: {}'.format(total_params))

    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #学習可能なパラメータ数
    print("Trainable parameters:", count_trainable_params(model))

    if cuda:
        model.cuda()

    #学習率
    lr=args.lr

    #バイナリ表現の誤差を測る関数 
    bi_loss = MaskedNLLLoss_2()

    #最適化手法
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=args.l2)

    #LambdaLR
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.schedular)


    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0, valid_num=args.valid_num)
    else:
        #人物情報付きMELD
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_c_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
        
    # --- 教師バイナリ表現CSVの読み込み ---
    if args.modal_mask_type == "correct":
        print("学習データにおける正答サンプルのみ蒸留")
        df_train = pd.read_csv("binary_data/binary/all_pred_merged_train_correct_new.csv")
    elif args.modal_mask_type == "good_at":
        print("学習データにおける正答かつ得意ラベルサンプルのみ蒸留")
        df_train = pd.read_csv("binary_data/binary/all_pred_merged_train_good_at_new.csv")
    # elif args.modal_mask_type == "good_at_2":
    #     print("学習データにおける正答かつ得意ラベルサンプルのみ蒸留")
    #     df_train = pd.read_csv("binary_data/all_pred_merged_train_good_at_2_unimodal.csv")
    else:
        print("全学習データを蒸留")
        df_train = pd.read_csv("binary_data/binary/all_pred_merged_train.csv")
    df_test = pd.read_csv("binary_data/binary/all_pred_merged_test.csv")

    merged_df_train = for_merged_df(df_train)
    merged_df_test = for_merged_df(df_test)



    best_acc, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    best_misclassified = None
    best_all_pred = None
    best_epoch = None

    for e in range(n_epochs):
        start_time = time.time()
 
        #学習
        train_loss, train_acc, _, _, _, train_fscore, hoge, _,\
        train_bi, train_ce,\
        train_bi_t, train_bi_a, train_bi_v,\
            = train_or_eval_model(model, loss_function, train_loader, e, optimizer, True, data_flag=args.Dataset, bi_loss=bi_loss, merged_df=merged_df_train)


        #検証，テスト(テストデータを使用)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, misclassified, all_pred,\
        test_bi, test_ce,\
        test_bi_t, test_bi_a, test_bi_v,\
            = train_or_eval_model(model, loss_function, test_loader, e, data_flag=args.Dataset, bi_loss=bi_loss, merged_df=merged_df_test)

        all_acc.append(test_acc)
        all_fscore.append(test_fscore)

        #最良モデルの更新
        if best_acc == None or best_acc < test_acc:
            best_acc = test_acc
            best_label, best_pred, best_mask = test_label, test_pred, test_mask
            best_misclassified = misclassified
            best_all_pred = all_pred
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
        history['train_bi_loss'].append(train_bi)
        history['train_ce_loss'].append(train_ce)
        history['val_bi_loss'].append(test_bi)
        history['val_ce_loss'].append(test_ce)
        history["lr"].append(optimizer.param_groups[0]['lr'])  # 現在の学習率を記録
        history['train_bi_loss_t'].append(train_bi_t)
        history['train_bi_loss_a'].append(train_bi_a)
        history['train_bi_loss_v'].append(train_bi_v)

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
        format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))

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
        fieldnames2 = ["vid", "utt_index", "character","pred", "true", "text", "all_prob"]
        writer = csv.DictWriter(f, fieldnames=fieldnames2)
        writer.writeheader()
        for row in best_all_pred:
            filtered_row = {key: row.get(key, "") for key in fieldnames2}
            writer.writerow(filtered_row)
        print("Saved all Pred examples to all_pred.csv")

    # ===== CSV読み込み =====
    df_ = pd.read_csv(args.out_path+"/all_pred.csv")

    # ===== 予測値と真値を抽出 =====
    y_pred = df_["pred"]
    y_true = df_["true"]
    
    #分類精度結果
    # labels = list(range(len(name_class)))
    report_dict = classification_report(y_true, y_pred, target_names=name_class, digits=4, output_dict=True)
    ##DataFrameに変換
    df_dict = pd.DataFrame(report_dict).transpose()

    ##Best Acc.のエポックを記録
    df_dict.loc["best_epoch"] = best_epoch

    ##CSVに出力
    df_dict.to_csv(args.out_path+'/classification_report.csv')

    #混同行列の作成
    make_cm(y_true, y_pred, best_mask, name_class, "cm")

    #学習の記録
    show_history(history, args.out_path)
    csv_history(history, args.out_path)


