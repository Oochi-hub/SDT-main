
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import E2EHierarchicalMCTN, E2EParallelMCTN, masked_contrastive_loss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score
import pandas as pd
import torch.nn as nn

import random

#seed値の設定
#default 42
seed = 42
torch.manual_seed(seed)
random.seed(seed)



#train_loss train_acc val_loss val_accの保存
def csv_history(history, output_folder):

    # 最大長を取得
    max_len = max(len(v) for v in history.values())

    # 長さを揃える
    history_fixed = {}

    for k, v in history.items():

        # 配列でない場合にも対応
        if not isinstance(v, list):
            v = list(v)

        history_fixed[k] = v + [None] * (max_len - len(v))

    # DataFrame化
    df = pd.DataFrame(history_fixed)

    # 出力先
    output_folder = os.path.join(output_folder, "train_results")
    os.makedirs(output_folder, exist_ok=True)

    path = os.path.join(output_folder, "history.csv")

    # 保存
    df.to_csv(path, index=False)

    print(f"Saved: {path}")

#パラメータ設定記録
def params_save_to_csv(params, output_folder):
    path = f"{output_folder}/arguemts.csv"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    df = pd.DataFrame([params])  # 引数をDataFrameに変換
    df.to_csv(path, index=False)  # CSVに保存

def save_plot(train_key, test_key, history, save_dir, title, filename,
              train_label="train", test_label="val"):

    train_data = history.get(train_key, [])
    test_data  = history.get(test_key, [])

    # 両方空なら出力しない
    if len(train_data) == 0 and len(test_data) == 0:
        return

    # 学習曲線を作成
    plt.figure()
    if len(train_data) > 0:
        plt.plot(train_data, label=train_label)
    if len(test_data) > 0:
        plt.plot(test_data, label=test_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

# 学習曲線のプロット
def plot_all(history, out_path):

    #ディレクトリを作成
    save_dir = out_path + "/train_results"
    os.makedirs(save_dir, exist_ok=True)

    # total loss
    save_plot("train_loss", "test_loss", history, save_dir,\
              "Total Loss", "total_loss.png")

    # classification
    save_plot("train_cls", "test_cls", history, save_dir,\
              "Classification Loss", "cls_loss.png")

    # translation
    save_plot("train_t1", "test_t1", history, save_dir,\
              "Translation t1 Loss", "trans_t1_loss.png")
    save_plot("train_t2", "test_t2", history, save_dir,\
              "Translation t2 Loss", "trans_t2_loss.png")

    # cycle
    save_plot("train_cycle", "test_cycle", history, save_dir,\
              "Cycle Loss", "cycle_loss.png")
    
    # parallel translation
    save_plot("train_t2a", "test_t2a", history, save_dir,\
              "Translation t2a Loss", "trans_t2a_loss.png")
    save_plot("train_t2v", "test_t2v", history, save_dir,\
              "Translation t2v Loss", "trans_t2v_loss.png")

    save_plot("train_a2t", "test_a2t", history, save_dir,\
              "Translation a2t Loss", "trans_a2t_loss.png")
    save_plot("train_a2v", "test_a2v", history, save_dir,\
              "Translation a2v Loss", "trans_a2v_loss.png")
    
    save_plot("train_v2t", "test_v2t", history, save_dir,\
              "Translation v2t Loss", "trans_v2t_loss.png")
    save_plot("train_v2a", "test_v2a", history, save_dir,\
              "Translation v2a Loss", "trans_v2a_loss.png")
    
    save_plot("train_trans", "test_trans", history, save_dir,\
              "Translation Loss", "trans_loss.png")
    
    # parallel cycle
    save_plot("train_cycle_t2a", "test_cycle_t2a", history, save_dir,\
              "Cycle t2a Loss", "cycle_t2a_loss.png")
    save_plot("train_cycle_t2v", "test_cycle_t2v", history, save_dir,\
              "Cycle t2v Loss", "cycle_t2v_loss.png")

    save_plot("train_cycle_a2t", "test_cycle_a2t", history, save_dir,\
              "Cycle a2t Loss", "cycle_a2t_loss.png")
    save_plot("train_cycle_a2v", "test_cycle_a2v", history, save_dir,\
              "Cycle a2v Loss", "cycle_a2v_loss.png")
    
    save_plot("train_cycle_v2t", "test_cycle_v2t", history, save_dir,\
              "Cycle v2t Loss", "cycle_v2t_loss.png")
    save_plot("train_cycle_v2a", "test_cycle_v2a", history, save_dir,\
              "Cycle v2a Loss", "cycle_v2a_loss.png")

    
    #contrasive loss
    save_plot("train_cont", "test_cont", history, save_dir,\
              "Contrasive Loss", "cont_loss.png")
    save_plot("train_cont_ta", "test_cont_ta", history, save_dir,\
              "Contrasive Text Audio Loss", "cont_ta_loss.png")
    save_plot("train_cont_tv", "test_cont_tv", history, save_dir,\
              "Contrasive Text Visual Loss", "cont_tv_loss.png")
    save_plot("train_cont_av", "test_cont_av", history, save_dir,\
              "Contrasive Audio Visual Loss", "cont_av_loss.png")

    # accuracy
    save_plot("train_acc", "test_acc", history, save_dir,\
              "Accuracy", "acc.png")

#分類結果
def save_report_csv(labels, preds, names, out_path):
    # dict形式で取得
    report_dict = classification_report(
        labels,
        preds,
        target_names=names,
        digits=4,
        output_dict=True
    )

    # DataFrame化（行＝クラス，列＝precisionなど）
    df = pd.DataFrame(report_dict).transpose()

    # 保存
    df.to_csv(out_path + "/report.csv", index=True)

#混同行列
def make_cm(labels, preds, names, out_path):
    # os.makedirs(out_path + "/cm", exist_ok=True)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    disp.plot(cmap=plt.cm.Blues)

    plt.savefig(out_path + "/cm.png")

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

def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, valid_num=0):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, valid_num)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=False,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=False,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def train_or_eval_mctn(model, dataloader, optimizer=None, train=False, modal_seq=None):

    config = {
        "w_t1": 0.1,
        "w_t2": 0.1,
        "w_cycle": 0.1,
        "w_cls": 1.0,
        "modal_seq": modal_seq
    }

    losses = []
    cls_losses, trans1_losses, trans2_losses, cycle_losses = [], [], [], []
    preds, labels = [], []

    device = next(model.parameters()).device

    mse = nn.MSELoss(reduction='none')
    ce = nn.CrossEntropyLoss(reduction='none')

    model.train() if train else model.eval()

    for data in dataloader:

        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = \
            [d.to(device) for d in data[:-2]]

        modal_dict = {
        "T": textf,
        "V": visuf,
        "A": acouf
    }

        # 例: "TAV"
        src_key = config["modal_seq"][0]
        tgt1_key = config["modal_seq"][1]
        tgt2_key = config["modal_seq"][2]

        x = modal_dict[src_key].permute(1, 0, 2)
        target_1 = modal_dict[tgt1_key].permute(1, 0, 2)
        target_2 = modal_dict[tgt2_key].permute(1, 0, 2)

        out1, cycle_out, out2, cls = model(x)

        cls_flat = cls.view(-1, cls.size(-1))
        label_flat = label.view(-1)

        #翻訳損失
        loss_trans1 = (mse(out1, target_1).mean(-1) * umask).mean()
        loss_trans2 = (mse(out2, target_2).mean(-1) * umask).mean()

        #逆翻訳損失
        if model.is_cycled:
            loss_cycle = (mse(cycle_out, x).mean(-1) * umask).mean()
        else:
            loss_cycle = torch.tensor(0.0).to(device) 

        #分類損失
        loss_cls = ce(cls_flat, label_flat)
        loss_cls = (loss_cls.view(umask.shape) * umask).mean()

        #全体の損失
        loss = (
            config["w_t1"] * loss_trans1 +
            config["w_t2"] * loss_trans2 +
            config["w_cycle"] * loss_cycle +
            config["w_cls"] * loss_cls
        )

        if train:
            loss.backward()
            optimizer.step()

        pred = torch.argmax(cls, dim=-1)
        pred = pred[umask == 1]
        label_ = label[umask == 1]

        preds.append(pred.detach().cpu().numpy())
        labels.append(label_.detach().cpu().numpy())

        #全体の損失，分類損失
        losses.append(loss.item())
        cls_losses.append(loss_cls.item())
        #翻訳損失，逆翻訳損失
        trans1_losses.append(loss_trans1.item())
        trans2_losses.append(loss_trans2.item())
        cycle_losses.append(loss_cycle.item())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    return (
        np.mean(losses),        #モデル全体の損失
        np.mean(cls_losses),    #分類損失
        np.mean(trans1_losses), #第一翻訳損失
        np.mean(trans2_losses), #第二翻訳損失
        np.mean(cycle_losses),  #第一逆翻訳損失
        accuracy_score(labels, preds) * 100,
        f1_score(labels, preds, average='weighted') * 100,
        labels, preds
    )

def train_or_eval_mctn_2(model, dataloader, optimizer=None, train=False, modal=None, const_flag=False, tau=1.00, is_cycle=True):

    losses = []
    cls_losses = []
    trans_losses = []
    trans_t2a_losses, trans_t2v_losses = [], []
    trans_a2t_losses, trans_a2v_losses = [], []
    trans_v2t_losses, trans_v2a_losses = [], []
    preds, labels = [], []

    const_ta_losses, const_tv_losses, const_av_losses = [], [], []
    const_losses = []

    cycle_losses = []
    cycle_t2a_losses, cycle_t2v_losses = [], []
    cycle_a2t_losses, cycle_a2v_losses = [], []
    cycle_v2t_losses, cycle_v2a_losses = [], []

    device = next(model.parameters()).device

    mse = nn.MSELoss(reduction='none')
    ce = nn.CrossEntropyLoss(reduction='none')

    model.train() if train else model.eval()

    for data in dataloader:

        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = \
            [d.to(device) for d in data[:-2]]
        
        textf = textf.permute(1, 0, 2)
        acouf = acouf.permute(1, 0, 2)
        visuf = visuf.permute(1, 0, 2)

        out_t2a, out_t2v, out_a2t, out_a2v, out_v2t, out_v2a, \
        cls, enc_t, enc_a, enc_v, \
        cycle_t2a, cycle_t2v, cycle_a2t, cycle_a2v, cycle_v2t, cycle_v2a = model(textf, acouf, visuf, modal, is_cycle)


        # --- class loss ---
        cls_flat = cls.view(-1, cls.size(-1))
        label_flat = label.view(-1)
        loss_cls = ce(cls_flat, label_flat)
        loss_cls = (loss_cls.view(umask.shape) * umask).mean()

        # --- trans loss ---
        ##text 2 other
        loss_trans_t2a = (mse(out_t2a, acouf).mean(-1) * umask).mean()
        loss_trans_t2v = (mse(out_t2v, visuf).mean(-1) * umask).mean()
        ##audio 2 other
        loss_trans_a2t = (mse(out_a2t, textf).mean(-1) * umask).mean()
        loss_trans_a2v = (mse(out_a2v, visuf).mean(-1) * umask).mean()
        ##visual 2 other
        loss_trans_v2t = (mse(out_v2t, textf).mean(-1) * umask).mean()
        loss_trans_v2a = (mse(out_v2a, acouf).mean(-1) * umask).mean()

        loss_trans = (loss_trans_t2a + loss_trans_t2v 
                   + loss_trans_a2t + loss_trans_a2v 
                   + loss_trans_v2t + loss_trans_v2a) / 6

        #損失の合計
        loss = (0.1 * loss_trans + 1.0 * loss_cls)

        # --- constract loss ---
        if const_flag:
            loss_cont_ta = masked_contrastive_loss(enc_t,enc_a,umask,tau)
            loss_cont_tv = masked_contrastive_loss(enc_t,enc_v,umask,tau)
            loss_cont_av = masked_contrastive_loss(enc_a,enc_v,umask,tau)

            loss_cont = (loss_cont_ta + loss_cont_tv + loss_cont_av) / 3
            loss += 0.1 * loss_cont

        # --- cycle loss ---
        if is_cycle:
            ##text 2 other
            loss_cycle_t2a = (mse(cycle_t2a, textf).mean(-1) * umask).mean()
            loss_cycle_t2v = (mse(cycle_t2v, textf).mean(-1) * umask).mean()
            ##audio 2 other
            loss_cycle_a2t = (mse(cycle_a2t, acouf).mean(-1) * umask).mean()
            loss_cycle_a2v = (mse(cycle_a2v, acouf).mean(-1) * umask).mean()
            ##visual 2 other
            loss_cycle_v2t = (mse(cycle_v2t, visuf).mean(-1) * umask).mean()
            loss_cycle_v2a = (mse(cycle_v2a, visuf).mean(-1) * umask).mean()

            loss_cycle = (loss_cycle_t2a + loss_cycle_t2v 
                   + loss_cycle_a2t + loss_cycle_a2v 
                   + loss_cycle_v2t + loss_cycle_v2a) / 6  

            loss += 0.1 * loss_cycle    

        if train:
            loss.backward()
            optimizer.step()

        pred = torch.argmax(cls, dim=-1)
        pred = pred[umask == 1]
        label_ = label[umask == 1]
        preds.append(pred.detach().cpu().numpy())
        labels.append(label_.detach().cpu().numpy())

        losses.append(loss.item())
        cls_losses.append(loss_cls.item())

        trans_losses.append(loss_trans.item())
        trans_t2a_losses.append(loss_trans_t2a.item())
        trans_t2v_losses.append(loss_trans_t2v.item())
        trans_a2t_losses.append(loss_trans_a2t.item())
        trans_a2v_losses.append(loss_trans_a2v.item())
        trans_v2t_losses.append(loss_trans_v2t.item())
        trans_v2a_losses.append(loss_trans_v2a.item())
        if const_flag:
            const_ta_losses.append(loss_cont_ta.item())
            const_tv_losses.append(loss_cont_tv.item())
            const_av_losses.append(loss_cont_av.item())
            const_losses.append(loss_cont.item())
        else:
            const_ta_losses.append(0)
            const_tv_losses.append(0)
            const_av_losses.append(0)
            const_losses.append(0)
        if is_cycle:
            cycle_losses.append(loss_cycle.item())
            cycle_t2a_losses.append(loss_cycle_t2a.item())
            cycle_t2v_losses.append(loss_cycle_t2v.item())
            cycle_a2t_losses.append(loss_cycle_a2t.item())
            cycle_a2v_losses.append(loss_cycle_a2v.item())
            cycle_v2t_losses.append(loss_cycle_v2t.item())
            cycle_v2a_losses.append(loss_cycle_v2a.item())
        else:
            cycle_losses.append(0)
            cycle_t2a_losses.append(0)
            cycle_t2v_losses.append(0)
            cycle_a2t_losses.append(0)
            cycle_a2v_losses.append(0)
            cycle_v2t_losses.append(0)
            cycle_v2a_losses.append(0)

    
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    return (
        np.mean(losses),           #モデル全体の損失
        np.mean(cls_losses),       #分類損失
        np.mean(trans_losses),     #翻訳損失
        np.mean(trans_t2a_losses), #t2a 翻訳損失
        np.mean(trans_t2v_losses), #t2v 翻訳損失
        np.mean(trans_a2t_losses), #a2t 翻訳損失
        np.mean(trans_a2v_losses), #a2v 翻訳損失
        np.mean(trans_v2t_losses), #v2t 翻訳損失
        np.mean(trans_v2a_losses), #v2a 翻訳損失
        np.mean(const_losses),     #対照損失
        np.mean(const_ta_losses),  #t,a 対照損失
        np.mean(const_tv_losses),  #t,v 対照損失
        np.mean(const_av_losses),  #a,v 対照損失
        np.mean(cycle_losses),     #逆翻訳損失
        np.mean(cycle_t2a_losses), #t2a 逆翻訳損失
        np.mean(cycle_t2v_losses), #t2v 逆翻訳損失
        np.mean(cycle_a2t_losses), #a2t 逆翻訳損失
        np.mean(cycle_a2v_losses), #a2v 逆翻訳損失
        np.mean(cycle_v2t_losses), #v2t 逆翻訳損失
        np.mean(cycle_v2a_losses), #v2a 逆翻訳損失
        accuracy_score(labels, preds) * 100,
        f1_score(labels, preds, average='weighted') * 100,
        labels, preds
    )

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ===== 引数 =====
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--tau', type=float, default=1.00)
    parser.add_argument('--Dataset', default='IEMOCAP')
    parser.add_argument('--out_path', default='demo')
    parser.add_argument('--valid_num', type=int, default=0)
    parser.add_argument('--modal_seq', type=str, default="TAV", help="modality order, if 階層型 e.g. TAV, ATV..., if 並列型 e.g T, A, V")
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--const_flag', default=False)
    parser.add_argument('--is_cycle', default=False)

    args = parser.parse_args()
    params_save_to_csv(vars(args), args.out_path) #引数の保存

    #cudaの確認
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if args.cuda else "cpu")

    batch_size = args.batch_size
    n_epochs = args.epochs

    # ===== データ設定 =====
    if args.Dataset == 'IEMOCAP':
        n_classes = 6
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(
            batch_size=batch_size, valid=0.0, valid_num=args.valid_num)
    else:
        n_classes = 7
        train_loader, valid_loader, test_loader = get_MELD_loaders(
            batch_size=batch_size, valid=0.0)
        
    modal_dims = {
    "T": 1024,  # text
    "V": 342,   # visual
    "A": 1582   # audio
    }

    #ソース，ターゲットモダリティの設定
    seq = args.modal_seq.upper()

    # ===== モデル =====
    if args.model_type == "hieral":

        seq = args.modal_seq.upper()

        assert len(seq) == 3
        assert set(seq) == set("TAV")

        input_dim  = modal_dims[seq[0]]
        second_dim = modal_dims[seq[1]]
        third_dim  = modal_dims[seq[2]]

        model = E2EHierarchicalMCTN(
            input_dim=input_dim,
            second_dim=second_dim,
            third_dim=third_dim,
            hidden_dim=args.hidden_dim,
            n_classes=n_classes
        ).to(device)
    elif args.model_type == "parallel":
        model = E2EParallelMCTN(
            text_dim=modal_dims["T"],
            audio_dim=modal_dims["A"],
            visual_dim=modal_dims["V"],
            hidden_dim=args.hidden_dim,
            n_classes=n_classes,
        ).to(device)

    print("Trainable parameters:", count_trainable_params(model))
    # for name, param in model.named_parameters():
    #     print(name, param.numel())
    # exit()

    #最適化手法
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
        eps=1e-7
    )

    best_acc = 0

    #損失保存用
    history = {
        "train_loss": [], "test_loss": [],
        "train_cls": [], "test_cls": [],

        "train_t1": [], "test_t1": [],
        "train_t2": [], "test_t2": [],
        "train_cycle": [], "test_cycle": [],

        "train_acc": [], "test_acc": [],
        "train_f1": [], "test_f1": [],

        "train_trans": [], "test_trans": [],
        "train_t2a": [], "test_t2a": [],
        "train_t2v": [], "test_t2v": [],
        "train_a2t": [], "test_a2t": [],
        "train_a2v": [], "test_a2v": [],
        "train_v2t": [], "test_v2t": [],
        "train_v2a": [], "test_v2a": [],

        "train_cont": [], "test_cont": [],
        "train_cont_ta": [], "test_cont_ta": [],
        "train_cont_tv": [], "test_cont_tv": [],
        "train_cont_av": [], "test_cont_av": [],

        "train_cycle_t2a": [], "test_cycle_t2a": [],
        "train_cycle_t2v": [], "test_cycle_t2v": [],
        "train_cycle_a2t": [], "test_cycle_a2t": [],
        "train_cycle_a2v": [], "test_cycle_a2v": [],
        "train_cycle_v2t": [], "test_cycle_v2t": [],
        "train_cycle_v2a": [], "test_cycle_v2a": [],
    }

    # ===== 学習 =====
    start_time = time.time() #総学習時間計測開始
    for e in range(n_epochs):
        epoch_time = time.time() #各エポック学習時間計測開始

        if args.model_type=="hieral": #階層型学習
            #学習
            train_loss, train_cls, train_t1, train_t2, train_cycle, train_acc, train_f1, _, _ = \
                train_or_eval_mctn(model, train_loader, optimizer, True, seq)
            #検証
            test_loss, test_cls, test_t1, test_t2, test_cycle, test_acc, test_f1, labels, preds = \
                train_or_eval_mctn(model, test_loader, None, False, seq)

            # 損失保存
            ##合計，分類損失
            history["train_loss"].append(train_loss), history["test_loss"].append(test_loss)
            history["train_cls"].append(train_cls), history["test_cls"].append(test_cls)

            #変換損失
            history["train_t1"].append(train_t1), history["test_t1"].append(test_t1)
            history["train_t2"].append(train_t2), history["test_t2"].append(test_t2)
            #自己回帰変換損失
            history["train_cycle"].append(train_cycle), history["test_cycle"].append(test_cycle)

            #Acc.，F値
            history["train_acc"].append(train_acc), history["test_acc"].append(test_acc)
            history["train_f1"].append(train_f1), history["test_f1"].append(test_f1)
        
        elif args.model_type=="parallel": #並列型学習

            #対照学習設定
            const_flag = args.const_flag
            tau = args.tau
            #自己回帰変換
            is_cycle = args.tau

            #学習
            train_loss, train_cls, \
            train_trans, train_t2a, train_t2v, train_a2t, train_a2v, train_v2t, train_v2a, \
            train_cont, train_cont_ta, train_cont_tv, train_cont_av, \
            train_cycle, train_cycle_t2a, train_cycle_t2v, train_cycle_a2t, train_cycle_a2v, train_cycle_v2t, train_cycle_v2a, \
            train_acc, train_f1, _, _ = \
                train_or_eval_mctn_2(model, train_loader, optimizer, True, seq[0], const_flag, tau, is_cycle)

            #検証
            test_loss, test_cls, \
            test_trans, test_t2a, test_t2v, test_a2t, test_a2v, test_v2t, test_v2a, \
            test_cont, test_cont_ta, test_cont_tv, test_cont_av, \
            test_cycle, test_cycle_t2a, test_cycle_t2v, test_cycle_a2t, test_cycle_a2v, test_cycle_v2t, test_cycle_v2a, \
            test_acc, test_f1, labels, preds = \
                train_or_eval_mctn_2(model, test_loader, None, False, seq[0], const_flag, tau, is_cycle)

            # 損失保存
            ##合計，分類損失
            history["train_loss"].append(train_loss), history["test_loss"].append(test_loss)
            history["train_cls"].append(train_cls), history["test_cls"].append(test_cls)

            ##変換損失
            history["train_trans"].append(train_trans), history["test_trans"].append(test_trans)
            history["train_t2a"].append(train_t2a), history["test_t2a"].append(test_t2a)
            history["train_t2v"].append(train_t2v), history["test_t2v"].append(test_t2v)
            history["train_a2t"].append(train_a2t), history["test_a2t"].append(test_a2t)
            history["train_a2v"].append(train_a2v), history["test_a2v"].append(test_a2v)
            history["train_v2t"].append(train_v2t), history["test_v2t"].append(test_v2t)
            history["train_v2a"].append(train_v2a), history["test_v2a"].append(test_v2a)

            #対照損失
            history["train_cont"].append(train_cont), history["test_cont"].append(test_cont)
            history["train_cont_ta"].append(train_cont_ta), history["test_cont_ta"].append(test_cont_ta)
            history["train_cont_tv"].append(train_cont_tv), history["test_cont_tv"].append(test_cont_tv)
            history["train_cont_av"].append(train_cont_av), history["test_cont_av"].append(test_cont_av)

            ##自己回帰 変換損失
            history["train_cycle"].append(train_cycle), history["test_cycle"].append(test_cycle)
            history["train_cycle_t2a"].append(train_cycle_t2a), history["test_cycle_t2a"].append(test_cycle_t2a)
            history["train_cycle_t2v"].append(train_cycle_t2v), history["test_cycle_t2v"].append(test_cycle_t2v)
            history["train_cycle_a2t"].append(train_cycle_a2t), history["test_cycle_a2t"].append(test_cycle_a2t)
            history["train_cycle_a2v"].append(train_cycle_a2v), history["test_cycle_a2v"].append(test_cycle_a2v)
            history["train_cycle_v2t"].append(train_cycle_v2t), history["test_cycle_v2t"].append(test_cycle_v2t)
            history["train_cycle_v2a"].append(train_cycle_v2a), history["test_cycle_v2a"].append(test_cycle_v2a)

            #Acc.，F値
            history["train_acc"].append(train_acc), history["test_acc"].append(test_acc)
            history["train_f1"].append(train_f1), history["test_f1"].append(test_f1)

        #最良モデル更新
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.out_path + "/best_mctn.pth")

        #各エポック結果を表示
        print(f"epoch: {e+1}, train_loss: {train_loss:.4f}, "
              f"train_acc: {train_acc:.2f}, test_acc: {test_acc:.2f}, "
              f"time: {round(time.time()-epoch_time,2)} sec")
        
    #学習時間を表示
    print("\n===== 学習時間 =====")
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"Runtime: {hours}時間 {minutes}分 {seconds:.2f}秒")  

    # ===== best model評価 =====
    print("\n===== Best Model Evaluation =====")
    #最良モデルを読み込む
    model.load_state_dict(torch.load(args.out_path + "/best_mctn.pth"))
    model.eval()

    #最良モデルテスト
    if args.model_type=="hieral":
        test_loss, test_cls, test_t1, test_t2, test_cycle, test_acc, test_f1, labels, preds = \
            train_or_eval_mctn(model, test_loader, None, False, seq)
    elif args.model_type=="parallel":
        const_flag = args.const_flag
        tau = args.tau
        is_cycle = args.is_cycle

        test_loss, test_cls, \
        test_trans, test_t2a, test_t2v, test_a2t, test_a2v, test_v2t, test_v2a, \
        test_cont, test_cont_ta, test_cont_tv, test_cont_av, \
        test_cycle, test_cycle_t2a, test_cycle_t2v, test_cycle_a2t, test_cycle_a2v, test_cycle_v2t, test_cycle_v2a, \
        test_acc, test_f1, labels, preds = \
            train_or_eval_mctn_2(model, test_loader, None, False, seq[0], const_flag, tau, is_cycle)
    #最良モデルテスト結果表示
    print(f"Best Test Accuracy: {test_acc:.2f}")
    print(f"Best Test F1-score: {test_f1:.2f}")

    # ===== 詳細評価 =====
    if args.Dataset == 'IEMOCAP':
        target_names = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
    else:
        target_names = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']

    plot_all(history, args.out_path) #学習曲線
    save_report_csv(labels, preds, target_names, args.out_path) #各ラベル分類精度記録
    make_cm(labels, preds, target_names, args.out_path) #混同行列
    csv_history(history, args.out_path) #各エポックの損失

