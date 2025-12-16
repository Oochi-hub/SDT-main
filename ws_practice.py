#ワッシャーシュタイン　練習
import torch
from model import WassersteinLoss
import random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import time

#seed値の設定
#default 42
seed = 42
torch.manual_seed(seed)
random.seed(seed)

def wasserstein_loss(p, q, M, mask, reg=0.005, num_epoch=300):
    """
    p:[batch, 最大発話数, class_num] 親表現の出力確率分布
    q:[batch, 最大発話数, class_num] 子表現の出力確率分布
    M:[class_num, class_num] 距離行列(バッチごとに共通)
    mask:[batch, 最大発話数,]
    reg: float 正則化パラメータ
    num_epoch: int shinkerhornアルゴリズムの試行回数
    """

    batch, t, class_num = p.shape #バッチサイズ，最大発話数，クラス数を取得
    device = p.device #デバイス情報の取得
    losses = []

    #行列を用意
    K_mat = torch.exp(-M/reg).unsqueeze(0).to(device) #[1, class_num, class_num]

    # Flattenして [B*T, C] に変換
    p_flat = p.reshape(batch*t, class_num)
    q_flat = q.reshape(batch*t, class_num)

    #バッチごとにu,vを1に初期化
    u = torch.ones((batch*t, class_num), device=device) #[batch*t, class_num]
    v = torch.ones((batch*t, class_num), device=device) #[batch*t, class_num]

    #shinkerborn
    for i in range(num_epoch):
        
        #uの更新 u = p / (K_mat @ v)
        ##torch.bmmでバッチ行列積を計算, vは列ベクトルに変換
        ##1e-16は数値安定化のため
        u = p_flat / (torch.bmm(K_mat.expand(batch*t, -1, -1), v.unsqueeze(2)).squeeze(2) + 1e-16) 

        #vの更新 v = q / (K_mat^T @ u)
        ##torch.bmmでバッチ行列積を計算, uは列ベクトルに変換
        ##1e-16は数値安定化のため
        v = q_flat / (torch.bmm(K_mat.transpose(1,2).expand(batch*t, -1, -1), u.unsqueeze(2)).squeeze(2) + 1e-16)

        #結合分布joint_distを求める 
        joint_dist = u.unsqueeze(2) * K_mat.expand(batch*t, -1, -1) * v.unsqueeze(1) #[batch*t, class_num, class_num]

        #質量*距離でバッチごとに変更コストを計算
        cost = (joint_dist * M.unsqueeze(0)).sum(dim=(1, 2)) #[batch*t]

        # マスクをフラット化
        mask_flat = mask.reshape(batch*t).float() #[batch*t]

        # マスク適用して平均
        masked_loss = cost * mask_flat
        loss = masked_loss.sum() / (mask_flat.sum() + 1e-16)

        losses.append(loss)

    return losses

def wasserstein_loss_log(p, q, M, mask, reg=0.005, num_epoch=300):
    """
    p:[batch, 最大発話数, class_num] 親表現の出力確率分布
    q:[batch, 最大発話数, class_num] 子表現の出力確率分布
    M:[class_num, class_num] 距離行列(バッチごとに共通)
    mask:[batch, 最大発話数,]
    reg: float 正則化パラメータ
    num_epoch: int sinkhornアルゴリズムの反復回数
    """

    batch, t, class_num = p.shape
    device = p.device

    losses = []

    # log_p, log_q の計算
    log_p = torch.log(p + 1e-16).reshape(batch*t, class_num)   # [batch*t, C]
    log_q = torch.log(q + 1e-16).reshape(batch*t, class_num)   # [batch*t, C]

    # Sinkhorn 用の行列
    K = -M / reg   # [C, C]

    # u,v の初期化
    log_u = torch.zeros((batch*t, class_num), device=device)   # [B*T, C]
    log_v = torch.zeros((batch*t, class_num), device=device)   # [B*T, C]

    for _ in range(num_epoch):
        # logsumexp の計算 (broadcast対応)
        log_u = log_p - torch.logsumexp(K.unsqueeze(0) + log_v.unsqueeze(1), dim=2)
        log_v = log_q - torch.logsumexp(K.unsqueeze(0) + log_u.unsqueeze(2), dim=1)

        # joint distribution
        joint_dist = torch.exp(log_u.unsqueeze(2) + log_v.unsqueeze(1) + K.unsqueeze(0))  # [B*T, C, C]

        # コスト計算
        cost = (joint_dist * M.unsqueeze(0)).sum(dim=(1, 2))  # [B*T]

        # マスクをフラット化
        mask_flat = mask.reshape(batch*t).float() #[batch*t]

        # マスク適用して平均
        masked_loss = cost * mask_flat
        loss = masked_loss.sum() / (mask_flat.sum() + 1e-16)

        losses.append(loss)

    return losses

# ====== 動作例 ======
# B, T, C = 2, 5, 6  # 2バッチ・最大5発話・6クラス

# # モデル出力（予測確率）
# p = torch.rand(B, T, C)
# p = p / p.sum(dim=2, keepdim=True)

# # 教師分布（one-hot）
# labels = torch.randint(0, C, (B, T))
# q = torch.nn.functional.one_hot(labels, num_classes=C).float()

B, T, C = 8, 30, 6  # 8 batches, 30 utterances, 6 classes

# ランダムなロジット（未正規化スコア）
student_logits = torch.randn(B, T, C)
teacher_logits = torch.randn(B, T, C)

# Softmax で確率分布に変換
p = F.softmax(student_logits, dim=-1)  # [B, T, C]
q = F.softmax(teacher_logits, dim=-1)  # [B, T, C]
# クラス間距離（0/1）
M = torch.ones(C, C) - torch.eye(C)
# iemo_matrix =[[0.70, 0.01, 0.02, 0.00, 0.16, 0.00],
#                 [0.01, 0.77, 0.03, 0.01, 0.00, 0.04],
#                 [0.09, 0.08, 0.74, 0.01, 0.04, 0.07],
#                 [0.00, 0.02, 0.01, 0.76, 0.00, 0.11],
#                 [0.18, 0.01, 0.05, 0.00, 0.75, 0.01],
#                 [0.01, 0.08, 0.13, 0.17, 0.02, 0.74]]
# iemo_matrix = torch.tensor(iemo_matrix, dtype=torch.float32)
# M = 0.8*torch.ones(C, C) - iemo_matrix

# パディングマスク（例: 2バッチ目は3発話まで有効）
# mask = torch.tensor([[1, 1, 1, 1, 1],
#                      [1, 1, 1, 0, 0]])
mask = torch.zeros(B, T)
for b in range(B):
    valid_len = random.randint(15, T)  # at least 15 utterances valid
    mask[b, :valid_len] = 1

# loss_fn = WassersteinLoss(M)
# loss = loss_fn(p, q, mask)
# # print(f"p:{p}")
# # print(f"q:{q}")
# print(f"M:{M}")
# print("ワッシャーシュタイン損失:", loss.item())

# 計測開始
start = time.time()

losses = wasserstein_loss(p,q,M,mask)

end = time.time()

print("elapsed time: {:.4f} sec".format(end - start))

print("ワッシャーシュタイン損失:", losses[-1])

# 折れ線グラフを描画
plt.plot(losses)

# タイトルとラベル
plt.title("Line Plot Example")
plt.xlabel("Index")
plt.ylabel("Value")

# グラフを表示
plt.show()