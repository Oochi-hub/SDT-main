import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import random

#seed値の設定
#default 42
seed = 42
torch.manual_seed(seed)
random.seed(seed)


class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()

    def forward(self, log_pred, target, mask):
        losses = F.kl_div(log_pred, target, reduction='none')  # shape: [B, C]
        masked_loss = (losses.sum(dim=1)) * mask.view(-1).float()  # sum over classes, then mask
        # print("KL損失")
        # print(masked_loss)
        return masked_loss.sum()
        #return masked_loss.sum() / mask.sum()
    
        # loss = self.loss(log_pred, target)  # shape: [batch, classes]
        # mask = mask.view(-1, 1).float()
        # loss = loss * mask
        # return loss.sum() / mask.sum()

#モダリティマスク KL損失
class MaskedKLDivLoss_2(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss_2, self).__init__()

    def forward(self, log_pred, target, mask, umask):
        """
        log_pred : [B, T, C]   → 各モダリティの log softmax 出力 (log p)
        target   : [B, T, C]   → 教師側 softmax 出力 (q)
        mask     : [B, T] または [B, T, 1] → 発話単位の有効フラグ (0/1)
                    → モダリティごとに mask_t, mask_a, mask_v が渡される
        """
        # KLDivLoss の要素ごとの出力を取得
        # reduction='none' により shape: [B * T, C]
        kl_per_class = F.kl_div(log_pred, target, reduction='none')

        # クラス次元を合計（KLの総和を各発話ごとに1値に）
        kl_per_sample = kl_per_class.sum(dim=-1)  # [B * T]

        mask_flatt = mask.view(-1).float() # [B * T]

        # マスク適用（0の部分を除外）
        masked_loss = kl_per_sample * mask_flatt

        # 平均化（マスクで有効な部分のみ）
        loss = masked_loss.sum() / (mask.sum() + 1e-8)
        return loss #/ umask.sum()


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    # def forward(self, pred, target, mask):
    #     """
    #     pred: shape (batch_size, num_classes) → log-probabilities
    #     target: shape (batch_size,) → class indices
    #     mask: shape (batch_size,) → 0 or 1
    #     """
    #     losses = F.nll_loss(pred, target, weight=self.weight, reduction='none')  # shape: [B]
    #     masked_loss = losses * mask.view(-1).float()
    #     return masked_loss.sum() / mask.sum()

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())

        return loss
    
class MaskedNLLLoss_2(nn.Module):
    def __init__(self, eps=1e-8):
        super(MaskedNLLLoss_2, self).__init__()
        self.eps = eps  # log(0)防止用の微小値

    def forward(self, pred, target, mask):
        """
        pred:   (B*T, D) 完全なバイナリ表現 (0 or 1)
        target: (B*T, D) 完全なバイナリ表現 (0 or 1)
        mask:   (B, T) または (B, T, 1) の0/1マスク
                → モダリティごとに渡される (mask_t, mask_a, mask_vなど)
        """

        # --- maskの整形 ---
        # mask_flat = mask.view(-1).float()  # (B*T, 1)

        # --- mask整形 ---
        mask_flat = mask.view(-1).float().unsqueeze(1)  # (B*T, 1)

        # #--- MSE計算 ---
        ce = (pred - target) ** 2  # (B*T, D)

        # --- マスク適用 ---
        ce_masked = ce * mask_flat  # 無効発話を除外 (B*T, D)

        # --- 平均化 ---
        loss = ce_masked.sum() / (mask_flat.sum() + self.eps)

        return loss
    
class WassersteinLoss(nn.Module):
    def __init__(self, M, reg=0.025, num_epoch=300):
        """
        M: [class_num, class_num] クラス間距離行列（0/1やユーザ定義距離）
        reg: Sinkhorn の正則化パラメータ
        num_epoch: Sinkhorn の反復回数
        """
        super().__init__()
        self.register_buffer("M", M)  #距離行列Mは0,1での表現だから学習対象外
        self.reg = reg
        self.num_epoch = num_epoch

        #行列を用意
        self.K_mat = torch.exp(-M / reg).unsqueeze(0)  #[1, class_num, class_num]

    def forward(self, p, q, mask):
        """
        p:[batch, 最大発話数, class_num] 親表現の出力確率分布
        q:[batch, 最大発話数, class_num] 子表現の出力確率分布
        M:[class_num, class_num] 距離行列(バッチごとに共通)
        mask:[batch, 最大発話数,]
        """
        batch, t, class_num = p.shape #バッチサイズ，最大発話数，クラス数を取得
        device = p.device #デバイス情報の取得

        K_mat = self.K_mat.to(device)

        # Flattenして [B*T, C] に変換
        p_flat = p.reshape(batch * t, class_num)
        q_flat = q.reshape(batch * t, class_num)

        # Sinkhorn　初期化
        u = torch.ones((batch * t, class_num), device=device) #[batch*t, class_num]
        v = torch.ones((batch * t, class_num), device=device) #[batch*t, class_num]

        # Sinkhornアルゴリズム u,vの更新
        ## 理想的には各行と各列の合計がそれぞれp,qの各要素と一致する
        for _ in range(self.num_epoch):
            #uの更新 u_k+1 = p / (K_mat @ v_K)
            ##torch.bmmでバッチ行列積を計算, vは列ベクトルに変換
            ##1e-16は数値安定化のため
            u = p_flat / (torch.bmm(K_mat.expand(batch*t, -1, -1),
                                    v.unsqueeze(2)).squeeze(2) + 1e-16)
            
            #vの更新 v_k+1 = q / (K_mat^T @ u_k+1)
            ##torch.bmmでバッチ行列積を計算, uは列ベクトルに変換
            ##1e-16は数値安定化のため
            v = q_flat / (torch.bmm(K_mat.transpose(1, 2).expand(batch*t, -1, -1),
                                    u.unsqueeze(2)).squeeze(2) + 1e-16)

        #結合分布joint_distを求める  #[batch*t, class_num, class_num]
        joint_dist = u.unsqueeze(2) * K_mat.expand(batch*t, -1, -1) * v.unsqueeze(1)

        # 質量*距離でバッチごとに変更コストを計算
        cost = (joint_dist * self.M.unsqueeze(0).to(device)).sum(dim=(1, 2)) #[batch*t]

        # パディングマスク適用
        mask_flat = mask.reshape(batch * t).float() #[batch*t]
        # マスクをフラット化
        masked_loss = cost * mask_flat
        # マスク適用して平均
        loss = masked_loss.sum() / (mask_flat.sum() + 1e-16)

        return loss
    
class LogWassersteinLoss(nn.Module):
    def __init__(self, M, reg=0.005, num_epoch=300):
        """
        M: [class_num, class_num] クラス間距離行列（0/1やユーザ定義距離）
        reg: Sinkhorn の正則化パラメータ
        num_epoch: Sinkhorn の反復回数
        """
        super().__init__()
        self.register_buffer("M", M)  #距離行列Mは0,1での表現だから学習対象外
        self.reg = reg
        self.num_epoch = num_epoch

        #行列を用意
        self.K = -M / reg #[class_num, class_num]

    def forward(self, p, q, mask):
        """
        p:[batch, 最大発話数, class_num] 親表現の出力確率分布
        q:[batch, 最大発話数, class_num] 子表現の出力確率分布
        M:[class_num, class_num] 距離行列(バッチごとに共通)
        mask:[batch, 最大発話数,]
        """
        batch, t, class_num = p.shape #バッチサイズ，最大発話数，クラス数を取得
        device = p.device #デバイス情報の取得

        k = self.K.to(device)

        # log_p, log_q の計算
        log_p = torch.log(p + 1e-16).reshape(batch*t, class_num)   # [batch*t, C]
        log_q = torch.log(q + 1e-16).reshape(batch*t, class_num)   # [batch*t, C]


        # Sinkhorn　初期化
        log_u = torch.zeros((batch*t, class_num), device=device)   #[batch*t, class_num]
        log_v = torch.zeros((batch*t, class_num), device=device)   #[batch*t, class_num]

        # Sinkhornアルゴリズム u,vの更新
        ## 理想的には各行と各列の合計がそれぞれp,qの各要素と一致する
        for _ in range(self.num_epoch):
            # logsumexp の計算 (broadcast対応)
            log_u = log_p - torch.logsumexp(k.unsqueeze(0) + log_v.unsqueeze(1), dim=2)
            log_v = log_q - torch.logsumexp(k.unsqueeze(0) + log_u.unsqueeze(2), dim=1)

        #結合分布joint_distを求める  #[batch*t, class_num, class_num]
        joint_dist = torch.exp(log_u.unsqueeze(2) + log_v.unsqueeze(1) + k.unsqueeze(0))
        # 質量*距離でバッチごとに変更コストを計算
        cost = (joint_dist * self.M.unsqueeze(0).to(device)).sum(dim=(1, 2)) #[batch*t]

        # パディングマスク適用
        mask_flat = mask.reshape(batch * t).float() #[batch*t]
        # マスクをフラット化
        masked_loss = cost * mask_flat
        # マスク適用して平均
        loss = masked_loss.sum() / (mask_flat.sum() + 1e-16)

        return loss    

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).\
                    contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)
        
        out = self.dropout(context) + inputs_b
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        else:
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b

#9/7 モダリティごと埋め込み用
class TransformerEncoder_2(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder_2, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb_a, speaker_emb_b):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb_b)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        else:
            x_a = self.pos_emb(x_a, speaker_emb_a)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb_b)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b

class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        if dataset == 'MELD':
            self.fc.weight.data.copy_(torch.eye(hidden_size, hidden_size))
            self.fc.weight.requires_grad = False

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep

class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        return final_rep



class Transformer_Based_Model(nn.Module):
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout, demo_charaID_flag=False):
        super(Transformer_Based_Model, self).__init__()
        self.temp = temp
        self.n_classes = n_classes
        if demo_charaID_flag:
            #人物ごとの固有埋め込みベクトルを作成
            self.n_speakers = 6
            padding_idx = 6
        else:
            #話者IDによる発話の区別
            self.n_speakers = n_speakers
            #IEMOCAP　ペア会話
            if self.n_speakers == 2:
                padding_idx = 2
            #MELD　最大9人によるマルチターン会話
            if self.n_speakers == 9:
                padding_idx = 9
        # #各特徴量と同じ次元の埋め込みベクトルを用意
        self.speaker_embeddings = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)

        #9/7 モダリティごとに埋め込みを行う
        #各特徴量と同じ次元の埋め込みベクトルを用意
        # self.speaker_embeddings_t = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        # self.speaker_embeddings_a = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        # self.speaker_embeddings_v = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        

        
        # Temporal convolutional layers
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        #self.textf_input = nn.Linear(D_text, hidden_dim, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.visuf_input = nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False)
        
        # Intra- and Inter-modal Transformers
        self.t_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.a_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.a_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.v_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.a_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        #9/7 モダリティごと埋め込み用
        # self.t_t = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        # self.a_t = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        # self.v_t = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        # self.a_a = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        # self.t_a = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        # self.v_a = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        # self.v_v = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        # self.t_v = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        # self.a_v = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        
        # Unimodal-level Gated Fusion
        self.t_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.a_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.v_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.features_reduce_t = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_a = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_v = nn.Linear(3 * hidden_dim, hidden_dim)

        # Multimodal-level Gated Fusion
        self.last_gate = Multimodal_GatedFusion(hidden_dim)


        # Emotion Classifier
        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.all_output_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, demo_charaID_flag):
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        elif self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        elif self.n_speakers == 6:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (6*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()

        spk_embeddings = self.speaker_embeddings(spk_idx)

        #9/7 モダリティごとに埋め込み
        # spk_embeddings_t = self.speaker_embeddings_t(spk_idx)
        # spk_embeddings_a = self.speaker_embeddings_a(spk_idx)
        # spk_embeddings_v = self.speaker_embeddings_v(spk_idx)

        # Temporal convolutional layers
        # print("textf.shape before Conv1d:", textf.shape)

        # print("NaN in textf:", torch.isnan(textf).any().item())
        # print("Inf in textf:", torch.isinf(textf).any().item())

        #textf = self.textf_input(textf.permute(1, 2, 0).contiguous()).transpose(1, 2)
        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)
        #textf = self.textf_input(textf.permute(0, 2, 1).contiguous()).transpose(1, 2)
        # textf = self.textf_input(textf)
        # acouf = self.acouf_input(acouf.permute(0, 2, 1).contiguous()).transpose(1, 2)
        # visuf = self.visuf_input(visuf.permute(0, 2, 1).contiguous()).transpose(1, 2)


        # Intra- and Inter-modal Transformers
        t_t_transformer_out = self.t_t(textf, textf, u_mask, spk_embeddings)
        a_t_transformer_out = self.a_t(acouf, textf, u_mask, spk_embeddings)
        v_t_transformer_out = self.v_t(visuf, textf, u_mask, spk_embeddings)

        a_a_transformer_out = self.a_a(acouf, acouf, u_mask, spk_embeddings)
        t_a_transformer_out = self.t_a(textf, acouf, u_mask, spk_embeddings)
        v_a_transformer_out = self.v_a(visuf, acouf, u_mask, spk_embeddings)

        v_v_transformer_out = self.v_v(visuf, visuf, u_mask, spk_embeddings)
        t_v_transformer_out = self.t_v(textf, visuf, u_mask, spk_embeddings)
        a_v_transformer_out = self.a_v(acouf, visuf, u_mask, spk_embeddings)

        # #9/7 モダリティごとに埋め込み
        # t_t_transformer_out = self.t_t(textf, textf, u_mask, spk_embeddings_t, spk_embeddings_t)
        # a_t_transformer_out = self.a_t(acouf, textf, u_mask, spk_embeddings_a, spk_embeddings_t)
        # v_t_transformer_out = self.v_t(visuf, textf, u_mask, spk_embeddings_v, spk_embeddings_t)

        # a_a_transformer_out = self.a_a(acouf, acouf, u_mask, spk_embeddings_a, spk_embeddings_a)
        # t_a_transformer_out = self.t_a(textf, acouf, u_mask, spk_embeddings_t, spk_embeddings_a)
        # v_a_transformer_out = self.v_a(visuf, acouf, u_mask, spk_embeddings_v, spk_embeddings_a)

        # v_v_transformer_out = self.v_v(visuf, visuf, u_mask, spk_embeddings_v, spk_embeddings_v)
        # t_v_transformer_out = self.t_v(textf, visuf, u_mask, spk_embeddings_t, spk_embeddings_v)
        # a_v_transformer_out = self.a_v(acouf, visuf, u_mask, spk_embeddings_a, spk_embeddings_v)

        # Unimodal-level Gated Fusion
        t_t_transformer_out = self.t_t_gate(t_t_transformer_out)
        a_t_transformer_out = self.a_t_gate(a_t_transformer_out)
        v_t_transformer_out = self.v_t_gate(v_t_transformer_out)

        a_a_transformer_out = self.a_a_gate(a_a_transformer_out)
        t_a_transformer_out = self.t_a_gate(t_a_transformer_out)
        v_a_transformer_out = self.v_a_gate(v_a_transformer_out)

        v_v_transformer_out = self.v_v_gate(v_v_transformer_out)
        t_v_transformer_out = self.t_v_gate(t_v_transformer_out)
        a_v_transformer_out = self.a_v_gate(a_v_transformer_out)

        t_transformer_out = self.features_reduce_t(torch.cat([t_t_transformer_out, a_t_transformer_out, v_t_transformer_out], dim=-1))
        a_transformer_out = self.features_reduce_a(torch.cat([a_a_transformer_out, t_a_transformer_out, v_a_transformer_out], dim=-1))
        v_transformer_out = self.features_reduce_v(torch.cat([v_v_transformer_out, t_v_transformer_out, a_v_transformer_out], dim=-1))

        # Multimodal-level Gated Fusion
        all_transformer_out = self.last_gate(t_transformer_out, a_transformer_out, v_transformer_out)


        # Emotion Classifier
        t_final_out = self.t_output_layer(t_transformer_out)
        a_final_out = self.a_output_layer(a_transformer_out)
        v_final_out = self.v_output_layer(v_transformer_out)
        all_final_out = self.all_output_layer(all_transformer_out)

        #各モダリティのlog softmax 損失計算用
        t_log_prob = F.log_softmax(t_final_out, 2)
        a_log_prob = F.log_softmax(a_final_out, 2)
        v_log_prob = F.log_softmax(v_final_out, 2)

        #各モダリティのsoft max 分析用
        t_prob = F.softmax(t_final_out, 2)
        a_prob = F.softmax(a_final_out, 2)
        v_prob = F.softmax(v_final_out, 2)

        #融合表現のlog_softmax. softmax
        all_log_prob = F.log_softmax(all_final_out, 2)
        all_prob = F.softmax(all_final_out, 2)

        #KL用 温度係数付き log_softmax
        kl_t_log_prob = F.log_softmax(t_final_out /self.temp, 2)
        kl_a_log_prob = F.log_softmax(a_final_out /self.temp, 2)
        kl_v_log_prob = F.log_softmax(v_final_out /self.temp, 2)

        kl_all_prob = F.softmax(all_final_out /self.temp, 2)

        #WS用 温度係数付き softmax
        ws_t_prob = F.softmax(t_final_out /self.temp, 2)
        ws_a_prob = F.softmax(a_final_out /self.temp, 2)
        ws_v_prob = F.softmax(v_final_out /self.temp, 2)

        ws_all_prob = F.softmax(all_final_out /self.temp, 2)

        return t_log_prob, a_log_prob, v_log_prob, all_log_prob, all_prob, \
               kl_t_log_prob, kl_a_log_prob, kl_v_log_prob, kl_all_prob, \
               ws_t_prob, ws_a_prob, ws_v_prob, ws_all_prob, \
               t_prob, a_prob, v_prob

#9/16 各モダリティ表現の分類性能，正解，不正解事例の分析用        
class Single_Modal_Transformer_Based_Model(nn.Module):
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout, demo_charaID_flag=False):
        super(Single_Modal_Transformer_Based_Model, self).__init__()
        self.temp = temp
        self.n_classes = n_classes
        if demo_charaID_flag:
            #人物ごとの固有埋め込みベクトルを作成
            self.n_speakers = 6
            padding_idx = 6
        else:
            #話者IDによる発話の区別
            self.n_speakers = n_speakers
            #IEMOCAP　ペア会話
            if self.n_speakers == 2:
                padding_idx = 2
            #MELD　最大9人によるマルチターン会話
            if self.n_speakers == 9:
                padding_idx = 9
        # #各特徴量と同じ次元の埋め込みベクトルを用意
        self.speaker_embeddings = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)

        #9/7 モダリティごとに埋め込みを行う
        #各特徴量と同じ次元の埋め込みベクトルを用意
        # self.speaker_embeddings_t = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        # self.speaker_embeddings_a = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        # self.speaker_embeddings_v = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        

        
        # Temporal convolutional layers
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.visuf_input = nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False)
        

        # Intra- and Inter-modal Transformers
        self.x_x = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.y_x = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.z_x = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)


        #9/7 モダリティごと埋め込み用
        # self.t_t = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        # self.a_t = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        # self.v_t = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        # self.a_a = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        # self.t_a = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        # self.v_a = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        # self.v_v = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        # self.t_v = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        # self.a_v = TransformerEncoder_2(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        
        # Unimodal-level Gated Fusion
        self.x_x_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.y_x_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.z_x_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.features_reduce = nn.Linear(3 * hidden_dim, hidden_dim)

        # Emotion Classifier
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, Modal):
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        elif self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        elif self.n_speakers == 6:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (6*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()

        spk_embeddings = self.speaker_embeddings(spk_idx)

        Module_flag = Modal

        #9/7 モダリティごとに埋め込み
        # spk_embeddings_t = self.speaker_embeddings_t(spk_idx)
        # spk_embeddings_a = self.speaker_embeddings_a(spk_idx)
        # spk_embeddings_v = self.speaker_embeddings_v(spk_idx)

        # Temporal convolutional layers
        # print("textf.shape before Conv1d:", textf.shape)

        # print("NaN in textf:", torch.isnan(textf).any().item())
        # print("Inf in textf:", torch.isinf(textf).any().item())

        #入力の特徴
        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)



        # Intra- and Inter-modal Transformers
        if Module_flag == "t":
            x_x_transformer_out = self.x_x(textf, textf, u_mask, spk_embeddings)
            y_x_transformer_out = self.y_x(acouf, textf, u_mask, spk_embeddings) #self attentionのみで　9/17
            z_x_transformer_out = self.z_x(visuf, textf, u_mask, spk_embeddings)
        elif Module_flag == "a":
            x_x_transformer_out = self.x_x(acouf, acouf, u_mask, spk_embeddings)
            y_x_transformer_out = self.y_x(textf, acouf, u_mask, spk_embeddings)
            z_x_transformer_out = self.z_x(visuf, acouf, u_mask, spk_embeddings)
        else:
            x_x_transformer_out = self.x_x(visuf, visuf, u_mask, spk_embeddings)
            y_x_transformer_out = self.y_x(textf, visuf, u_mask, spk_embeddings)
            z_x_transformer_out = self.z_x(acouf, visuf, u_mask, spk_embeddings)

        # #9/7 モダリティごとに埋め込み
        # t_t_transformer_out = self.t_t(textf, textf, u_mask, spk_embeddings_t, spk_embeddings_t)
        # a_t_transformer_out = self.a_t(acouf, textf, u_mask, spk_embeddings_a, spk_embeddings_t)
        # v_t_transformer_out = self.v_t(visuf, textf, u_mask, spk_embeddings_v, spk_embeddings_t)

        # a_a_transformer_out = self.a_a(acouf, acouf, u_mask, spk_embeddings_a, spk_embeddings_a)
        # t_a_transformer_out = self.t_a(textf, acouf, u_mask, spk_embeddings_t, spk_embeddings_a)
        # v_a_transformer_out = self.v_a(visuf, acouf, u_mask, spk_embeddings_v, spk_embeddings_a)

        # v_v_transformer_out = self.v_v(visuf, visuf, u_mask, spk_embeddings_v, spk_embeddings_v)
        # t_v_transformer_out = self.t_v(textf, visuf, u_mask, spk_embeddings_t, spk_embeddings_v)
        # a_v_transformer_out = self.a_v(acouf, visuf, u_mask, spk_embeddings_a, spk_embeddings_v)

        # Unimodal-level Gated Fusion
        x_x_transformer_out = self.x_x_gate(x_x_transformer_out) #torch.Size([16, 74, 1024])
        y_x_transformer_out = self.y_x_gate(y_x_transformer_out) #torch.Size([16, 74, 1024])
        z_x_transformer_out = self.z_x_gate(z_x_transformer_out) #torch.Size([16, 74, 1024])

        transformer_out = self.features_reduce(torch.cat([x_x_transformer_out, y_x_transformer_out, z_x_transformer_out], dim=-1)) #torch.Size([16, 74, 1024])

        # Emotion Classifier
        final_out = self.output_layer(transformer_out)
        # # ##9/17 self attentionのみ
        # final_out = self.output_layer(x_x_transformer_out)

        #各モダリティのlog softmax 損失計算用
        log_prob = F.log_softmax(final_out, 2)
        #各モダリティのsoft max 分析用
        prob = F.softmax(final_out, 2)

        #9/29追加
        #autoencoder KL損失計算用
        kl_prob = F.softmax(final_out /self.temp, 2)

        return log_prob, prob, kl_prob, transformer_out
    

#autoencoder model
class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 順伝播では0/1に丸める
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        # 逆伝播ではそのまま勾配を流す（STE）
        return grad_output

def binary_ste(x):
    return BinaryActivation.apply(x)

#非対称オートエンコーダ
#バイナリ表現から直接復元
class AutoEncoder(nn.Module):
    def __init__(self, learned_model, hidden_dim, latent_dim, dropout, n_classes, temp):
        super().__init__()

        self.temp = temp

        #学習済みモデルをロード
        self.learned_model = torch.load(learned_model)
        for param in self.learned_model.parameters():
                param.requires_grad = False
        # self.learned_model.eval()  #学習済みモデルは更新しない

        # --- AutoEncoder部分 ---
        # 特徴量次元をhidden_dimに統一（学習済みモデルのfeatures次元に合わせる）
        self.encoder = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, hidden_dim)
        # Encoder: 1024 → 512 → 256
        # self.encoder = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim//2, latent_dim),
        # )

        # # Decoder: 256 → 512 → 1024
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim//2, hidden_dim)
        # )

        # バイナリ潜在表現から予測するヘッド
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_classes)
            )

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, Modal):
        # 学習済みモデルから予測と特徴量を取得
        _, _, target_pred, features = self.learned_model(
            textf, visuf, acouf, u_mask, qmask, dia_len, Modal
        )

        # --- AutoEncoder ---
        # エンコード
        z_cont = torch.sigmoid(self.encoder(features))  # 連続値 [0,1]
        z = binary_ste(z_cont)                         # STEで完全0/1表現
        # 復元
        h_hat = self.decoder(z)


        # 予測再構成
        binary_out = self.output_layer(z)
        binary_prob = F.softmax(binary_out, 2)
        kl_log_prob = F.log_softmax(binary_out /self.temp, 2)

        return target_pred, features, z, h_hat, binary_prob, kl_log_prob
    
#デコード部分をエンコードと対称的に
#デコード手順 2値=>連続値, 次元拡張
class Symmetry_AutoEncoder(nn.Module):
    def __init__(self, learned_model, hidden_dim, latent_dim, dropout, n_classes, temp, finetune_flag):
        super().__init__()

        self.temp = temp

        # 学習済みモデルをロード
        self.learned_model = torch.load(learned_model)

        for param in self.learned_model.parameters():
                    param.requires_grad = finetune_flag
        # self.learned_model.eval()

        # --- AutoEncoder部分 ---
        self.encoder = nn.Linear(hidden_dim, latent_dim)

        # バイナリ表現を連続値に直す（Linear + ReLU）
        self.bin2cont = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )

        # 復元用デコーダ
        self.decoder = nn.Linear(latent_dim, hidden_dim)
        # self.encoder = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim//2, latent_dim),
        # )

        # # Decoder: 256 → 512 → 1024
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim//2, hidden_dim)
        # )

        # 予測再構成（復元特徴量から分類）
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_classes)
        )

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, Modal):
        # 学習済みモデルから予測と特徴量を取得
        _, _, target_pred, features = self.learned_model(
            textf, visuf, acouf, u_mask, qmask, dia_len, Modal
        )

        # --- AutoEncoder ---
        # エンコード + バイナリ化
        z_cont = torch.sigmoid(self.encoder(features))
        z = binary_ste(z_cont)  # 完全0/1表現

        # バイナリ表現を連続値に直す
        z_recon = self.bin2cont(z)

        # 特徴量復元
        h_hat = self.decoder(z_recon)

        # 予測再構成
        binary_out = self.output_layer(z)
        binary_prob = F.softmax(binary_out, 2)
        kl_log_prob = F.log_softmax(binary_out /self.temp, 2)

        return target_pred, features, z, h_hat, binary_prob, kl_log_prob
    
#パディングマスク付きL2損失(平均)
class MaskedL2Loss(nn.Module):
    def __init__(self):
        super(MaskedL2Loss, self).__init__()

    def forward(self, feat1, feat2, mask):
        """
        feat1: (B, T, D)
        feat2: (B, T, D)
        mask:  (B, T)  → 0 or 1
        """
        # L2距離 (二乗平均 MSE)
        diff = (feat1 - feat2) ** 2          # (B, T, D)
        l2_per_utt = diff.mean(dim=-1)  # [B, T]

        # マスク適用
        masked_loss = l2_per_utt * mask.float()
        # print("L2損失")
        # print(masked_loss)

        # 正規化 (有効発話数で割る)
        return masked_loss.sum()
        #return masked_loss.sum() / mask.sum()
    

    
class autoencoder_and_multimodal_Model_with_flag(nn.Module):
    def __init__(self, autoeocoder_t, autoeocoder_a, autoeocoder_v,
                 hidden_dim, binarry_dim, n_classes, finetune_flag, temp):
        super(autoencoder_and_multimodal_Model_with_flag, self).__init__()

        self.temp = temp
        self.binarry_dim = binarry_dim
        self.hidden_dim = hidden_dim

        # ===== Load pretrained autoencoders =====
        self.autoeocoder_t = torch.load(autoeocoder_t)
        self.autoeocoder_a = torch.load(autoeocoder_a)
        self.autoeocoder_v = torch.load(autoeocoder_v)

        for model in [self.autoeocoder_t, self.autoeocoder_a, self.autoeocoder_v]:
            for param in model.parameters():
                param.requires_grad = finetune_flag

        # ===== Feature fusion =====
        self.features_reduce = nn.Linear(3 * hidden_dim, hidden_dim)

        # ===== Output layers =====
        self.all_output_layer = nn.Sequential(
            nn.Linear(3 * binarry_dim + hidden_dim, n_classes),
             nn.ReLU()
        )

        # self.output_layer_t = nn.Linear(binarry_dim, n_classes)
        # self.output_layer_a = nn.Linear(binarry_dim, n_classes)
        # self.output_layer_v = nn.Linear(binarry_dim, n_classes)

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, modality_flags=None):
        """
        modality_flags: torch.Tensor [batch, 3]
            各サンプルのモダリティ有効フラグ (0または1)
            例: [[1,1,1], [1,0,1], [0,1,0], ...]
        """

        # ===== Extract features from each autoencoder =====
        _, features_t, binary_t, _, _, kl_log_prob_t = self.autoeocoder_t(textf, visuf, acouf, u_mask, qmask, dia_len, "t")
        _, features_a, binary_a, _, _, kl_log_prob_a = self.autoeocoder_a(textf, visuf, acouf, u_mask, qmask, dia_len, "a")
        _, features_v, binary_v, _, _, kl_log_prob_v = self.autoeocoder_v(textf, visuf, acouf, u_mask, qmask, dia_len, "v")

        # ===== Concatenate and reduce =====
        all_features_multi = torch.cat([features_t, features_a, features_v], dim=-1)
        features_multi = self.features_reduce(all_features_multi)


        # ===== Apply modality mask (0/1 flags) =====
        # → ブロードキャストで同じ形に変換してマスク適用
        mask_t = modality_flags[:, :, 0].unsqueeze(-1)  # [batch, utt, 1]
        mask_a = modality_flags[:, :, 1].unsqueeze(-1)
        mask_v = modality_flags[:, :, 2].unsqueeze(-1)

        binary_t = binary_t * mask_t
        binary_a = binary_a * mask_a
        binary_v = binary_v * mask_v

        # ===== Concatenate binary + fused features =====
        final_features = torch.cat([binary_t, binary_a, binary_v, features_multi], dim=-1)
        # shape: [batch, seq_len, 3*binarry_dim + hidden_dim]

        # ===== Final outputs =====
        final_out = self.all_output_layer(final_features)
        binary_out_t = self.output_layer_t(binary_t)
        binary_out_a = self.output_layer_a(binary_a)
        binary_out_v = self.output_layer_v(binary_v)

        # ===== Probabilities =====
        final_prob = F.softmax(final_out, dim=2)
        final_log_prob = F.log_softmax(final_out, dim=2)

        kl_log_prob_t = F.log_softmax(binary_out_t / self.temp, dim=2)
        kl_log_prob_a = F.log_softmax(binary_out_a / self.temp, dim=2)
        kl_log_prob_v = F.log_softmax(binary_out_v / self.temp, dim=2)

        return final_prob, final_log_prob, kl_log_prob_t, kl_log_prob_a, kl_log_prob_v

class autoencoder_and_multimodal_Model_with_flag_binary(nn.Module):
    def __init__(self, autoeocoder_t, autoeocoder_a, autoeocoder_v,
                 hidden_dim, binarry_dim, n_classes, finetune_flag, temp):
        super(autoencoder_and_multimodal_Model_with_flag_binary, self).__init__()

        self.temp = temp
        self.binarry_dim = binarry_dim
        self.hidden_dim = hidden_dim

        # ===== Load pretrained autoencoders =====
        self.autoeocoder_t = torch.load(autoeocoder_t)
        self.autoeocoder_a = torch.load(autoeocoder_a)
        self.autoeocoder_v = torch.load(autoeocoder_v)

        for model in [self.autoeocoder_t, self.autoeocoder_a, self.autoeocoder_v]:
            for param in model.parameters():
                param.requires_grad = finetune_flag

        # ===== Feature fusion =====
        self.features_reduce = nn.Linear(3 * hidden_dim, hidden_dim)

        # ===== Output layers =====
        self.all_output_layer = nn.Sequential(
            nn.Linear(3 * binarry_dim + hidden_dim, n_classes),
             nn.ReLU()
        )


    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, modality_flags=None):
        """
        modality_flags: torch.Tensor [batch, 3]
            各サンプルのモダリティ有効フラグ (0または1)
            例: [[1,1,1], [1,0,1], [0,1,0], ...]
        """

        # ===== Extract features from each autoencoder =====
        _, features_t, binary_t, _, _, kl_log_prob_t = self.autoeocoder_t(textf, visuf, acouf, u_mask, qmask, dia_len, "t")
        _, features_a, binary_a, _, _, kl_log_prob_a = self.autoeocoder_a(textf, visuf, acouf, u_mask, qmask, dia_len, "a")
        _, features_v, binary_v, _, _, kl_log_prob_v = self.autoeocoder_v(textf, visuf, acouf, u_mask, qmask, dia_len, "v")

        # ===== Concatenate and reduce =====
        all_features_multi = torch.cat([features_t, features_a, features_v], dim=-1)
        features_multi = self.features_reduce(all_features_multi)


        # ===== Apply modality mask (0/1 flags) =====
        # → ブロードキャストで同じ形に変換してマスク適用
        # mask_t = modality_flags[:, :, 0].unsqueeze(-1)  # [batch, utt, 1]
        # mask_a = modality_flags[:, :, 1].unsqueeze(-1)
        # mask_v = modality_flags[:, :, 2].unsqueeze(-1)

        # binary_t_mask = binary_t * mask_t
        # binary_a_mask = binary_a * mask_a
        # binary_v_mask = binary_v * mask_v
        binary_t_mask = binary_t
        binary_a_mask = binary_a 
        binary_v_mask = binary_v

        # ===== Concatenate binary + fused features =====
        final_features = torch.cat([binary_t_mask, binary_a_mask, binary_v_mask, features_multi], dim=-1)
        # shape: [batch, seq_len, 3*binarry_dim + hidden_dim]

        # ===== Final outputs =====
        final_out = self.all_output_layer(final_features)

        # ===== Probabilities =====
        final_prob = F.softmax(final_out, dim=2)
        final_log_prob = F.log_softmax(final_out, dim=2)

        return final_prob, final_log_prob, binary_t_mask, binary_a_mask, binary_v_mask

###demo
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- 基本設定 ---
    max_utt = 21   # バッチ内の最大発話数
    batch_size = 16
    max_speakers = 2  # MELD: 9人 IEMOCAP: 2人

    # --- テンソル作成 ---
    textf = torch.randn(max_utt, batch_size, 1024)  # テキスト特徴量
    visuf = torch.randn(max_utt, batch_size, 342)   # ビジュアル特徴量
    acouf = torch.randn(max_utt, batch_size, 1582)   # オーディオ特徴量

    # --- マスク類 ---
    # 各会話の実際の発話数 (1〜max_uttの範囲でランダム)
    lengths = np.random.randint(5, max_utt + 1, size=batch_size)

    # umask: 各会話で有効な発話位置=1, それ以降は0
    umask = torch.zeros(batch_size, max_utt)
    for i, l in enumerate(lengths):
        umask[i, :l] = 1

    # qmask: 各会話内で各発話の話者IDをワンホットで表す (shape: [batch, utt, speakers])
    qmask = torch.zeros(batch_size, max_utt, max_speakers)
    for i, l in enumerate(lengths):
        # 各発話にランダムに話者IDを割り当て
        speaker_ids = np.random.randint(0, max_speakers, size=l)
        qmask[i, np.arange(l), speaker_ids] = 1

    # 教師ラベル（感情クラス）
    num_classes = 6
    label = torch.full((batch_size, max_utt), fill_value=-1)  # padding部は-1
    for i, l in enumerate(lengths):
        label[i, :l] = torch.randint(0, num_classes, (l,))

    # --- 結果確認 ---
    # print("textf:", textf.shape)
    # print("visuf:", visuf.shape)
    # print("acouf:", acouf.shape)
    # print("qmask:", qmask.shape)
    # print("umask:", umask.shape)
    # print("label:", label.shape)
    # print("lengths:", lengths.tolist())

    # モダリティフラグ (0/1)
    modality_flags = torch.randint(0, 2, (batch_size, max_utt, 3)).float().to(device)
    print("modality_flags.shape =", modality_flags[0][0])

    # === データを同じデバイスへ転送 ===
    textf = textf.to(device)
    visuf = visuf.to(device)
    acouf = acouf.to(device)
    umask = umask.to(device)
    qmask = qmask.to(device)

    #モデルの定義
    hidden_dim = 1024
    binary_dim = 256
    finetune_flag = True
    temp = 1
    auto_t = "experience_results/1009_autoencoder/text/weights/model_weights_last.pth"
    auto_a = "experience_results/1009_autoencoder/audio/weights/model_weights_last.pth"
    auto_v = "experience_results/1009_autoencoder/visual/weights/model_weights_last.pth"
    model = autoencoder_and_multimodal_Model_with_flag_binary(auto_t, auto_a, auto_v, hidden_dim, binary_dim, num_classes, finetune_flag, temp).to(device)

    model(textf, visuf, acouf, umask, qmask, lengths, modality_flags)

    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", count_trainable_params(model))