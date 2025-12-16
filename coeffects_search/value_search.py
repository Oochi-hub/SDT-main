#07/24~
# ビジュアルを狙った動的な係数導入のためのﾃｽﾄ

import pandas as pd
import matplotlib.pyplot as plt
import os

#history_coeffの保存
def csv_history_coeff(dict, path, name):
    # DataFrameに変換
    df = pd.DataFrame(dict)


    # 最初の行にキーを設定
    df.insert(0, 'Metric', df.index)

    output_folder = path + "/"

    file = f"{output_folder}/{name}_history_coeff.csv"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df.to_csv(file, index=False)

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
def moment(loss_kl_t_1, loss_kl_a_1, loss_kl_v_1, loss_kl_t_2, loss_kl_a_2, loss_kl_v_2, velocity, beta=0.90):
    #各モダリティの速度 = 前回の速度 + 直前の変化量
    velocity_t = beta * velocity["t"] + (1 - beta) * (loss_kl_t_1 - loss_kl_t_2)
    velocity_a = beta * velocity["a"] + (1 - beta) * (loss_kl_a_1 - loss_kl_a_2)
    velocity_v = beta * velocity["v"] + (1 - beta) * (loss_kl_v_1 - loss_kl_v_2)

    return velocity_t, velocity_a, velocity_v



def main(df, hoge):
    ##各モダリティの学習データに対するkl損失(list)
    loss_kl_t = df["train_kl_loss_t"]
    loss_kl_a = df["train_kl_loss_a"]
    loss_kl_v = df["train_kl_loss_v"]

    #学習回数
    epochs = len(loss_kl_t)

    #更新を検討する間隔
    change_epochs = 50

    #保存用
    history_coeff = {"rate_t":[], "rate_a":[], "rate_v":[],
            "ave_chan_t":[], "ave_chan_a":[], "ave_chan_v":[],
            "velocity_t":[], "velocity_a":[], "velocity_v":[]
            }

    velocity = {"t":0.0, "a":0.0, "v":0.0}

    for e in range(hoge,epochs):

        ####入力を用意################
        """
        loss_kl_{modal}_1: eエポック時(現在)のkl損失
        """
        loss_kl_t_1, loss_kl_a_1, loss_kl_v_1 = loss_kl_t[e], loss_kl_a[e], loss_kl_v[e]


        loss_kl_t_1_2, loss_kl_a_1_2, loss_kl_v_1_2 = loss_kl_t[e-1], loss_kl_a[e-1], loss_kl_v[e-1]

        #モーメントの計算
        velocity_t, velocity_a, velocity_v = moment(loss_kl_t_1, loss_kl_a_1, loss_kl_v_1, loss_kl_t_1_2, loss_kl_a_1_2, loss_kl_v_1_2, velocity)

        ####モーメントを保存
        velocity["t"] = velocity_t
        velocity["a"] = velocity_a
        velocity["v"] = velocity_v

        t=change_epochs

        #change_epochsごとに更新を検討
        if (e+1)%change_epochs == 0:

            if (e+1) == change_epochs:
                t = change_epochs - hoge

            ####入力を用意2################
            """
            1:現在の損失
            2:tエポック前の損失
            """
            print(f"1:{e}, 2:{e-(t-1)}")

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
            # print(ave_chan_t, ave_chan_a, ave_chan_v)

            history_coeff["ave_chan_t"].append(ave_chan_t)
            history_coeff["ave_chan_a"].append(ave_chan_a)
            history_coeff["ave_chan_v"].append(ave_chan_v)


            ##現在のモーメンタムを保存
            # print(velocity_t, velocity_a, velocity_v)
            history_coeff["velocity_t"].append(velocity_t)
            history_coeff["velocity_a"].append(velocity_a)
            history_coeff["velocity_v"].append(velocity_v)

    return history_coeff

for hoge in [10, 20, 30]:

    #損失の値
    df = pd.read_csv(f"../experience_results/0724/t_1.0/train_results/history.csv")

    path = "result_csv/T_test"
    name = f"T_{hoge}"

    history_coeff = main(df, hoge)

    #保存
    csv_history_coeff(history_coeff, path, name)