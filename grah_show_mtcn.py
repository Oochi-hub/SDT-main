import os
import pandas as pd
import matplotlib.pyplot as plt


def save_plot(train_key, test_key, history, save_dir, title, filename,
              train_label="train", test_label="test",
              y_lim=None):

    train_data = history.get(train_key, [])
    test_data  = history.get(test_key, [])

    # 両方空ならスキップ
    if len(train_data) == 0 and len(test_data) == 0:
        return

    plt.figure()

    if len(train_data) > 0:
        plt.plot(train_data, label=train_label)

    if len(test_data) > 0:
        plt.plot(test_data, label=test_label)

    # 縦軸範囲を指定
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def plot_all(history, out_path, y_limits=None):

    # 保存先ディレクトリ
    save_dir = os.path.join(out_path)
    os.makedirs(save_dir, exist_ok=True)

    # y_limitsが未指定なら空dict
    if y_limits is None:
        y_limits = {}

    # =========================
    # total loss
    # =========================
    save_plot(
        "train_loss", "test_loss", history, save_dir,
        "Total Loss", "total_loss.png",
        y_lim=y_limits.get("total_loss")
    )

    # =========================
    # classification
    # =========================
    save_plot(
        "train_cls", "test_cls", history, save_dir,
        "Classification Loss", "cls_loss.png",
        y_lim=y_limits.get("cls_loss")
    )

    # =========================
    # translation
    # =========================
    save_plot(
        "train_t1", "test_t1", history, save_dir,
        "Translation_1 Loss(v to t)", "trans_t1_loss.png",
        y_lim=y_limits.get("trans_t1_loss")
    )

    save_plot(
        "train_t2", "test_t2", history, save_dir,
        "Translation_2 Loss(emb to a)", "trans_t2_loss.png",
        y_lim=y_limits.get("trans_t2_loss")
    )

    # =========================
    # cycle
    # =========================
    save_plot(
        "train_cycle", "test_cycle", history, save_dir,
        "Cycle Loss(t to v)", "cycle_loss.png",
        y_lim=y_limits.get("cycle_loss")
    )

    # =========================
    # parallel translation
    # =========================
    save_plot(
        "train_t2a", "test_t2a", history, save_dir,
        "Translation t2a Loss", "trans_t2a_loss.png",
        y_lim=y_limits.get("trans_t2a_loss")
    )

    save_plot(
        "train_t2v", "test_t2v", history, save_dir,
        "Translation t2v Loss", "trans_t2v_loss.png",
        y_lim=y_limits.get("trans_t2v_loss")
    )

    save_plot(
        "train_a2t", "test_a2t", history, save_dir,
        "Translation a2t Loss", "trans_a2t_loss.png",
        y_lim=y_limits.get("trans_a2t_loss")
    )

    save_plot(
        "train_a2v", "test_a2v", history, save_dir,
        "Translation a2v Loss", "trans_a2v_loss.png",
        y_lim=y_limits.get("trans_a2v_loss")
    )

    save_plot(
        "train_v2t", "test_v2t", history, save_dir,
        "Translation v2t Loss", "trans_v2t_loss.png",
        y_lim=y_limits.get("trans_v2t_loss")
    )

    save_plot(
        "train_v2a", "test_v2a", history, save_dir,
        "Translation v2a Loss", "trans_v2a_loss.png",
        y_lim=y_limits.get("trans_v2a_loss")
    )

    # total translation
    save_plot(
        "train_trans", "test_trans", history, save_dir,
        "Translation Total Loss", "trans_loss.png",
        y_lim=y_limits.get("trans_loss")
    )

    # =========================
    # contrastive loss
    # =========================
    save_plot(
        "train_cont", "test_cont", history, save_dir,
        "Contrastive Loss", "cont_loss.png",
        y_lim=y_limits.get("cont_loss")
    )

    save_plot(
        "train_cont_ta", "test_cont_ta", history, save_dir,
        "Contrastive Text-Audio Loss", "cont_ta_loss.png",
        y_lim=y_limits.get("cont_ta_loss")
    )

    save_plot(
        "train_cont_tv", "test_cont_tv", history, save_dir,
        "Contrastive Text-Visual Loss", "cont_tv_loss.png",
        y_lim=y_limits.get("cont_tv_loss")
    )

    save_plot(
        "train_cont_av", "test_cont_av", history, save_dir,
        "Contrastive Audio-Visual Loss", "cont_av_loss.png",
        y_lim=y_limits.get("cont_av_loss")
    )

    # =========================
    # accuracy
    # =========================
    save_plot(
        "train_acc", "test_acc", history, save_dir,
        "Accuracy", "acc.png",
        y_lim=y_limits.get("acc")
    )

    # =========================
    # F1
    # =========================
    save_plot(
        "train_f1", "test_f1", history, save_dir,
        "F1 Score", "f1.png",
        y_lim=y_limits.get("f1")
    )


if __name__ == "__main__":

    # =========================
    # CSVファイル
    # =========================
    csv_path = "experiment_results/0513_mctn/VTA/dim_512/train_results/history.csv"

    # =========================
    # 出力先
    # =========================
    out_path = "experiment_results/0514_zemi/v_hiera"

    # =========================
    # 縦軸スケール設定
    # (min, max)
    # =========================
    y_limits = {

        "total_loss": (0, 2),
        "cls_loss": (0, 3.0),

        "trans_t1_loss": (0, 0.60),
        "trans_t2_loss": (0, 0.80),

        "cycle_loss": (0, 0.06),

        # parallel translation
        "trans_t2a_loss": (0, 0.7),
        "trans_t2v_loss": (0, 0.06),

        "trans_a2t_loss": (0, 0.60),
        "trans_a2v_loss": (0, 0.06),

        "trans_v2t_loss": (0, 0.60),
        "trans_v2a_loss": (0, 0.70),

        "trans_loss": (0, 3),

        # contrastive
        "cont_loss": (0, 10),
        "cont_ta_loss": (0, 10),
        "cont_tv_loss": (0, 10),
        "cont_av_loss": (0, 10),

        "acc": (0, 100),
        "f1": (0, 100),
    }

    # CSV読み込み
    df = pd.read_csv(csv_path)

    # dictへ変換
    history = {}

    for col in df.columns:
        history[col] = df[col].dropna().tolist()

    # グラフ作成
    plot_all(history, out_path, y_limits)

    print("グラフを保存しました")