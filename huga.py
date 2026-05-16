import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ===== ラベル定義 =====
label_mapping = {
    0: "happy",
    1: "sad",
    2: "neutral",
    3: "angry",
    4: "excited",
    5: "frustrated",
}

labels = list(label_mapping.keys())
label_names = list(label_mapping.values())

for modal in ["text", "audio", "visual"]:
    # ===== CSV読み込み =====
    df = pd.read_csv(f"fusion_misclassified_in_{modal}.csv")

    # ===== 混同行列作成 =====
    cm = confusion_matrix(
        df["true_fusion"],   # 行：正解
        df["pred_fusion"],   # 列：予測
        labels=labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names
    )

    disp.plot(cmap=plt.cm.Blues)

    # ===== カラースケール固定 =====
    im = disp.ax_.images[0]
    im.set_clim(vmin=0, vmax=15)   # データ規模に応じて調整

    # ===== セル内数値を整数表示 =====
    for txt in disp.text_.flat:
        try:
            v = int(float(txt.get_text()))
            txt.set_text(f"{v}")
        except Exception:
            pass

    # ===== 体裁 =====
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (Fusion Model)")

    plt.tight_layout()

    # ===== 保存 =====
    plt.savefig(f"confusion_matrix_{modal}.png", dpi=300)
    plt.close()
