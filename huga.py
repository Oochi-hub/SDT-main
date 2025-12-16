import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

label_mapping = {
    0:"happy",
    1:"sad",
    2:"neutral",
    3:"angry",
    4:"excited",
    5:"frustrated",
}

for modal in ["text", "audio", "visual"]:

    file1 = pd.read_csv(f"unique_correct_{modal}.csv")
    fusion = pd.read_csv("demo/all_pred.csv")

    file1["id"] = file1["vid"].astype(str) + "_" + file1["utt_index"].astype(str)
    fusion["id"] = fusion["vid"].astype(str) + "_" + fusion["utt_index"].astype(str)

    fusion["is_correct"] = fusion["pred"] == fusion["true"]
    fusion_wrong = fusion[fusion["is_correct"]==False][["id","pred","true"]]

    merged = file1.merge(
        fusion_wrong, on="id", how="inner",
        suffixes=("_file1", "_fusion")
    )

    # ==== 混同行列 ====
    cm = confusion_matrix(
        merged["pred_file1"],
        merged["pred_fusion"]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(label_mapping.values())
    )
    disp.plot(cmap=plt.cm.Blues)

    # ---- カラースケールを固定する重要ポイント ----
    im = disp.ax_.images[0]
    im.set_clim(vmin=0, vmax=15)  # ★ここで色の範囲を指定

    # ---- セル表示整形 ----
    for txt in disp.text_.flat:
        try:
            v = int(float(txt.get_text()))
            txt.set_text(f"{v}")
        except:
            pass
    save_path = f"confusion_matrix_{modal}.png"
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
