import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- 感情ラベル対応表 ---
label_mapping = {
    0: "Happy",
    1: "Sad",
    2: "Neutral",
    3: "Angry",
    4: "Excited",
    5: "Frustrated",
}
labels = list(label_mapping.values())
modalities = ["text", "audio", "visual"]

# --- すべての混同行列を事前計算 ---
cms = []
max_val = 0
for modal in modalities:
    df = pd.read_csv(f"experience_results/0929_train_for_autoencoder/{modal}/all_pred.csv")
    y_true = df["true"].map(label_mapping)
    y_pred = df["pred"].map(label_mapping)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cms.append(cm)
    max_val = max(max_val, cm.max())  # 最大値を更新

# --- Figure 準備（constrained_layout=True を使用）---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

for ax, modal, cm in zip(axes, modalities, cms):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    im = disp.plot(
        cmap=plt.cm.Blues,
        xticks_rotation=45,
        ax=ax,
        colorbar=False,
        values_format="d",
    ).im_

    # --- 色スケールを統一 ---
    im.set_clim(0, max_val)

    # --- 各図タイトル ---
    ax.set_title(f"{modal.capitalize()}", fontsize=14)

# --- 共通カラーバー ---
cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
cbar.set_label("Count", fontsize=12)

# --- 保存 ---
plt.savefig("confusion_matrix_all_common_scale_labels.png", dpi=300)
plt.close()
