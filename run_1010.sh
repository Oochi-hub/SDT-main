#!/bin/bash

#出力ディレクトリ作成
outdir="log"

mkdir -p "$outdir"
logfile_iemo="$outdir/run_1022.log"


echo "==== 実行ログ ====" > "$logfile_iemo"

for type in normal correct good_at; do
    # for alpha in 0.50 0.80 0.85 0.95 0.90 1.00; do
    for alpha in 0.80 0.85 0.95 0.90; do
        # for lr in 0.0001 0.00001 0.000001; do
        for lr in 0.000001; do
        #確認用
        echo "Running: " >> "$logfile_iemo"

        python train_for_binary_model.py --lr $lr --batch-size 16 --epochs 100 --temp 1 --modal_mask_type $type --alpha $alpha \
        --out_path "experience_results/1020_result/binary_newmodel_100epo_nomask/$type/alpha_$alpha/lr_$lr" >> "$logfile_iemo"

        echo "" >> "$logfile_iemo"
        done
    done
done

# for type in good_at_2; do
#     # for alpha in 0.80 0.85 0.95 0.90 1.00; do
#     for alpha in 0.50; do
#         for lr in 0.00001 0.000001; do
#         #確認用
#         echo "Running: " >> "$logfile_iemo"

#         python train_for_newmodel.py --lr $lr --batch-size 16 --epochs 50 --temp 1 --modal_mask_type good_at_2 --alpha $alpha \
#         --out_path "experience_results/1016_result/newmodel_50epo/$type/lr_$lr" >> "$logfile_iemo"

#         echo "" >> "$logfile_iemo"
#         done
#     done
# done

