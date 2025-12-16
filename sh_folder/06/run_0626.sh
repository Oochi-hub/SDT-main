#!/bin/bash
#6/26~ 自己蒸留損失項のモダリティごとの係数調整実験用スクリプト やり直し

#出力ディレクトリ作成
outdir="log"

mkdir -p "$outdir"
logfile_iemo="$outdir/run_0626_selfdis_coeff_ex_coeff_kl_iemo.log"
logfile_meld="$outdir/run_0626_selfdis_coeff_ex_coeff_kl_meld.log"

echo "==== 実行ログ ====" > "$logfile_iemo"
echo "==== 実行ログ ====" > "$logfile_meld"

for coeff_t in 0.5 1.0 1.5 2.0; do
    for coeff_a in 0.5 1.0 1.5 2.0; do
        for coeff_v in 0.5 1.0 1.5 2.0; do
            coeffs="1.0, 1.0, 1.0, $coeff_t, $coeff_a, $coeff_v"

            #確認用
            #echo $coeffs
            echo "Running: [$coeffs]" >> "$logfile_iemo"
            echo "Running: [$coeffs]" >> "$logfile_meld"
            
            python train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --out_path "experience_results/0626_selfdis_coeff_ex/coeff_kl/comb_${coeffs}/IEMO" --dist_coefficients "[$coeffs]" >> "$logfile_iemo"
            python train.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --out_path "experience_results/0626_selfdis_coeff_ex/coeff_kl/comb_${coeffs}/MELD" --dist_coefficients "[$coeffs]" >> "$logfile_meld"
            echo "" >> "$logfile_iemo"
            echo "" >> "$logfile_meld"
        done
    done
done

# logfile_iemo2="$outdir/run_0626_selfdis_coeff_ex_coeff_cekl_iemo.log"
# logfile_meld2="$outdir/run_0626_selfdis_coeff_ex_coeff_cekl_meld.log"

# echo "==== 実行ログ ====" > "$logfile_iemo2"
# echo "==== 実行ログ ====" > "$logfile_meld2"

# for coeff_t in 0.5 1.0 1.5 2.0; do
#     for coeff_a in 0.5 1.0 1.5 2.0; do
#         for coeff_v in 0.5 1.0 1.5 2.0; do
#             coeffs="$coeff_t, $coeff_a, $coeff_v, 1.0, 1.0, 1.0"

#             #確認用
#             #echo $coeffs
#             echo "Running: [$coeffs]" >> "$logfile_iemo2"
#             echo "Running: [$coeffs]" >> "$logfile_meld2"
            
#             python train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --out_path "experience_results/0626_selfdis_coeff_ex/coeff_ce/comb_${coeffs}/IEMO" --dist_coefficients "[$coeffs]" >> "$logfile_iemo2"
#             python train.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --out_path "experience_results/0626_selfdis_coeff_ex/coeff_ce/comb_${coeffs}/MELD" --dist_coefficients "[$coeffs]" >> "$logfile_meld2"
#             echo "" >> "$logfile_iemo2"
#             echo "" >> "$logfile_meld2"
#         done
#     done
# done


##demo##
# coeffs=$(printf "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f" $val $val $val $val $val $val)
# python train.py --epoch 2 --dist_coefficients "[$coeffs]"