#!/bin/bash

#6/30 モダリティごとのCEとKLの関係調査

#出力ディレクトリ作成

outdir="log"

mkdir -p "$outdir"

logfile_iemo="$outdir/run_0630_selfdis_coeff_ex_coeff_t_iemo.log"

logfile_meld="$outdir/run_0630_selfdis_coeff_ex_coeff_t_meld.log"

echo "==== 実行ログ ====" > "$logfile_iemo"

echo "==== 実行ログ ====" > "$logfile_meld"

for coeff_1 in 0.5 1.0 1.5 2.0; do

    for coeff_2 in 0.5 1.0 1.5 2.0; do

        coeffs="$coeff_1, 1.0, 1.0, $coeff_2, 1.0, 1.0"

        #確認用

        #echo $coeffs

        echo "Running: [$coeffs]" >> "$logfile_iemo"

        echo "Running: [$coeffs]" >> "$logfile_meld"

        python train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --out_path "experience_results/0630_selfdis_coeff_ex/coeff_t/comb_${coeffs}/IEMO" --dist_coefficients "[$coeffs]" >> "$logfile_iemo"

        python train.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --out_path "experience_results/0630_selfdis_coeff_ex/coeff_t/comb_${coeffs}/MELD" --dist_coefficients "[$coeffs]" >> "$logfile_meld"

        echo "" >> "$logfile_iemo"

        echo "" >> "$logfile_meld"

    done

done

logfile_iemo="$outdir/run_0630_selfdis_coeff_ex_coeff_a_iemo.log"

logfile_meld="$outdir/run_0630_selfdis_coeff_ex_coeff_a_meld.log"

echo "==== 実行ログ ====" > "$logfile_iemo"

echo "==== 実行ログ ====" > "$logfile_meld"

for coeff_1 in 0.5 1.0 1.5 2.0; do

    for coeff_2 in 0.5 1.0 1.5 2.0; do

        coeffs="1.0, $coeff_1, 1.0, 1.0, $coeff_2, 1.0"

        #確認用

        #echo $coeffs

        echo "Running: [$coeffs]" >> "$logfile_iemo"

        echo "Running: [$coeffs]" >> "$logfile_meld"

        python train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --out_path "experience_results/0630_selfdis_coeff_ex/coeff_a/comb_${coeffs}/IEMO" --dist_coefficients "[$coeffs]" >> "$logfile_iemo"

        python train.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --out_path "experience_results/0630_selfdis_coeff_ex/coeff_a/comb_${coeffs}/MELD" --dist_coefficients "[$coeffs]" >> "$logfile_meld"

        echo "" >> "$logfile_iemo"

        echo "" >> "$logfile_meld"

    done

done

logfile_iemo="$outdir/run_0630_selfdis_coeff_ex_coeff_v_iemo.log"

logfile_meld="$outdir/run_0630_selfdis_coeff_ex_coeff_v_meld.log"

echo "==== 実行ログ ====" > "$logfile_iemo"

echo "==== 実行ログ ====" > "$logfile_meld"

for coeff_1 in 0.5 1.0 1.5 2.0; do

    for coeff_2 in 0.5 1.0 1.5 2.0; do

        coeffs="1.0, 1.0, $coeff_1, 1.0, 1.0, $coeff_2"

        #確認用

        #echo $coeffs

        echo "Running: [$coeffs]" >> "$logfile_iemo"

        echo "Running: [$coeffs]" >> "$logfile_meld"

        python train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --out_path "experience_results/0630_selfdis_coeff_ex/coeff_v/comb_${coeffs}/IEMO" --dist_coefficients "[$coeffs]" >> "$logfile_iemo"

        python train.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --out_path "experience_results/0630_selfdis_coeff_ex/coeff_v/comb_${coeffs}/MELD" --dist_coefficients "[$coeffs]" >> "$logfile_meld"

        echo "" >> "$logfile_iemo"

        echo "" >> "$logfile_meld"

    done

done