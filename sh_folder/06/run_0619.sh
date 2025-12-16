#!/bin/bash
#6/19~ 自己蒸留損失項のモダリティごとの係数調整実験用スクリプト

# 任意の値（1.00と切り替える値）を指定
val=1.20
outdir="log"

# 出力ディレクトリ作成
mkdir -p "$outdir"
logfile_iemo="$outdir/run_0620_selfdis_coeff_ex_coeff_1.2_iemo.log"
logfile_meld="$outdir/run_0620_selfdis_coeff_ex_coeff_1.2_meld.log"

#echo "==== 実行ログ ====" > "$logfile_iemo"
echo "==== 実行ログ ====" > "$logfile_meld"

for i in {1..63}; do
    binary=$(printf "%06d" "$(bc <<< "obase=2;$i")")
    coeffs=$(echo "$binary" | sed "s/./& /g" | awk -v v=$val '{for(i=1;i<=NF;i++) printf($i=="1"?"1.00":v)(i<NF?",":"");}')

    # すべて1.00かどうか判定
    if [ "$coeffs" == "1.00,1.00,1.00,1.00,1.00,1.00" ]; then
    coeffs=$(printf "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f" $val $val $val $val $val $val)
    fi

    #echo "Running: [$coeffs]" >> "$logfile_iemo"
    echo "Running: [$coeffs]" >> "$logfile_meld"
    # echo "hoge/comb_${coeffs}/IEMO" >> "$logfile"
    #python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --out_path "experience_results/0619_selfdis_coeff_ex/coeff_1p2/comb_${coeffs}/IEMO" --dist_coefficients "[$coeffs]" >> "$logfile_iemo"
    python train.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --out_path "experience_results/0619_selfdis_coeff_ex/coeff_1p2/comb_${coeffs}/MELD" --dist_coefficients "[$coeffs]" >> "$logfile_meld"
    #echo "" >> "$logfile_iemo"
    echo "" >> "$logfile_meld"

done

# val2=1.50

# logfile_iemo2="$outdir/run_0620_selfdis_coeff_ex_coeff_1.5_iemo.log"
# logfile_meld2="$outdir/run_0620_selfdis_coeff_ex_coeff_1.5_meld.log"

# echo "==== 実行ログ ====" > "$logfile_iemo2"
# echo "==== 実行ログ ====" > "$logfile_meld2"

# for i in {1..63}; do
#     binary=$(printf "%06d" "$(bc <<< "obase=2;$i")")
#     coeffs=$(echo "$binary" | sed "s/./& /g" | awk -v v=$val2 '{for(i=1;i<=NF;i++) printf($i=="1"?"1.00":v)(i<NF?",":"");}')

#     # すべて1.00かどうか判定
#     if [ "$coeffs" == "1.00,1.00,1.00,1.00,1.00,1.00" ]; then
#     coeffs=$(printf "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f" $val2 $val2 $val2 $val2 $val2 $val2)
#     fi

#     echo "Running: [$coeffs]" >> "$logfile_iemo2"
#     echo "Running: [$coeffs]" >> "$logfile_meld2"
#     # echo "hoge/comb_${coeffs}/IEMO" >> "$logfile"
#     python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --out_path "experience_results/0619_selfdis_coeff_ex/coeff_1p5/comb_${coeffs}/IEMO" --dist_coefficients "[$coeffs]" >> "$logfile_iemo2"
#     python train.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --out_path "experience_results/0619_selfdis_coeff_ex/coeff_1p5/comb_${coeffs}/MELD" --dist_coefficients "[$coeffs]" >> "$logfile_meld2"
#     echo "" >> "$logfile_iemo2"
#     echo "" >> "$logfile_meld2"

# done

##demo##
# coeffs=$(printf "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f" $val $val $val $val $val $val)
# python train.py --epoch 2 --dist_coefficients "[$coeffs]"