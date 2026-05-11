#!/bin/bash

# 出力ディレクトリ
outdir="log"
mkdir -p "$outdir"

logfile="$outdir/run_mctn_parallel.log"
echo "-- run start --" > "$logfile"

# モダリティ順序（6通り）
modal_seqs=(
  "T"
  "A"
  "V"
)

# hidden_dim候補
dims=(1024 512 256 128 64 32)

# ループ
for seq in "${modal_seqs[@]}"
do
    for dim in "${dims[@]}"
    do
        echo "Running: seq=${seq}, dim=${dim}"

        python train_mctn.py \
            --epochs 200 \
            --batch-size 32 \
            --modal_seq "$seq" \
            --hidden_dim "$dim" \
            --model_type parallel \
            --out_path "experiment_results/0510_parallel_mctn/${seq}/dim_${dim}" \
            >> "$logfile" 2>&1

    done
done