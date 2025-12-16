# echo =======================
for lr in 0.0000005 0.00000005 0.000003 0.000001; do
    python train.py --lr $lr --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --out_path "experience_results/0717_MELD_decay/lr_$lr/ori"
done

for hoge in 0.990 0.970; do
    for lr in 0.0000005 0.00000005 0.000003 0.000001; do
        python train.py --lr $lr --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --schedular $hoge --out_path "experience_results/0717_MELD_decay/lr_$lr/decay_$hoge"
    done   
done

