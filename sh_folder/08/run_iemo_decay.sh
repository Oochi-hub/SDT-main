# echo =======================
for lr in 0.00001 0.000001 0.0000001; do
    python train.py --lr $lr --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --out_path "experience_results/0717_iemo_decay/lr_$lr/ori"
done

for hoge in 0.990 0.970; do
    for lr in 0.00001 0.000001 0.0000001; do
        python train.py --lr $lr --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --schedular $hoge --out_path "experience_results/0717_iemo_decay/lr_$lr/decay_$hoge"
    done   
done

