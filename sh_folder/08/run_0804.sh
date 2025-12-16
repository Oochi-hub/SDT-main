#動的な係導入
#IEMOCAPにおいて，学習減衰と組み合わせ

##変更タイミングの回数による結果の違い
outdir="log/0804"

mkdir -p "$outdir"

logfile_iemo="$outdir/run_0804.log"
#logfile_meld="$outdir/run_0728_meld.log"
echo "==== 実行ログ ====" >> "$logfile_iemo"
for change_epoch in 30 40 50;do
    for add_kl in 0.50;do
        for schedular in 0.990;do

            python train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' \
            --change_epoch $change_epoch --add_kl $add_kl --schedular $schedular\
            --out_path "experience_results/for_visu/0804_iemocap_decay/decay_$schedular/change_epoch_$change_epoch" >> "$logfile_iemo"

            echo "" >> "$logfile_iemo"
        done
    done
done

