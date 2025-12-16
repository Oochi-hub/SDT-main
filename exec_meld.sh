# echo =======================
# for iter in 1 2 3 4 5 6 7 8 9 10
# do
#     echo --- $iter ---
# for hoge in 0.990 0.970; do
#     python train.py --lr 0.000005 --batch-size 8 --epochs 100 --temp 8 --Dataset 'MELD' --flag_2 True --flag_3 True --schedular $hoge \
#     --out_path "experience_results/0714_reduce_2/redu_$hoge/MELD/gannma_2_3"

#     python train.py --lr 0.000005 --batch-size 8 --epochs 100 --temp 8 --Dataset 'MELD' --flag_2 True --schedular $hoge \
#     --out_path "experience_results/0714_reduce_2/redu_$hoge/MELD/gannma_2"

#     python train.py --lr 0.000005 --batch-size 8 --epochs 100 --temp 8 --Dataset 'MELD' --flag_3 True --schedular $hoge \
#     --out_path "experience_results/0714_reduce_2/redu_$hoge/MELD/gannma_3"
# done
# done > sdt_meld.txt 2>&1 &

# echo =======================
# echo --- demo ---
# python -u train.py --lr 0.000005 --batch-size 8 --epochs 50 --temp 8 --Dataset 'MELD' --out_path "MELD_test"

python train.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --model "t" --out_path "experience_results/0926_ex_3/text/MELD"
python train.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --model "a" --out_path "experience_results/0926_ex_3/audio/MELD"
python train.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --model "v" --out_path "experience_results/0926_ex_3/visual/MELD"

# python test.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --model "t" --weight "experience_results/0925_evaldata_training_ex/text/MELD/weights/model_weights_best.pth" --out_path "experience_results/0925_evaldata_training_ex/valid_test/text/MELD"
# python test.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --model "a" --weight "experience_results/0925_evaldata_training_ex/audio/MELD/weights/model_weights_best.pth" --out_path "experience_results/0925_evaldata_training_ex/valid_test/audio/MELD"
# python test.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --model "v" --weight "experience_results/0925_evaldata_training_ex/visual/MELD/weights/model_weights_best.pth" --out_path "experience_results/0925_evaldata_training_ex/valid_test/visual/MELD"

# python test.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --model "t" --weight "experience_results/0916_ex/text/MELD/weights/model_weights_best.pth" --out_path "experience_results/0925_ex/text/MELD"
# python test.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --model "a" --weight "experience_results/0916_ex/audio/MELD/weights/model_weights_best.pth" --out_path "experience_results/0925_ex/audio/MELD"
# python test.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --model "v" --weight "experience_results/0916_ex/visual/MELD/weights/model_weights_best.pth" --out_path "experience_results/0925_ex/visual/MELD"

# python test.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --model "t" --weight "experience_results/0917_ex/text_only/MELD/weights/model_weights_best.pth" --out_path "experience_results/0925_ex_only/text/MELD"
# python test.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --model "a" --weight "experience_results/0917_ex/audio_only/MELD/weights/model_weights_best.pth" --out_path "experience_results/0925_ex_only/audio/MELD"
# python test.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD' --model "v" --weight "experience_results/0917_ex/visual_only/MELD/weights/model_weights_best.pth" --out_path "experience_results/0925_ex_only/visual/MELD"

# python train.py --lr 0.000005 --batch-size 8 --epochs 200 --temp 8 --Dataset 'MELD_c' --out_path "experience_results/0826_ex/MELD_characters"