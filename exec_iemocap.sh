python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "t" --out_path "experience_results/0930_train_for_autoencoder/text/"
python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "a" --out_path "experience_results/0930_train_for_autoencoder/audio/"
python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "v" --out_path "experience_results/0930_train_for_autoencoder/visual/"


# for iter in 9
# do
# python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "t" --valid_num $iter --out_path "experience_results/0926_cross_valid/valid$iter/text/IEMOCAP/"
# # python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "a" --valid_num $iter --out_path "experience_results/0926_cross_valid/valid$iter/audio/IEMOCAP/"
# # python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "v" --valid_num $iter --out_path "experience_results/0926_cross_valid/valid$iter/visual/IEMOCAP/"
# done

# python -u test.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "t" --weight "experience_results/0925_evaldata_training_ex/text/IEMOCAP/weights/model_weights_best.pth" --out_path "experience_results/0925_evaldata_training_ex/valid_test/text/IEMOCAP/"
# python -u test.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "a" --weight "experience_results/0925_evaldata_training_ex/audio/IEMOCAP/weights/model_weights_best.pth" --out_path "experience_results/0925_evaldata_training_ex/valid_test/audio/IEMOCAP/"
# python -u test.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "v" --weight "experience_results/0925_evaldata_training_ex/visual/IEMOCAP/weights/model_weights_best.pth" --out_path "experience_results/0925_evaldata_training_ex/valid_test/visual/IEMOCAP/"


# python -u test.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "t" --weight "experience_results/0916_ex/text/IEMOCAP/weights/model_weights_best.pth" --out_path "experience_results/0925_ex/text/IEMOCAP/"
# python -u test.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "a" --weight "experience_results/0916_ex/audio/IEMOCAP/weights/model_weights_best.pth" --out_path "experience_results/0925_ex/audio/IEMOCAP/"
# python -u test.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "v" --weight "experience_results/0916_ex/visual/IEMOCAP/weights/model_weights_best.pth" --out_path "experience_results/0925_ex/visual/IEMOCAP/"

# python -u test.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "t" --weight "experience_results/0917_ex/text_only/IEMOCAP/weights/model_weights_best.pth" --out_path "experience_results/0925_ex/text_only/IEMOCAP/"
# python -u test.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "a" --weight "experience_results/0917_ex/audio_only/IEMOCAP/weights/model_weights_best.pth" --out_path "experience_results/0925_ex/audio_only/IEMOCAP/"
# python -u test.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "v" --weight "experience_results/0917_ex/visual_only/IEMOCAP/weights/model_weights_best.pth" --out_path "experience_results/0925_ex/visual_only/IEMOCAP/"

