# source exec_iemocap.sh
# source exec_meld.sh
# for iter in 0 1 2 3 4
# do
# python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "t" --valid_num $iter --schedular 0.99 --out_path "experience_results/0927_cross_valid_sche/epoch200/valid$iter/text/"
# python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "a" --valid_num $iter --schedular 0.99 --out_path "experience_results/0927_cross_valid_sche/epoch200/valid$iter/audio/"
# python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "v" --valid_num $iter --schedular 0.99 --out_path "experience_results/0927_cross_valid_sche/epoch200/valid$iter/visual/"
# done

# for iter in 0 1 2 3 4
# do
# python -u train.py --lr 0.0001 --batch-size 16 --epochs 100 --temp 1 --Dataset 'IEMOCAP' --model "t" --valid_num $iter --out_path "experience_results/0927_cross_valid/epoch100/valid$iter/text/IEMOCAP/"
# python -u train.py --lr 0.0001 --batch-size 16 --epochs 100 --temp 1 --Dataset 'IEMOCAP' --model "a" --valid_num $iter --out_path "experience_results/0927_cross_valid/epoch100/valid$iter/audio/IEMOCAP/"
# python -u train.py --lr 0.0001 --batch-size 16 --epochs 100 --temp 1 --Dataset 'IEMOCAP' --model "v" --valid_num $iter --out_path "experience_results/0927_cross_valid/epoch100/valid$iter/visual/IEMOCAP/"
# done

# python train_for_autoencoder.py --learned_model experience_results/1205_train_for_autoencoder/text/weights/model_weights_best.pth --model "t" --temp 1 --epochs 100 --out_path "experience_results/1211_autoencoder/text/"
# python train_for_autoencoder.py --learned_model experience_results/1205_train_for_autoencoder/audio/weights/model_weights_best.pth --model "a" --temp 1 --epochs 100 --out_path "experience_results/1211_autoencoder/audio/"
# python train_for_autoencoder.py --learned_model experience_results/1205_train_for_autoencoder/visual/weights/model_weights_best.pth --model "v" --temp 1 --epochs 100 --out_path "experience_results/1211_autoencoder/visual/"

python train_for_autoencoder.py --learned_model experience_results/0929_train_for_autoencoder/text/weights/model_weights_best.pth --model "t" --temp 1 --epochs 100 --out_path "experience_results/1211_autoencoder_2/text/"
python train_for_autoencoder.py --learned_model experience_results/0929_train_for_autoencoder/audio/weights/model_weights_best.pth --model "a" --temp 1 --epochs 100 --out_path "experience_results/1211_autoencoder_2/audio/"
python train_for_autoencoder.py --learned_model experience_results/0929_train_for_autoencoder/visual/weights/model_weights_best.pth --model "v" --temp 1 --epochs 100 --out_path "experience_results/1211_autoencoder_2/visual/"

# python train_for_autoencoder.py --learned_model experience_results/0929_train_for_autoencoder/text/weights/model_weights_best.pth --model "t" --temp 2 --epochs 100 --autoencoder not --out_path "experience_results/0929_autoencoder/not_fine_tune_last/temp2/100epoch/asymmetry/text/"
# python train_for_autoencoder.py --learned_model experience_results/0929_train_for_autoencoder/audio/weights/model_weights_best.pth --model "a" --temp 2 --epochs 100 --autoencoder not --out_path "experience_results/0929_autoencoder/not_fine_tune_last/temp2/100epoch/asymmetry/audio/"
# python train_for_autoencoder.py --learned_model experience_results/0929_train_for_autoencoder/visual/weights/model_weights_best.pth --model "v" --temp 2 --epochs 100 --autoencoder not --out_path "experience_results/0929_autoencoder/not_fine_tune_last/temp2/100epoch/asymmetry/visual/"


# python -u test.py --batch-size 16  --temp 1 --Dataset 'IEMOCAP' --model "t" --weight "experience_results/0929_train_for_autoencoder/text/weights/model_weights_best.pth" --out_path "experience_results/0930_evaldata_training_ex/model/text/"
# python -u test.py --batch-size 16  --temp 1 --Dataset 'IEMOCAP' --model "a" --weight "experience_results/0929_train_for_autoencoder/audio/weights/model_weights_best.pth" --out_path "experience_results/0930_evaldata_training_ex/model/audio/"
# python -u test.py --batch-size 16  --temp 1 --Dataset 'IEMOCAP' --model "v" --weight "experience_results/0929_train_for_autoencoder/visual/weights/model_weights_best.pth" --out_path "experience_results/0930_evaldata_training_ex/model/visual/"


# python test_for_autoencoder.py --learned_model experience_results/0929_train_for_autoencoder/text/weights/model_weights_best.pth --model "t" --temp 1 --weight "experience_results/1009_autoencoder/text/weights/model_weights_last.pth" --out_path "experience_results/for_newmodel_result/test/text/"
# python test_for_autoencoder.py --learned_model experience_results/0929_train_for_autoencoder/audio/weights/model_weights_best.pth --model "a" --temp 1 --weight "experience_results/1009_autoencoder/audio/weights/model_weights_last.pth" --out_path "experience_results/for_newmodel_result/test/audio/"
# python test_for_autoencoder.py --learned_model experience_results/0929_train_for_autoencoder/visual/weights/model_weights_best.pth --model "v" --temp 1 --weight "experience_results/1009_autoencoder/visual/weights/model_weights_last.pth" --out_path "experience_results/for_newmodel_result/test/visual/"

# python train_for_newmodel.py --epochs 100 --out_path "experience_results/1010_result/newmodel/lr_3" --lr 0.001
# python train_for_newmodel.py --epochs 100 --out_path "experience_results/1010_result/newmodel/lr_6" --lr 0.000001
# python train_for_newmodel.py --epochs 100 --out_path "experience_results/1010_result/newmodel/lr_7" --lr 0.0000001

# python -u test.py --batch-size 16  --temp 1 --Dataset 'IEMOCAP' --model "t" --weight "experience_results/0929_train_for_autoencoder/text/weights/model_weights_best.pth" --out_path "experience_results/1010_result/unimodal_model/text/"
# python -u test.py --batch-size 16  --temp 1 --Dataset 'IEMOCAP' --model "a" --weight "experience_results/0929_train_for_autoencoder/audio/weights/model_weights_best.pth" --out_path "experience_results/1010_result/unimodal_model/audio/"
# python -u test.py --batch-size 16  --temp 1 --Dataset 'IEMOCAP' --model "v" --weight "experience_results/0929_train_for_autoencoder/visual/weights/model_weights_best.pth" --out_path "experience_results/1010_result/unimodal_model/visual/"

# python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "t" --out_path "experience_results/1205_train_for_autoencoder/text/"
# python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "a" --out_path "experience_results/1205_train_for_autoencoder/audio/"
# python -u train.py --lr 0.0001 --batch-size 16 --epochs 200 --temp 1 --Dataset 'IEMOCAP' --model "v" --out_path "experience_results/1205_train_for_autoencoder/visual/"