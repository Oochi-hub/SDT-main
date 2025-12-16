#!/bin/bash
declare -A dict

dict=(
    ["text"]="t" 
    ["audio"]="a"
    ["visual"]="v" 
)

for modal in text audio visual; do
    python test_for_autoencoder.py --learned_model experience_results/0929_train_for_autoencoder/$modal/weights/model_weights_best.pth --model ${dict[$modal]}\
    --temp 1 --weight "experience_results/demo_1205/text/weights/model_weights_last.pth" --out_path "experience_results/1205_demo_a/text_ae/$modal"
    # python test_for_autoencoder.py --learned_model experience_results/0929_train_for_autoencoder/$modal/weights/model_weights_best.pth --model ${dict[$modal]}\
    # --temp 1 --weight "experience_results/1009_autoencoder/text/weights/model_weights_last.pth" --out_path "experience_results/1205_demo/text_ae/$modal"
    # python test_for_autoencoder.py --learned_model experience_results/0929_train_for_autoencoder/$modal/weights/model_weights_best.pth --model ${dict[$modal]}\
    # --temp 1 --weight "experience_results/1009_autoencoder/audio/weights/model_weights_last.pth" --out_path "experience_results/1205_demo/audio_ae/$modal"
    # python test_for_autoencoder.py --learned_model experience_results/0929_train_for_autoencoder/$modal/weights/model_weights_best.pth --model ${dict[$modal]}\
    # --temp 1 --weight "experience_results/1009_autoencoder/visual/weights/model_weights_last.pth" --out_path "experience_results/1205_demo/visual_ae/$modal"
done
