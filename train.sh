export CUDA_VISIBLE_DEVICES=0,1,2
nohup python train.py --gpus 3 --name Mamba-SRx8 --model Mamba \
    --Gnorm "bn" --lr 0.0004 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot /data/caojianan/CelebA/celeba_train --dataset_name celeba \
    --batch_size 32 --total_epochs 100 --visual_freq 100 --print_freq 10 \
    --save_latest_freq 500 &
    # --continue_train