export CUDA_VISIBLE_DEVICES=4
python test.py --gpus 1 --model Mamba --name Mamba-SRx8 \
    --load_size 128 --dataset_name single --dataroot /data/caojianan/Helen/LR \
    --pretrain_model_path ./ckpt/latest_demo_model.pt \
    --save_as_dir result/Mamba-helen

export CUDA_VISIBLE_DEVICES=4
python test.py --gpus 1 --model Mamba --name Mamba-SRx8 \
   --load_size 128 --dataset_name single --dataroot /data/caojianan/CelebA/celeba_test/LR \
   --pretrain_model_path ./ckpt/latest_demo_model.pt \
   --save_as_dir result/Mamba-celeba

