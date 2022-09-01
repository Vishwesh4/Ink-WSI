#!/bin/bash

# name="tiger_pix2pix_domaindata_alone"
# name="tiger_pix2pix_domaindata_alone_norm"
# name="tiger_pix2pix_norm"
# name="tiger_norm_domain_transfer"
name="norm_mixed"


# nohup python train.py \
#  --name $name \
#  --direction AtoB \
#  --use_wandb \
#  --dataset_mode dcisink \
#  --gpu_ids 0 \
#  --checkpoints_dir "/localdisk3/ramanav/Results/Pix2Pix" \
#  --model pix2pix \
#  --dataroot "/labs3/amartel_data3/tiger_dataset/SSL_training/" \
#  --stride_h 4 \
#  --stride_w 4 \
#  --batch_size 2 > ./Results/log_$name.out &

# nohup python train.py \
#  --name $name \
#  --direction AtoB \
#  --use_wandb \
#  --dataset_mode dcisink \
#  --gpu_ids 0 \
#  --checkpoints_dir "/localdisk3/ramanav/Results/Pix2Pix" \
#  --model pix2pix \
#  --dataroot "/labs3/amartel_data3/tiger_dataset/SSL_training/" \
#  --stride_h 4 \
#  --stride_w 4 \
#  --batch_size 2 \
#  --continue_train \
#  --epoch_count 18 > ./Results/log_$name.out &

#  nohup python train.py \
#  --name $name \
#  --direction AtoB \
#  --use_wandb \
#  --dataset_mode dcisink \
#  --gpu_ids 0 \
#  --checkpoints_dir "/localdisk3/ramanav/Results/Pix2Pix" \
#  --model pix2pix \
#  --dataroot "/labs3/amartel_data3/tiger_dataset/SSL_training/" \
#  --stride_h 4 \
#  --stride_w 4 \
#  --batch_size 2 \
#  --do_norm \
#  --load_size 256 \
#  --preprocess none > ./Results/log_$name.out &

# nohup python train.py \
# --name $name \
# --direction AtoB \
# --use_wandb \
# --dataset_mode tigerink \
# --gpu_ids 0 \
# --checkpoints_dir "/localdisk3/ramanav/Results/Pix2Pix" \
# --model pix2pix \
# --dataroot "/labs3/amartel_data3/tiger_dataset/SSL_training/" \
# --stride_h 8.5 \
# --stride_w 8.5 \
# --batch_size 2 \
# --do_norm \
# --load_size 256 \
# --preprocess none \
# --continue_train \
# --epoch_count 19  > ./Results/log_$name.out &

#  nohup python train.py \
#  --name $name \
#  --direction AtoB \
#  --use_wandb \
#  --dataset_mode dcisink \
#  --gpu_ids 0 \
#  --checkpoints_dir "/localdisk3/ramanav/Results/Pix2Pix" \
#  --model pix2pix \
#  --dataroot "/labs3/amartel_data3/tiger_dataset/SSL_training/" \
#  --stride_h 4 \
#  --stride_w 4 \
#  --batch_size 2 \
#  --do_norm \
#  --load_size 256 \
#  --preprocess none \
#  --continue_train  > ./Results/log_cont_$name.out &

#  nohup python train.py \
#  --name $name \
#  --direction AtoB \
#  --use_wandb \
#  --dataset_mode mixed \
#  --gpu_ids 0 \
#  --checkpoints_dir "/localdisk3/ramanav/Results/Pix2Pix" \
#  --model pix2pix \
#  --dataroot "/labs3/amartel_data3/tiger_dataset/SSL_training/" \
#  --batch_size 2 \
#  --do_norm \
#  --load_size 256 \
#  --preprocess none > ./Results/log_$name.out &

 nohup python train.py \
 --name $name \
 --direction AtoB \
 --use_wandb \
 --dataset_mode mixed \
 --gpu_ids 0 \
 --checkpoints_dir "/localdisk3/ramanav/Results/Pix2Pix" \
 --model pix2pix \
 --dataroot "/labs3/amartel_data3/tiger_dataset/SSL_training/" \
 --batch_size 2 \
 --do_norm \
 --load_size 256 \
 --preprocess none \
 --continue_train \
 --epoch_count 13 > ./Results/log_cont_$name.out &


