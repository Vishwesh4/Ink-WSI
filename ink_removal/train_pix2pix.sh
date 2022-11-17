#!/bin/bash

# name="norm_mixed"
# checkpoints_dir_path="/localdisk3/ramanav/Results/Pix2Pix/"

#  nohup python train.py \
#  --name $name \
#  --direction AtoB \
#  --use_wandb \
#  --dataset_mode mixed \
#  --gpu_ids 0 \
#  --checkpoints_dir $checkpoints_dir_path \
#  --model pix2pix \
#  --batch_size 2 \
#  --do_norm \
#  --load_size 256 \
#  --preprocess none > ./Results/log__$name.out &

name="norm_v2_nobrown"
checkpoints_dir_path="/localdisk3/ramanav/Results/Pix2Pix/"

 nohup python train.py \
 --name $name \
 --direction AtoB \
 --use_wandb \
 --dataset_mode mixed \
 --gpu_ids 0 \
 --checkpoints_dir $checkpoints_dir_path \
 --model pix2pix \
 --batch_size 2 \
 --do_norm \
 --load_size 256 \
 --preprocess none > ./Results/log_$name.out &


