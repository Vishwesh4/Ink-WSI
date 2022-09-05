#!/bin/bash

name="norm_mixed"
results_dir_path="/localdisk3/ramanav/Results/Pix2Pix/"
checkpoints_dir_path="/localdisk3/ramanav/Results/Pix2Pix/"

nohup python /home/ramanav/Projects/Ink-WSI/ink_removal/test_ink.py \
 --model pix2pix \
 --checkpoints_dir $checkpoints_dir_path \
 --results_dir $results_dir_path \
 --gpu_ids 0 \
 --dataset_mode pairink \
 --direction AtoB \
 --stride_h 1 \
 --stride_w 1 \
 --load_size 256 \
 --preprocess none \
 --do_norm \
 --eval \
 --num_test 30000 \
 --name $name > /home/ramanav/Projects/Ink-WSI/ink_removal/Results/logtest_full_$name.out &