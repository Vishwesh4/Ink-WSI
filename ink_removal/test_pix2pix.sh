#!/bin/bash

name="norm_mixed"
results_dir_path=""
checkpoints_dir_path=""
pair_csv_pth=""
ink_slide_path=""
clean_path=""

nohup python test_ink.py \
 --model pix2pix \
 --checkpoints_dir $checkpoints_dir_path \
 --results_dir $results_dir_path \
 --gpu_ids 0 \
 --dataset_mode pairink \
 --direction AtoB \
 --stride_h 1 \
 --stride_w 1 \
 --pair_csv $pair_csv_pth \
 --ink_slide_pth $ink_slide_path \
 --clean_slide_pth $clean_path \
 --load_size 256 \
 --preprocess none \
 --do_norm \
 --eval \
 --num_test 30000 \
 --name $name > ./Results/logtest_downstream_$name.out &