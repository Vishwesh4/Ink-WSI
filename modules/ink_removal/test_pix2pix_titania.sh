#!/bin/bash

# name="tiger_pix2pix_domaindata_alone"
# name="tiger_pix2pix_domaindata_alone_norm"
# name="tiger_pix2pix_norm"
# name="tiger_norm_domain_transfer"
name="norm_mixed"



# nohup python test_ink.py \
#  --dataroot "/labs3/amartel_data3/tiger_dataset/SSL_training/" \
#  --model pix2pix \
#  --checkpoints_dir "/localdisk3/ramanav/Results/Pix2Pix/" \
#  --results_dir "/localdisk3/ramanav/Results/Pix2Pix/" \
#  --gpu_ids 0 \
#  --dataset_mode pairink \
#  --direction AtoB \
#  --name $name > ./Results/logtest_$name.out &

# nohup python test_ink.py \
#  --dataroot "/labs3/amartel_data3/tiger_dataset/SSL_training/" \
#  --model pix2pix \
#  --checkpoints_dir "/localdisk3/ramanav/Results/Pix2Pix/" \
#  --results_dir "/localdisk3/ramanav/Results/Pix2Pix/" \
#  --gpu_ids 1 \
#  --dataset_mode pairink \
#  --direction AtoB \
#  --stride_h 1 \
#  --stride_w 1 \
#  --load_size 256 \
#  --preprocess none \
#  --do_norm \
#  --eval \
#  --num_test 25000 \
#  --name $name > ./Results/logtest_$name.out &

nohup python /home/ramanav/Projects/Ink-WSI/modules/ink_removal/test_ink.py \
 --dataroot "/labs3/amartel_data3/tiger_dataset/SSL_training/" \
 --model pix2pix \
 --checkpoints_dir "/localdisk3/ramanav/Results/Pix2Pix/" \
 --results_dir "/localdisk3/ramanav/Results/Pix2Pix/" \
 --gpu_ids 0 \
 --dataset_mode pairink \
 --direction AtoB \
 --stride_h 1 \
 --stride_w 1 \
 --load_size 256 \
 --preprocess none \
 --do_norm \
 --eval \
 --num_test 500 \
 --name $name > /home/ramanav/Projects/Ink-WSI/modules/ink_removal/Results/logtest_$name.out &