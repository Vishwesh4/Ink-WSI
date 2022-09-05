#!/bin/bash

name="norm_mixed"
version="191534" #The version denotes the directory where you want to load calculated results from, the directory are named as {versionid}_test_latest
results_dir_path="/localdisk3/ramanav/Results/Pix2Pix/"

nohup python /home/ramanav/Projects/Ink-WSI/ink_removal/calc_metrics.py \
 --model pix2pix \
 --dataset_mode pairink \
 --version $version \
 --results_dir $results_dir_path \
 --name $name > /home/ramanav/Projects/Ink-WSI/ink_removal/Results/logmetric_$name.out &