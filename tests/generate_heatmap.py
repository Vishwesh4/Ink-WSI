from collections import OrderedDict
import torch
import torchvision
import torch.nn as nn
import os
import glob
import sys
sys.path.append("/amartel_data4/temp/lukasz_test/blur/")
sys.path.append("/home/ramanav/Projects/Ink-WSI")

import pandas as pd
from pathlib import Path
#import dptools.config.slides
#dptools.config.slides.slidelibrary = 'tiffslide'

import trainer
from modules import train_filter

from dptools.slides.processing.wsipredictions import WSIPredictions
from dptools.slides.processing.wsimask import WSIMask
from dptools.slides.processing.wsiheatmap import WSIHeatmap
from dptools.slides.utils.wsi import get_wsi_id

# MODEL_PATH = '/home/ramanav/Projects/Ink-WSI/Results/filter/Checkpoint_28Jul12_19_55_1.00.pt'
MODEL_PATH = '/home/ramanav/Projects/Ink-WSI/Results/filter/Checkpoint_27Jul18_05_09_1.00.pt'
INPUT_DIR = '/amartel_data4/Vishwesh_Generated_Heatmap/'
OUTPUT_DIR = '/home/ramanav/Projects/Ink-WSI/Results/heatmaps'

df = pd.read_excel("~/Downloads/pairs.ods")
ink_slide_path = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/"
clean_path = "/labs3/amartel_data3/histology/Data/DCIS_cohort/PRECISE_NoRT/"

def get_model():
    model = trainer.Model.create("ink")
    model.load_model_weights(MODEL_PATH,torch.device("cpu"))
    return model

def get_classifier():
    return classifier_function_blur

def classifier_function_clear(result):
    probabilities = torch.nn.functional.softmax(result, dim=1)
    output = probabilities[:,0]
    return output

def classifier_function_blur(result):
    probabilities = torch.nn.functional.softmax(result, dim=1)
    output = probabilities[:,-1]
    return output

def find_files_by_types(file_types, input_dir):
    """ Given a list of file types/extensions, find files in directory """
    files_found = []
    for file_type in file_types:
        recursive = '**'
        extension = '*.' + file_type.strip()
        scan_pattern = os.path.join(input_dir, recursive, extension)
        files_found.extend(sorted(glob.glob(scan_pattern, recursive=True)))
    return files_found

model = get_model()
classifier = get_classifier()
# files = find_files_by_types(['svs'], INPUT_DIR)
files =  [str( Path(ink_slide_path) / (str(df["Ink Slides"][i])+".svs" ) ) for i in range(len(df))]
predictions = WSIPredictions(model, classifier, patch_size=256, patch_stride=1, num_workers=4, batch_size=128, foreground_threshold=0.95, tensor_dataset=True)
files = ["/home/ramanav/Downloads/02-0484.svs"]
print(files)
for f in files:
    wsi_id = get_wsi_id(f)
    wsi_file = f
    mask_file = os.path.join(OUTPUT_DIR, wsi_id + '_mask_array.npy')
    mask = WSIMask(wsi_file, min_size=800, mode='lab', threshold=0.1)
    mask.save_array(mask_file)
    mask.save_png(os.path.join(OUTPUT_DIR, wsi_id + '_mask_image.png'))

    predictions.process_wsi(wsi_file, mask_file)
    predictions_file = os.path.join(OUTPUT_DIR, wsi_id + '_predictions_array.npy')
    predictions.save_array(predictions_file)
    predictions.save_png(os.path.join(OUTPUT_DIR, wsi_id + '_predictions_image.png'))
    print("Data size =", predictions.datasize)

    heatmap = WSIHeatmap(predictions_file, normalize=False)
    heatmap.save_png(os.path.join(OUTPUT_DIR, wsi_id + '_heatmap.png'))
    heatmap.save_bar_png(os.path.join(OUTPUT_DIR, wsi_id + '_heatmap_bar.png'))