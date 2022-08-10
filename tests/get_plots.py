import os
import sys
sys.path.append("/home/ramanav/Projects/Ink-WSI")
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import torchvision

from modules.patch_extraction import ExtractAnnotations
from modules.train_filter import InkGenerator, Handwritten


def save(name,image):
    cv2.imwrite("/home/ramanav/Projects/Ink-WSI/images/"+name,cv2.cvtColor(image,cv2.COLOR_RGB2BGR))


df = pd.read_excel("~/Downloads/pairs.ods")
ink_slide_path = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/"
clean_path = "/labs3/amartel_data3/histology/Data/DCIS_cohort/PRECISE_NoRT/"

#Compare with single patch extraction annotation
ink_slide = str( Path(ink_slide_path) / (str(df["Ink Slides"][0])+".svs" ) )
clean_slide = str( Path(clean_path) / (str(df["Clean Slides"][0])+".svs" ) )
annotation_dir = str( Path(ink_slide_path) / Path("sedeen") )

TILE_H = 256
TILE_W = 256
TILE_STRIDE_FACTOR_H = 1
TILE_STRIDE_FACTOR_W = 1
LWST_LEVEL_IDX = 0
TRANSFORM = torchvision.transforms.ToTensor()
# SPACING = 0.2526
SPACING = None

ink_labelset = {"clean":"#00ff00ff","ink":"#ff0000ff"}


singledataset = ExtractAnnotations(
        image_pth=ink_slide,
        annotation_dir=annotation_dir,
        renamed_label=ink_labelset,
        tile_h=TILE_H,
        tile_w=TILE_W,
        tile_stride_factor_h=TILE_STRIDE_FACTOR_H,
        tile_stride_factor_w=TILE_STRIDE_FACTOR_W,
        spacing=None,
        lwst_level_idx=LWST_LEVEL_IDX,
        mode="train",
        train_split=1,
        transform=TRANSFORM,
        threshold=0.7,
        sample_threshold=30
        )


template = Handwritten(path="/localdisk3/ramanav/backup_codes/Ink_project/Projects/Dataset/by_class/",n=500)

colors = [["black","#28282B"],["#002d04","#2a7e19"],["#000133","skyblue"],["#1f0954","#6d5caf"],["#a90308","#ff000d"],["#005558","#90DCD5"],["#001769","#005CC9"],["#3C1C16","#A05745"]]

ink_generator = InkGenerator(ink_template=template,colors=colors)

img = singledataset.all_image_tiles_hr[485]

for i in range(100):
    crop,color_matrix2,noise_img,mask,flag,alpha = ink_generator.generate(img)