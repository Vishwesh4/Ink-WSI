# from math import frexp
# import os
# import sys
# sys.path.append("/home/ramanav/projects/rrg-amartel/ramanav/Projects/InkFilter")

# from pathlib import Path
# import torch
# import random
# import numpy as np
# from tqdm import tqdm

# from modules.train_filter import *
# import trainer

# random_seed = 2022
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)
# random.seed(random_seed)

# dataset = trainer.Dataset.create("ink",
#                                  path="/localscratch/ramanav.39148973.0/SSL_training",
#                                  test_batch_size=64,
#                                  train_batch_size=64,
#                                  template_pth="/localscratch/ramanav.39148973.0/by_class",
#                                  tile_h=256,
#                                  tile_w=256,
#                                  tile_stride_factor_h=4,
#                                  tile_stride_factor_w=4,
#                                  colors=[("black","#28282B"),("#002d04","#2a7e19"),("#000133","skyblue"),("#1f0954","#6d5caf"),("#a90308","#ff000d")],
#                                  train_split=0.85,
#                                  n_template=10000
#                                 )

# for i in range(100):
#     print(i)
#     for data in dataset.trainloader:
#         pass
#     # for data in dataset.testloader:
#     #     pass
# print("Something")


import os
import sys
sys.path.append("/home/ramanav/Projects/Ink-WSI")
from pathlib import Path

import pandas as pd
import numpy as np
import torchvision
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import torchmetrics

from modules.patch_extraction import SedeenAnnotationParser, ExtractAnnotations
from modules.deploy import Pairwise_ExtractAnnot
from modules.train_filter import InkGenerator, Handwritten
import trainer
from modules import train_filter
from modules.metrics import ssim, psnr, mse, pbvif


df = pd.read_excel("~/Downloads/pairs.ods")
ink_slide_path = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/"
clean_path = "/labs3/amartel_data3/histology/Data/DCIS_cohort/PRECISE_NoRT/"


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

pair_list = [(str( Path(clean_path) / (str(df["Clean Slides"][i])+".svs" ) ),str( Path(ink_slide_path) / (str(df["Ink Slides"][i])+".svs" ) ))
                for i in range(len(df))]
# print(pair_list)
# print(pair_list[5])

pairdataset = Pairwise_ExtractAnnot(pair_pths=pair_list[-2:],
                                annotation_dir=annotation_dir,
                                renamed_label=ink_labelset,
                                tile_h=TILE_H,
                                tile_w=TILE_W,
                                tile_stride_factor_h=TILE_STRIDE_FACTOR_H, 
                                tile_stride_factor_w=TILE_STRIDE_FACTOR_W, 
                                lwst_level_idx=LWST_LEVEL_IDX, 
                                mode="train", 
                                train_split=1, 
                                transform=TRANSFORM,
                                sample_threshold=30)