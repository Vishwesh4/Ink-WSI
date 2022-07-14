import os
import sys
sys.path.append("/home/ramanav/Projects/Ink-WSI")

import torch
import numpy as np
from pathlib import Path
import torchvision

from modules.deploy import Pairwise_ExtractPatches

# INPUT_FILE = "/labs3/amartel_data3/histology/Data/DCIS_cohort/PRECISE_NoRT/114793.svs"
# INPUT_FILE = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/121504.svs"
# INPUT_FILE = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/121504.svs"
DEST_FILE = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/121504.svs"
SRC_FILE = "/labs3/amartel_data3/histology/Data/DCIS_cohort/PRECISE_NoRT/114793.svs"

# OUTPUT_DIR = "/home/ramanav/Projects/Ink-WSI/tests/Results/Ink_121393"

# if not Path(OUTPUT_DIR).exists():
#     os.mkdir(OUTPUT_DIR)

TILE_H = 256
TILE_W = 256
TILE_STRIDE_FACTOR_H = 1
TILE_STRIDE_FACTOR_W = 1
LWST_LEVEL_IDX = 0
TRANSFORM = torchvision.transforms.ToTensor()
# SPACING = 0.2526
SPACING = None

dataset = Pairwise_ExtractPatches(
        pair_pths=(SRC_FILE,DEST_FILE),
        tile_h=TILE_H,
        tile_w=TILE_W,
        tile_stride_factor_h=TILE_STRIDE_FACTOR_H,
        tile_stride_factor_w=TILE_STRIDE_FACTOR_W,
        spacing=None,
        lwst_level_idx=LWST_LEVEL_IDX,
        mode="train",
        train_split=1,
        transform=TRANSFORM,
        threshold=0.7
)

print(len(dataset))
