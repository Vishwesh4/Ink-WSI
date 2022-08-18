import os

import torch
import numpy as np
from pathlib import Path
import torchvision

from .utils import Ink_deploy
from ..patch_extraction import ExtractPatches

# INPUT_FILE = "/labs3/amartel_data3/histology/Data/DCIS_cohort/PRECISE_NoRT/114793.svs"
# INPUT_FILE = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/121504.svs"
# INPUT_FILE = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/121504.svs"
INPUT_FILE = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/121393.svs"

OUTPUT_DIR = "/home/ramanav/Projects/Ink-WSI/tests/Results"

DEVICE = torch.device("cuda:3")
TILE_H = 256
TILE_W = 256
TILE_STRIDE_FACTOR_H = 1
TILE_STRIDE_FACTOR_W = 1
LWST_LEVEL_IDX = 0
TRANSFORM = torchvision.transforms.ToTensor()
SPACING = 0.456694
INK_PATH = "/localdisk3/ramanav/Results/Ink_WSI/Ink_filter/Checkpoint_12Jul12_46_16_1.00.pt"
PIX2PIX_NAME = "tiger_pix2pix_domaindata_alone_norm"

inkdeploy = Ink_deploy(model_path=INK_PATH,
                       output_dir=OUTPUT_DIR,
                       remover_name=PIX2PIX_NAME,
                       device=DEVICE)

slide_name = f"Slide_{Path(INPUT_FILE).stem}_pix2pix"
output_loc = str(Path(OUTPUT_DIR) / slide_name)

dataset = ExtractPatches(
        image_pth=INPUT_FILE,
        tile_h=TILE_H,
        tile_w=TILE_W,
        tile_stride_factor_h=TILE_STRIDE_FACTOR_H,
        tile_stride_factor_w=TILE_STRIDE_FACTOR_W,
        spacing=SPACING,
        output_pth=output_loc,
        lwst_level_idx=LWST_LEVEL_IDX,
        mode="train",
        train_split=1,
        transform=TRANSFORM,
        threshold=0.7
)

dataset, template = inkdeploy.filter(dataset = dataset, slide_name=slide_name, template = dataset.template)
