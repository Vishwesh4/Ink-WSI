import os
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent)

import torch
import numpy as np
import torchvision

from utils import Ink_deploy
from modules.patch_extraction import ExtractPatches


INPUT_FILE = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/121504.svs"
OUTPUT_DIR = str(Path(__file__).parent.parent / "tests/Results")
INK_PATH = str(Path(__file__).parent.parent / "Ink_removal_weights/filter_weights.pt")
PIX2PIX_PATH = str(Path(__file__).parent.parent / "Ink_removal_weights/latest_net_G.pth")
DEVICE = torch.device("cuda:3")
TILE_H = 256
TILE_W = 256
TILE_STRIDE_FACTOR_H = 1
TILE_STRIDE_FACTOR_W = 1
LWST_LEVEL_IDX = 0
TRANSFORM = torchvision.transforms.ToTensor()
SPACING = 0.456694

inkdeploy = Ink_deploy(filter_path=INK_PATH,
                       output_dir=OUTPUT_DIR,
                       pix2pix_path=PIX2PIX_PATH,
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
        threshold=0.7,
        get_template=True
)

dataset, template, filter_predictions = inkdeploy.filter(dataset = dataset, slide_name=slide_name, template = dataset.template)

#SPECIFY THE DOWNSTREAM TASK BASED ON THE DATASET