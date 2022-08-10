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
import pickle as pkl
import matplotlib

from modules.patch_extraction import SedeenAnnotationParser, ExtractAnnotations
from modules.deploy import Pairwise_ExtractAnnot
from modules.train_filter import InkGenerator, Handwritten
import trainer
from modules import train_filter

def log_results(template, all_patch_results, output_dir, wsi_name):
    """
    Plots heatmap of tumourbed and TIL score
    """
    template = template.astype(np.float64)
    template[template==0] = np.nan
    fill_tissue = template.flatten().copy()
    fill_tissue[np.where(fill_tissue >= 1)[0]] = all_patch_results
    tissue_heatmap = np.reshape(fill_tissue, np.shape(template))

    cmap = matplotlib.cm.jet
    cmap.set_bad('white',1.)
    # ax.imshow(masked_array, interpolation='nearest', cmap=cmap)
    im_ratio = tissue_heatmap.shape[0] / tissue_heatmap.shape[1]
    plt.figure()
    plt.title("Ink tissue map")
    im = plt.imshow(tissue_heatmap, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, fraction=0.046 * im_ratio, pad=0.04)
    plt.savefig(str(Path(output_dir)/f"{wsi_name}_heatmap_v2.png"))

MODEL_PATH = '/home/ramanav/Projects/Ink-WSI/Results/filter/Checkpoint_27Jul18_05_09_1.00.pt'
OUTPUT_DIR = "/home/ramanav/Projects/Ink-WSI/Results/heatmaps"
device = torch.device("cuda:3")

model = trainer.Model.create("ink")
model.load_model_weights(MODEL_PATH,torch.device("cpu"))
model.to(device)


# model_path = "/home/ramanav/Projects/Ink-WSI/Results/filter/Checkpoint_27Jul18_05_09_1.00.pt"
# # model_path = "/home/ramanav/Projects/Ink-WSI/Results/filter/Checkpoint_28Jul12_19_55_1.00.pt"
# device = torch.device("cuda:3")
# model = trainer.Model.create("ink")
# model.load_model_weights(model_path,torch.device("cpu"))
# model.to(device)

# ink_slide_path = "/localdisk3/ramanav/02-0484.svs"
# ink_slide_path = "/localdisk3/ramanav/121455.svs"
# ink_slide_path = "/localdisk3/ramanav/02-1693.svs"
# ink_slide_path = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/121819.svs"
# ink_slide_path = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/121516.svs"
ink_slide_path = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/121804.svs"



annotation_dir = str(Path(ink_slide_path).parent / "sedeen")

TILE_H = 256
TILE_W = 256
TILE_STRIDE_FACTOR_H = 1
TILE_STRIDE_FACTOR_W = 1
LWST_LEVEL_IDX = 0
TRANSFORM = torchvision.transforms.ToTensor()
# SPACING = 0.2526
SPACING = None

ink_labelset = {"mask":"#0000ffff"}

singledataset = ExtractAnnotations(
        image_pth=ink_slide_path,
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
        sample_threshold=10,
        get_template=True
        )

template = singledataset.template.copy()

dataloader = torch.utils.data.DataLoader(singledataset,batch_size=64)

model.eval()

with torch.no_grad():
    all_patch_results = []
    for data in tqdm(dataloader):
        img, label = data
        img = img.to(device)
        outputs = model(img)
        outputs = torch.nn.functional.softmax(outputs,dim=1)
        # _, predicted = torch.max(outputs.data, 1)
        predicted = outputs[:,1]>=0.95
        preds = torch.squeeze(predicted.cpu()).numpy()
        all_patch_results.extend(preds)


log_results(template,all_patch_results,OUTPUT_DIR,Path(ink_slide_path).stem)