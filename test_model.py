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

from modules.patch_extraction import SedeenAnnotationParser
from modules.patch_extraction import ExtractAnnotations
import modules
import trainer
from modules import train_filter

df = pd.read_excel("~/Downloads/pairs.ods")
ink_slide_path = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/"
clean_path = "/labs3/amartel_data3/histology/Data/DCIS_cohort/PRECISE_NoRT/"

slide_path = [str( Path(ink_slide_path) / (str(df["Ink Slides"][i])+".svs" ) ) for i in range(len(df))]
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

dataset_list = []
for slides in slide_path:
        dataset = ExtractAnnotations(
                image_pth=slides,
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
                threshold=0.7
                )
        dataset_list.append(dataset)
        # break

all_dataset = torch.utils.data.ConcatDataset(dataset_list)

#Ink filter model
device = torch.device("cuda:0")
model = trainer.Model.create("ink")
model.load_model_weights("/home/ramanav/Projects/Ink-WSI/Results/filter/Checkpoint_23Jul17_05_28_1.00.pt",torch.device("cpu"))
model.to(device)
model.eval()


cm = torchmetrics.ConfusionMatrix(num_classes=2)
auroc = torchmetrics.AUROC(pos_label=1)


dataloader = torch.utils.data.DataLoader(all_dataset,batch_size=64)
with torch.no_grad():
    for data in tqdm(dataloader):
        img, label = data
        img = img.to(device)
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        preds = torch.squeeze(predicted.cpu())
        soft = torch.nn.functional.softmax(outputs,dim=1).cpu()
        cm(preds,label)
        auroc(soft[:,1],label)
        
confusion = cm.compute()
tn = confusion[0,0]
fp = confusion[0,1]
fn = confusion[1,0]
tp = confusion[1,1]

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2*(precision*recall)/(precision+recall)
accuracy = (tp+tn)/(tp+fp+fn+tn)

auc = auroc.compute()


print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nAUCROC: {auc}\nConfusion matrix : {confusion}")
