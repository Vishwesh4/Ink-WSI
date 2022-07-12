import os
import trainer
import sys

sys.path.append("/home/ramanav/projects/rrg-amartel/ramanav/Projects/InkFilter")
import utils
from pathlib import Path
import torch
import random
import numpy as np

from utils.inkgeneration import InkGenerator
from utils import Handwritten
from matplotlib import pyplot as plt

# parent_path = Path("/localscratch")
# imgs_path = parent_path / Path([p for p in os.listdir(parent_path) if "ramanav" in p ][0]) / "SSL_training"
random_seed = 2022
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


dataset = trainer.Dataset.create("ink",
                                 path="/localscratch/ramanav.38885233.0/SSL_training",
                                 test_batch_size=16,
                                 train_batch_size=16,
                                 template_pth="by_class",
                                 tile_h=256,
                                 tile_w=256,
                                 tile_stride_factor_h=4,
                                 tile_stride_factor_w=4,
                                 colors=[("black","#28282B"),("#002d04","#2a7e19"),("#000133","skyblue"),("#1f0954","#6d5caf"),("#a90308","#ff000d")],
                                 train_split=0.8
)
# print(len(dataset.trainset))
# ink_templates = Handwritten(path="/localscratch/ramanav.38885233.0/by_class",n=10000)
# print(ink_templates[1256].shape)
# print(len(ink_templates))

for i in range(100):
    print(i)
    for data in dataset.testloader:
        pass
    # for data in dataset.testloader:
    #     pass
print("Something")