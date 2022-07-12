from math import frexp
import os
import sys
sys.path.append("/home/ramanav/projects/rrg-amartel/ramanav/Projects/InkFilter")

from pathlib import Path
import torch
import random
import numpy as np
from tqdm import tqdm

import utils
from utils.inkgeneration import InkGenerator
import trainer

random_seed = 2022
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

dataset = trainer.Dataset.create("ink",
                                 path="/localscratch/ramanav.39148973.0/SSL_training",
                                 test_batch_size=64,
                                 train_batch_size=64,
                                 template_pth="/localscratch/ramanav.39148973.0/by_class",
                                 tile_h=256,
                                 tile_w=256,
                                 tile_stride_factor_h=4,
                                 tile_stride_factor_w=4,
                                 colors=[("black","#28282B"),("#002d04","#2a7e19"),("#000133","skyblue"),("#1f0954","#6d5caf"),("#a90308","#ff000d")],
                                 train_split=0.85,
                                 n_template=10000
                                )

for i in range(100):
    print(i)
    for data in dataset.trainloader:
        pass
    # for data in dataset.testloader:
    #     pass
print("Something")