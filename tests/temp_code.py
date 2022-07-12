# import os
# import sys
# from pathlib import Path
# import torchvision

# sys.path.append("../")
# from utils.dataloader import Vectorize_WSIs

# parent_path = Path("/localscratch")
# imgs_path = parent_path / Path([p for p in os.listdir(parent_path) if "ramanav" in p ][0]) / "SSL_training"

# dataset = Vectorize_WSIs(
#                         image_pth=str(imgs_path / "images/122S.tif"),
#                         mask_pth=str(imgs_path / "masks"),
#                         tile_h=256,
#                         tile_w=256,
#                         tile_stride_factor_h=3,
#                         tile_stride_factor_w=3,
#                         colors=[("black","#28282B"),("#002d04","#2a7e19"),("#000133","skyblue"),("#1f0954","#6d5caf"),("#a90308","#ff000d")],
#                         transform=torchvision.transforms.ToTensor()
# )

# print("Something")

import os
import trainer
import sys
sys.path.append("/home/vishwesh/Projects/Ink-WSI")
# sys.path.append("/home/ramanav/projects/rrg-amartel/ramanav/Projects/InkFilter")
import utils
from pathlib import Path
import torch
import random
import numpy as np

from utils.inkgeneration import InkGenerator
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
                                 path="/home/vishwesh/Projects/testinput_104S",
                                 test_batch_size=16,
                                 train_batch_size=16,
                                 template_pth="/home/vishwesh/Projects/Ink_Correction/Data/by_class",
                                 tile_h=256,
                                 tile_w=256,
                                 tile_stride_factor_h=3,
                                 tile_stride_factor_w=3,
                                 colors=[("black","#28282B"),("#002d04","#2a7e19"),("#000133","skyblue"),("#1f0954","#6d5caf"),("#a90308","#ff000d")],
                                 train_split=0.8
)
print(len(dataset.trainset))


# for i in range(100):
#     print(i)
#     for data in dataset.trainloader:
#         pass
#     # for data in dataset.testloader:
#     #     pass
# print("Something")

# img = dataset.trainset.all_image_tiles_hr[523]
# for i in range(100):
#     dataset.trainset.ink_generator.get_plots(img)
#     # plt.show()