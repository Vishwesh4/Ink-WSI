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
sys.path.append("/home/ramanav/projects/rrg-amartel/ramanav/Projects/InkFilter")
import utils
from pathlib import Path

parent_path = Path("/localscratch")
imgs_path = parent_path / Path([p for p in os.listdir(parent_path) if "ramanav" in p ][0]) / "SSL_training"

dataset = trainer.Dataset.create("ink",
                                 path=str(imgs_path),
                                 test_batch_size=64,
                                 train_batch_size=64,
                                image_pth=str(imgs_path / "images/122S.tif"),
                                mask_pth=str(imgs_path / "masks"),
                                template_pth=str(imgs_path.parent/"by_class"),
                                tile_h=256,
                                tile_w=256,
                                tile_stride_factor_h=3,
                                tile_stride_factor_w=3,
                                colors=[("black","#28282B"),("#002d04","#2a7e19"),("#000133","skyblue"),("#1f0954","#6d5caf"),("#a90308","#ff000d")],
                                train_split=0.8
)

print("Something")
