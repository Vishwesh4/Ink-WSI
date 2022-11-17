import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import torchvision
from skimage import io
from PIL import Image
from tqdm import tqdm


from utils import Ink_deploy
from modules.patch_extraction import ExtractAnnotations

INPUT_FILE = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/121694.svs"

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
annotation_dir = str( Path(INPUT_FILE).parent / Path("sedeen") )

#Inked WSI often gives a bad tissue mask, better to provide a mask
ink_labelset = {"mask":"#0000ffff"}

dataset = ExtractAnnotations(
        image_pth=INPUT_FILE,
        annotation_dir=annotation_dir,
        renamed_label=ink_labelset,
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
        sample_threshold=10,
        get_template=True,
        get_coordinates=True
)

coords = dataset.all_coordinates[0]

dataset, template, filter_predicted = inkdeploy.filter(dataset = dataset, slide_name=slide_name, template = dataset.template)

#Load full wsi into memory for editing
img_temp = io.imread(INPUT_FILE)

#ink patches
patches_id = np.where(filter_predicted==1)[0]
for i in tqdm(patches_id,desc="Stitching restored images"):
    x, y = coords[i]
    img_temp[y:y+TILE_W,x:x+TILE_H,:]= np.asarray(255*dataset[i].permute(1,2,0).cpu().numpy(),np.uint8)

print("Saving...")
img_temp = Image.fromarray(img_temp).convert("RGB")

new_size = (np.array((img_temp.size)) * (1/40)).astype(int)
img_temp = img_temp.resize(new_size, Image.ANTIALIAS)
img_temp.save(Path(output_loc)/"restored.png")