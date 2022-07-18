import sys
from pathlib import Path
sys.path.append("/home/ramanav/Projects/Ink-WSI")

import torchvision

from modules.patch_extraction import SedeenAnnotationParser
from modules.patch_extraction import ExtractAnnotations

# slide_path = Path("/home/ramanav/Downloads/121484.svs")
slide_path = Path("/home/ramanav/Downloads/103732.svs")
xml_path = slide_path.parent / Path(slide_path.stem + ".session.xml")

# label_set = {"Clean":"#00ff00ff", "Ink":"#ff0000ff"}
label_set = {"tils":"#ffff00ff","tumor":"#00ff00ff","calcification":"#000000ff","cell":"#00ffffff"}

annotations = SedeenAnnotationParser(renamed_label=label_set, annular_color="#00ff00ff")
ext_annotation = annotations.parse(xml_path)

print(ext_annotation[0].geometry.area)


# Test annotation extraction
slide_path = "/home/ramanav/Downloads/121484.svs"
annotation_dir = "/home/ramanav/Downloads"

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

ink_labelset = {"clean":"#00ff00ff","ink":"#ff0000ff"}
dataset = ExtractAnnotations(
        image_pth=slide_path,
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

print(len(dataset))
print(dataset[0][0])
print(dataset[0][1])