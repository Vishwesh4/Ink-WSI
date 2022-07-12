import sys
sys.path.append("/home/vishwesh/Projects/Ink-WSI")

from matplotlib import pyplot as plt
import staintools
import cv2
from skimage.metrics import structural_similarity as ssim

from utils import Pairwise_Extractor

size_img = (256,256)
#Read slides
img_ink_path = "/home/vishwesh/Projects/Ink-WSI/images/121504.svs"
img_noink_path = "/home/vishwesh/Projects/Ink-WSI/images/114793.svs"
patch_extractor = Pairwise_Extractor.from_path(src_path=img_noink_path, dest_path=img_ink_path, plot=True)

#Pair wise patch extraction
# x_point,y_point = (30773, 15864)
x_point,y_point = (36184, 19835)

ink_patch, src_patch, reg_patch = patch_extractor.extract(x_point,y_point,size_img)

#Color Normalization
template_img = cv2.cvtColor(cv2.imread("/home/vishwesh/Projects/Ink-WSI/utils/staintemplate.png"),cv2.COLOR_BGR2RGB)
normalizer = staintools.StainNormalizer(method="vahadane")
normalizer.fit(template_img)

ink_patch_norm = normalizer.transform(ink_patch)
reg_patch_norm = normalizer.transform(reg_patch)

_,axes = plt.subplots(1,3)
axes[0].imshow(template_img)
axes[1].imshow(ink_patch_norm)
axes[2].imshow(reg_patch_norm)
plt.show()

src_patch = cv2.resize(src_patch,(256,256))
src_normalize = normalizer.transform(src_patch)
#SSIM calculation
ssim_without_normalization = ssim(ink_patch, reg_patch, data_range=255, channel_axis=-1)
ssim_normalization = ssim(ink_patch_norm, reg_patch_norm, data_range=255, channel_axis=-1)
ssim_src = ssim(ink_patch, src_patch, data_range=255, channel_axis=-1)
ssim_src_norm = ssim(ink_patch_norm, src_normalize, data_range=255, channel_axis=-1)


ssim_ = ssim(ink_patch_norm, ink_patch_norm, data_range=255, channel_axis=-1)
# print(ssim_)

print("Without normalization: {}\nWith normalization: {}".format(ssim_without_normalization,ssim_normalization))
print("ssim_src: {}\nwith normalization: {}".format(ssim_src,ssim_src_norm))