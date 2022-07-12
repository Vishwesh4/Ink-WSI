import os
import sys

import cv2
from matplotlib import pyplot as plt
import numpy as np

sys.path.append("/home/vishwesh/Projects/Ink-WSI")
from utils import ImageRegister

#Read images
orig_image = cv2.imread("/home/vishwesh/Projects/Ink-WSI/images/1.jpg")
reg_image = cv2.imread("/home/vishwesh/Projects/Ink-WSI/images/2.jpg")

# two image object for registration
ImageRegister.set_downsample_percent(100)
I_realHE = ImageRegister(image=orig_image)
I_virtualHE = ImageRegister(image=reg_image)

print(I_realHE.scale_percent)

# prepare two images for registration
I_virtualHE.prepare_img_registration()
I_realHE.prepare_img_registration()

fig = plt.figure(num="Orig Images")
plt.subplot(1,2,1)
plt.imshow(I_realHE.prepared)
plt.subplot(1,2,2)
plt.imshow(I_virtualHE.prepared)
plt.show()


# perform registration
good, M = I_virtualHE.perform_registration(I_realHE, draw_match=True)

# perform transformation to get registered image
I_realHE = ImageRegister(image=orig_image)
I_virtualHE = ImageRegister(image=reg_image)

fig = plt.figure(num="Registered Images")
plt.subplot(1,3,1)
plt.imshow(I_virtualHE.im)
I_virtualHE.warp_img(M, (I_realHE.im.shape[1], I_realHE.im.shape[0]))
plt.subplot(1,3,2)
plt.imshow(I_virtualHE.warped)
plt.subplot(1,3,3)
plt.imshow(I_realHE.im)


print(f"M:{M}")
print('d')
plt.show()

