import os
import sys
from typing import Tuple

import cv2
from matplotlib import pyplot as plt
import numpy as np
import openslide

sys.path.append("/home/vishwesh/Projects/Ink-WSI")
from utils import ImageRegister

def get_homography(src_img:np.array,
                   dest_img:np.array,
                   plot:bool=True
                   ) ->Tuple[np.array,np.array]:
    """
    Given two images , gets projection matrix such that dest_img = H.src_img
    Returns:
        H: Projection matrix
        proj_img: Projected image
    """
    src_object = ImageRegister(image=src_img)
    dest_object = ImageRegister(image=dest_img)
    
    # prepare two images for registration
    src_object.prepare_img_registration()
    dest_object.prepare_img_registration()

    # perform registration
    good, M = src_object.perform_registration(dest_object, draw_match=plot)

    #Obtain projected image
    proj_object = ImageRegister(image=src_img)
    proj_object.warp_img(M, (dest_img.shape[1], dest_img.shape[0]))

    return M, proj_object

def wsi_registration(src_slide:openslide.OpenSlide,
                     dest_slide:openslide.OpenSlide,
                     plot:bool = True) -> np.array:
    """
    Registers two whole slides and outputs the project matrix that can be used for the original slide
    """
    src_dimension = src_slide.dimensions
    dest_dimension = dest_slide.dimensions

    #Extract thumbnail from given slide
    src_thumb = np.asarray(src_slide.get_thumbnail((512,512)).convert("RGB"))
    dest_thumb = np.asarray(dest_slide.get_thumbnail((512,512)).convert("RGB"))

    #Get the scaling for getting M
    src_scaling = np.array([
                            [src_dimension[1]/src_thumb.shape[0], 0, 0],
                            [0,src_dimension[0]/src_thumb.shape[1],0],
                            [0,0,1]]
                            )
    dest_scaling = np.array([
                            [dest_dimension[1]/dest_thumb.shape[0], 0, 0],
                            [0,dest_dimension[0]/dest_thumb.shape[1],0],
                            [0,0,1]]
                            )
    
    #Get projection matrix for thumbnails
    M, proj_object = get_homography(src_thumb,dest_thumb,plot=plot)
    M_adjusted = dest_scaling @ M @ np.linalg.inv(src_scaling)
    
    return M_adjusted

def warp_slide(slide:openslide.OpenSlide,
               M:np.array) -> openslide.ImageSlide:
    """
    Given slide, applies projection matrix on the slide and returns the warped slide
    """
    img = slide.read_region((0,0),0,slide.dimensions)
    plt.imshow(img)
    plt.show()

    pass

img_ink = openslide.OpenSlide("/home/vishwesh/Projects/Ink-WSI/images/121504.svs")
img_noink = openslide.OpenSlide("/home/vishwesh/Projects/Ink-WSI/images/114793.svs")

thumb_ink = np.asarray(img_ink.get_thumbnail((512,512)).convert("RGB"))
thumb_noink = np.asarray(img_noink.get_thumbnail((512,512)).convert("RGB"))

fig = plt.figure("Orig Slides")
plt.subplot(1,2,1)
plt.imshow(thumb_ink)
plt.subplot(1,2,2)
plt.imshow(thumb_noink)
# plt.show()


M, proj_object = get_homography(thumb_noink,thumb_ink,plot=False)

fig = plt.figure(num="Registered Images")
plt.subplot(1,3,1)
plt.imshow(thumb_ink)
plt.subplot(1,3,2)
plt.imshow(proj_object.warped)
plt.subplot(1,3,3)
plt.imshow(thumb_noink)
# plt.show()


# print(f"M:{M}")

M_adj = wsi_registration(img_noink,img_ink)
print("M_adjusted: {}".format(M_adj))

plt.show()
# #Some experiments
# given_patch = img_noink.read_region((0,0),0,(512,512))
# reg_patch = ImageRegister(image=given_patch).warp_img(M_adj,(1024,1024)).warped
# plt.imshow(reg_patch)
# plt.show()
