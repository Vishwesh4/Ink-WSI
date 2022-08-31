import sys

from matplotlib import pyplot as plt

sys.path.append("/home/vishwesh/Projects/Ink-WSI")
from modules.register import Pairwise_Extractor
import time




size_img = (256,256)

#Read slides
img_ink_path = "/home/vishwesh/Projects/Ink-WSI/images/121504.svs"
img_noink_path = "/home/vishwesh/Projects/Ink-WSI/images/114793.svs"

# #Read region from ink
# x_point,y_point = (30773, 15864)
# # x_point,y_point = (28059,21613)
# x_point,y_point = (23346,24651)
# x_point,y_point = (11367,9926)
# x_point,y_point = (19967,14363)
# x_point,y_point = (46188,11040)
x_point,y_point = (14349,12345)


# img_ink_path = "/home/vishwesh/Projects/Ink-WSI/images/121393.svs"
# img_noink_path = "/home/vishwesh/Projects/Ink-WSI/images/114758.svs"

#Read region from ink
# x_point,y_point = (53315, 22135)
# x_point,y_point = (49461, 14263)


patch_extractor = Pairwise_Extractor.from_path(src_path=img_noink_path, dest_path=img_ink_path, plot=True)

start = time.time()
ink_img, src_patch, reg_noink = patch_extractor.extract(x_point,y_point,size_img)
end = time.time()
print("Time taken:{}".format(end-start))


plt.show()

#Time comparison
#0.045 for direct and no secondary registration
#0.16 for secondary registration



# def adjust_origin(x_shift:int,y_shift:int,M:np.array)->np.array:
#     """
#     Adjusts the projection matrix based on given shifted coordinates
#     """
#     M_adj = np.copy(M)
#     M_adj[:,-1] = M_adj[:,-1] + M_adj[:,:-1] @ np.array([x_shift,y_shift])
#     return M_adj

# def adjust_perspective(x_shift:int,y_shift:int,M:np.array)->np.array:
#     M_adj = np.copy(M)
#     M_adj[:,-1] = M_adj[:,-1] + M_adj[:,:-1] @ np.array([x_shift,y_shift])
#     x_shift_proj,y_shift_proj = transform_coords(x_shift,y_shift,M)
#     M_adj[0,:] = M_adj[0,:] - x_shift_proj*M_adj[-1,:]
#     M_adj[1,:] = M_adj[1,:] - y_shift_proj*M_adj[-1,:]
#     return M_adj
#Step 3: Projection using shifted projection matrix
# M_shift = adjust_origin(x_corner,y_corner,M)
# print(transform_coords(x_corner,y_corner,M_shift))
# no_ink_patch = np.asarray(img_noink.read_region((x_corner,y_corner),0,(h,w)).convert("RGB"))
# shift_object = ImageRegister(image=no_ink_patch)
# shift_object.warp_img(M_shift,size_img)

# def M_new_calulate(self,src_corner:np.array, dest_corner:np.array,M:np.array):
    
#     M_adj = np.array([
#                       [M[0,0]-M[2,0]*dest_corner[0], M[0,1]-M[2,1]*dest_corner[0], M[0,2]-M[2,2]*dest_corner[0] + M[0,:].T @ np.array([src_corner[0],src_corner[1],1])],
#                       [M[1,0]-M[2,0]*dest_corner[1], M[1,1]-M[2,1]*dest_corner[1], M[1,2]-M[2,2]*dest_corner[1] + M[1,:].T @ np.array([src_corner[0],src_corner[1],1])],
#                       [M[2,0], M[2,1], M[2,2]+M[2,:].T @ np.array([src_corner[0],src_corner[1],1])]
#                      ])
#     return M_adj