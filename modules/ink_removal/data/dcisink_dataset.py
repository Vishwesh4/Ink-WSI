import os
import sys
sys.path.append("/home/ramanav/Projects/Ink-WSI")
# sys.path.append("/home/ramanav/Projects/pytorch-CycleGAN-and-pix2pix")

from pathlib import Path
import torch.utils.data as data
import os, math, torch
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms

import modules
from modules.patch_extraction import ExtractPatches
from modules.train_filter import Handwritten
from modules.train_filter import InkGenerator
# from data.extract_patches import Vectorize_WSIs
from data.base_dataset import BaseDataset, get_transform

class DcisinkDataset(BaseDataset, ExtractPatches):
    """A template dataset class for you to implement custom datasets."""
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.add_argument('--mode',type=str,default="train",help="Train/Test")
        parser.add_argument('--stride_h',type=float,default=5,help="Stride factor with tile size 256 in y direction")
        parser.add_argument('--stride_w',type=float,default=5,help="Stride factor with tile size 256 in x direction")

        if is_train==False:
            parser.set_defaults(mode="test")  # specify dataset-specific default values

        return parser
    
    def __init__(self, 
                 opt, 
                 tile_h=256, 
                 tile_w=256, 
                 lwst_level_idx=0,
                 train_split=1, 
                 threshold=0.7, 
                 transform=transforms.ToTensor()
                 ):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        image_pth = "/amartel_data4/Flow/DCIS_prediction/DCIS_unknown"
        template_pth = "/localdisk3/ramanav/backup_codes/Ink_project/Projects/Dataset"
        with open("/home/ramanav/Projects/pytorch-CycleGAN-and-pix2pix/datasets/newdomain_slides.txt","r") as f:
            image_ids = f.readlines()
        # mask_pth = str(Path(image_pth) / "masks")
        # image_pth = str(Path(image_pth) / "images")
        
        # self.colors = [["black","#28282B"],["#002d04","#2a7e19"],["#000133","skyblue"],["#1f0954","#6d5caf"],["#a90308","#ff000d"]]
        self.colors = [["black","#28282B"],["#002d04","#2a7e19"],["#000133","skyblue"],["#1f0954","#6d5caf"],["#a90308","#ff000d"],["#005558","#90DCD5"],["#001769","#005CC9"],["#3C1C16","#A05745"]]
        #For ink stains``
        self.ink_templates =  Handwritten(path=template_pth,n=10000)
        self.n_templ = len(self.ink_templates)
        self.ink_generator = InkGenerator(ink_template=self.ink_templates,
                                          colors=self.colors
                                         )

        self.ink_generator.ALPHA = [0.5,0.85]
        
        BaseDataset.__init__(self, opt)
        
        self.do_norm = opt.do_norm

        image_pths = [str(Path(image_pth)/i.strip("\n")) for i in image_ids]
        
        ExtractPatches.__init__(self,        
                                image_pths,
                                tile_h,
                                tile_w,
                                opt.stride_h,
                                opt.stride_w,
                                spacing=0.5,
                                mask_pth=None,
                                # output_pth=str(Path(opt.checkpoints_dir)/opt.name),
                                output_pth=opt.checkpoints_dir,
                                lwst_level_idx=lwst_level_idx,
                                mode=opt.mode,
                                train_split=train_split,
                                threshold=threshold,
                                transform=transform,
                                get_template=False,
                                )

        # save the option and dataset root
        #Basic Transforms
        # self.transform = transforms.Compose([transforms.ToTensor(),Normalize])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        img =  self.all_image_tiles_hr[index]
        #Get fake images
        gen_image,label = self.add_inkstain(img)
        data_A =  self.transform(Image.fromarray((gen_image*255).astype(np.uint8)))
        data_B =  self.transform(Image.fromarray(img))
        
        return {'A': self.normalize(data_A), 'B': self.normalize(data_B), 'A_paths': "tiger_dataset_{}".format(label), 'B_paths':  "tiger_dataset_{}".format(label)}

    def add_inkstain(self,img):
        """
        For adding artificial ink stains on a given image
        """
        #For classification
        p = torch.rand(1).item()
        if p<0.5: #50% chance for clean and ink stained data
            label = 0
            noise_img = img.copy()/255
        else:
            _,_,noise_img,_,_,_ = self.ink_generator.generate(img)
            label = 1
        
        return noise_img, label
    
    def normalize(self,img):
        if self.do_norm:
            return 2*img - 1
        else:
            return img

    def __len__(self):
        """Return the total number of images."""
        return len(self.all_image_tiles_hr)