import os
import sys
from xml.sax.handler import property_declaration_handler
sys.path.append("/home/ramanav/Projects/Ink-WSI")

from pathlib import Path
import torch.utils.data as data
import os, math, torch
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms
import pandas as pd

import modules
from modules.patch_extraction import Pairwise_ExtractAnnot
from data.base_dataset import BaseDataset, get_transform


class PairinkDataset(BaseDataset, Pairwise_ExtractAnnot):
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
                 mode="train", 
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
        # image_pth = "/labs3/amartel_data3/tiger_dataset/SSL_training"
        # template_pth = "/home/ramanav/Projects/Ink_project/Projects/Dataset"
        # mask_pth = str(Path(image_pth) / "masks")
        # image_pth = str(Path(image_pth) / "images")

        df = pd.read_excel("~/Downloads/pairs.ods")
        ink_slide_path = "/amartel_data4/Flow/DCIS_prediction/DCIS_Precise_20x/"
        clean_path = "/labs3/amartel_data3/histology/Data/DCIS_cohort/PRECISE_NoRT/"

        # ink_slide = str( Path(ink_slide_path) / (str(df["Ink Slides"][0])+".svs" ) )
        # clean_slide = str( Path(clean_path) / (str(df["Clean Slides"][0])+".svs" ) )

        pair_list = [(str( Path(clean_path) / (str(df["Clean Slides"][i])+".svs" ) ),str( Path(ink_slide_path) / (str(df["Ink Slides"][i])+".svs" ) ))
                for i in range(len(df))]
        annotation_dir = str( Path(ink_slide_path) / Path("sedeen") )
        ink_labelset = {"clean":"#00ff00ff","ink":"#ff0000ff"}

        BaseDataset.__init__(self, opt)

        self.do_norm = opt.do_norm
        
        Pairwise_ExtractAnnot.__init__(self,
                                pair_pths=pair_list,
                                annotation_dir=annotation_dir,
                                renamed_label=ink_labelset,
                                tile_h=tile_h,
                                tile_w=tile_w,
                                tile_stride_factor_h=opt.stride_h, 
                                tile_stride_factor_w=opt.stride_w, 
                                lwst_level_idx=lwst_level_idx, 
                                mode=mode, 
                                train_split=train_split, 
                                transform=transform,
                                threshold=threshold,
                                sample_threshold=50
                                )
        
        print(len(self.all_image_tiles_hr))
        # ExtractPatches.__init__(self,image_pth, tile_h, tile_w, tile_stride_factor_h, tile_stride_factor_w, spacing, mask_pth, output_pth, lwst_level_idx, opt.mode, train_split, threshold, transform)

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
        ink_img, clean_img = self.all_image_tiles_hr[index]
        label = self.all_labels[index]
        #Get fake images
        # gen_image,label = self.add_inkstain(img)
        # data_A =  self.transform(Image.fromarray((gen_image*255).astype(np.uint8)))
        data_A =  self.transform(Image.fromarray(ink_img))
        data_B =  self.transform(Image.fromarray(clean_img))
        
        return {'A': self.normalize(data_A), 'B': self.normalize(data_B), 'A_paths': "tiger_dataset_{}_{}".format(label,index), 'B_paths':  "tiger_dataset_{}_{}".format(label,index), 'label': label}

    def add_inkstain(self,img):
        """
        For adding artificial ink stains on a given image
        """
        #For classification
        p = torch.rand(1).item()
        if p<0.3: #30% chance for clean and ink stained data
            label = 0
            noise_img = img.copy()/255
        else:
            _,_,noise_img,_,_,_ = self.ink_generator.generate(img)
            label = 1
        
        return noise_img, label
    
    def __len__(self):
        """Return the total number of images."""
        return len(self.all_image_tiles_hr)

    def normalize(self,img):
        if self.do_norm:
            return 2*img - 1
        else:
            return img

    @property
    def labels(self):
        return self.all_labels
        