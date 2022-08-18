import os
import sys
# sys.path.append("/home/ramanav/Projects/Ink-WSI")
# sys.path.append("/home/ramanav/Projects/pytorch-CycleGAN-and-pix2pix")

from pathlib import Path
import torch.utils.data as data
import os, math, torch
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms

# from data.extract_patches import Vectorize_WSIs
from data.base_dataset import BaseDataset

class JuangdataDataset(BaseDataset):
    """For reading small set of patches given in https://github.com/smujiang/WSIPenMarkingRemoval """
    
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

        if is_train==False:
            parser.set_defaults(mode="test")  # specify dataset-specific default values

        return parser
    
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        image_pth = "/home/ramanav/Downloads/WSIPenMarkingRemoval/img_samples/"

        BaseDataset.__init__(self, opt)
        
        self.do_norm = opt.do_norm

        self.image_pths = list(Path(image_pth).glob("**/*.jpg"))

        # save the option and dataset root
        #Basic Transforms
        self.transform = transforms.ToTensor()

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
        img =  cv2.cvtColor(cv2.imread(str(self.image_pths[index])),cv2.COLOR_BGR2RGB)
        #Get the images
        data_A =  self.transform(Image.fromarray(img[:,256:,:])) #ink image
        data_B =  self.transform(Image.fromarray(img[:,:256,:])) #clean image
        
        return {'A': self.normalize(data_A), 'B': self.normalize(data_B), 'A_paths': "tiger_dataset_{}".format(1), 'B_paths':  "tiger_dataset_{}".format(1)}

    def normalize(self,img):
        if self.do_norm:
            return 2*img - 1
        else:
            return img

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_pths)