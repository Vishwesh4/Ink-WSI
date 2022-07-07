from pathlib import Path
import os
from os.path import exists
import glob
import random

import torch
import numpy as np
import openslide
from tqdm import tqdm
import torch.utils.data as data
from skimage import io
from PIL import Image
import cv2
import random

from utils.inkgeneration import InkGenerator


class Handwritten(data.Dataset):
    ''' Gets the handwritten dataset, but can be used for any random simple image dataset, whose data can be loaded into the memory'''
    def __init__(self,path,n=10000,transform=None):
        super(Handwritten,self).__init__()
        self.path = path
        self.master_path = list(Path(self.path).glob("**/*.png"))
        self.transform = transform
        random.shuffle(self.master_path)
        self.master_path = self.master_path[:n]
        self.data = self.read_data()

    def __len__(self):
        return len(self.master_path)
    
    def __getitem__(self,index):
        if self.transform!=None:
            return self.transform(Image.fromarray(self.data[index]))
        else:
            return self.data[index]
    
    def read_data(self):
        print("Loading the data...")
        data = []
        for i in tqdm(range(len(self.master_path))):
            path = str(self.master_path[i])
            data.append(cv2.imread(path,cv2.IMREAD_GRAYSCALE))
        return data


class Vectorize_WSIs(data.Dataset):
    """ WSI dataset preparation for ink filter for TIGER"""

    def __init__(self,
                 image_pth,
                 mask_pth,
                 template_pth,
                 tile_h,
                 tile_w,
                 tile_stride_factor_h,
                 tile_stride_factor_w,
                 colors,
                 lwst_level_idx=0,
                 mode="train",
                 train_split=0.8,
                 transform=None
                 ):

        """
        Args:
            image_pth (str): path to wsi/folder of wsi.
            mask_pth(str): path to mask folder
            template_pth(str): path to ink template folder
            tile_h (int): tile height
            tile_w (int): tile width
            tile_stride_factor_h (int): stride height factor, height will be tile_height * factor
            tile_stride_factor_w (int): stride width factor, width will be tile_width * factor
            colors (List[Tuple(str,str),]): List of tuples consisting of two colors, giving a range to color
            lwst_level_idx (int): lowest level for patch indexing
            mode (str): train or val, split the slides into trainset and val set
            train_split(float): Between 0-1, ratio of split between train and val set
        """

        self.image_path = image_pth
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.tile_stride_h = tile_h*tile_stride_factor_h
        self.tile_stride_w = tile_w*tile_stride_factor_w
        self.hr_level =  lwst_level_idx
        self.mask_path = mask_pth
        self.transform = transform
        self.mode = mode
        self.colors = colors
        self.train_split = train_split
        self.all_image_tiles_hr,self.coords = self.tiles_array()
        #For ink stains
        self.n_templ = 10000
        self.ink_templates = Handwritten(path=template_pth,
                                         n=self.n_templ
                                        )
        self.ink_generator = InkGenerator(ink_template=self.ink_templates,
                                          colors=self.colors
                                         )

    def __len__(self):
        return len(self.all_image_tiles_hr)

    def __getitem__(self, index):
        img =  self.all_image_tiles_hr[index]
        #Get fake images
        gen_image,label = self.add_inkstain(img)
        if self.transform is not None:
            return self.transform(Image.fromarray((gen_image*255).astype(np.uint8))), label
        else:
            return gen_image, label

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

    def tiles_array(self):

        # Check image
        if not exists(self.image_path):
            raise Exception('WSI file does not exist in: %s' % str(self.image_path))

        all_wsipaths = []
        if Path(self.image_path).suffix[1:] in ["tif","svs"]:
            all_wsipaths.append(self.image_path)
        for file_ext in ['tif', 'svs']:
            all_wsipaths = all_wsipaths + glob.glob('{}/*.{}'.format(self.image_path, file_ext))
        random.shuffle(all_wsipaths)
        
        #Select subset of slides for training/val setup
        if len(all_wsipaths)>5:
            if self.mode=="train":
                wsipaths = all_wsipaths[:int(self.train_split*len(all_wsipaths))]
            else:
                wsipaths = all_wsipaths[int(self.train_split*len(all_wsipaths)):]
        else:
            wsipaths = all_wsipaths

        with tqdm(enumerate(sorted(wsipaths))) as t:

            all_image_tiles_hr = []
            all_coords = []

            for wj, wsipath in t:
                t.set_description('Loading wsis.. {:d}/{:d}'.format(1 + wj, len(wsipaths)))

                'generate tiles for this wsi'
                image_tiles_hr,coords = self.get_wsi_patches(wsipath)

                # Check if patches are generated or not for a wsi
                if len(image_tiles_hr) == 0:
                    print('bad wsi, no patches are generated for', str(wsipath))
                    continue
                else:
                    all_image_tiles_hr.append(image_tiles_hr)
                    all_coords.extend(coords)


            # Stack all patches across images
            all_image_tiles_hr = np.concatenate(all_image_tiles_hr)

        return all_image_tiles_hr,all_coords

    def load_mask(self,wsipth):
        """
        Loads tissue mask
        """
        filename, file_extension = os.path.splitext(Path(wsipth).name)
        tissue_mask_path = Path(self.mask_path)/f"{filename}_tissue{file_extension}"
        #Read at level 0
        mask = io.imread(str(tissue_mask_path))
        return mask

    def _getpatch(self, scan, x, y):

        'read low res. image'
        'hr patch'
        image_tile_hr = scan.read_region((x, y), self.hr_level, (self.tile_w, self.tile_h)).convert('RGB')
        image_tile_hr = np.array(image_tile_hr).astype('uint8')

        return image_tile_hr

    def get_wsi_patches(self, wsipth):

        'read the wsi scan'
        scan = openslide.OpenSlide(wsipth)
        mask = self.load_mask(wsipth)

        'downsample multiplier'
        '''
        due to the way pyramid images are stored,
        it's best to use the lower resolution to
        specify the coordinates then pick high res.
        from that (because low. res. pts will always
        be on high res image but when high res coords
        are downsampled, you might lose that (x,y) point)
        '''

        iw, ih = scan.level_dimensions[self.hr_level]
        sh, sw = self.tile_stride_h, self.tile_stride_w
        ph, pw = self.tile_h, self.tile_w

        patch_id = 0
        image_tiles_hr = []
        coords = []

        for ypos in range(sh, ih - 1 - ph, sh):
            for xpos in range(sw, iw - 1 - pw, sw):
                if self._isforeground(xpos, ypos, mask):  # Select valid foreground patch
                    coords.append((xpos,ypos))
                    image_tile_hr = self._getpatch(scan, xpos, ypos)

                    image_tiles_hr.append(image_tile_hr)

                    patch_id = patch_id + 1

        # Concatenate
        if len(image_tiles_hr) == 0:
            image_tiles_hr == []
        else:
            image_tiles_hr = np.stack(image_tiles_hr, axis=0).astype('uint8')

        return image_tiles_hr,coords

    def _isforeground(self,x,y,mask,threshold=0.95):
        patch = mask[y:y+self.tile_w,x:x+self.tile_w]
        return np.sum(patch)/float(self.tile_w*self.tile_h)>threshold
