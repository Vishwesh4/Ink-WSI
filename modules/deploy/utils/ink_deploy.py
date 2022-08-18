#
# --------------------------------------------------------------------------------------------------------------------------
# Created on Fri Jun 17 2022 at University of Toronto
#
# Author: Vishwesh Ramanathan
# Email: vishwesh.ramanathan@mail.utoronto.ca
# Description: This script is about ink filter
# Modifications (date, what was modified):
#   1. Aug 10: Modified for using the current version of ink filter
#   2. Aug 17. Modified to include ink removal module using pix2pix
# --------------------------------------------------------------------------------------------------------------------------
#
import sys
from typing import Tuple, Union, List
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch, os
import torch._utils
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib
from matplotlib import pyplot as plt

import trainer
from ...train_filter import *
from ...ink_removal.options.test_options import TestOptions
from ...ink_removal.models import create_model
from ...ink_removal.util.visualizer import save_images
from ...ink_removal.util import html

class Ink_deploy:
    """
    Class for filtering given patches into clean vs ink. Additionally has a method filter to go through the entire dataset and get a new dataset
    with filtered images with a new template for plotting. Given the option can perform ink removal using pix2pix
    """
    NUM_CLASSES = 2
    CLASSES = ["Clean","Ink Region"]
    TRANSFORM = transforms.ToTensor()
    def __init__(self, model_path:str, output_dir:str=None, remover_name:str=None, device=torch.device("cpu")) -> None:
       """
       Initializes model of ink filter and if remover_name is not none then pix2pix model
       Parameters:
            model_path: saved path for ink filter
            output_dir: path for saving results, specify none if you dont want any plots
            remover_name: Name of the pix2pix model for model retrieval
       """
       self.device = device
       device_id = self.device.index
       self.output_dir = output_dir
       self.remover_name = remover_name

       #build ink detection network
       self.build_network(model_path)

       #build ink removal network
       if remover_name is not None:
            print("Loading ink removal module...")
            #Modify few parameters
            Ink_remover.OPTIONS["results_dir"] = output_dir
            Ink_remover.OPTIONS["gpu_ids"] = [device_id]
            Ink_remover.OPTIONS["name"] = remover_name
            
            self.ink_remover = Ink_remover.build_opt()

        
    def build_network(self,model_path:str) -> None:
        print("Loading ink filter module...")
        self.model = trainer.Model.create("ink")
        self.model.load_model_weights(model_path,torch.device("cpu"))
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self,img) -> torch.Tensor:
        """
        For a given img of dimension 3x256x256, outputs the class. It is assumed that the image is a tensor
        """
        # img = self.normalize(img).to(self.device)
        img = img.to(self.device)
        output = self.model(img.unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
        return predicted[0]
    
    def predict_batch(self,images) -> torch.Tensor:
        """
        For a given batch of images of dimensions Nx3x256x256, outputs the prediction. It is assumed that the image is a tensor
        """
        # images = self.normalize(images).to(self.device)
        images = images.to(self.device)
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        return predicted 

    def filter(self, dataset:Dataset, slide_name:str=None, template:np.array=None) -> Tuple[torch.utils.data.Dataset, Union[None,np.array]]:
        """
        Given a Dataset object, creates a new Dataset object with filtered set of images
        """
        new_template = template
        full_dataset = dataset.all_image_tiles_hr.copy()
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,shuffle=False)
        predicted_labels = []
        #Predict labels
        with torch.no_grad():
            for data in tqdm(dataloader,desc="Filtering out Ink"):
                images = data.to(self.device)
                labels = self.predict_batch(images)
                predicted_labels.extend(labels.cpu().numpy())
        predicted_labels = np.array(predicted_labels)

        if self.output_dir is not None:
            #Plot and save the results
            self.plot_results(predicted_labels, slide_name, template)

        #ink dataset
        ink_dataset = full_dataset[np.where(predicted_labels==1)[0]]

        if self.remover_name is not None:
            removed_dataset = self.ink_remover.remove(ink_dataset, slide_name)
            full_dataset[np.where(predicted_labels==1)[0]] = removed_dataset
        else:
            full_dataset = full_dataset[np.where(predicted_labels==0)[0]]  

        full_dataset = [Ink_deploy.TRANSFORM(img) for img in full_dataset]
        
        #Change template if only filter is present
        if (template is not None) and (self.remover_name is None):
            template_flat = template.flatten()
            template_info = template_flat[template_flat>0]
            #Make locations with ink as 0
            template_info[np.where(predicted_labels==1)[0]] = 0
            template_info[np.where(predicted_labels==0)[0]] = np.arange(1,len(full_dataset)+1)
            template_flat[template_flat>0] = template_info
            new_template = template_flat.reshape(template.shape)

        return full_dataset, new_template


    def plot_results(self,preds:np.array,slide_name:str,template:np.array):
        """
        Given template, plots the results from filter method
        """
        template = template.astype(np.float64)
        template[template==0] = np.nan
        fill_ink = template.flatten().copy()
        fill_ink[np.where(fill_ink >= 1)[0]] = preds
        ink_heatmap = np.reshape(fill_ink, np.shape(template))
        # cv2.imwrite(str(Path(output_dir) / "Ink_map_viewer.png"),ink_heatmap*255)

        cmap = matplotlib.cm.jet
        cmap.set_bad('white',1.)
        # ax.imshow(masked_array, interpolation='nearest', cmap=cmap)
        plt.figure()
        plot_name = "Ink_map"
        plot_var = ink_heatmap
        # plt.title(plot_name)
        plt.imshow(plot_var, interpolation="nearest")
        plt.axis('off')
        # plt.imshow(plot_var, interpolation="nearest")
        plt.savefig(str(Path(self.output_dir) / Path(slide_name) / f"{plot_name}_viewer.png"))


class Ink_remover:
    """
    Class for deploying pix2pix on test images
    """
    SAVE_FREQ = 100
    OPTIONS = {
        "num_threads" : 0, 
        "serial_batches" : True, 
        "no_flip" : True,
        "display_id" : -1,
        "checkpoints_dir" : "/localdisk3/ramanav/Results/Pix2Pix/",
        "results_dir" : "/localdisk3/ramanav/Results/Pix2Pix/",
        "gpu_ids" : [0],
        "batch_size" : 1,
        "direction" : "AtoB",
        "load_size" : 256,
        "preprocess" : "none",
        "do_norm" : True,
        "eval": True,
        "name" : "random_name"
    }
    
    def __init__(self,opt) -> None:
        self.model = create_model(opt)      # create a model given opt.model and other options
        self.model.setup(opt)
        self.opt = opt
        
        if self.opt.eval:
            self.model.eval() 
        
    @classmethod
    def build_opt(cls):
        opt = TestOptions().parse()
        for key,item in Ink_remover.OPTIONS.items():
            setattr(opt,key,item)
        return cls(opt)

    def normalize(self,img) -> torch.Tensor:
        if self.opt.do_norm:
            return 2*img - 1
        else:
            return img

    def form_dataset(self,dataset:np.array) -> torch.utils.data.Dataset:
        new_dataset = []
        for i in range(len(dataset)):
            temp = {}
            img = self.normalize(Ink_deploy.TRANSFORM(dataset[i]))
            temp["A"] = img
            temp["B"] = img*0 + 1 #White image
            temp["A_paths"] = f"filter_{i}"
            temp["B_paths"] = f"filter_{i}"
            new_dataset.append(temp)

        return new_dataset

    def remove(self,dataset:np.array, slide_name:str=None) -> List[np.array]:
        new_dataset = self.form_dataset(dataset)
        rectified_dataset = []
        dataloader = torch.utils.data.DataLoader(new_dataset, batch_size=self.opt.batch_size,shuffle=False)
        
        if self.opt.results_dir is not None:
            # create a website
            web_dir = os.path.join(self.opt.results_dir,slide_name, self.opt.name, '{}_{}'.format(self.opt.phase, self.opt.epoch))  # define the website directory
            if self.opt.load_iter > 0:  # load_iter is 0 by default
                web_dir = '{:s}_iter{:d}'.format(web_dir, self.opt.load_iter)
            print('creating web directory', web_dir)
            self.webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (self.opt.name, self.opt.phase, self.opt.epoch))
        
        
        for i,data in tqdm(enumerate(dataloader),desc="Removing Ink"):
            self.model.set_input(data)
            self.model.test()
            visuals = self.model.get_current_visuals()  # get image results
            img_path = self.model.get_image_paths()     # get image paths
        
            #Collect rectified images
            rectified_dataset.extend(visuals["fake_B"].permute(0,2,3,1).cpu().numpy()*255)

            if self.opt.results_dir is not None:
                #Save images at a constant interval
                if i%Ink_remover.SAVE_FREQ==0:
                    save_images(self.webpage,(0,0), visuals, img_path, aspect_ratio=self.opt.aspect_ratio, width=self.opt.display_winsize, use_wandb=self.opt.use_wandb)

        
        if self.opt.results_dir is not None:
            self.webpage.save()

        return rectified_dataset
