#
# --------------------------------------------------------------------------------------------------------------------------
# Created on Fri Jun 17 2022 at University of Toronto
#
# Author: Vishwesh Ramanathan
# Email: vishwesh.ramanathan@mail.utoronto.ca
# Description: This script is about ink filter
# Modifications (date, what was modified):
#   1. Copied from tiger_deploy
#   2. Removed normalization
# --------------------------------------------------------------------------------------------------------------------------
#

import cv2
import numpy as np
import torch, os
import torch._utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path

import trainer
from ...train_filter import *

class Ink_filter:
    """
    Class for filtering given patches into clean vs ink. Additionally has a method filter to go through the entire dataset and get a new dataset
    with filtered images with a new template for plotting
    """
    NUM_CLASSES = 2
    CLASSES = ["Clean","Ink Region"]
    def __init__(self, model_path:str, model_name:str, device=torch.device("cpu")) -> None:
       self.device = device
       #build ink detection network
       self.model = trainer.Model.create(model_name)
       self.model.load_model_weights(model_path = model_path, device = self.device)
       self.model.to(self.device)
       self.model.eval()
    
    def predict(self,img):
        """
        For a given img of dimension 3x256x256, outputs the class. It is assumed that the image is a tensor
        """
        # img = self.normalize(img).to(self.device)
        img = img.to(self.device)
        output = self.model(img.unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
        return predicted[0]
    
    def predict_batch(self,images):
        """
        For a given batch of images of dimensions Nx3x256x256, outputs the prediction. It is assumed that the image is a tensor
        """
        # images = self.normalize(images).to(self.device)
        images = images.to(self.device)
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        return predicted 

    def filter(self, dataset:Dataset, output_dir:str=None, template:np.array=None):
        """
        Given a Dataset object, creates a new Dataset object with filtered set of images
        """
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,shuffle=False)
        predicted_labels = []
        #Predict labels
        with torch.no_grad():
            for data in tqdm(dataloader):
                images = data.to(self.device)
                labels = self.predict_batch(images)
                predicted_labels.extend(labels.cpu().numpy())
        self.predicted_labels = np.array(predicted_labels)

            #Change dataset
        new_dataset = dataset.all_image_tiles_hr[np.where(self.predicted_labels ==0)[0]]
        
        if output_dir is not None:            
            #Plot and save the results
            self.plot_results(self.predicted_labels , template, output_dir)

        #Change template
        if template is not None:
            template_flat = template.flatten()
            template_info = template_flat[template_flat>0]
            #Make locations with ink as 0
            template_info[np.where(self.predicted_labels ==1)[0]] = 0
            template_info[np.where(self.predicted_labels ==0)[0]] = np.arange(1,len(new_dataset)+1)
            template_flat[template_flat>0] = template_info
            new_template = template_flat.reshape(template.shape)
            if output_dir is not None:
                cv2.imwrite(str(Path(output_dir)/"template_ink.png"),255.0*(template>0))
            return new_dataset, new_template
        else:
            return new_dataset

    def plot_results(self,preds:np.array,template:np.array, output_dir):
        """
        Given template, plots the results from filter method
        """
        template = template.astype(np.float64)
        template[template==0] = np.nan
        fill_ink = template.flatten().copy()
        fill_ink[np.where(fill_ink >= 1)[0]] = preds
        ink_heatmap = np.reshape(fill_ink, np.shape(template))

        cmap = matplotlib.cm.jet
        cmap.set_bad('white',1.)
        # ax.imshow(masked_array, interpolation='nearest', cmap=cmap)
        plt.figure()
        plot_name = "Ink map"
        plot_var = ink_heatmap
        plt.title(plot_name)
        plt.imshow(plot_var, interpolation="nearest", cmap=cmap)
        plt.savefig(str(Path(output_dir) / f"{plot_name}_viewer.png"))