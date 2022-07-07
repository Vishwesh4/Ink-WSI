from typing import Union, Tuple, Any
from pathlib import Path

import numpy as np
import torch
import torchmetrics
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import wandb

import trainer
from utils import Vectorize_WSIs

@trainer.Metric.register("ink")
class Test_Metric(trainer.Metric):
    def get_metrics(self):
        metricfun = torchmetrics.MetricCollection(
            [torchmetrics.Accuracy(),
             torchmetrics.ConfusionMatrix(num_classes=2)]
        )
        return metricfun


@trainer.Dataset.register("ink")
class Mnist_Dataset(trainer.Dataset):
    IMG_SIZE = 256
    NUM_OPS = 3


    def get_transforms(self) -> Tuple[Any, Any]:
        train_augs = transforms.Compose([
                transforms.Resize(size=(self.IMG_SIZE, self.IMG_SIZE)),
                transforms.RandAugment(num_ops=self.NUM_OPS),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.ToTensor(),
            ])

        test_augs = transforms.Compose([
                transforms.Resize(size=(self.IMG_SIZE, self.IMG_SIZE)),
                transforms.ToTensor(),
            ])
        return train_augs, test_augs

    def get_loaders(self):
        
        image_pth = str(Path(self.path)/"images")
        mask_pth = str(Path(self.path)/"masks")

        trainset = Vectorize_WSIs(image_pth=image_pth,
                                  mask_pth=mask_pth,
                                  template_pth=self.kwargs["template_pth"],
                                  tile_h=self.kwargs["tile_h"],
                                  tile_w=self.kwargs["tile_w"],
                                  tile_stride_factor_h=self.kwargs["tile_stride_factor_h"],
                                  tile_stride_factor_w=self.kwargs["tile_stride_factor_w"],
                                  mode="train",
                                  train_split=self.kwargs["train_split"],
                                  transform=self.train_transform,
                                  colors=self.kwargs["colors"])
        
        testset = Vectorize_WSIs(image_pth=image_pth,
                                  mask_pth=mask_pth,
                                  template_pth=self.kwargs["template_pth"],
                                  tile_h=self.kwargs["tile_h"],
                                  tile_w=self.kwargs["tile_w"],
                                  tile_stride_factor_h=self.kwargs["tile_stride_factor_h"],
                                  tile_stride_factor_w=self.kwargs["tile_stride_factor_w"],
                                  mode="val",
                                  train_split=self.kwargs["train_split"],
                                  transform=self.test_transform,
                                  colors=self.kwargs["colors"])


        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch_size, num_workers=4,shuffle=True,pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch_size,shuffle=True, num_workers=4,pin_memory=True)

        return trainset, trainloader, testset, testloader


@trainer.Logger.register("ink")
class Mnist_logger(trainer.Logger):
    SOFTMAX = torch.nn.Softmax()
    def log_table(self, input, output, label, epoch, metrics):
        con = metrics["val_ConfusionMatrix"]
        columns = ["id", "image", "real class", "calculated class", "probability", "accuracy","recall","precision","TN","FP","FN","TP"]
        table = wandb.Table(columns=columns)
        _, preds = torch.max(output.data, 1)
        probs = Mnist_logger.SOFTMAX(output.data)
        n = min(16,len(probs))
        for i in range(n):
            idx = f"{epoch}_{i}"
            image = wandb.Image(input[i].permute(1, 2, 0).cpu().numpy())
            accuracy = (con[1,1]+con[0,0])/(con[0,0]+con[0,1]+con[1,0]+con[1,1])
            precision = con[1,1]/(con[1,1]+con[0,1])
            recall = con[1,1]/(con[1,1]+con[1,0])
            table.add_data(idx, image, preds[i], label[i],probs[i,int(preds[i].item())],accuracy,recall,precision,con[0,0],con[0,1],con[1,0],con[1,1])
        self.log({"table_key": table})

@trainer.Model.register("ink")
class Mnist_model(trainer.Model):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.__dict__["resnet18"](pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        output = self.model(x)
        return output

