"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Please see test_pix2pix.sh for example on how to run for the set of experiments

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import sys
import random
import time
from pathlib import Path
sys.path.append(Path(__file__).parent.parent)

from tqdm import tqdm

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import wandb
import numpy as np
import pickle as pkl
import torch
from modules.metrics import ssim, psnr, mse, pbvif
import train_filter.utils
import trainer


def calc_imagemmetrics(img_ref,img_src):
    ssim_calc = ssim(img_ref, img_src)
    psnr_calc = psnr(img_ref, img_src)
    vif_calc = pbvif(img_ref, img_src)
    return (ssim_calc,psnr_calc,vif_calc)

random_seed = 2022
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    # opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    time_curr = time.localtime()
    time_str = f"{time_curr.tm_mday}{time_curr.tm_mon}{time_curr.tm_hour}{time_curr.tm_min}"

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}_{}'.format(time_str,opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    if opt.eval:
        model.eval()
    all_calc = []
    ink_index = []
    all_label = []
    
    MODEL_PATH = "../Ink_removal_weights/filter_weights.pt"
    device = torch.device(f"cuda:{opt.gpu_ids[0]}")
    model_filter = trainer.Model.create("ink")
    model_filter.load_model_weights(MODEL_PATH,torch.device("cpu"))
    model_filter.to(device)
    model_filter.eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataset)):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths

            outputs = model_filter(visuals["real_A"])
            _, predicted = torch.max(outputs.data, 1)
            preds = torch.squeeze(predicted.cpu()).item()
            if preds==1:
                ink_index.append(i)
            
            real_ink =  np.asarray(visuals["real_A"][0].permute(1,2,0).cpu().numpy()*255,np.uint8)
            real_clean = np.asarray(visuals["real_B"][0].permute(1,2,0).cpu().numpy()*255,np.uint8)
            ink_removed = np.asarray(visuals["fake_B"][0].permute(1,2,0).cpu().numpy()*255,np.uint8)

            reference_calc = calc_imagemmetrics(real_clean,real_ink)
            removal_calc = calc_imagemmetrics(real_clean,ink_removed)

            all_calc.append((reference_calc,removal_calc))
            all_label.append(data['label'])
            if i % 50 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
                save_images(webpage,(reference_calc,removal_calc), visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        webpage.save()  # save the HTML

    with open(os.path.join(web_dir, f'{opt.name}_ref_remove_imagemetrics.pkl'), 'wb') as f:
        pkl.dump(all_calc, f)

    with open(os.path.join(web_dir, f'{opt.name}_filtered_indx.pkl'),"wb") as f:
        pkl.dump(ink_index,f)
    
    with open(os.path.join(web_dir, f'{opt.name}_labels.pkl'), 'wb') as f:
        pkl.dump(all_label, f)

    print("done")