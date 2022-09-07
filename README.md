# Ink Removal in Whole Slide Images using Hallucinated Data

This repository contains the code for **Ink Removal in Whole Slide Images using Hallucinated Data**. The trained model weights are at this [link](https://drive.google.com/drive/folders/1KRqQz7Xf9P7qfPceerSlODwFidPkLvti?usp=sharing)

## Description

This project is about identifying and removing ink markings from histopathology whole slides for aiding downstream computational analysis. The algorithm requires no annotation or manual curation of data and requires only clean slides, making it easy to adapt and deploy in new set of histopathology slides.

## Methodology
The methodlogy consists of two networks:-
1. Ink filter: A binary classifier with Resnet 18 backbone
2. Ink corrector: Pix2pix module for removing ink from a patch by image to image translation 
An overview of the methodology and its results are shown below

<img src="https://github.com/Vishwesh4/Ink-WSI/blob/master/images/methodology_overview.png" align="center" width="880" ><figcaption>Fig.1 - Methodology overview</figcaption></a>

<img src="https://github.com/Vishwesh4/Ink-WSI/blob/master/images/inkfilter.png" align="center" width="880" ><figcaption>Fig.2 - Ink filter output</figcaption></a> 

<img src="https://github.com/Vishwesh4/Ink-WSI/blob/master/images/pix2pix_results.png" align="center" width="880" ><figcaption>Fig.3 - Pix2pix output</figcaption></a> 

## Getting Started

### Dependencies

```
opencv
dominate
visdom
trainer - https://github.com/Vishwesh4/TrainerCode
pytorch-gpu
wandb
openslide
scikit-learn
scipy
scikit-image
```
### Modules

The project has 6 modules:-  
1. Ink filter module - `./train_filter`
2. Ink removal module (Pix2pix) - `./ink_removal`
3. Patch Extraction - `./modules/patch_extraction`
4. Image Metric Calculate - `./modules/metrics`
5. Registration - `./modules/register`
6. Deployment of methodology over new slides - `./deploy`

#### Ink Filter module
1. The model can be trained by modifying `config.yml` file, specifying the location of path of clean slides to be used, and set of colors to be used
2. The training can be done by using
```
python train.py -c [CONFIG FILE LOCATION]
```
#### Ink Removal module
1. The code has been taken from the original repository [link](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
2. For training with your own dataset, please follow a similar code structure to `./ink_removal/data/dcisink_dataset.py` or `./ink_removal/data/tiger_dataset.py`. Mixture of the two datasets was used for the given model `./ink_removal/data/mixed_dataset.py`
3. The model can be trained by using
```
./train_pix2pix.sh
```
4. The model can be tested by using
```
./test_pix2pix.sh
```
For testing, corresponding ink and clean slides should be available
5. The image metrics can be calculated by using
```
./run_calc_metrics.sh
```
The test model name has to be specified

#### Deploy module
1. The modules can be deployed using the class `Ink_deploy`. An example is shown in `./deploy/process.py`. It also has a script `./deploy/construct_wsi.py` for running algorithm over a whole slide image, however it expects sedeen annotation.
```python
ink_deploy = Ink_deploy(filter_path:str=INK_PATH,
                        output_dir:str=None, 
                        pix2pix_path:str=PIX2PIX_PATH, 
                        device=torch.device("cpu"))
```
### 
## Authors
- Vishwesh Ramanathan ([@Vishwesh4](https://github.com/Vishwesh4))
## Contact
If you want to contact, you can reach the authors by raising an issue or
 email at vishweshramanathan@mail.utoronto.ca

## Acknowledgments
- The pix2pix code was taken from [link](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- The `./modules/metrics/quality_metrics.py` code was taken from [link](https://github.com/smujiang/WSIPenMarkingRemoval)