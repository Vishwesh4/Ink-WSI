# Ink Detection and Removal

The README is for the reference as to what I did and what files are for what reason. The project is locally source controlled by git. You can view git log.

## Description

In this project, I worked on ink detection and its removal in the histopathology data (trained on CIHR-invasive BC data, tested on DCIS data).

## Methodology
The basic idea is simulation of ink marks by the use of handwritten dataset. Different colors and textures are used. This way you get ink stained images. Resnet-18 was used to train a classification model. Based on the results, the simulated patches represent the true distribution of real ink stained patches.  

Now the paired data is used to train pix2pix, so as to correct the ink stained patches. As of now, the criteria for filtering the bad results from pix2pix is to use some threshold, which is got from basic boxplot 0.75 percentile+1.5*IQR

## Getting Started

### Dependencies

* Work on torch environment present in the local computer

### Executing program

The project has 4 components:-  
1. Curation of dataset
2. Ink detection
3. Ink Correction using Pix2Pix
4. Analysis

#### Curation of dataset
1. First curate the dataset by annotation/viewing using Sedeen and extract patches. The extraction of patches
can be done using ```python Patch_Extraction_WSIs/extractTiles-ws.py --slide [FILELOC] -out [OUTLOC] --px [FILESIZE] --um [MPP_X*FILESIZE]```
2. If the file is annotated one you can use  

```
python ROI_Patch_Extraction/extract_patches_.py --images_path /home/osha/Vishwesh/Data/Dirty Data/ --output_path /home/osha/Vishwesh/Data/annot_patch/ --annot_path /home/osha/Vishwesh/Data/Dirty Data/sedeen/ --sampling grid
```
3. Run ```python utils/Get_Dataset_ready.py``` to get the dataset set up, this creates a file named as ```files.txt```, read about the options
4. You might also want to call ```python utils/random_distribute.py``` to randomize the dataset so as to curate . In the end you should have a text file with name "train/val/test".txt. This is important for training of ink detection model or pix2pix training model.
5. For Pix2pix, the training dataset is similar to the training set of ink detection module. For testing, the process of curating is passing set of images through ```python ink_filter.py```. It will give clean path and ink path text. The ink path text will be the testing set for Pix2Pix.
6. A bash file called ```./dataset_prepare.sh [PATH] [1/0]``` is also there, which does all this process automatically except random_distribute. Based on the option ($1=[PATH]) ($2=[1/0]), it will curate dataset for pix2pix or till the ```utils/Get_Dataset_ready point.py```
####  Ink Detection  
1. Run ```python train.py --hyp [HYPERPARAMETERS.json] --log [LOGID]``` and it should work, it has two arguments hyperparameters and the log directory
2. You can test the model over train/val/test set using ```test_model.py```, this generates all metrics like confidence intervals, dataset biases, accuracies and additionally also stores all the False positives and False negatives in `\Results\Failure_cases`
3. ```ink_filter.py``` should be run, which is basically output of the trained model
#### Pix2Pix
1. For pix2pix training, you have to run the ```./pytorch-CycleGAN-and-pix2pix/train.py [opt]```. Lot of options are there, refer to the original repo for inspiration [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
2. For testing you have to run ```./pytorch-CycleGAN-and-pix2pix/test.py```
2. The options used in the current implementation for pix2pix is in the bash file. You should run the bash file ```./pix2pix.sh [DATAPATH] [NAMEOFEXP(should be same as the training name)] [1/0] [RESULTDIR(only for testing)]``` where 0 is train mode and 1 is test mode  
Note: All training results are stored in ```./Results/checkpoints```
#### Analysis
1. For TSNE plots, you can use ```python generate_tsne_plots.py```. You have to manually edit the code for fetching saved model path and which images to be loaded. I manually made a text file like the ones in ```./Data/pix2pix_pics_fake.txt``` and ```./Data/TSNE_dataset.txt```, to decide which images to be included
2. For heatmap you have to first run get the slide you want to run heatmap on. First run to get the mask, use ```python ./util/Mask_generation.py```. After getting the masks, run ```python ./util/Heatmap.py``` to get the heatmaps   

NOTE: All the results will be stored in ```Results``` directory
### Experiments
1. `Results/logs/run1.txt` and `Results/logs/run2.txt` - some problem with torch.dataloader and np random
2. `Results/logs/run3.txt` - Ozan's pretrained model used for classification
3. `Results/logs/run3_wopretrain.txt` - Imagenet pretrained used
4. `Results/logs/run3_begin.txt` - No pretraining
5. For pix2pix training results are found in `/checkpoints` and testing results are found in `/Results/pix2pix_results` 
6. `Results/exp2_run1.txt` - had good results but small bug in ink generator
7. `Results/exp2_run2.txt` - Latest model with best results
8. `Results/Inkheatmap_results` - Ink heat map results
9. `Results/images` - contains all relevant result images

All the logs are in `/Results` folder  
You can additionally check out the tensorboard files for checking the results  

### Datasets
1. `Data/annot_patch`: Extracted patches of the `Data/Dirty Data` directory, Used for training for black ink markers, `run1\2\3` uses that dataset
2. `Data/test_patchs`: Extracted patches of the `Data/Test_slide` directory, Used for testing for black ink markers, used for sole purpose of testing the dataset
3. `Data/Experiment2`: Dataset collected for latest successfull run of ink generation
4. `Data/Experiment2_pix2pix` : Dataset collected for running pix2pix   
Rest of the files are unimportant
5. `Data/Exp3_DCIS` : Dataset for full testing pipeline on DCIS data
6. `Data/by_class` : Dataset used by ink generation algorithm

### 
## Authors

Vishwesh Ramanathan (vishweshramanathan@mail.utoronto.ca) 

## Acknowledgments

All the extraction codes and TSNE, heatmap belongs to Chetan