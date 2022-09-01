#!/bin/bash
#SBATCH --account=rrg-amartel    #rrg-amartel, def-amartel    ---  6 CPU cores per P100 GPU (p100 and p100l) and no more than 8 CPU cores per V100 GPU (v100l).

#SBATCH --gres=gpu:v100l:1   # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=160G          # Memory proportional to GPUs: 32000 Cedar
#SBATCH --time=23:00:00      #23:00:00
#SBATCH --output=/home/ramanav/projects/rrg-amartel/ramanav/Projects/pytorch-CycleGAN-and-pix2pix/Results/pix2pix_sample_%j.log

# SBATCH --mail-user=vishwesh.ramanathan@mail.utoronto.ca # Where to send mail
# SBATCH --mail-type=ALL           # Enable email

# module load gcc opencv openslide python scipy-stack cuda cudnn
export WANDB_API_KEY=0255d24671ed1d02a9ff41388513f5b58097c7a0
module load python/3.8 openslide opencv cuda cudnn geos
source /home/ramanav/projects/rrg-amartel/ramanav/Projects/InkFilter/ink/bin/activate
export LD_LIBRARY_PATH=$EBROOTOPENSLIDE/lib

SOURCEDIR=/home/ramanav/projects/rrg-amartel/ramanav/Projects/pytorch-CycleGAN-and-pix2pix

cd $SLURM_TMPDIR
echo "Copying and Unziping..."
echo $SLURM_TMPDIR
#cp /home/ramanav/projects/rrg-amartel/Digital_Pathology/Public_Datasets/tiger-training-data/seg_patches.zip .
if [ -d $SLURM_TMPDIR/by_class ] 
then
    echo "Folder already exists"
else
    unzip -q ~/projects/rrg-amartel/Digital_Pathology/Public_Datasets/tiger-training-data/by_class.zip -d ./ &
    # unzip -q ~/projects/rrg-amartel/Digital_Pathology/Public_Datasets/tiger-training-data/ink_filter_tiger.zip -d ./
    cp -r ~/projects/rrg-amartel/Digital_Pathology/Public_Datasets/tiger-training-data/SSL_training ./
fi
echo "Beginning Training..."

cd $SOURCEDIR

# python train.py -l $SLURM_TMPDIR -c /home/ramanav/projects/rrg-amartel/ramanav/Projects/KI67/Classification/train_cell/config.yml
# python train.py -l $SLURM_TMPDIR -c /home/ramanav/projects/rrg-amartel/ramanav/Projects/KI67/Classification/train_cell/config_twoway.yml
python train.py --dataroot $SLURM_TMPDIR --name tiger_pix2pix_try --direction AtoB --dataset_mode tigerink --gpu_ids 0 --checkpoints_dir "/home/ramanav/projects/rrg-amartel/Digital_Pathology/Public_Datasets/tiger-training-data/Results/pix2pix/" --model pix2pix