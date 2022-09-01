#
# --------------------------------------------------------------------------------------------------------------------------
# Created on Wed Jun 23 2022 at University of Toronto
#
# Author: Vishwesh Ramanathan
# Email: vishwesh.ramanathan@mail.utoronto.ca
# Description: This script shows example of how to run training on Ki67 using the modules
# Modifications (date, what was modified):
#   1. Code based on example from MNIST example code
# --------------------------------------------------------------------------------------------------------------------------
#
import sys

from pathlib import Path
import yaml
import argparse

import utils
import trainer

parser = argparse.ArgumentParser()
parser.add_argument("-c", help="config location",required=True)
parser.add_argument("-l", help="location modifier for compute canada",required=True)
args = parser.parse_args()

config_path = args.c
location_mod = args.l

# config_path = "/home/ramanav/projects/rrg-amartel/ramanav/Projects/InkFilter/config.yml"
# location_mod = "/localscratch/ramanav.38885233.0"

mnist_trainer = utils.TrainEngine(config_pth=config_path,location_mod=location_mod)
mnist_trainer.run()
