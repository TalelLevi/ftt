#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ffcv

# Run some
#python main.py

#python fast_cnn/train_imagenet.py --config-file fast_cnn/rn18_configs/rn18_16_epochs.yaml --data.train_dataset=/home/talel-levi/datasets/imagenet/igor/train_500_0.50_90.ffcv --data.val_dataset=/home/talel-levi/datasets/imagenet/val_500_0.5_90.ffcv --data.num_workers=2 --data.in_memory=0 --logging.folder=/home/talel-levi/fft/log_folder

python main.py --config-file fast_cnn/rn18_configs/rn18_16_epochs.yaml --data.train_dataset=/home/talel-levi/datasets/imagenet/igor/train_500_0.50_90.ffcv --data.val_dataset=/home/talel-levi/datasets/imagenet/val_500_0.5_90.ffcv --data.num_workers=2 --data.in_memory=0 --logging.folder=/home/talel-levi/fft/log_folder