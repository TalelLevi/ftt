#!/bin/bash

export WRITE_DIR=/home/talel-levi/fft/data/ffcv/imagenette2
export DATA_DIR=/home/talel-levi/fft/data/imagenette2

write_dataset () {
    write_path=$WRITE_DIR/${1}_${2}_${3}_${4}.ffcv
    echo "Writing ImageNet ${1} dataset to ${write_path}"
    python write_imagenet.py \
        --cfg.dataset=imagenet \
        --cfg.split=${1} \
        --cfg.data_dir=$DATA_DIR/${1} \
        --cfg.write_path=$write_path \
        --cfg.max_resolution=${2} \
        --cfg.write_mode=proportion \
        --cfg.compress_probability=${3} \
        --cfg.jpeg_quality=$4
}

write_dataset train $1 $2 $3
write_dataset val $1 $2 $3
