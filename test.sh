#!/bin/bash

echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python solve.py \
  --img_size 768 \
  --img_path samples/imagenet/person.JPEG \
  --prompt "a photo of a closed face" \
  --task colorization\
  --deg_scale 12 \
  --efficient_memory



