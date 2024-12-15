#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=20
bs=24
lr=0.000005
encoder=vits
dataset=nyu # vkitti hypersim
img_size=518
min_depth=0.001
max_depth=10 # 80 for virtual kitti 20 for hypersim
pretrained_from=../checkpoints/depth_anything_v2_${encoder}.pth
save_path=exp/nyu # exp/vkitti exp/hypersim

mkdir -p $save_path

python3 train_signal.py \
    --epochs $epoch \
    --encoder $encoder \
    --bs $bs \
    --lr $lr \
    --save-path $save_path \
    --dataset $dataset \
    --img-size $img_size \
    --min-depth $min_depth \
    --max-depth $max_depth \
    --pretrained-from $pretrained_from \
    2>&1 | tee -a $save_path/$now.log