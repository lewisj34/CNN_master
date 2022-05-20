#!/bin/sh
python train.py --model_name NewFusionNetwork --batch_size 2 --speed_test True
python train.py --model_name new_fusion_zed --batch_size 2 --speed_test True 
python train.py --model_name NewFusionNetwork --batch_size 2 --speed_test True --image_height 512 --image_width 512
python train.py --model_name new_fusion_zed --batch_size 2 --speed_test True --image_height 512 --image_width 512