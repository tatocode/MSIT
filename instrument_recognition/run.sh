#!/bin/bash

python main.py --type_name vgg19bn_test --pre_train result/vgg19bn_train/231010_225146/checkpoint/best.pth
python main.py --type_name densenet201_test --pre_train result/densenet201_train/231010_230139/checkpoint/best.pth
python main.py --type_name resnet50_test --pre_train result/r50_train/230727_143125/checkpoint/best.pth
python main.py --type_name resnet101_test --pre_train result/r101_train/230914_172855/checkpoint/best.pth
python main.py --type_name vit_test --pre_train result/vit_train/231010_205424/checkpoint/best.pth
python main.py --type_name swin_test --pre_train result/swin_train/231010_223916/checkpoint/best.pth
python main.py --type_name banet_r50_test --pre_train result/banet_r50_train/231106_123630/checkpoint/best.pth
python main.py --type_name banet_r101_test --pre_train result/banet_r101_train/231106_154632/checkpoint/best.pth
