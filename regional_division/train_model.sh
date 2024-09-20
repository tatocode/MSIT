#!/bin/bash
# FCN
python -W ignore main.py --task_type train --type_name train-FCN --model FCN
# UNet
python -W ignore main.py --task_type train --type_name train-UNet --model UNet
# SegNet
python -W ignore main.py --task_type train --type_name train-SegNet --model SegNet
# PSPNet
python -W ignore main.py --task_type train --type_name train-PSPNet --model PSPNet
# RefineNet
python -W ignore main.py --task_type train --type_name train-RefineNet --model RefineNet
# DeepLabv3+
python -W ignore main.py --task_type train --type_name train-DeepLabv3+ --model DeepLabv3+
# SETR
python -W ignore main.py --task_type train --type_name train-SETR --model SETR