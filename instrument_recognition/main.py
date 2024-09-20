import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CURL_CA_BUNDLE'] = ''
import torch
import argparse, random, numpy as np
import sys
import torchvision.models
from models.BANet.banet import Banet

import datasets.temp
from datasets.temp import InstrumentDataset
import albumentations as A
from torch.utils.data import DataLoader
from torch import nn
from tools.evaluate import MultiClassificationMetric
from tools.trainer import MyTrainer
from util.file import *
from util.functions import *
from tools.tester import MyTester
import timm

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_type', type=str, default='test', help='Training or testing')
    parser.add_argument('--type_name', type=str, default='banet_r101_train', help='Name the current training procedure')
    parser.add_argument('--data_root', type=str, default=r"C:\Users\Tatocode\Documents\desk\dataset\tool_recognition", help='Specifies where the data set is stored')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes to classification')
    parser.add_argument('--fold', type=int, default=4, help='Number of fold-th')

    parser.add_argument('--batch_size', type=int, default=4, help='Number of batch size')
    parser.add_argument('--num_worker', type=int, default=2, help='Number of threads reading data')
    parser.add_argument('--epoch_num', type=int, default=500, help='Number of epoch num')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Number of learning rate')
    parser.add_argument('--pre_trained', type=str, default='', help='Pre-train weight file location')

    return parser




if __name__ == '__main__':
    # parameter

    param = get_argparser().parse_args()


    # random seed
    seed = 2023
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministirc = True

    current_time = get_datetime_str()


    assert param.task_type in ['train', 'test'], '请使用 --task_type 可选参数指定当前任务为 train 或者 test'

     # process
    A_transform = A.Compose([
        # A.Resize(512, 512),
        A.Resize(224, 224),
        A.Flip(),
        # A.CenterCrop(256, 256)
    ])

    if param.task_type == 'train':
        if param.type_name.strip() == '':
            train_name = 'train'
        else:
            train_name = param.type_name.strip()
        assert param.data_root != '', '请使用 --data_root 可选参数指定数据集位置'
        data_root = param.data_root
        batch_size = param.batch_size
        num_worker = param.num_worker
        epoch_num = param.epoch_num
        learning_rate = param.learning_rate
        assert param.num_classes != 0, '请使用 --num_classes 可选参数指定要分类的类别数'
        num_classes = param.num_classes
        pre_trained = param.pre_trained


        train_dataset = InstrumentDataset(data_root=data_root, transform=A_transform, dataset_type='train')
        val_dataset = InstrumentDataset(data_root=data_root, transform=A_transform, dataset_type='val')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=False)

        if train_name.startswith("vgg"):
            model = torchvision.models.vgg19_bn(weights=torchvision.models.VGG19_BN_Weights)
            model.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, num_classes), nn.Softmax(dim=1))
        elif train_name.startswith("densenet"):
            model = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif train_name.startswith("resnet50"):
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif train_name.startswith("resnet101"):
            model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif train_name.startswith("vit"):
            model = timm.create_model(model_name="vit_base_patch16_384", pretrained=False, num_classes=4)
            model = torchvision.models.vit_b_16(pretrained=True)
            model.num_classes = num_classes
        elif train_name.startswith("swin"):
            model = torchvision.models.swin_b(weights=torchvision.models.Swin_B_Weights)
            model.num_classes = num_classes
        elif train_name.startswith("banet"):
            model = Banet(num_classes, backbone=train_name.split("_")[1])
        else:
            sys.exit()

        # 加载 pre_trained 权重
        if pre_trained != '':
            model.load_state_dict(torch.load(pre_trained))

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.65)
        criterion = nn.CrossEntropyLoss()
        metric = MultiClassificationMetric(num_classes, ["scissors", "forceps", "gauze", "kidney dish"], normalize=False)

        train_config_dict = {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'epoch_num': epoch_num,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'criterion': criterion,
            'metric': metric,
            'type': 'binary' if param.num_classes == 2 else 'multi'
        }

        trainer = MyTrainer(train_config_dict)


        home_dir = os.path.join('result', train_name, current_time)
        make_dir(home_dir)

        trainer.train(home_dir)

    else:
        if param.type_name.strip() == '':
            test_name = 'test'
        else:
            test_name = param.type_name.strip()
        assert param.data_root != '', '请使用 --data_root 可选参数指定数据集位置'
        data_root = param.data_root
        batch_size = param.batch_size
        num_worker = param.num_worker
        assert param.num_classes != 0, '请使用 --num_classes 可选参数指定要分割的类别数'
        num_classes = param.num_classes
        pre_trained = param.pre_trained.strip()
        metric = MultiClassificationMetric(num_classes, ['Scissors', 'Forceps', 'Gauze', 'Kidney-dish'], normalize=True)

        test_dataset = InstrumentDataset(data_root=data_root, transform=A_transform, is_test=True, dataset_type='val')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=False)

        if test_name.startswith("vgg"):
            model = torchvision.models.vgg19_bn()
            model.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, num_classes), nn.Softmax(dim=1))
        elif test_name.startswith("densenet"):
            model = torchvision.models.densenet201()
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif test_name.startswith("resnet50"):
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif test_name.startswith("resnet101"):
            model = torchvision.models.resnet101()
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif test_name.startswith("vit"):
            model = timm.create_model(model_name="vit_base_patch16_384", pretrained=False, num_classes=4)
            model = torchvision.models.vit_b_16(pretrained=True)
            model.num_classes = num_classes
        elif test_name.startswith("swin"):
            model = torchvision.models.swin_b()
            model.num_classes = num_classes
        elif test_name.startswith("banet"):
            model = Banet(num_classes, backbone=test_name.split("_")[1])
        else:
            sys.exit()

        assert pre_trained != '', f'测试时需使用 --pre_trained 可选参数传入训练好的权重文件'
        # 加载 pre_trained 权重
        model.load_state_dict(torch.load(pre_trained, map_location='cuda' if torch.cuda.is_available() else 'cpu'))

        test_config_dict = {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'model': model,
            'test_loader': test_loader,
            'metric': metric,
            'type': 'binary' if param.num_classes == 2 else 'multi'
        }

        tester = MyTester(test_config_dict)

        home_dir = os.path.join('result', test_name, get_datetime_str())
        make_dir(home_dir)

        tester.test(home_dir)


