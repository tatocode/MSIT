import torch
import argparse
import datasets.dataset
from datasets.dataset import TempDataset
import albumentations as A
from torch.utils.data import DataLoader
from models.UNet.unet_model import UNet
from models.FCN.fcn import *
# from models.maskrcnn import maskrcnn_resnet50
from models.PSPNet.pspnet import PSPNet
from models.segnet.segnet import SegNet
from models.refinenet.refinenet.refinenet_4cascade import RefineNet4Cascade
from models.deeplabv3plus import modeling
from models.SETR import SETR
from models.Polynet.polynet import PolyNet
from torch import nn
from tools.evaluate import SegmentationMetric
from tools.trainer import MyTrainer
from util.file import *
from util.functions import *
from tools.tester import MyTester
import random, math

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_type', type=str, default='', help='Training or testing')
    parser.add_argument('--type_name', type=str, default='', help='Name the current training procedure')
    parser.add_argument('--model', type=str, default='', help='Which model will be used')
    parser.add_argument('--data_root', type=str, default='/home/dell/tao/dataset_20230630/regional_division', help='Specifies where the data set is stored')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes to segmentation')
    parser.add_argument('--fold', type=int, default=4, help='The fold number of Cross Validation')

    parser.add_argument('--batch_size', type=int, default=4, help='Number of batch size')
    parser.add_argument('--num_worker', type=int, default=2, help='Number of threads reading data')
    parser.add_argument('--epoch_num', type=int, default=200, help='Number of epoch num')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='Number of learning rate')
    parser.add_argument('--pre_trained', type=str, default='', help='Pre-train weight file location')

    return parser


IMAGE_SIZE = 512

def my_loss(mask_out, mask, point_out, point, current_epoch, all_epoch):
    # print(mask_out.shape, mask.shape)
    CE = nn.CrossEntropyLoss()(mask_out, mask)
    L1 = nn.L1Loss()(point_out, point)
    time = current_epoch/all_epoch
    return (1-time)*CE + time*L1



if __name__ == '__main__':

    # 随机数种子
    SEED = 2023
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)




    # parameter

    param = get_argparser().parse_args()

    assert param.task_type in ['train', 'test'], '请使用 --task_type 可选参数指定当前任务为 train 或者 test'

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
        assert param.num_classes != 0, '请使用 --num_classes 可选参数指定要分割的类别数'
        num_classes = param.num_classes
        pre_trained = param.pre_trained
        current_time = get_datetime_str()

        # process
        A_transform = A.Compose([
            # A.Flip(),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(IMAGE_SIZE , IMAGE_SIZE ),
        ], keypoint_params=A.KeypointParams('xy'))

        for fold in range(param.fold):
            print('******************************')
            print(f'{param.model} Fold {fold}')
            print('******************************')

            train_dataset = TempDataset(data_root=data_root, transform=A_transform, fold=fold, dataset_type='train')
            val_dataset = TempDataset(data_root=data_root, transform=A_transform, fold=fold, dataset_type='val')

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=False)

            if param.model == 'UNet':
                model = UNet(3, num_classes)
            elif param.model == 'FCN':
                vgg_model = VGGNet(requires_grad=True)
                model = FCN8s(pretrained_net=vgg_model, n_class=num_classes)
            elif param.model == 'SegNet':
                model = SegNet(3, num_classes)
            elif param.model == 'PSPNet':
                model = PSPNet(n_classes=num_classes, backend='resnet50', pretrained=False)
            elif param.model == 'RefineNet':
                model = RefineNet4Cascade(input_shape=(3, 512), num_classes=num_classes)
            elif param.model == 'DeepLabv3+':
                model = modeling.deeplabv3plus_resnet101(2, 16)
            elif param.model == 'SETR':
                _, model = SETR.SETR_MLA_S(dataset='desk')
            elif param.model == 'PolyNet':
                model = PolyNet(img_size=IMAGE_SIZE )
            else:
                raise Exception(f'Mismatch this model: {param.model}')

            # 加载 pre_trained 权重
            if pre_trained != '':
                model.load_state_dict(torch.load(pre_trained))

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.65)
            if param.model == 'PolyNet':
                # criterion = nn.MSELoss()
                criterion = my_loss
            else:
                criterion = nn.CrossEntropyLoss()
            
            metric = SegmentationMetric(num_classes)

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
                'model_name': param.model
            }

            trainer = MyTrainer(train_config_dict)

            home_dir = os.path.join('result', train_name, current_time, str(fold))
            make_dir(home_dir)

            trainer.train(home_dir, datasets.dataset.CMAP)

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

        test_dataset = TempDataset(data_root=data_root, transform=A.Resize(IMAGE_SIZE , IMAGE_SIZE ), dataset_type='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=False)

        # model = UNet(3, num_classes)

        # vgg_model = VGGNet(requires_grad=True)
        # model = FCN8s(pretrained_net=vgg_model, n_class=num_classes)

        # model = SegNet(3, num_classes)

        # model = PSPNet(n_classes=num_classes, backend='resnet50', pretrained=False)

        # model = RefineNet4Cascade(input_shape=(3, 512), num_classes=num_classes)

        # model = modeling.deeplabv3plus_resnet101(2, 16, False)

        # _, model = SETR.SETR_MLA_S(dataset='desk')

        model = PolyNet(img_size=512)


        assert pre_trained != '', f'测试时需使用 --pre_trained 可选参数传入训练好的权重文件'
        # 加载 pre_trained 权重
        model.load_state_dict(torch.load(pre_trained))

        metric = SegmentationMetric(num_classes)

        test_config_dict = {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'model': model,
            'test_loader': test_loader,
            'metric': metric,
        }

        tester = MyTester(test_config_dict)

        home_dir = os.path.join('result', test_name, get_datetime_str())
        make_dir(home_dir)

        tester.test(home_dir, datasets.dataset.CMAP)


