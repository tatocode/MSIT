import numpy as np
import torch, json
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import albumentations as A

CMAP = np.array([
    [0, 0, 0],
    [255, 255, 255]
])

IMAGE_SIZE = 512

class TempDataset(Dataset):
    """
    data_root:
        images:
            img00001.png
            img00002.png
            ...
        masks:
            img00001.png
            img00002.png
            ...
        annos:
            img00001.json
            img00002.json
            ...
        fold_0.txt
        fold_1.txt
        fold_2.txt
        fold_3.txt
        test.txt
    """

    def __init__(self, data_root: str, transform, dataset_type: str, fold: int, normalization_tensor: bool = True,
                 mean=None, std=None) -> None:
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]

        self.images_dir = Path(data_root) / 'images'
        self.masks_dir = Path(data_root) / 'masks'
        self.annos_dir = Path(data_root) / 'annos'

        self.transform = transform
        self.dataset_type = dataset_type
        self.have_nt = normalization_tensor
        self.mean = mean
        self.std = std


        if dataset_type in ['train', 'val']:
            if dataset_type == 'train':
                data_idx = [0, 1, 2, 3]
                data_idx.remove(fold)
            else:
                data_idx = [fold]

            self.data = []
            for di in data_idx:
                fold_file_path = Path(data_root) / f'fold_{di}.txt'
                with fold_file_path.open('r') as f:
                    self.data += [i.strip() for i in f.readlines()]
        elif dataset_type == 'test':
            file_path = Path(data_root) / r'test.txt'
            with file_path.open('r') as f:
                self.data = [i.strip() for i in f.readlines()]

        else:
            raise Exception(f'dataset_type 的值应在 "train", "val", "test" 中, 现传入 {dataset_type}')

    def __len__(self):
        return len(self.data)
    
    def adjust_position(self, vertices):
        # 1   2
        # 4   3
        # print(len(vertices))
        ret = [0]*4
        vertices.sort(key=lambda x:x[0]**2 + x[1]**2)
        ret[0] = vertices[0]
        ret[1], ret[3] = (vertices[1], vertices[2]) if vertices[1][0] > vertices[2][0] else (vertices[2], vertices[1])
        ret[2] = vertices[3]
        return ret
    
    def make_tgt_token(self, vertices):
        ret = [0]
        for item in vertices:
            ret.append(item[0] * IMAGE_SIZE + item[1] + 2)
        return ret
    
    def make_gt_token(self, vertices):
        ret = [[item[0] / IMAGE_SIZE, item[1] / IMAGE_SIZE] for item in vertices]
        # print(f'ret: {ret}')
        return ret

    def __getitem__(self, item):
        if self.dataset_type in ['train', 'val']:
            image_path = self.images_dir / (self.data[item]+'.png')
            mask_path = self.masks_dir / (self.data[item]+'.png')
            anno_path = self.annos_dir / (self.data[item]+'.json')

            image = np.array(Image.open(image_path))
            mask = np.array(Image.open(mask_path))

            with anno_path.open('r') as f:
                vertices = json.load(f)['shapes'][0]['points']
            assert len(vertices) == 4

            # print(f'vertices: {vertices}')

            assert self.transform is not None, f'在 train 或 val 时不能传入为 None 的数据增强方案'

            comm_transform = self.transform(image=image, mask=mask, keypoints=vertices)

            image_transform, mask_transform, vertices = comm_transform['image'], comm_transform['mask'], comm_transform['keypoints']

            # print(f'vertices: {vertices}')
            vertices = self.adjust_position(vertices)


            if self.have_nt:
                T_image = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=self.mean, std=self.std)
                ])
                return T_image(image_transform).type(torch.float32), torch.from_numpy(mask_transform).type(torch.long), torch.tensor(self.make_tgt_token(vertices)).type(torch.int32), torch.tensor(self.make_gt_token(vertices)).type(torch.float32)
            else:
                return image_transform.type(torch.float32), mask_transform.type(torch.long),  torch.from_numpy(mask_transform).type(torch.long), torch.tensor(self.make_tgt_token(vertices)).type(torch.int32), torch.tensor(self.make_gt_token(vertices)).type(torch.float32)

        elif self.dataset_type == 'test':
            image_path = self.images_dir / (self.data[item]+'.png')
            mask_path = self.masks_dir / (self.data[item]+'.png')
            anno_path = self.annos_dir / (self.data[item]+'.json')

            image = np.array(Image.open(image_path))
            mask = np.array(Image.open(mask_path))

            with anno_path.open('r') as f:
                vertices = json.load(f)['shapes'][0]['points']
            assert len(vertices) == 4
            
            # print(len(vertices), self.data[item])

            if self.transform is None:
                T_image = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=self.mean, std=self.std)
                ])
                return T_image(image).type(torch.float32), torch.from_numpy(mask).type(torch.long),  torch.from_numpy(mask_transform).type(torch.long), torch.tensor(self.make_tgt_token(vertices)).type(torch.int32), torch.tensor(self.make_gt_token(vertices)).type(torch.float32)
            else:
                comm_transform = self.transform(image=image, mask=mask, keypoints=vertices)
                image_transform = comm_transform['image']
                mask_transform = comm_transform['mask']
                vertices = comm_transform['keypoints']
                vertices = self.adjust_position(vertices)

                T_image = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=self.mean, std=self.std)
                ])
                return T_image(image_transform).type(torch.float32), torch.from_numpy(mask_transform).type(torch.long),  torch.from_numpy(mask_transform).type(torch.long), torch.tensor(self.make_tgt_token(vertices)).type(torch.int32), torch.tensor(self.make_gt_token(vertices)).type(torch.float32)
        else:
            raise Exception(f'dataset_type 的值应在 "train", "val", "test" 中, 现传入 {self.dataset_type}')


# 测试功能
if __name__ == '__main__':
    A_transform = A.Compose([
        A.Resize(512, 512),
        # A.Flip(),
    ],  keypoint_params=A.KeypointParams(format='xy'))
    ds_train = TempDataset(r'/home/dell/tao/dataset_20230630/regional_division', A_transform, 'train', 0, True)
    ds_val = TempDataset(r'/home/dell/tao/dataset_20230630/regional_division', A_transform, 'val', 0, True)
    ds_test = TempDataset(r'/home/dell/tao/dataset_20230630/regional_division', A_transform, 'test', 0, True)

    print(f'>>> {len(ds_train), len(ds_val), len(ds_test)}')
    print(f'>>> {ds_train[0][0].shape}, {ds_train[0][1].shape} {ds_train[0][2].shape}, {ds_val[0][0].shape}, {ds_val[0][1].shape}, {ds_val[0][2].shape}, {ds_test[0][0].shape}, {ds_test[0][1].shape}, {ds_test[0][2].shape}')
    print(f'>>> {ds_train[0][0]}, {ds_train[0][1]}, {ds_train[0][2]}, {ds_val[0][0]}, {ds_val[0][1]}, {ds_val[0][2]}, {ds_test[0][0]}, {ds_test[0][1]}, {ds_test[0][2]}')
