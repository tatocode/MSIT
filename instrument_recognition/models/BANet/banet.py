import torch, torchvision
from torch import nn
from torchvision.models import resnet50, resnet101


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Banet(nn.Module):
    def __init__(self, num_classes, backbone) -> None:
        super(Banet, self).__init__()
        if backbone == "r101":
            res101 = resnet101(pretrained=True)
            moudules = list(res101.children())[:-1]
        elif backbone == "r50":
            res50 = resnet50(pretrained=True)
            moudules = list(res50.children())[:-1]
        self.backbone = nn.Sequential(*moudules)

        self.sa = SpatialAttention()
        self.ca = nn.ModuleList([ChannelAttention(i) for i in (256, 512, 1024, 2048)])

        self.conv = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.module_list = nn.ModuleList([nn.Linear(2048, 2048) for _ in range(3)])

        # atten
        self.batch_atten = nn.MultiheadAttention(2048, 8, 0.2, batch_first=True)

        self.final_fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
            nn.Softmax()
        )

    def forward_backbone(self, img):
        idx = 0
        for md in self.backbone:
            img = md(img)
            if isinstance(md, nn.Sequential):
                img = self.ca[idx](img.clone()) * img
                img = self.sa(img.clone()) * img
                idx += 1
        return img

    def forward(self, img):
        f1 = self.forward_backbone(img)  # [8, 2048, 1, 1]
        f2 = torch.flatten(f1, start_dim=1)  # [8, 2048]
        f3 = f2.unsqueeze(dim=0)  # [1, 8, 2048]
        query, key, value = [module(f3) for module in self.module_list]
        f3, _ = self.batch_atten(query, key, value)  # [1, 8, 2048]
        f3 = f3.squeeze(dim=0)  # [8, 2048]
        f4 = torch.cat([f2, f3], dim=1)  # [8, 4096]
        f4 = self.final_fc(f4)  # [8, 4]
        return f4


if __name__ == "__main__":
    banet = Banet(num_classes=4)

    img = torch.randn((8, 3, 224, 224))
    print(banet(img).shape)
