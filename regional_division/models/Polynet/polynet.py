import warnings
warnings.filterwarnings('ignore')  # 不要输出警告信息
import torch
from torch import nn
# from torchvision import models
# from timm.models.swin_transformer import swin_base_patch4_window12_384  # window_size=8 depth=[2,2,6,2]
import numpy as np
from models.Polynet.swin import SwinTransformer
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channel, out_channel)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class PolyNet(nn.Module):
    def __init__(self, img_size=384, d_model=768, n_head=4, num_layers=6, out_indices=[0, 1, 2, 3]) -> None:
        super(PolyNet, self).__init__()
        self.img_size = img_size
        # self.encoder = swin_base_patch4_window12_384(pretrained=True, features_only=True,
        #                                             out_indices=out_indices)  # out_indices=[3]，仅输出swin transformer的最后一个子模块的输出
        
        self.encoder = SwinTransformer(pretrain_img_size=384, window_size=12, embed_dim=128,
                                                out_indices=[0, 1, 2, 3],
                                                depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.encoder.init_weights(pretrained='models/Polynet/swin_base_patch4_window12_384_22k.pth')
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 2, 1),
            nn.Softmax()
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, 3, 1, 1),
        )
        self.ph_row = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(16, 4),
            nn.Sigmoid()
        )

        self.ph_column = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(16, 4),
            nn.Sigmoid()
        )
        

    def forward(self, img):
        x = self.encoder(img)
        x0, x1, x2, x3 = x['layer0'], x['layer1'], x['layer2'], x['layer3']
        # print(x0.shape, x1.shape, x2.shape, x3.shape)
        o = self.up1(x3, x2)
        o = self.up2(o, x1)
        o = self.up3(o, x0)
        seg_out = self.seg_head(o)
        t = self.conv_block(x3)
        t = torch.flatten(t, 1)
        point_out = torch.stack([self.ph_row(t), self.ph_column(t)], dim=2)
        return seg_out, point_out



if __name__ == '__main__':
    model = PolyNet()
    img = torch.randn(4, 3, 224, 224)
    pos_lab = torch.abs(torch.randn(4, 4, 2) * 10).int()
    # print(pos_lab)
    pos_mask = torch.from_numpy(np.triu(np.ones(4), k=1).astype('uint8')) == 1
    out = model(img, pos_lab)
    print(out.shape)
    print(out)