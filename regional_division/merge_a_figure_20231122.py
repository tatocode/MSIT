import os
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    origin = "for_plot/images"
    gt = "temp2/mask"
    ours = "temp2/Polynet"
    fcn = "temp2/FCN"
    setr = "temp2/SETR"
    unet = "temp2/UNet"
    pspnet = "temp2/PSPNet"
    segnet = "temp2/SegNet"
    deeplabv3 = "temp2/DeepLabv3+"
    refinenet = "temp2/RefineNet"

    # all_dir = [origin, gt, ours, unet, pspnet, segnet, deeplabv3, refinenet]
    all_dir = [origin, gt, ours, fcn, unet, segnet, deeplabv3]

    plt.figure(figsize=(30, 14))
    for col in range(7):
        img_dir = all_dir[col]
        for idx, img_name in enumerate(sorted(os.listdir(img_dir))):
            plt.subplot(6, 7, 7 * idx + col + 1)
            plt.axis("off")
            plt.imshow(cv2.imread(os.path.join(img_dir, img_name))[:, :, ::-1])
    plt.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.01, hspace=0.01)  # 调整子图间距
    plt.savefig("all2.png")
