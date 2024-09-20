import torchvision.transforms as T
import cv2, os, torch
import albumentations as A
from torch import nn
import torchvision.models

if __name__ == '__main__':

    IMAGE_PATH = r'Desktop/9.png'

    A_transform = A.Compose([
        A.Resize(512, 512),
        A.Flip(),
        A.CenterCrop(256, 256)
    ])

    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    T_image = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    image = cv2.imread(IMAGE_PATH, 1)

    inp = T_image(A_transform(image=image)['image']).unsqueeze(dim=0)
    print(inp.shape)

    model = torchvision.models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 4)

    model.load_state_dict(torch.load(r'result/r50_train/230725_120015/checkpoint/best.pth'))

    model.eval()
    with torch.no_grad():
        out = model(inp).cpu()
        print(out[0])
        prediction = torch.argmax(out, dim=1)
        # print(prediction.shape)
        if prediction[0] == 0:
            print('剪刀')
        elif prediction[0] == 1:
            print('镊子')
        elif prediction[0] == 2:
            print('纱布')
        else:
            print('弯盘')