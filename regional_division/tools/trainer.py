import tools.logging as logging
import os
import random, numpy as np, cv2

# import torchvision.util
# import draw
import torch
from util.file import *
from util.data_process import *
from util.draw import *
import torch.nn.functional as F

IMAGE_SIZE = 512

class MyTrainer:
    """
    train_config_dict:
        device: 使用 GPU 还是 CPU
        model: 网络结构
        train_loader: 训练数据 loader
        val_loader: 验证数据 loader
        epoch_num: 迭代次数
        optimizer: 优化器
        scheduler: 调度器
        criterion: 损失函数
        metric: 验证集上评价
    """

    def __init__(self, train_config_dict: dict):
        self.device = train_config_dict['device']
        self.model = train_config_dict['model'].to(self.device)
        self.train_loader = train_config_dict['train_loader']
        self.val_loader = train_config_dict['val_loader']
        self.epoch_num = train_config_dict['epoch_num']
        self.optimizer = train_config_dict['optimizer']
        self.scheduler = train_config_dict['scheduler']
        self.criterion = train_config_dict['criterion']
        self.metric = train_config_dict['metric']
        self.model_name = train_config_dict['model_name']


    def rebuild_mask(self, prediction, img_size=IMAGE_SIZE):
        stack = []
        for idx in range(len(prediction)):
            prediction[prediction>=1] = 1
            prediction[prediction<=0] = 0
            bg = np.zeros((img_size, img_size), dtype=np.int32)
            contour = (prediction[idx]*img_size).numpy().astype(np.int32)
            # print(f'contour type: {type(contour)}, contour shape: {contour.shape}')
            img = cv2.fillConvexPoly(bg, contour, 1)
            stack.append(torch.from_numpy(img))
        return torch.stack(stack, dim=0)

    def train(self, home_dir, cmap, mean=None, std=None):
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        print(f'>>> {self.device} 可用!')

        checkpoint_dir = os.path.join(home_dir, 'checkpoint')
        save_val_dir = os.path.join(home_dir, 'save_val')

        make_dir(home_dir)
        make_dir(checkpoint_dir)
        make_dir(save_val_dir)

        logger = logging.getLogger(home_dir)

        if self.model_name == 'PolyNet':
            record = Record(self.epoch_num, ['L1', 'mAcc', 'mIoU', 'FWIoU', 'Dice'], os.path.join(home_dir, r'record.jpg'))
        else:
            record = Record(self.epoch_num, ['mAcc', 'mIoU', 'FWIoU', 'Dice'], os.path.join(home_dir, r'record.jpg'))

        # train
        mIoU_best = 0
        for epoch in range(self.epoch_num):
            self.model.train()

            loss_epoch = 0
            time_interval = 0
            for image, mask, tgt_token, gt_token in self.train_loader:

                image, mask = image.to(self.device, dtype=torch.float32), mask.to(self.device, dtype=torch.long)

                self.optimizer.zero_grad()
                if self.model_name == 'PSPNet':
                    prediction, _ = self.model(image)
                elif self.model_name == 'SETR':
                    prediction,_ = self.model(image, [1, 4, 8, 12])
                elif self.model_name == 'PolyNet':
                    tgt_token, gt_token = tgt_token.to(self.device), gt_token.to(self.device) 
                    mask_out, point_out = self.model(image)
                    # print(f'prediction shape: {prediction.shape}, prediction: {prediction}')
                else:
                    prediction = self.model(image)
                # print(f'prediction: {prediction.shape}') # [B, 5, 2]
                if self.model_name == 'PolyNet':
                    # print(f'prediction: {prediction[0]}, gt: {gt_token}')
                    loss = self.criterion(mask_out, mask, point_out, gt_token, epoch, self.epoch_num)
                else:
                    loss = self.criterion(prediction, mask)
                loss.backward()
                self.optimizer.step()

                loss_np = loss.detach().cpu().numpy()
                loss_epoch += loss_np.item()
                time_interval += 1

            loss_mean = loss_epoch/time_interval
            logger.info(f'train | epoch: {epoch + 1}\t\t\t loss: {loss_mean:.6f}')

            self.model.eval()
            with torch.no_grad():
                if self.model_name == 'PolyNet':
                    L1_sum = 0
                mAcc_sum = 0
                mIoU_sum = 0
                FWIoU_sum = 0
                Dice_sum = 0
                time_interval = 0
                for image, mask, tgt_token, gt_token in self.val_loader:
                    image, mask = image.to(self.device, dtype=torch.float32), mask.to(self.device, dtype=torch.long)
                    if self.model_name == 'PSPNet':
                        prediction, _ = self.model(image)
                        prediction = torch.argmax(prediction.cpu(), dim=1)
                    elif self.model_name == 'SETR':
                        prediction, _ = self.model(image, [1, 4, 8, 12])
                        prediction = torch.argmax(prediction.cpu(), dim=1)
                    elif self.model_name == 'PolyNet':
                        tgt_token, gt_token = tgt_token.to(self.device), gt_token.to(self.device) 
                        mask_out, point_out = self.model(image)
                        prediction = point_out
                    else:
                        prediction = torch.argmax(self.model(image).cpu(), dim=1)
                    
                    if self.model_name == 'PolyNet':
                        L1_sum += F.l1_loss(prediction, gt_token).cpu()
                        rebuild = self.rebuild_mask(prediction.cpu())
                        # print(f'mask shape: {mask.shape}, mask type: {type(mask)}, mask[0]:\n{mask[0]}')
                        self.metric.addBatch(rebuild, mask.cpu())
                    else:
                        self.metric.addBatch(prediction, mask.cpu())
                    mAcc_sum += self.metric.meanPixelAccuracy()
                    mIoU_sum += self.metric.meanIntersectionOverUnion()
                    FWIoU_sum += self.metric.Frequency_Weighted_Intersection_over_Union()
                    Dice_sum += self.metric.Dice_score()
                    self.metric.reset()

                    if epoch % 20 == 0:
                        # 画图
                        idx_random = random.randint(0, image.shape[0]-1)

                        i = de_normalization(image[idx_random].cpu(), mean=mean, std=std).numpy().transpose(1, 2, 0)*255
                        # print(f'i type: {type(i)}, i shape: {i.shape}')
                        m = mask[idx_random].cpu().numpy()
                        if self.model_name == 'PolyNet':
                            p = rebuild[idx_random].cpu().numpy()
                        else:
                            p = prediction[idx_random].cpu().numpy()

                        save_val_prediction(i, m, p, cmap, os.path.join(save_val_dir, f'epoch_{epoch+1}.jpg'))
                    time_interval += 1

                if self.model_name == 'PolyNet':
                    L1_mean = L1_sum/time_interval
                mAcc_mean = mAcc_sum/time_interval
                mIoU_mean = mIoU_sum/time_interval
                FWIoU_mean = FWIoU_sum/time_interval
                Dice_mean = Dice_sum/time_interval

                if self.model_name == 'PolyNet':
                    logger.info(f' val  | epoch: {epoch + 1}\t\t\t L1:{L1_mean:.6f}, mAcc: {mAcc_mean:.6f}, mIoU: {mIoU_mean:.6f}, FWIoU: {FWIoU_mean:.6f}, Dice: {Dice_mean:.6f}')

                else:
                    logger.info(f' val  | epoch: {epoch + 1}\t\t\t mAcc: {mAcc_mean:.6f}, mIoU: {mIoU_mean:.6f}, FWIoU: {FWIoU_mean:.6f}, Dice: {Dice_mean:.6f}')

                torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'last.pth'))
                if mIoU_mean > mIoU_best:
                    mIoU_best = mIoU_mean
                    torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, f'best.pth'))

            self.scheduler.step()

            # 记录训练过程并画图
            if self.model_name == 'PolyNet':
                metric_dict = {
                    'L1': L1_mean,
                    'mAcc': mAcc_mean,
                    'mIoU': mIoU_mean,
                    'FWIoU': FWIoU_mean,
                    'Dice': Dice_mean
                }
            else:
                metric_dict = {
                    'mAcc': mAcc_mean,
                    'mIoU': mIoU_mean,
                    'FWIoU': FWIoU_mean,
                    'Dice': Dice_mean
                }



            record.add_data_mean(loss_mean, metric_dict)
            record.draw()
        del logger


