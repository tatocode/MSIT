import tools.logging as logging
import logging as lgging
from util.draw import *
from util.file import *


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
        self.type = train_config_dict['type']

    def train(self, home_dir):
        print(f'>>> {self.device} 可用!')

        checkpoint_dir = os.path.join(home_dir, 'checkpoint')
        save_val_dir = os.path.join(home_dir, 'save_val')

        make_dir(home_dir)
        make_dir(checkpoint_dir)
        make_dir(save_val_dir)

        logger = logging.getLogger(home_dir)

        if self.type == 'binary':
            record = Record(self.epoch_num, ['mAcc', 'Precision', 'Recall', 'F1-score'],
                            os.path.join(home_dir, r'record.jpg'))
        else:
            record = Record(self.epoch_num, ['mAcc'],
                            os.path.join(home_dir, r'record.jpg'))

        # train
        mAcc_best = 0
        for epoch in range(self.epoch_num):
            self.model.train()

            loss_epoch = 0
            time_interval = 0
            for image, label in self.train_loader:
                image, label = image.to(self.device, dtype=torch.float32), label.to(self.device, dtype=torch.long)

                self.optimizer.zero_grad()
                prediction = self.model(image)
                loss = self.criterion(prediction, label)
                loss.backward()
                self.optimizer.step()

                loss_np = loss.detach().cpu().numpy()
                loss_epoch += loss_np.item()
                time_interval += 1

            loss_mean = loss_epoch / time_interval
            logger.info(f'train | epoch: {epoch + 1}\t\t\t loss: {loss_mean:.4f}')

            self.model.eval()
            with torch.no_grad():
                Acc_sum = 0
                Precision_sum = 0
                Recall_sum = 0
                F1_score_sum = 0
                time_interval = 0
                for image, label in self.val_loader:
                    image, label = image.to(self.device, dtype=torch.float32), label.to(self.device, dtype=torch.long)
                    prediction = torch.argmax(self.model(image).cpu(), dim=1)
                    # print(label, prediction)
                    self.metric.addBatch(prediction, label.cpu())
                    Acc_sum += self.metric.getAcc()
                    if self.type == 'binary':
                        Precision_sum += self.metric.getPrecision()
                        Recall_sum += self.metric.getRecall()
                        F1_score_sum += self.metric.getF1Score()
                    self.metric.reset()

                    time_interval += 1

                mAcc_mean = Acc_sum / time_interval
                if self.type == 'binary':
                    Precision_mean = Precision_sum / time_interval
                    Recall_mean = Recall_sum / time_interval
                    F1_score_mean = F1_score_sum / time_interval

                if self.type == 'binary':
                    logger.info(
                        f' val  | epoch: {epoch + 1}\t\t\t mAcc: {mAcc_mean:.2f}, Precision: {Precision_mean:.2f}, Recall: {Recall_mean:.2f}, F1-Score: {F1_score_mean:.2f}')
                else:
                    logger.info(f' val  | epoch: {epoch + 1}\t\t\t mAcc: {mAcc_mean:.2f}')

                torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'last.pth'))
                if mAcc_mean > mAcc_best:
                    mAcc_best = mAcc_mean
                    torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))

            self.scheduler.step()

            # 记录训练过程并画图
            if self.type == 'binary':
                metric_dict = {
                    'mAcc': mAcc_mean,
                    'Precision': Precision_mean,
                    'Recall': Recall_mean,
                    'F1-Score': F1_score_mean,
                }
            else:
                metric_dict = {
                    'mAcc': mAcc_mean,
                }
            record.add_data_mean(loss_mean, metric_dict)
            record.draw()
        lgging.shutdown()