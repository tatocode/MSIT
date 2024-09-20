from util.file import *
from util.data_process import *
from util.draw import *
from tqdm import tqdm


class MyTester:
    """
    test_config_dict:
        device: 使用 GPU 还是 CPU
        model: 网络结构
        test_loader: 测试数据 loader
    """

    def __init__(self, test_config_dict: dict):
        self.device = test_config_dict['device']
        self.model = test_config_dict['model'].to(self.device)
        self.test_loader = test_config_dict['test_loader']
        self.metric = test_config_dict['metric']

    def test(self, home_dir, cmap, mean=None, std=None):
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        print(f'>>> {self.device} 可用!')

        make_dir(home_dir)

        # test
        self.model.eval()
        with torch.no_grad():
            mAcc_sum = 0
            mIoU_sum = 0
            FWIoU_sum = 0
            times = 0

            for image, mask in tqdm(self.test_loader):
                image = image.to(self.device, dtype=torch.float32)
                # prediction = torch.argmax(self.model(image).cpu(), dim=1)
                prediction, _ = self.model(image, [1, 4, 8, 12])
                prediction = torch.argmax(prediction.cpu(), dim=1)
                self.metric.addBatch(prediction, mask.cpu())

                mAcc_sum += self.metric.meanPixelAccuracy()
                mIoU_sum += self.metric.meanIntersectionOverUnion()
                FWIoU_sum += self.metric.Frequency_Weighted_Intersection_over_Union()
                self.metric.reset()

                # 画图
                if times % 20 == 0:
                    for idx in range(image.shape[0]):
                        i = de_normalization(image[idx].cpu(), mean=mean, std=std).numpy().transpose(1, 2, 0)[:, :, ::-1]*255
                        p = prediction[idx].cpu().numpy()

                        save_test_prediction(i, p, cmap, os.path.join(home_dir, f'test_{(times*image.shape[0])+idx+1:06d}.png'))

                times += 1

            mAcc_mean = mAcc_sum / times
            mIoU_mean = mIoU_sum / times
            FWIoU_mean = FWIoU_sum / times

            print(fr'>>> 测试集上结果：mAcc_mean: {mAcc_mean:.4f}, mIoU_mean: {mIoU_mean:.4f}, FWIoU_mean: {FWIoU_mean:.4f}')
            with open(os.path.join(home_dir, 'result.txt'), 'w') as f:
                f.write(fr'>>> 测试集上结果：mAcc_mean: {mAcc_mean:.4f}, mIoU_mean: {mIoU_mean:.4f}, FWIoU_mean: {FWIoU_mean:.4f}')