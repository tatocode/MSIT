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
        self.type = test_config_dict['type']

    def test(self, home_dir):
        print(f'>>> {self.device} 可用!')

        make_dir(home_dir)

        # test
        self.model.eval()
        with torch.no_grad():

            prob_lst = []
            label_lst = []

            for image, label in tqdm(self.test_loader):
                image, label = image.to(self.device, dtype=torch.float32), label.to(self.device, dtype=torch.long)
                if self.type == 'binary':
                    prob_lst += self.model(image).cpu()[:, 1].tolist()
                    label_lst += label.cpu().tolist()
                prediction = torch.argmax(self.model(image).cpu(), dim=1)

                self.metric.addBatch(prediction, label.cpu())
            if self.type == 'binary':
                ro_curve(prob_lst, label_lst, os.path.join(home_dir, 'AUC curve.png'))
            self.metric.drawMatrix(os.path.join(home_dir, 'confusion matrix.png'))

            Acc = self.metric.getAcc()
            Precision = self.metric.getPrecision()
            Recall = self.metric.getRecall()
            F1_score = self.metric.getF1Score()


            with open(os.path.join(home_dir, 'result_metric_test.txt'), 'w') as f:
                f.write(f'****************metric result****************\n')
                f.write(
                    f'mAcc: {Acc:.4f}, Precision: {Precision:.4f}, Recall: {Recall:.4f}, F1-Score: {F1_score:.4f}\n')

