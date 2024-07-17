import numpy as np
import os
import matplotlib.pyplot as plt

class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtst = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtst.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""

    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def pr2val(self):
        entries = "Val: "
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def plot_performance_per_fold(acc_scores, auc_scores, folds, output_dir, filename="performance.png"):
    """
    绘制模型的准确率、AUC和损失随每折的变化的折线图，并将图像保存到指定目录。

    参数:
    acc_scores (list): 每折的准确率列表。
    auc_scores (list): 每折的AUC列表。
    loss_values (list): 每折的损失列表。
    folds (int): 折数，即k值。
    output_dir (str): 图像要保存的目录。
    filename (str): 保存的文件名。
    """

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建图像
    plt.figure(figsize=(12, 6))
    # 绘制准确率
    plt.plot(range(1, folds + 1), acc_scores, marker='o', label='Accuracy')
    # 标记每个准确率数据点
    for i, txt in enumerate(acc_scores):
        plt.annotate('{:.2f}'.format(txt), (i + 1, acc_scores[i]))
    # 绘制AUC
    plt.plot(range(1, folds + 1), auc_scores, marker='s', label='AUC')
    # 标记每个AUC数据点
    for i, txt in enumerate(auc_scores):
        plt.annotate('{:.2f}'.format(txt), (i + 1, auc_scores[i]))
    # 设置图例
    plt.legend()
    # 设置标题和轴标签
    plt.title('Model Performance per Fold')
    plt.xlabel('Fold')
    plt.ylabel('Score / Loss')

    # 设置x轴刻度标签
    plt.xticks(range(1, folds + 1))

    # 保存图像到指定目录
    plt.savefig(os.path.join(output_dir, filename), format='png')

    # 关闭图像，释放资源
    plt.close()
