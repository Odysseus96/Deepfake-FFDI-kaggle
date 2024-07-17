
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torchvision
from torchvision import transforms

class FFDIDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        label = self.img_label[index]
        img = Image.open(self.img_path[index]).convert('RGB')

        if label != 0:
            self.transform = transforms.Compose([
                        # transforms.ToPILImage(),
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

        if self.transform is not None:
            img = self.transform(img)

        # print(f"img shape: {img.shape}, label shape: {label_one_hot.shape}")
        return img, torch.from_numpy(np.array(label))

    def __len__(self):
        return len(self.img_path)


class FFDIDatasetv2(Dataset):
    def __init__(self, img_path, img_label, transform=None, alpha=32.0, mixup_prob=0.5, imbalance_correction=True):
        self.img_path = img_path
        self.img_label = img_label
        self.alpha = alpha
        self.mixup_prob = mixup_prob

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

        self.num_classes = len(np.unique(img_label))

        if imbalance_correction:
            # 计算每个类别的权重
            class_sample_count = np.array([len(np.where(img_label == t)[0]) for t in np.unique(img_label)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in img_label])
            samples_weight /= samples_weight.sum()
            self.samples_weight = torch.from_numpy(samples_weight)
            self.samples_weight = self.samples_weight.double()

    def __getitem__(self, index):
        img, label = self.load_image_and_label(index)

        if np.random.rand() < self.mixup_prob:
            img, label = self.mixup(img, label)
        else:
            img, label = self.cutmix(img, label)

        return img, torch.from_numpy(np.array(label))

    def load_image_and_label(self, index):
        label = self.img_label[index]
        img = Image.open(self.img_path[index]).convert('RGB')
        # if label != 0:
        #     self.transform = transforms.Compose([
        #                 # transforms.ToPILImage(),
        #                 transforms.Resize((256, 256)),
        #                 transforms.ToTensor(),
        #                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #             ])
        if self.transform:
            img = self.transform(img)
        return img, label

    def mixup(self, img, label):
        img2, label2 = self.random_sample()
        lam = np.random.beta(self.alpha, self.alpha)
        img2 = self.transform(img2)
        mixed_img = lam * img + (1 - lam) * img2
        mixed_label = self.smooth_labels(label, label2, lam)
        return mixed_img, mixed_label

    def cutmix(self, img, label):
        img2, label2 = self.random_sample()
        lam = np.random.beta(self.alpha, self.alpha)
        img2 = self.transform(img2)

        bx1, by1, bx2, by2 = self.rand_bbox(img.size(), lam)
        img[:, bx1:bx2, by1:by2] = img2[:, bx1:bx2, by1:by2]
        lam = 1 - ((bx2 - bx1) * (by2 - by1) / (img.size(-1) * img.size(-2)))
        mixed_label = self.smooth_labels(label, label2, lam)
        return img, mixed_label

    def smooth_labels(self, label, label2, lam):
        # 创建标签平滑向量
        y1 = torch.zeros(self.num_classes).scatter_(0, torch.tensor(label), 1)
        y2 = torch.zeros(self.num_classes).scatter_(0, torch.tensor(label2), 1)
        return lam * y1 + (1 - lam) * y2

    def random_sample(self):
        index = np.random.choice(len(self.img_path), p=self.samples_weight)
        label = self.img_label[index]
        img = Image.open(self.img_path[index]).convert('RGB')
        return img, label

    def rand_bbox(self, size, lam):
        W = size[1]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __len__(self):
        return len(self.img_path)


def create_weighted_loader(dataset, batch_size=128):
    sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.samples_weight, len(dataset.samples_weight))
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)


if __name__ == '__main__':
    import os.path as osp
    import pandas as pd

    IMG_SIZE = 256

    dataset_dir = "/home/wangyaozhong/data/phase1"

    train_label = pd.read_csv(osp.join(dataset_dir, "trainset_label.txt"))
    val_label = pd.read_csv(osp.join(dataset_dir, "valset_label.txt"))

    train_label['path'] = osp.join(dataset_dir, 'trainset/') + train_label['img_name']
    val_label['path'] = osp.join(dataset_dir, 'valset/') + val_label['img_name']

    train_loader = torch.utils.data.DataLoader(
        FFDIDataset(train_label['path'].head(1000), train_label['target'].head(1000),
                    transforms.Compose([
                        # transforms.ToPILImage(),
                        transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        # transforms.ToDtype(torch.float32, scale=True),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    ), batch_size=256, shuffle=True, num_workers=4, pin_memory=True
    )

    image, labels = next(iter(train_loader))
    print(labels.shape)

