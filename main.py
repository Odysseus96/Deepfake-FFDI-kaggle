import os
import argparse
import torch
import timm
import pandas as pd
import numpy as np
import os.path as osp
import torch.nn as nn
import torchvision.transforms as transforms
import datetime
from sklearn.model_selection import train_test_split, KFold

from utils.dataset import *
from utils.utils import *
from val import validate
from train import train


def run(dataset_dir, epochs=5, save_dir='weights', imgsz=256, batch_size=128, folds=5):

    if not osp.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    train_label = pd.read_csv(osp.join(dataset_dir, "trainset_label.txt"))
    val_label = pd.read_csv(osp.join(dataset_dir, "valset_label.txt"))

    train_label['path'] = osp.join(dataset_dir, 'trainset/') + train_label['img_name']
    val_label['path'] = osp.join(dataset_dir, 'valset/') + val_label['img_name']

    ffdi_dataset = pd.concat([train_label, val_label], ignore_index=True)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        ffdi_dataset['path'],
        ffdi_dataset['target'],
        test_size=0.2,
        random_state=233,
        stratify=ffdi_dataset['target']
    )

    kf = KFold(n_splits=folds, shuffle=True, random_state=233)

    X_train_val = np.array(X_train_val)
    y_train_val = np.array(y_train_val)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    tesst_loader = torch.utils.data.DataLoader(
        FFDIDataset(X_test, y_test,
                    transforms.Compose([
                        transforms.Resize((imgsz, imgsz)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    ), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    acc_scores = []
    auc_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_val, y_train_val)):

        print(f"第 {fold+1} 个fold")

        best_acc_dir = osp.join(save_dir, f'fold{fold+1}/best_acc_fold')
        if not osp.exists(best_acc_dir):
            os.makedirs(best_acc_dir, exist_ok=True)

        best_auc_dir = osp.join(save_dir, f'fold{fold+1}/best_auc_fold')
        if not osp.exists(best_auc_dir):
            os.makedirs(best_auc_dir, exist_ok=True)

        train_data = X_train_val[train_index]
        train_label = y_train_val[train_index]

        val_data = X_train_val[val_index]
        val_label = y_train_val[val_index]

        train_loader = torch.utils.data.DataLoader(
            FFDIDataset(train_data, train_label,
                        transforms.Compose([
                            # transforms.ToPILImage(),
                            transforms.Resize((imgsz, imgsz)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ColorJitter(brightness=.5, hue=.3),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        ), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            FFDIDataset(val_data, val_label,
                        transforms.Compose([
                            transforms.Resize((imgsz, imgsz)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        ), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=args.num_classes,
                                  checkpoint_path=args.ckp)

        model.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), 0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)

        best_acc = 0.0
        best_auc = 0.0
        best_model = model
        for epoch in range(epochs):

            print('Epoch: ', epoch)

            train(train_loader, model, criterion, optimizer, device)
            val_acc, val_auc = validate(val_loader, model, criterion, device)

            if val_acc.avg.item() > best_acc:
                best_acc = round(val_acc.avg.item(), 2)
                torch.save(model.state_dict(), f'{best_acc_dir}/model_{best_acc}.pt')

            if val_auc.avg > best_auc:
                best_auc = round(val_auc.avg, 2)
                torch.save(model.state_dict(), f'{best_auc_dir}/model_{best_auc}.pt')
                best_model = model

            scheduler.step()

        print(f"\nFold {fold+1} test: ")
        top1, roc_auc = validate(tesst_loader, best_model, criterion, device)
        print("\n")
        acc_scores.append(top1.avg.detach().cpu().numpy())
        auc_scores.append(roc_auc.avg)

    plot_performance_per_fold(acc_scores, auc_scores, folds, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, default='efficientnet_b1', help='model name')
    parser.add_argument('--data_dir', '-d', type=str, default='/home/wangyaozhong/data/phase1', help='dataset dir')
    parser.add_argument('--pretrained', default=False, type=bool, help='whether use pretrained model')
    parser.add_argument('--ckp', default='weights/timm-efficientnet-b1/efficientnet_b1.pth', type=str, help='pretrained model')
    parser.add_argument('--num_classes', '-n', default=2, type=int, help='class number')
    parser.add_argument('--batch_size', '-b', default=128, type=int, help='batch size')
    parser.add_argument('--imgsz', default=256, type=int, help='img size of input')
    parser.add_argument('--epochs', '-e', default=5, type=int, help='training epochs')
    parser.add_argument('--folds', '-f', default=5, type=int, help='training epochs')

    args = parser.parse_args()

    timestamp = datetime.datetime.now().timestamp()
    timestamp_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%dT%H-%M-%S')

    save_dir = f"res/{timestamp_str}_{args.model_name}"
    run(args.data_dir, args.epochs, save_dir, args.imgsz, args.batch_size, args.folds)

