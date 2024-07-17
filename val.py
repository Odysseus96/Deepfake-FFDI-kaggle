import torch
import time
import argparse
import pandas as pd
import timm
import os.path as osp
from tqdm import tqdm
from utils.utils import *
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
import torch.nn as nn

from utils.dataset import FFDIDataset


def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    roc_auc = AverageMeter('AUC', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, roc_auc)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc = (output.argmax(1).view(-1) == target.float().view(-1)).float().mean() * 100
            auc = roc_auc_score(target.float().view(-1).detach().cpu().numpy(),
                                output.argmax(1).view(-1).detach().cpu().numpy()).mean() * 100
            losses.update(loss.item(), input.size(0))
            top1.update(acc, input.size(0))
            roc_auc.update(auc, input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        print('Val: Avg Time {batch_time.avg:.3f}\t '
              'Avg Loss {losses.avg:.4e}\t '
              'Avg Acc@1 {top1.avg:.3f}\t '
              'Avg AUC {roc_auc.avg:.3f}'
              .format(batch_time=batch_time,
                      losses=losses,
                      top1=top1,
                      roc_auc=roc_auc))
        return top1, roc_auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, default='efficientnet_b1', help='model name')
    parser.add_argument('--data_dir', '-d', type=str, default='/home/wangyaozhong/data/phase1', help='dataset dir')
    parser.add_argument('--pretrained', default=False, type=bool, help='whether use pretrained model')
    parser.add_argument('--ckp', default='res/efficientnet_b1/model_91.6.pt', type=str, help='checkpoint path')
    parser.add_argument('--num_classes', '-n', default=2, type=int, help='class number')
    parser.add_argument('--batch_size', '-b', default=40, type=int, help='batch size')
    parser.add_argument('--imgsz', default=256, type=int, help='img size of input')

    args = parser.parse_args()
    val_label = pd.read_csv(osp.join(args.data_dir, "valset_label.txt"))
    val_label['path'] = osp.join(args.data_dir, 'valset/') + val_label['img_name']

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=args.num_classes, checkpoint_path=args.ckp)
    model = model.to(device)
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        FFDIDataset(val_label['path'], val_label['target'],
                    transforms.Compose([
                        transforms.Resize((args.imgsz, args.imgsz)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    ), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    criterion = nn.CrossEntropyLoss().to(device)
    validate(test_loader, model, criterion, device)

    
