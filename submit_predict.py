import timm
import torch
import os.path as osp
import argparse
import pandas as pd
from torchvision import transforms

from utils.dataset import FFDIDataset
from predict import predict


def submit_predict(dataset_dir, model_name, ckp, num_classes, pretrained=False, imgsz=256, batch_size=40):
    val_label = pd.read_csv(osp.join(dataset_dir, "valset_label.txt"))
    val_label['path'] = osp.join(dataset_dir, 'valset/') + val_label['img_name']

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, checkpoint_path=ckp)
    model = model.cuda()
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        FFDIDataset(val_label['path'], val_label['target'],
                transforms.Compose([
                            transforms.Resize((imgsz, imgsz)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    val_label['y_pred'] = predict(test_loader, model, 1)[:, 1]
    val_label[['img_name', 'y_pred']].to_csv('submit.csv', index=None)

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

    submit_predict(args.data_dir,
                   args.model_name,
                   args.ckp,
                   args.num_classes,
                   args.pretrained,
                   args.imgsz,
                   args.batch_size
                   )