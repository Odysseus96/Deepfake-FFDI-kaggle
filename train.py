import time
from sklearn.metrics import roc_auc_score

from utils.utils import AverageMeter, ProgressMeter


def train(train_loader, model, criterion, optimizer, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    roc_auc = AverageMeter('AUC', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1, roc_auc)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        acc = (output.argmax(1).view(-1) == target.float().view(-1)).float().mean() * 100
        auc = roc_auc_score(target.float().view(-1).detach().cpu().numpy(), output.argmax(1).view(-1).detach().cpu().numpy()).mean() * 100

        top1.update(acc, input.size(0))
        roc_auc.update(auc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 500 == 0:
            progress.pr2int(i)