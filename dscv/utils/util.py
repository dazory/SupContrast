from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy2(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if len(output.shape) > 2: # Case: output=(B, N, D), target=(B,D)
            num = output.shape[1]
            output = output.permute(1, 0, 2).reshape(-1, output.shape[-1]) # (N*B,D)
            output_list = list(torch.chunk(output, num, dim=0)) # [N*(B,D)]
            target = target.repeat(num) # (N*B)
            target_list = list(torch.chunk(target, num, dim=0)) # [N*(B,)]

        res_list = []
        for i in range(num):
            _, pred = output_list[i].topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target_list[i].view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            res_list.append(res) # res = (acc1, acc5)
            # res_list = [ (acc1, acc5), (acc1, acc5), (acc1, acc5) ]
        res = torch.mean(torch.Tensor(res_list), dim=0) # (num, len(topk)) -> (len(topk))
        res = list(res.unsqueeze(1))

        return res

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if len(output.shape) > 2: # Case: output=(B, N, D), target=(B,D)
            num = output.shape[1]
            output = output.permute(1, 0, 2).reshape(-1, output.shape[-1]) # (N*B,D)
            output_list = list(torch.chunk(output, num, dim=0)) # [N*(B,D)]
            target = target.repeat(num) # (N*B)
            target_list = list(torch.chunk(target, num, dim=0)) # [N*(B,)]

            output = torch.cat(output_list, dim=0)
            target = torch.cat(target_list, dim=0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def adjust_learning_rate(cfg, optimizer, epoch):
    lr = cfg.learning_rate
    if cfg.lr_config.cosine:
        eta_min = lr * (cfg.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / cfg.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(cfg.lr_decay_epochs))
        if steps > 0:
            lr = lr * (cfg.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(cfg, epoch, batch_id, total_batches, optimizer):
    if cfg.warm and epoch <= cfg.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (cfg.warm_epochs * total_batches)
        lr = cfg.warmup_from + p * (cfg.warmup_to - cfg.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(cfg, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.learning_rate,
                          momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    return optimizer


def save_model(model, optimizer, cfg, epoch, save_file):
    print('==> Saving...')
    state = {
        'cfg': cfg,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
