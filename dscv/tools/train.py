import os
import sys
import argparse
import time
import math
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets
import torch.optim as optim

from dscv.utils.util import TwoCropTransform
from dscv.utils.util import AverageMeter
from dscv.utils.util import accuracy
from dscv.utils.util import save_model
from dscv.models.models.resnet_big import SupCEResNet
from dscv.models.losses.sup_con_loss import SupConLoss
from dscv.utils.config import Config, replace_cfg_vals
from dscv.utils.logger import Logger


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', type=str,
                        default=f'/ws/data/dshong/supcontrast/')
    opt = parser.parse_args()
    return opt


def get_model_name(cfg):
    model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'. \
        format(cfg.data.dataset,
               cfg.model.type,
               cfg.optimizer.learning_rate,
               cfg.optimizer.weight_decay,
               cfg.data.batch_size,
               cfg.runner.trial)
    if hasattr(cfg.optimizer, 'lr_config'):
        if cfg.optimizer.lr_config.cosine:
            model_name = '{}_cosine'.format(model_name)
        if cfg.optimizer.lr_config.warm:
            model_name = '{}_warm'.format(model_name)
    return model_name


def set_loader(cfg):
    train_transform = cfg.train.transform
    val_transform = cfg.val.transform
    if (cfg.method == 'SupCon') or (cfg.method == 'SimCLR'):
        train_transform = TwoCropTransform(train_transform)

    # Construct data loader
    if cfg.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=cfg.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=cfg.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif cfg.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=cfg.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=cfg.data_folder,
                                        train=False,
                                        transform=val_transform)
    else:
        raise ValueError(cfg.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


class Runner():
    def __init__(self, cfg):
        self.log_cfg = cfg.log_config
        self.optim_cfg = cfg.optimizer
        self.config_path = cfg.config_path
        self.method = cfg.model.method

        self.n_cls = cfg.data.n_cls
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.logger = None

        self._max_epochs = cfg.runner.epochs
        self._best_acc = 0
        self._epoch = 0
        self._iter = 0

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value
        self.logger._epoch = value

    @property
    def iter(self):
        return self._iter

    @iter.setter
    def iter(self, value):
        self._iter = value
        self.logger._iter = value

    def update_epoch(self):
        self.logger._epoch = self._epoch
        return self._epoch

    def set_model(self, cfg):
        model = SupCEResNet(name=cfg.type, num_classes=self.n_cls)
        if cfg.method == 'SupCE':
            criterion = torch.nn.CrossEntropyLoss()
        elif cfg.method == 'SupCon' or 'SimCLR':
            criterion = SupConLoss(temperature=cfg.temp)
        else:
            raise ValueError(cfg.method)

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True

        self.model = model
        self.criterion = criterion

    def set_optimizer(self, cfg):
        optimizer = optim.SGD(self.model.parameters(),
                              lr=cfg.learning_rate,
                              momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
        self.optimizer = optimizer

    def set_logger(self, cfg):
        self.logger = Logger(cfg)

    def save_file(self, filename=''):
        save_file = os.path.join(
            self.log_cfg.save_folder, filename)
        save_model(self.model, self.optimizer, self.config_path, self.epoch, save_file)

    def update_best_acc(self, acc):
        if acc > self._best_acc:
            self._best_acc = acc

    def adjust_learning_rate(self):
        lr = self.optim_cfg.learning_rate
        if self.optim_cfg.lr_config.cosine:
            eta_min = lr * (self.optim_cfg.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (
                    1 + math.cos(math.pi * self.epoch / self._max_epochs)) / 2
        else:
            steps = np.sum(self.epoch > np.asarray(self.optim_cfg.lr_decay_epochs))
            if steps > 0:
                lr = lr * (self.optim_cfg.lr_decay_rate ** steps)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def warmup_learning_rate(self, total_batches):
        lr_config = self.optim_cfg.lr_config
        if lr_config.warm and self.epoch <= lr_config.warm_epochs:
            p = (self.iter + (self.epoch - 1) * total_batches) / \
                (lr_config.warm_epochs * total_batches)
            lr = lr_config.warmup_from + p * (lr_config.warmup_to - lr_config.warmup_from)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train(self, train_loader, do_val=False, val_loader=None):
        self.logger.before_run()
        self.epoch = 1

        while(self.epoch < self._max_epochs + 1):
            self.adjust_learning_rate()

            # Train for one epoch
            self.logger.before_train_epoch()
            time1 = time.time()

            loss, train_acc = self.train_step(train_loader)

            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(self.epoch, time2 - time1))
            self.logger.after_train_epoch()

            # Evaluation
            if do_val:
                assert val_loader is not None
                self.logger.before_val_epoch()

                loss, val_acc = self.validate(val_loader)

                self.logger.update_data(dict(val_loss=loss, val_acc=val_acc))
                self.logger.after_val_epoch()
                self.update_best_acc(val_acc)

                if self.epoch % self.log_cfg.save_freq == 0:
                    self.save_file('ckpt_epoch_{epoch}.pth'.format(epoch=self.epoch))

            self.epoch += 1

        # save the last model
        self.save_file('last.pth')
        print('best accuracy: {:.2f}'.format(self._best_acc))
        self.logger.after_run()

    def train_step(self, data_loader):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        end = time.time()
        for idx, (images, labels) in enumerate(data_loader):
            self.iter = idx
            data_time.update(time.time() - end)
            self.logger.before_train_iter()

            if torch.is_tensor(images):
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            elif isinstance(images, list):
                len_image = len(images)
                images = torch.cat(images, dim=0)
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
            else:
                raise TypeError(self.method)
            bsz = labels.shape[0]

            # warm-up learning rate
            self.warmup_learning_rate(len(data_loader))

            # Compute loss
            output = self.model(images)

            if output.shape[0] != bsz:
                assert output.shape[0] % bsz == 0
                num = output.shape[0] // bsz
                outputs = list(torch.chunk(output, num, dim=0))
                for i in range(num):
                    outputs[i] = outputs[i].unsqueeze(1)
                output = torch.cat(outputs, dim=1)

            if self.method == 'SimCLR':
                loss = self.criterion(output, None)
            else:
                loss = self.criterion(output, labels)

            # Update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # SGD
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print info
            if (self.iter + 1) % self.log_cfg.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    self.epoch, self.iter + 1, len(data_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))
                sys.stdout.flush()
            self.logger.after_train_iter()

        return losses.avg, top1.avg

    def validate(self, data_loader):
        """validation"""
        self.model.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        with torch.no_grad():
            end = time.time()
            for idx, (images, labels) in enumerate(data_loader):
                self.logger.before_val_iter()
                images = images.float().cuda()
                labels = labels.cuda()
                bsz = labels.shape[0]

                # forward
                output = self.model(images)
                loss = self.criterion(output, labels)

                # update metric
                losses.update(loss.item(), bsz)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1.update(acc1[0], bsz)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % self.log_cfg.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        idx, len(data_loader), batch_time=batch_time,
                        loss=losses, top1=top1))
                self.logger.after_val_iter()

        val_loss = losses.avg
        val_acc = top1.avg

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

        return val_loss, val_acc


def main():
    # Build config
    opt = parse_option()
    cfg = Config.fromfile(opt.config)
    cfg = replace_cfg_vals(cfg)
    cfg = cfg._cfg_dict
    cfg.model_name = get_model_name(cfg)

    cfg.config_path = opt.config
    cfg.log_config['save_folder'] = os.path.join(f"{opt.work_dir}/{cfg.data.dataset}_models", cfg.model_name)
    for logger in cfg.log_config.loggers:
        if logger['type'] == 'tb_logger':
            logger['tb_folder'] = os.path.join(f"{opt.work_dir}/{cfg.data.dataset}_tensorboard",
                                               cfg.model_name)
            os.makedirs(logger.tb_folder, exist_ok=True)
        elif logger['type'] == 'wandb_logger':
            logger.init_kwargs.update(name=cfg.model_name)
            logger.init_kwargs['config'].update(dict(dataset=cfg.data.dataset,
                                                     model=cfg.model.type,
                                                     method=cfg.model.method,
                                                     epochs=cfg.runner.epochs,))
    os.makedirs(cfg.log_config.save_folder, exist_ok=True)


    # Build runner
    runner = Runner(cfg)

    # Build data loader
    cfg.data.method = cfg.model.method
    train_loader, val_loader = set_loader(cfg.data)

    # Build model and criterion
    runner.set_model(cfg.model)

    # Build optimizer
    runner.set_optimizer(cfg.optimizer)

    # Logger
    runner.set_logger(cfg.log_config)

    # Training routine
    runner.train(train_loader, do_val=True, val_loader=val_loader)


if __name__ == '__main__':
    main()
