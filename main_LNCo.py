from __future__ import print_function

import os
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import gradcheck

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, FixedSupConResNet, SupCEResNet
from loss.losses import cross_entropy_loss
from loss.SupConLoss import LNSupConLoss
from hpra import parse_option
import dataloaders.dataset as ds

device = torch.device('cuda:0')

def setup_seed(seed):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        # mean = eval(opt.mean)
        # std = eval(opt.std)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root="./data/cifar10",
                                         transform=TwoCropTransform(train_transform),
                                         train=True,
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          train=True,
                                          download=True)
    elif opt.dataset == 'path':
        # train_dataset = datasets.ImageFolder(root=opt.data_folder,
        #                                     transform=TwoCropTransform(train_transform))
        train_dataset, _1, _2, _3, _4 = ds.isic_dataloaders_sample(opt, is_sample=False)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader

def set_model(opt):
    model = FixedSupConResNet(name=opt.model, num_classes=opt.cls_num)
    criterion_co = LNSupConLoss(opt, temperature=opt.temp, neg_size=opt.neg_size)
    criterion_ce = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        model = model.to(device)
        criterion_ce = criterion_ce.to(device)
        criterion_co = criterion_co.to(device)
        cudnn.benchmark = True

    if opt.resume:
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'], strict=False)
        if opt.resume_co:
            checkpoint_co = torch.load(opt.resume_co)
            criterion_co.load_state_dict(checkpoint_co['model'], strict=False)
        # print(checkpoint['optimizer']['param_groups']['params'])
    elif opt.ckpt:
        checkpoint = torch.load(opt.ckpt)
        if opt.ckpt_co:
            criterion_co.load_state_dict(torch.load(opt.ckpt_co)['model'], strict=False)
        
    return model, criterion_co, criterion_ce

def train(train_loader, model, criterion, optimizer, epoch, opt, writer, iter_num):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - end)


        images = torch.cat([images[0], images[1]], dim=0)  # [2*bs, ...]
        if torch.cuda.is_available():
            images = images.to(device)
            labels = labels.to(device)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features, logits = model(images)  # [2*bs, feat] [2*bs, num_cls]
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)  # [bs, feat]
        logit_1, logit_2 = torch.split(logits, [bsz, bsz], dim=0)  # [bs, num_cls]
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # [bs, 2, feat]
        
        # test = gradcheck(criterion, (features, ))
        if opt.method == 'SupCon':
            loss = criterion(features, labels, epoch)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            iter_num += 1
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr:.3f}\t'
                  'loss {loss.val:.6f} ({loss.avg:.6f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, lr=optimizer.param_groups[0]['lr'], loss=losses))
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('train_loss', loss, iter_num)
            sys.stdout.flush()

    return losses.avg, iter_num, writer


def main():
    setup_seed(2024)

    opt = parse_option()

    writer = SummaryWriter(opt.tb_folder)

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion_co, criterion_ce = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    print(opt.epochs, opt.batch_size, opt.learning_rate, opt.lr_decay_rate, opt.lr_decay_epochs, opt.cosine)

    iter_num = 0
    # training routine
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, iter_num, writer = train(train_loader, model, criterion_co, optimizer, epoch, opt, writer, iter_num)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
        if epoch % opt.save_freq_co == 0:
            save_file_co = os.path.join(
                opt.save_folder, 'ckpt_epoch_co_{epoch}.pth'.format(epoch=epoch))
            save_model(criterion_co, optimizer, opt, epoch, save_file_co)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    writer.close()

if __name__ == '__main__':
    main()
