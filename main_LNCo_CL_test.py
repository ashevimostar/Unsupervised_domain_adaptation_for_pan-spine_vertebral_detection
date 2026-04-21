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
from networks.resnet_big import SupConResNet, LinearClassifier, FixedSupConResNet, SupCEResNet
from loss.losses import cross_entropy_loss
from loss.SupConLoss import LNSupConLoss
from hpra_test import parse_option
import dataloaders.dataset as ds

from metric_process import Preds

# try:
#     import apex
#     from apex import amp, optimizers
# except ImportError:
#     pass
DEVICE_NAME = 'cuda:0'

device = torch.device(DEVICE_NAME)

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
        train_dataset, test_dataset, _2, _3, _4, t_test_dataset = ds.isic_dataloaders_sample(opt, is_sample=False)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)
    t_test_loader = torch.utils.data.DataLoader(
        t_test_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)
    return train_loader, test_loader, t_test_loader


def set_model(opt):
    model = FixedSupConResNet(name=opt.model, num_classes=opt.cls_num)
    criterion_co = LNSupConLoss(opt, temperature=opt.temp, neg_size=opt.neg_size)
    criterion_ce = torch.nn.CrossEntropyLoss()
    # model = SupCEResNet(ame=opt.model, num_classes=opt.cls_num)
    # criterion = cross_entropy_loss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
        #     model.encoder = torch.nn.DataParallel(model.encoder)
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
        model.load_state_dict(checkpoint['model'], strict=False)
        if opt.ckpt_co:
            criterion_co.load_state_dict(torch.load(opt.ckpt_co)['model'], strict=False)
        
    return model, criterion_co, criterion_ce


def train(train_loader, model, criterion_co, criterion_ce, optimizer, epoch, opt, writer, iter_num):
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
            loss_co = criterion_co(features, labels, epoch)
        elif opt.method == 'SimCLR':
            loss_co = criterion_co(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        loss_ce = criterion_ce(logit_1, labels) + criterion_ce(logit_2, labels)
        loss = opt.co_weight * loss_co + opt.ce_weight * loss_ce

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if epoch == 8 and idx == 340 :
            print()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            iter_num += 1
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr:.8f}\t'
                  'loss {loss.val:.3f} ({loss.avg:.6f})\t'
                  'loss_ce {lossce:.6}\t'
                  'loss_co {lossco:.8}\t'
                  .format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, lr=optimizer.param_groups[0]['lr'],
                    loss=losses, lossce=loss_ce, lossco=loss_co))
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('train_loss', loss, iter_num)
            writer.add_scalar('train_loss_ce', loss_ce, iter_num)
            writer.add_scalar('train_loss_co', loss_co, iter_num)
            sys.stdout.flush()

    return losses.avg, iter_num, writer

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        target = target.argmax(-1)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, ( _, _, images, labels) in enumerate(val_loader):
            images = images.float().to(device)
            labels = labels.to(device)
            # labels = labels.argmax(-1)
            bsz = labels.shape[0]

            # forward
            features, logits = model(images)
            loss = criterion(logits, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if idx % opt.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #            idx, len(val_loader), batch_time=batch_time,
            #            loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg

def test_main(model_dir=True):
    setup_seed(2022)
    opt = parse_option()
    _, eval_loader, test_loader = set_loader(opt)
    print(opt.epochs, opt.batch_size, opt.learning_rate, opt.lr_decay_rate, opt.lr_decay_epochs, opt.cosine)

    # ISIC
   
    ## lnco
    # model_dir_path = '/home/xunxun/workspace/lnco/save/SupCon/path_models/isic_co_5_plus_2_SupCon_path_resnet50_lr_0.0001_decay_0.0001_bsz_64_temp_0.07_trial_0_memolr_0.3_cosine'

    ## supcon
    # model_dir_path = '/home/xunxun/workspace/lnco/save/SupCon/path_models/isic_co_3_plus_1_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_64_temp_0.07_trial_0_memolr_0.3_cosine'
    
    ## simclr
    # model_dir_path = '/home/xunxun/workspace/lnco/save/SupCon/path_models/isic_simclr_co_1_plus_1_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_32_temp_0.07_trial_0_memolr_0.3_cosine'
    
    ## ce
    # model_dir_path = '/home/xunxun/workspace/lnco/save/SupCon/path_models/isic_ce_2_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_16_temp_0.07_trial_0_memolr_0.3_cosine'
    
    # APTOS
    ## lnco
    # model_dir_path = '/home/xunxun/workspace/lnco/save/SupCon/path_models/aptos_co_ce_1_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_32_temp_0.07_trial_0_memolr_0.3_cosine'

    ## supcon
    # model_dir_path = '/home/xunxun/workspace/lnco/save/SupCon/path_models/aptos_supcon_co_1_plus_1_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_32_temp_0.07_trial_0_memolr_0.3_cosine'

    ## simclr
    # model_dir_path = '/home/xunxun/workspace/lnco/save/SupCon/path_models/aptos_simclr_co_1_plus_1_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_32_temp_0.07_trial_0_memolr_0.3_cosine'

    ## ce
    # model_dir_path = '/home/xunxun/workspace/lnco/save/SupCon/path_models/aptos_ce_1_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_32_temp_0.07_trial_0_memolr_0.3_cosine'
    
    ## adco
    model_dir_path = '/home/xunxun/workspace/lnco/save/SupCon/path_models/aptos_adco_2_plus_2_SupCon_path_resnet50_lr_0.001_decay_0.001_bsz_16_temp_0.07_trial_0_memolr_0.3_cosine'

    ep = 25
    opt.ckpt = model_dir_path + '/ckpt_epoch_' + str(ep) + '.pth'
    while os.path.exists(opt.ckpt):
        model, criterion_co, criterion_ce = set_model(opt)
        model.eval()
        prr = Preds()
        for idx, ( _, _, images, labels) in enumerate(test_loader):
            images = images.float().to(device)
            labels = labels.to(device)
            features, logits = model(images)
            prr.got_ll(logits, labels)
        print(str(ep) + '_results :' + str(prr.cal_metrics()) + "\n")
        prr.clear_list()
        ep += 25
        opt.ckpt = model_dir_path + '/ckpt_epoch_' + str(ep) + '.pth'


def main():
    setup_seed(2022)

    opt = parse_option()

    writer = SummaryWriter(opt.tb_folder)

    # build data loader
    _, test_loader = set_loader(opt)

    # build model and criterion
    model, criterion_co, criterion_ce = set_model(opt)

    # build optimizer
    # optimizer = set_optimizer(opt, model)

    print(opt.epochs, opt.batch_size, opt.learning_rate, opt.lr_decay_rate, opt.lr_decay_epochs, opt.cosine)
    
    loss, val_acc = validate(test_loader, model, criterion_ce, opt)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    # iter_num = 0
    # training routine
    # for epoch in range(opt.start_epoch, opt.epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        # time1 = time.time()
        # loss, iter_num, writer = train(train_loader, model, criterion_co, criterion_ce, optimizer, epoch, opt, writer, iter_num)
        # time2 = time.time()
        # print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        # logger.log_value('loss', loss, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        # loss, val_acc = validate(test_loader, model, criterion_ce, opt)
        # writer.add_scalar('val_acc', val_acc, iter_num)
        # if epoch % opt.save_freq == 0:
        #     save_file = os.path.join(
        #         opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     save_model(model, optimizer, opt, epoch, save_file)
        # if epoch % opt.save_freq_co == 0:
        #     save_file_co = os.path.join(
        #         opt.save_folder, 'ckpt_epoch_co_{epoch}.pth'.format(epoch=epoch))
        #     save_model(criterion_co, optimizer, opt, epoch, save_file_co)

    # save the last model
    # save_file = os.path.join(
    #     opt.save_folder, 'last.pth')
    # save_model(model, optimizer, opt, opt.epochs, save_file)

    # writer.close()

if __name__ == '__main__':
    test_main()
