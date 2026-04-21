import argparse
import os
import sys
import time
import math

DEVICE_NAME = 'cuda:1'

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=20,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=25,
                        help='save frequency')
    parser.add_argument('--save_freq_co', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='start epoch')
    parser.add_argument('--cls_num', type=int, default=7, help='the number of class')    # class num
    parser.add_argument('--resume', type=str, default=None, help='model to resume')
    parser.add_argument('--resume_co', type=str, default=None, help='model to resume')
    parser.add_argument('--device', type=str, default=DEVICE_NAME, help='model to resume')
    parser.add_argument('--ckpt', type=str, default=None, help='model to resume')
    parser.add_argument('--ckpt_co', type=str, default=None, help='model to resume')

    # optimization
    # parser.add_argument('--epochs', type=int, default=800,
    #                     help='number of training epochs')
    # parser.add_argument('--lr_decay_epochs', type=str, default='200,400,500,600,700', # '200,300,400,500'
    #                     help='where to decay lr, can be a list')
    parser.add_argument('--epochs', type=int, default=320,
                        help='number of training epochs')
    parser.add_argument('--lr_decay_epochs', type=str, default='80,160,240,280', # '200,300,400,500'
                        help='where to decay lr, can be a list')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--neg_size', type=int, default=10, help='negative bank for contrastive')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    # parser.add_argument('--cosine', action='store_true',
    #                     help='using cosine annealing')
    parser.add_argument('--cosine', type=bool, default=True,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--nce_p', default=5, type=int,
                        help='number of positive samples for NCE')
    parser.add_argument('--nce_k', default=100, type=int,
                        help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float,
                        help='temperature parameter for softmax')
    parser.add_argument('--mem_t', default=0.02, type=float,
                        help='temperature for memory bank(default: 0.07)')
    parser.add_argument('--mem_wd', default=1e-4, type=float,
                        help='weight decay of memory bank (default: 0)')
    parser.add_argument('--nce_m', default=0.5, type=float,
                        help='momentum for non-parametric updates')
    parser.add_argument('--memory_lr', type=float, default=100, help="learning rate for adversial memory bank")
    parser.add_argument('--momentum_SGD', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--p_replace', type=str, default="norep", choices=['rep', 'norep'])
    parser.add_argument('--n_replace', type=str, default="rep", choices=['rep', 'norep'])

    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax', 'multi_pos'])
    parser.add_argument('--CCD_mode', type=str, default="sup", choices=['sup', 'unsup'])

    ## ISIC
    parser.add_argument('--root_path', type=str, default='./data/data_ISIC/ISIC2018_Task3_Training_Input/')
    parser.add_argument('--csv_file_path', type=str, default='./data/data_ISIC/cv_split_dataset/')

    # # APTOS
    # parser.add_argument('--root_path', type=str, default='./data/data_aptos/images/')
    # parser.add_argument('--csv_file_path', type=str, default='./data/data_aptos/cv_split_dataset/')

    parser.add_argument('--which_split', type=str, default='split1', help='model_name')

    parser.add_argument('--desc', type=str, default="isic_co_5_plus_11_que_p50")

    '''
        self.t = 0.7
        self.ln_t = 0.07
        self.queL = 4
        self.ng_queL = 3
        self.po_queL = 2
        self.wd = 1e-4
        self.memo_lr = 3  # learning rate for adversial memory bank
        self.momentum = 0.9 
    '''
    parser.add_argument('--t', type=float, default=0.07)
    parser.add_argument('--ln_t', type=float, default=0.1)
    parser.add_argument('--queL', type=int, default=4)
    parser.add_argument('--ng_queL', type=int, default=300)
    parser.add_argument('--po_queL', type=int, default=50)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--memo_lr', type=float, default=3e-1) # 3e-1
    parser.add_argument('--mini_bank_size', type=int, default=7) # 7
    parser.add_argument('--rand_vec', type=float, default=0.1) # 0.1
    parser.add_argument('--memo_lr_decay', type=float, default=0.1)
    parser.add_argument('--memo_decay_epoch', type=int, default=120) # 0.5

    
    parser.add_argument('--co_weight', type=float, default=1)
    parser.add_argument('--ce_weight', type=float, default=1)
    
    
    opt = parser.parse_args()
    opt.neg_num_4_con = (opt.cls_num - 1) * opt.mini_bank_size


    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_memolr_{}'.\
        format(opt.desc, opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial, opt.memo_lr)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)


    ###################
    # opt.resume = '/home/xunxun/workspace/lnco/save/SupCon/cifar10_models/ISIC_dataset_LeNeg_SupCon_cifar10_resnet50_lr_0.1_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine/last.pth'
    # opt.ckpt = '/home/xunxun/workspace/lnco/save/SupCon/path_models/isic_supcon_co_1_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_64_temp_0.07_trial_0_memolr_0.3_cosine/ckpt_epoch_800.pth'
    # opt.ckpt = '/home/xunxun/workspace/lnco/save/SupCon/path_models/aptos_co_1_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_32_temp_0.07_trial_0_memolr_0.3_cosine/ckpt_epoch_800.pth'
    # opt.ckpt_co = '/home/xunxun/workspace/lnco/save/SupCon/path_models/aptos_co_1_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_32_temp_0.07_trial_0_memolr_0.3_cosine/ckpt_epoch_co_800.pth'


    # SimCLR
    # opt.ckpt = '/home/xunxun/workspace/lnco/save/SupCon/path_models/aptos_simclr_co_1_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_32_temp_0.07_trial_0_memolr_0.3_cosine/ckpt_epoch_800.pth'
    # opt.ckpt = '/home/xunxun/workspace/lnco/save/SupCon/path_models/isic_simclr_co_1_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_64_temp_0.07_trial_0_memolr_0.3_cosine/ckpt_epoch_300.pth'

    # lnco
    opt.ckpt = '/home/xunxun/workspace/lnco/save/SupCon/path_models/isic_co_5_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_64_temp_0.07_trial_0_memolr_0.3_cosine/ckpt_epoch_800.pth'
    # opt.ckpt_co = '/home/xunxun/workspace/lnco/save/SupCon/path_models/isic_co_5_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_64_temp_0.07_trial_0_memolr_0.3_cosine/ckpt_epoch_co_800.pth'

    # adco
    # opt.ckpt = '/home/xunxun/workspace/lnco/save/SupCon/path_models/aptos_adco_co_2_SupCon_path_resnet50_lr_0.001_decay_0.0001_bsz_32_temp_0.07_trial_0_memolr_0.3_cosine/ckpt_epoch_800.pth'

    # test
    # opt.ckpt = '/home/xunxun/workspace/lnco/save/SupCon/path_models/isic_co_5_plus_2_SupCon_path_resnet50_lr_0.0001_decay_0.0001_bsz_64_temp_0.07_trial_0_memolr_0.3_cosine/ckpt_epoch_co_300.pth'


    return opt