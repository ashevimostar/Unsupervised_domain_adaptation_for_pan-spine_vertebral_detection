from tensorboard.backend.event_processing import event_accumulator
 
'''
Train: [8][60/84]	BT 2.226 (2.539)	DT 2.141 (2.453)	LR 0.00099846	loss 1.133 (0.911707)	loss_ce 1.13339	loss_co 4.202466	
Train: [8][80/84]	BT 3.118 (2.542)	DT 3.033 (2.456)	LR 0.00099846	loss 1.705 (0.933386)	loss_ce 1.70527	loss_co 4.0560694	
epoch 8, total time 212.17
 * Acc@1 71.400

'''
def get_data_list_from_tb(tbpath='/home/xunxun/workspace/ubcl/unbiased-contrastive-learning-master/logs/infonce_out_isic_resnet50_sgd_bsz32_lr0.01_cosine_t0.1_eps0.0_lr-eps0.0001_feat128_identity_alpha1.0_beta0_lambda0.0_kld1.0_abc_mlp_lr0.001_mlp_optimizer_adam_trial0/events.out.tfevents.1711708284.be29036b0622.2594639.0'):
    #加载日志数据'
    ea=event_accumulator.EventAccumulator(tbpath) 
    ea.Reload()
    val_psnr=ea.scalars.Items('test/acc@1')
    print(len(val_psnr))
    print([(i.step,i.value) for i in val_psnr])
    res_list = []
    for i in val_psnr:
        res_list.append(i.value)
    return res_list

def get_data_list_from_log(log_path='/home/xunxun/workspace/lnco/isic_co_5_plus_2.log', dec=0.):
    records = []
    with open(log_path, 'r') as f:
        line = f.readline()
        while line:
            line = f.readline()
            if 'Acc@1' in line:
                words = line.split(' ')
                accuracy = words[-1]
                records.append(float(accuracy))
    return records

def transxy(vlist, freq):
    y_list = [0.]
    x_list = [0]
    idx = freq
    while idx < len(vlist):
        x_list.append(idx)
        y_list.append(vlist[idx])
        idx += freq
    return x_list, y_list

if __name__ == '__main__':

    # isic 
    isic_tb_paths = {
        'UBCo' : '/home/xunxun/workspace/ubcl/unbiased-contrastive-learning-master/logs/infonce_out_isic_resnet50_sgd_bsz32_lr0.01_cosine_t0.1_eps0.0_lr-eps0.0001_feat128_identity_alpha1.0_beta0_lambda0.0_kld1.0_abc_mlp_lr0.001_mlp_optimizer_adam_trial0/events.out.tfevents.1711708284.be29036b0622.2594639.0'

    }
    isic_log_paths = {
        'LNCo' : '/home/xunxun/workspace/lnco/isic_co_5_plus_2.log',
        'ce' : '/home/xunxun/workspace/lnco/isic_ce_2.log',
        'SimCLR' : '/home/xunxun/workspace/lnco/isic_simclr_co_1_plus_1.log',
        'SupCon' : '/home/xunxun/workspace/lnco/isic_supcon_co_1_plus_1.log',
        'AdCo' : '/home/xunxun/workspace/lnco/isic_adco_co_1_plus_wo_1.log'
    }



    # aptos 
    aptos_tb_paths = {
        'UBCo' : '/home/xunxun/workspace/ubcl/unbiased-contrastive-learning-master/logs/infonce_out_aptos_resnet50_sgd_bsz16_lr0.01_cosine_t0.1_eps0.0_lr-eps0.0001_feat128_identity_alpha1.0_beta0_lambda0.0_kld1.0_abc_mlp_lr0.001_mlp_optimizer_adam_trial0/events.out.tfevents.1711866739.be29036b0622.2717253.0'
    }
    aptos_log_paths = {
        'LNCo' : '/home/xunxun/workspace/lnco/aptos_co_1_plus_1.log',
        'ce' : '/home/xunxun/workspace/lnco/aptos_ce_1.log',
        'SimCLR' : '/home/xunxun/workspace/lnco/aptos_simclr_co_1_plus_1.log',
        'SupCon' : '/home/xunxun/workspace/lnco/aptos_supcon_co_1_plus_1.log',
        'AdCo' : '/home/xunxun/workspace/lnco/aptos_adco_2_plus_2.log'
    }

    record_freq = 17

    from matplotlib import pyplot as plt
    #设置中文显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    #设置图形大小
    plt.figure(figsize=(10,8),dpi=80)
    plt.rcParams.update({'font.size': 16})


    ################### isic #####################

    for k,v in isic_tb_paths.items():
        xs, ys = transxy(get_data_list_from_tb(v), record_freq)
        plt.plot(xs,ys,label=k,
                # color="orange",
                # linestyle=':',
                linewidth=3,
                alpha=0.8)
    for k,v in isic_log_paths.items():
        xs, ys = transxy(get_data_list_from_log(v), record_freq)
        plt.plot(xs,ys,label=k,
                # color="orange",
                # linestyle=':',
                linewidth=3,
                alpha=0.8)
    plt.xlabel('Training steps(epochs)')
    plt.ylabel('ACC (%)')


    ################### aptos #####################

    # for k,v in aptos_tb_paths.items():
    #     xs, ys = transxy(get_data_list_from_tb(v), record_freq)
    #     plt.plot(xs,ys,label=k,
    #             # color="orange",
    #             # linestyle=':',
    #             linewidth=3,
    #             alpha=0.8)
    # for k,v in aptos_log_paths.items():
    #     xs, ys = transxy(get_data_list_from_log(v), record_freq)
    #     plt.plot(xs,ys,label=k,
    #             # color="orange",
    #             # linestyle=':',
    #             linewidth=3,
    #             alpha=0.8)
    # plt.xlabel('训练时长 (epochs)')
    # plt.ylabel('准确率 (%)')



    #画图
    #label设置图例标签;
    #color设置颜色;
    #linestyle设置线型;
    #linewidth设置线的粗细
    #alpha设置线的透明度
    # plt.plot(x,y_1,label="自己",
    #         color="orange",
    #         linestyle=':',
    #         linewidth=5,
    #         alpha=0.8)
    # plt.plot(x,y_2,label="同桌",
    #         color="cyan",
    #         linestyle='-.',
    #         linewidth=5,
    #         alpha=0.8)

    #设置坐标
    # _xtick_labels = ["{}岁".format(i) for i in x]
    # plt.xticks(x,_xtick_labels )


    #绘制网格
    #alpha表示调节网格透明度
    plt.grid(alpha=0.4)

    #添加图例
    #loc表示设置图例放在什么位置
    plt.legend(loc="lower right")

    #展示
    plt.show()
    plt.savefig("./isic_sota_compare_17.png")