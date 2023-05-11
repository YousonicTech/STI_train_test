# -*- coding: utf-8 -*-
"""
@file      :  test_resnet50_500Hz.py
@Time      :  2022/8/20 17:54
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
from tqdm import tqdm, trange
import math

import time
import torch
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.tensorboard import SummaryWriter
# 决定使用哪块GPU进行训练
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv

from data_load_sti import Dataset_dict, collate_fn



from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
import torch.optim as optim
import datetime

import argparse
from model.attentionFPN import FPN

torch.backends.cudnn.benchmark = True


def parse_args():
    # Test路径见下面 val_dict_root
    parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
    parser.add_argument('--model_path', type=str, default='save_model/0408_atenFPNawl/')
    parser.add_argument('--epoch_for_save', type=int, default=11)  # 记得改epoch数

    parser.add_argument('--start_freq', type=int, default=0)
    parser.add_argument('--end_freq', type=int, default=29)
    parser.add_argument('--ln_out', type=int, default=1)
    parser.add_argument('--SERVER', type=bool, default=True)  # 在服务器上运行记得改成True
    parser.add_argument('--outputresult_dir', type=str, default="./0409_test/epoch11") # 记得改输出路径，不用带/应该

    args = parser.parse_args()
    return args


def load_checkpoint(checkpoint_path=None, trained_epoch=None, model=None, device=None):
    save_model = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(save_model['model'])
    trained_epoch_load = save_model['epoch']
    # trained_epoch = state['epoch']
    print('model loaded from %s' % checkpoint_path)
    return_epoch = 0
    if not trained_epoch is None:
        return_epoch = trained_epoch
    else:
        return_epoch = trained_epoch_load

    return model, return_epoch


def output_result_analysis(result_dict, output_dir, test_path):
    test_dataset = test_path.split("/")[-2]
    if not test_dataset:
        test_dataset = "hahahah"
    csv_file = os.path.join(output_dir, test_dataset + "epoch" +str(args.epoch_for_save )+".csv")

    f = open(csv_file, "w")
    csv_writer_normal = csv.writer(f)
    # bias_lst = []
    for key, value in result_dict.items():

        freq_name = 'STI'
        csv_writer = csv_writer_normal
        csv_writer.writerow([0, freq_name])
        csv_writer.writerow([str(key)])

        for k in range(len(result_dict[key]["output_list"])):
            csv_writer.writerow(["output%d" % (k)] + result_dict[key]["output_list"][k].cpu().numpy().tolist())
        csv_writer.writerow(["mean_output"] + result_dict[key]["mean_output"].cpu().numpy().tolist())
        csv_writer.writerow(["gt"] + result_dict[key]["gt"].cpu().numpy().tolist())
        csv_writer.writerow(["mean_bias"] + result_dict[key]["mean_bias"].cpu().numpy().tolist())
        # bias_lst.extend(result_dict[key]["mean_bias"].cpu().numpy().tolist())
        csv_writer.writerow(["mean_mse"] + (result_dict[key]["mean_bias"] ** 2).cpu().numpy().tolist())
        csv_writer.writerow([" "])
        csv_writer.writerow([" "])


def test_net(net, epoch, val_loader, writer, batch_size):
    result_dict = dict()
    with torch.no_grad():
        net.eval()

        total_mean_loss = torch.zeros((1, args.ln_out))
        total_mean_bias = torch.zeros((1, args.ln_out))

        progress_bar = tqdm(val_loader)
        useless_count = 0
        for j, datas in enumerate(progress_bar):

            images_reshape = datas['image'].to(torch.float32).to(device)
            valid_len = datas['validlen']

            gt_sti = datas['sti'].reshape(-1, 1).to(torch.float32).to(device)  # [total_slices, 1]

            names = datas['name']

            output_pts, _ = net(images_reshape, valid_len)

            bias = gt_sti - output_pts
            rsquare_error = torch.sqrt(bias ** 2)

            if not torch.isnan(rsquare_error).all():
                total_mean_loss += torch.mean(torch.mean(rsquare_error, dim=0), dim=0).cpu().detach()
                total_mean_bias += torch.mean(torch.mean(bias, dim=0), dim=0).cpu().detach()

            else:
                useless_count += 1

            for i in range(len(valid_len)):
                start_num = 0
                if i > 0:
                    start_num = sum(valid_len[0:i])

                output_list = [output_pts[k] for k in range(start_num, start_num + valid_len[i])]
                mean_output = torch.mean(output_pts[start_num:start_num + valid_len[i]], dim=0)
                mean_bias = torch.mean(bias[start_num:start_num + valid_len[i]], dim=0)
                mean_rsquare_error = torch.mean(rsquare_error[start_num:start_num + valid_len[i]], dim=0)
                mean_gt = torch.mean(gt_sti[start_num:start_num + valid_len[i]], dim=0)
                result_dict[names[i][0]] = {"mean_output": mean_output, "mean_bias": mean_bias, "gt": mean_gt,
                                            "square_error": mean_rsquare_error, "output_list": output_list}

        mean_loss = total_mean_loss / (len(val_loader) - batch_size * useless_count)
        mean_bias = total_mean_bias / (len(val_loader) - batch_size * useless_count)
        if not writer is None:
            writer.add_scalar('val/mean_loss', mean_loss, epoch)
            writer.add_scalar('val/mean_bias', mean_bias, epoch)

        print("epoch:", args.epoch_for_save, "Mean loss:", mean_loss)
        print("epoch:", args.epoch_for_save, "Mean bias:", mean_bias)
        print("epoch:", args.epoch_for_save, "RMSE:", math.sqrt(mean_loss))
        # print("Epoch %d Evaluation: mean loss:%f,mean bias:%f" %(epoch,mean_loss,mean_bias))

        return result_dict


def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend(fontsize=20)


def get_boxplot(test_res):
    room_dict = {}

    for key, value in test_res.items():
        if '.pt' not in key:
            continue
        room_config, room_name = key.split('/')[-2:]
        temp_room = room_name.split(room_config)[1].strip('_')  # 'mine_site1_1way_bformat_2'

        if temp_room not in room_dict.keys():
            # then create a new room_dict
            room_dict[temp_room] = {'gt': value['gt'].tolist(), 'mean_output': value['mean_output'].tolist(),
                                    'mean_bias': value['mean_bias'].tolist()}

        else:
            room_dict[temp_room]['gt'].extend(value['gt'].tolist())
            room_dict[temp_room]['mean_output'].extend(value['mean_output'].tolist())
            room_dict[temp_room]['mean_bias'].extend(value['mean_bias'].tolist())

    """"aaa"""
    fig = plt.figure(figsize=(40, 20))
    ticks = list(room_dict.keys())
    for i in range(len(ticks)):
        if i // 2 == 0:
            ticks[i] = '\n' + ticks[i]
    gt_plot = plt.boxplot([v['gt'] for _, v in room_dict.items()],
                          positions=np.array(np.arange(len(room_dict.values()))) * 2.0-0.3, widths=0.3)
    mean_output_plot = plt.boxplot([v['mean_output'] for _, v in room_dict.items()],
                                   positions=np.array(np.arange(len(room_dict.values()))) * 2.0 + 0.3, widths=0.3)
    #mean_bias_plot = plt.boxplot([v['mean_bias'] for _, v in room_dict.items()],
    #                             positions=np.array(np.arange(len(room_dict.values()))) * 2.0 + 0.6, widths=0.3)

    # setting colors for each groups
    define_box_properties(gt_plot, 'black', 'gt')
    define_box_properties(mean_output_plot, '#D7191C', 'mean_output')
    # define_box_properties(mean_bias_plot, '#2C7BB6', 'mean_bias')

    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks, fontsize=12)
    plt.yticks(fontsize=20)
    # plt.tight_layout()
    # set the title
    # plt.xticks(rotation=-45)

    fig_name = 'Test_' + freq_name + '_epoch' + str(args.epoch_for_save)
    plt.title(fig_name, fontsize=30)  # 标题，并设定字号大小
    plt.show()
    plt.savefig(os.path.join(outputresult_dir,fig_name + '.png'), dpi=fig.dpi, pad_inches=4)
    
    #plt.savefig(outputresult_dir + '/' + fig_name + '.png', dpi=fig.dpi, pad_inches=4)



if __name__ == "__main__":

    args = parse_args()

    DEBUG = args.SERVER

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path
    net = FPN(num_blocks=[2, 4, 23, 3], num_classes=3, back_bone="resnet50", pretrained=False)

    start_time = time.time()
    net, trained_epoch = load_checkpoint(model_path, args.epoch_for_save, net, device)
    print('Successfully Loaded model: {}'.format(model_path))
    print('Finished Initialization in {:.3f}s!!!'.format(
        time.time() - start_time))

    net.to(device)

    # print(net)

    criterion = torch.nn.MSELoss()

    data_transform = transforms.Compose([transforms.Resize([224, 224])])

    # optimizer = optim.Adam([{'params':net.parameters(),'initial_lr':0.0001}], lr=0.0001,weight_decay=0.0001)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {"Total": total_num, "Trainable": trainable_num}


    print(get_parameter_number(net))

    if DEBUG == False:
        val_dict_root = "/Users/bajianxiang/Desktop/internship/filter_down_spec_dataset"
        val_batch_size = 3
        failed_file = "/Users/bajianxiang/Desktop/internship/filter_down_spec_dataset/koli-national-park-winter/koli-national-park-winter_koli_snow_site4_1way_bformat_1_koli-national-park-winter_东北话男声_1_TIMIT_a005_100_110_10dB-0.pt"
    else:
        val_dict_root = "/data/xbj/0929_STI_catTIMIT_withNOISE_DATA/val"
        failed_file = "/data/xbj/0929_STI_catTIMIT_withNOISE_DATA/val/Seven_Config/Seven_Config_junzheng310hui01_right_Seven_Config_DR4_FDKN0_TIMIT_S_1000dB-0.pt"
        val_batch_size = 10
        #failed_file = "/data/xbj/0902_1000hz_with_clean/val/creswell-crags/creswell-crags_1_s_mainlevel_r_mainlevel2_1_creswell-crags_东北话男声_1_TIMIT_a098_290_300_20dB-0.pt"


    val_transformed_dataset = Dataset_dict(root_dir=val_dict_root, transform=data_transform, start_freq=args.start_freq,
                                           end_freq=args.end_freq, failed_file=failed_file, random_choose_slice=False)

    print("len of val dataset:", len(val_transformed_dataset))

    # print('Number of images: ', len(transformed_dataset))

    if DEBUG == False:
        val_loader = torch.utils.data.DataLoader(val_transformed_dataset, shuffle=False, num_workers=1,
                                                 batch_size=val_batch_size, drop_last=True,
                                                 collate_fn=collate_fn)
    else:
        val_loader = torch.utils.data.DataLoader(val_transformed_dataset,
                                                 shuffle=True, num_workers=4,
                                                 batch_size=val_batch_size, drop_last=True, prefetch_factor=100,
                                                 collate_fn=collate_fn)
    print("after train loader init")

    test_output_result = test_net(net, trained_epoch, val_loader, None, val_batch_size)
    outputresult_dir = args.outputresult_dir
    if not os.path.exists(outputresult_dir):
        os.makedirs(outputresult_dir)

    freq_name = "STI"

    output_result_analysis(test_output_result, outputresult_dir, val_dict_root)
    get_boxplot(test_res=test_output_result)
    save_path = outputresult_dir + '/' + str(args.epoch_for_save) + '.pt'
    torch.save(test_output_result, save_path)

