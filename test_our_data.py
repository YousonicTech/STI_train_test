# -*- coding: utf-8 -*-
"""
@file      :  test_resnet50_500Hz.py
@Time      :  2022/8/20 17:54
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import math
import torchvision.transforms

# from workspace_utils import active_session
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')
import valdata_meant60
from valdata_meant60 import Val_meanT60
from torch.utils.tensorboard import SummaryWriter
# 决定使用哪块GPU进行训练
import csv
import os
import glob
from new_data_load_original_for_our_data import Dataset_dict, collate_fn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device_ids = [0, 1]
## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim
import datetime
from valdata_meant60 import ValDataset, Val_meanT60
import argparse
from model.FPN import FPN

torch.backends.cudnn.benchmark = True


def parse_args():
    # Test路径见下面 val_dict_root
    parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
    parser.add_argument('--trained_epoch', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='save_model/0914_1000hz_fpn_lre5/')
    # parser.add_argument('--model_path', type=str, default='/Users/bajianxiang/Desktop/internship/new_dataset_t60_attention/save_model/resnet_1000Hz/t60_predict_model_0_fullT60_rir_timit_noise.pt')
    parser.add_argument('--model_path', type=str, default='save_model/0914_1000hz_fpn_lre5/')
    parser.add_argument('--epoch_for_save', type=int, default=64)  # 记得改epoch数
    parser.add_argument('--load_pretrain', type=bool, default=True)
    # 125Hz:(0, 7); 250Hz:(1, 10); 500Hz:(2, 13); 1kHz:(3, 16); 2kHz:(4, 19)
    parser.add_argument('--which_freq', type=int, default=3)   # 记得改成对应的图编号
    parser.add_argument('--freq_loc', type=int, default=16)    # 记得改成对应的倍频程
    parser.add_argument('--start_freq', type=int, default=0)
    parser.add_argument('--end_freq', type=int, default=29)
    parser.add_argument('--ln_out', type=int, default=1)
    parser.add_argument('--SERVER', type=bool, default=True)  # 在服务器上运行记得改成True
    parser.add_argument('--outputresult_dir', type=str, default="./0921_test/epoch64") # 记得改输出路径，不用带/应该

    # parser.add_argument('--epoch',type = int,default = 170)
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


def net_sample_output():
    for i, sample in enumerate(val_loader):

        images = sample['image']
        ddr = sample['ddr']
        t60 = sample['t60']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = torch.tensor(images.clone().detach(), dtype=torch.float32, device=device)

        # forward pass to get net output
        t60_predict = net(images)

        # break after first image is tested
        if i == 0:
            return images, t60_predict, t60


def val_net(net, epoch, val_loader, writer):
    with torch.no_grad():
        net.eval()
        total_mean_loss = 0
        total_mean_bias = 0
        progress_bar = tqdm(val_loader)
        for j, datas in enumerate(progress_bar):
            images_reshape = datas['image'].to(torch.float32).to(device)  # 服务器上这个to(device)改了，得在CPU上运算
            valid_len = datas['validlen']
            gt_t60_reshape = datas['t60'].to(torch.float32).to(device).unsqueeze(1)

            gt_t60_reshape = gt_t60_reshape.T[args.freq_loc].T  # [total_slices, 1]
            output_pts = net(images_reshape)

            loss = criterion(output_pts, gt_t60_reshape)
            bias = torch.sum((gt_t60_reshape - output_pts)) / output_pts.shape[0]

            total_mean_loss += loss.item()
            total_mean_bias += bias.item()

        mean_loss = total_mean_loss / len(val_loader)
        mean_bias = float(total_mean_bias) / len(val_loader)
        if not writer is None:
            writer.add_scalar('val/mean_loss', mean_loss, epoch)
            writer.add_scalar('val/mean_bias', mean_bias, epoch)

        print("Epoch %d Evaluation: mean loss:%f,mean bias:%f" % (epoch, mean_loss, mean_bias))


def output_result_analysis(result_dict, output_dir, test_path):
    test_dataset = test_path.split("/")[-2]
    if not test_dataset:
        test_dataset = "hahahah"
    csv_file = os.path.join(output_dir, test_dataset + ".csv")

    f = open(csv_file, "w")
    csv_writer_normal = csv.writer(f)
    # bias_lst = []
    for key, value in result_dict.items():

        freq_name = 125 * 2 ** args.which_freq
        if freq_name < 1000:
            freq_name = str(freq_name) + 'Hz'
        else:
            freq_name = str(int(freq_name / 1000)) + 'k Hz'
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

            gt_t60_reshape = datas['t60'].to(torch.float32).to(device).unsqueeze(1)

            valid_len = datas['validlen']
            names = datas['name']

            gt_t60_reshape = gt_t60_reshape.T[args.freq_loc].T  # [total_slices, 1]
            output_pts, _ = net(images_reshape)

            bias = gt_t60_reshape - output_pts
            rsquare_error = torch.sqrt(bias ** 2)
            abs_bias = torch.abs(gt_t60_reshape - output_pts)

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
                mean_abs_bias = torch.mean(abs_bias[start_num:start_num + valid_len[i]], dim=0)
                mean_bias = torch.mean(bias[start_num:start_num + valid_len[i]], dim=0)
                mean_rsquare_error = torch.mean(rsquare_error[start_num:start_num + valid_len[i]], dim=0)
                mean_gt = torch.mean(gt_t60_reshape[start_num:start_num + valid_len[i]], dim=0)
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


class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.warmup = False
        self.warmup_epochs = 10
        self.min_lr = min(lr)
        self.max_lr = max(lr)
        # self.lr = lr
        # self.lr_epochs = lr_epochs

        # self.lr_warmup = lambda epoch_in : min(self.lr)+0.5*(max(self.lr) - min(self.lr))*(1+np.cos((epoch_in-self.warmup_epochs)*PI/(2*self.warmup_epochs)))
        self.lr_warmup = lambda epoch_in: self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + np.cos((epoch_in - self.warmup_epochs) * PI / (2 * self.warmup_epochs)))
        if self.warmup == True:
            self.lr_epochs = [self.warmup_epochs] + [i + self.warmup_epochs for i in lr_epochs]
            self.lr = [self.lr_warmup] + lr
        else:
            self.lr_epochs = lr_epochs
            self.lr = lr

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        if self.warmup == True and idx == 0:
            return self.lr[idx](epoch)
        else:
            return self.lr[idx]

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
        # print("config:", room_config, "--name:", room_name)
        temp_room = room_name.strip('.pt')  # 'mine_site1_1way_bformat_2'

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
    plt.savefig(outputresult_dir + '/' + fig_name + '.png', dpi=fig.dpi, pad_inches=4)



if __name__ == "__main__":

    args = parse_args()

    DEBUG = args.SERVER
    LOAD_PRETRAIN = args.load_pretrain

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path + 't60_predict_model_' + str(args.epoch_for_save) + '_fullT60_rir_timit_noise.pt'
    net = FPN(num_blocks=[2, 4, 23, 3], num_classes=3, back_bone="resnet50", pretrained=False)

    if LOAD_PRETRAIN == True:
        start_time = time.time()
        net, trained_epoch = load_checkpoint(model_path, 99, net, device)
        print('Successfully Loaded model: {}'.format(model_path))
        print('Finished Initialization in {:.3f}s!!!'.format(
            time.time() - start_time))
    else:
        trained_epoch = 0
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
        val_dict_root = "/data/xbj/0921_OUR_DATA_FOR_TEST/Test_pt"
        failed_file = "/data/xbj/0921_OUR_DATA_FOR_TEST/Test_pt/Junzhen/junzheng3dahui02_left-0.pt"
        val_batch_size = 1
        #failed_file = "/data/xbj/0902_1000hz_with_clean/val/creswell-crags/creswell-crags_1_s_mainlevel_r_mainlevel2_1_creswell-crags_东北话男声_1_TIMIT_a098_290_300_20dB-0.pt"


    val_transformed_dataset = Dataset_dict(root_dir=val_dict_root, transform=data_transform, start_freq=args.start_freq,
                                           end_freq=args.end_freq, which_freq=args.which_freq, failed_file=failed_file, random_choose_slice=False)

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
    trained_epoch = args.trained_epoch
    # train_net(trained_epoch,n_epochs, train_loader,val_loader,batch_size,args)
    test_output_result = test_net(net, trained_epoch, val_loader, None, val_batch_size)
    outputresult_dir = args.outputresult_dir
    if not os.path.exists(outputresult_dir):
        os.makedirs(outputresult_dir)

    freq_name = 125 * 2 ** args.which_freq
    if freq_name < 1000:
        freq_name = str(freq_name) + 'Hz'
    else:
        freq_name = str(int(freq_name / 1000)) + 'kHz'

    output_result_analysis(test_output_result, outputresult_dir, val_dict_root)
    get_boxplot(test_res=test_output_result)
    save_path = outputresult_dir + '/' + str(args.epoch_for_save) + '.pt'
    torch.save(test_output_result, save_path)

