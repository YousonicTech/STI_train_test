# -*- coding: utf-8 -*-
"""
@file      :  train_resnet50_500Hz.py
@Time      :  2022/8/19 15:27
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""

# CUDA_VISIBLE_DEVICES="2" python xxxx.py

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import torchvision.transforms
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy('file_system')
import valdata_meant60
from valdata_meant60 import Val_meanT60
from torch.utils.tensorboard import SummaryWriter
import os
import glob
from new_data_load_original import Dataset_dict, collate_fn
from SSIMLoss import ssim

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model.swinTrans import FPN
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import Rescale, RandomCrop
from torchvision.transforms import Normalize, ToTensor
import torch.optim as optim
import datetime
from valdata_meant60 import ValDataset, Val_meanT60
import argparse

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Whatever')
    parser.add_argument('--trained_epoch', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='save_model/0923_STI_FPN_lre5_alpha1_beta05')  # 需要改一下倍频程
    parser.add_argument('--model_path', type=str, default='save_model/')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--ln_out', type=int, default=1)
    parser.add_argument('--start_freq', type=int, default=0)
    parser.add_argument('--end_freq', type=int, default=29)
    parser.add_argument('--freq_loc', type=int, default=10)  # STI_gt所在的倍频程位置
    parser.add_argument('--alpha', type=float, default=1)  # MSE loss
    parser.add_argument('--beta', type=float, default=0.5)  # SSIM Loss

    parser.add_argument('--SERVER', type=bool, default=True)  # 在服务器上运行记得改成True

    # parser.add_argument('--which_freq', type=int, default=4)  # 换倍频程
    args = parser.parse_args()
    return args


def load_checkpoint(checkpoint_path=None, trained_epoch=None, model=None, device=None):
    save_model = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(save_model['model'])
    trained_epoch_load = save_model['epoch']
    print('model loaded from %s' % checkpoint_path)
    if not trained_epoch is None:
        return_epoch = trained_epoch
    else:
        return_epoch = trained_epoch_load
    return model, return_epoch


def net_sample_output():
    for i, sample in enumerate(val_loader):

        images = sample['image']
        t60 = sample['t60']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        images = torch.tensor(images.clone().detach(), dtype=torch.float32, device=device)
        t60_predict = net(images)
        if i == 0:
            return images, t60_predict, t60


def preprocess_img(image):
    # input = [total_slice, 1, 100, 1000]
    lst = []
    for i in range(image.shape[0]):
        temp_img = image[i]  # [1, 100, 1000]
        temp_img = torch.from_numpy(cv2.resize(temp_img.squeeze().numpy(), (224, 224)))  # [224,224]
        temp_img = torch.cat([temp_img.unsqueeze(0)] * 3, dim=0)  # [3, 224, 224]
        lst.append(temp_img.unsqueeze(0))  # [1, 3, 224, 224]
    return torch.cat(lst, dim=0)


def val_net(net, epoch, val_loader, writer):
    with torch.no_grad():
        net.eval()
        total_mse_loss = 0
        total_ssim_loss = 0
        total_mean_loss = 0
        total_mean_bias = 0
        progress_bar = tqdm(val_loader)
        for j, datas in enumerate(progress_bar):
            images_reshape = datas['image'].to(torch.float32).to(device)
            valid_len = datas['validlen']
            gt_t60_reshape = datas['t60'].to(torch.float32).to(device).unsqueeze(1)
            clean_reshape = datas['clean'].to(torch.float32).to(device)
            gt_t60_reshape = gt_t60_reshape.T[args.freq_loc].T  # [total_slices, 1]

            output_pts, dereverb_out = net(images_reshape)

            mse_loss = criterion(output_pts, gt_t60_reshape)
            ssim_loss = (1 - ssim(dereverb_out, clean_reshape))
            loss = args.alpha * mse_loss + args.beta * ssim_loss

            bias = torch.sum((gt_t60_reshape - output_pts)) / output_pts.shape[0]

            total_mean_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_ssim_loss += ssim_loss.item()
            total_mean_bias += bias.item()

        mean_loss = total_mean_loss / len(val_loader)
        mean_bias = float(total_mean_bias) / len(val_loader)
        mean_mse_loss = total_mse_loss / len(val_loader)
        mean_ssim_loss = total_ssim_loss / len(val_loader)
        writer.add_scalar('val/mean_loss', mean_loss, epoch)
        writer.add_scalar('val/mean_bias', mean_bias, epoch)
        writer.add_scalar('val/mean_mse_loss', mean_mse_loss, epoch)
        writer.add_scalar('val/mean_ssim_loss', mean_ssim_loss, epoch)

        print("Epoch %d Evaluation: mean loss:%f,mean bias:%f" % (epoch, mean_loss, mean_bias))


def train_net(start_epoch, n_epochs, train_loader, val_loader, batch_size, args):
    model_dir = args.save_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = torch.optim.lr_scheduler.StepLR(optimizer,
                                         step_size=25,
                                         gamma=0.5, last_epoch=start_epoch)

    print("lr at beginning:", lr.get_last_lr())
    net.train()
    lr_list = list()
    print("the training process from epoch{}...".format(start_epoch))
    dt = datetime.datetime.now()

    log_dir = os.path.join("./log", dt.strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    for epoch in range(start_epoch, n_epochs):
        print("Training:epoch ", epoch, "lr:", lr.get_last_lr()[-1])
        lr_list.append(lr.get_last_lr())
        total_mse_loss = 0
        total_ssim_loss = 0
        total_mean_loss = 0
        total_mean_bias = 0
        progress_bar = tqdm(train_loader)

        for batch_i, datas in enumerate(progress_bar):
            # images_reshape = [batch, 3, 224, 224]
            images_reshape = datas['image'].to(torch.float32).to(device)
            gt_t60_reshape = datas['t60'].to(torch.float32).to(device).unsqueeze(1)
            clean_reshape = datas['clean'].to(torch.float32).to(device)
            # gt_t60_reshape / output_pts = [28, 1]
            gt_t60_reshape = gt_t60_reshape.T[args.freq_loc].T  # [total_slices, 1]

            output_pts, dereverb_out = net(images_reshape)  # [total_slice, 1], [total_slice, 3, 224, 224]

            mse_loss = criterion(output_pts, gt_t60_reshape)
            ssim_loss = (1 - ssim(dereverb_out, clean_reshape))

            loss = args.aalpha * mse_loss + args.beta * ssim_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=0.1, norm_type=2)

            optimizer.step()
            bias = torch.sum((gt_t60_reshape - output_pts)) / output_pts.shape[0]  # /sum(valid_len)
            total_mean_loss = total_mean_loss + loss.item()
            total_mean_bias = total_mean_bias + bias.item()
            total_mse_loss += mse_loss.item()
            total_ssim_loss += ssim_loss.item()

        mean_loss = total_mean_loss / len(train_loader)
        mean_bias = total_mean_bias / len(train_loader)
        mean_mse_loss = total_mse_loss / len(train_loader)
        mean_ssim_loss = total_ssim_loss / len(train_loader)
        writer.add_scalar('train/mean_loss', mean_loss, epoch)
        writer.add_scalar('train/mean_mse_loss', mean_mse_loss, epoch)
        writer.add_scalar('train/mean_ssim_loss', mean_ssim_loss, epoch)
        writer.add_scalar('train/mean_bias', mean_bias, epoch)
        writer.add_scalar('lr/lr', lr.get_last_lr()[-1], epoch)

        print('in case no t60, last t60_gt is:', gt_t60_reshape)
        print("In training, epoch {},mse is {},bias is {}".format(epoch, mean_loss, mean_bias))
        lr.step()

        if epoch % 5 == 0:
            print("eval")
            val_net(net, epoch, val_loader, writer)
            net.train()

        if True:
            # './Checkpoints/rir_timit_noise_new_0509_alldata/'
            model_name = 't60_predict_model_%d_fullT60_rir_timit_noise.pt' % (epoch)

            # after training, save your model parameters in the dir 'saved_models'
            state = {"model": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                     "lr": lr.state_dict()}
            torch.save(state, os.path.join(model_dir, model_name))
            print('Finished Training')

    writer.close()


class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.warmup = False
        self.warmup_epochs = 10
        self.min_lr = min(lr)
        self.max_lr = max(lr)
        PI = 3.14159
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


if __name__ == "__main__":

    args = parse_args()
    DEBUG = args.SERVER
    LOAD_PRETRAIN = args.load_pretrain  # False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path
    net = FPN(num_blocks=[2, 4, 23, 3], num_classes=3, back_bone="resnet50", pretrained=False)

    ### 这个判断的作用，用来计时
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
    optimizer = optim.Adam([{'params': net.parameters(), 'initial_lr': 1e-5}], lr=1e-5, weight_decay=0.000001)
    n_epochs = 100
    data_transform = transforms.Compose([transforms.Resize([256, 256])])


    # 网络参数数量的作用
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {"Total": total_num, "Trainable": trainable_num}


    print(get_parameter_number(net))

    if DEBUG == False:
        train_dict_root = "/Users/bajianxiang/Desktop/internship/STI_dataset"
        val_dict_root = "/Users/bajianxiang/Desktop/internship/STI_dataset"
        batch_size = 3
        val_batch_size = 3
        failed_file = "/Users/bajianxiang/Desktop/internship/STI_dataset/Nine_Config_Amaranth_left_Nine_Config_DR4_FKDW0_SI1891_TIMIT_S_1000dB-0.pt"
    else:
        train_dict_root = "/data/xbj/0922_STI_FPN/train"
        # train_dict_root = "/mnt/sda/xbj/0815_GEN_DATASET/train"
        val_dict_root = "/data/xbj/0922_STI_FPN/val"
        batch_size = 7
        val_batch_size = 7
        failed_file = "/data/xbj/0922_STI_FPN/train/Nine_Config/Nine_Config_Amaranth_left_Nine_Config_DR4_FJXP0_SI1752_TIMIT_S_1000dB-0.pt"

    print("train_dir:", train_dict_root)

    train_transformed_dataset = Dataset_dict(root_dir=train_dict_root, transform=data_transform,
                                             start_freq=args.start_freq, end_freq=args.end_freq,
                                             failed_file=failed_file, random_choose_slice=False)

    print("len of train dataset:", len(train_transformed_dataset))

    val_transformed_dataset = Dataset_dict(root_dir=val_dict_root, transform=data_transform, start_freq=args.start_freq,
                                           end_freq=args.end_freq, failed_file=failed_file, random_choose_slice=False)

    print("len of val dataset:", len(val_transformed_dataset))

    if DEBUG == False:
        train_loader = torch.utils.data.DataLoader(train_transformed_dataset,
                                                   shuffle=False, num_workers=1,
                                                   batch_size=batch_size, drop_last=True,
                                                   collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_transformed_dataset, shuffle=False, num_workers=1,
                                                 batch_size=val_batch_size, drop_last=True,
                                                 collate_fn=collate_fn)
    else:
        train_loader = torch.utils.data.DataLoader(train_transformed_dataset,
                                                   shuffle=True, num_workers=6,
                                                   batch_size=batch_size, drop_last=True,  # prefetch_factor=batch_size,
                                                   collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_transformed_dataset,
                                                 shuffle=False, num_workers=1,
                                                 batch_size=val_batch_size, drop_last=True, prefetch_factor=100,
                                                 collate_fn=collate_fn)
    print("after train loader init")
    trained_epoch = args.trained_epoch
    train_net(trained_epoch, n_epochs, train_loader, val_loader, batch_size, args)

    # TODO 训练时要改dict，权重名字，验证时保存的文件名和路径
